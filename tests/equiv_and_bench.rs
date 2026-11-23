#![forbid(unsafe_code)]

use zero_cc::{
    compile_kernel, compile_rust_source, emit_ntf1, export_kernel_ntf1, optimize_kernel,
    parse_ntf1, parse_rust_kernel, verify_ntf1_airlock, BufAccess, CompileError, Domain, Kernel,
    Ntf1Error, Op, RegBudget, LANES,
};

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Deterministic Zipfian (s = 1.2) sampler mapped onto `[lo, hi]` with edge values included.
struct ZipfSampler {
    cdf: Vec<f64>,
    rng: SplitMix64,
}

impl ZipfSampler {
    fn new(support: usize, s: f64, seed: u64) -> Self {
        let mut cdf = Vec::with_capacity(support);
        let mut acc = 0.0f64;
        for k in 1..=support {
            acc += (k as f64).powf(-s);
            cdf.push(acc);
        }
        for v in &mut cdf {
            *v /= acc;
        }
        Self { cdf, rng: SplitMix64(seed) }
    }

    fn sample_range(&mut self, lo: i64, hi: i64) -> i64 {
        let u = (self.rng.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        let rank = self.cdf.partition_point(|&p| p < u);
        let span = (hi - lo) as u64;
        if span == 0 {
            return lo;
        }
        // Mix Zipfian hot ranks with exact boundary/modular edge stress points.
        let val = match rank {
            0 => lo,
            1 => hi,
            2 => lo + (span.min(250)) as i64,
            3 => lo + (span.min(251)) as i64,
            4 => hi - (span.min(1)) as i64,
            _ => {
                let hashed = ((rank as u64).wrapping_mul(2654435761)) % (span + 1);
                lo + hashed as i64
            }
        };
        if val < lo { lo } else if val > hi { hi } else { val }
    }
}

/// Kernel 1: Modular arithmetic + LUT + `noalias` store-forwarding + load-hoisting + bounds check.
fn build_modular_hash_lut() -> Kernel {
    let mut k = Kernel::new("modular_hash_lut");
    let a = k.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 65535));
    let b = k.add_buf("b", BufAccess::ReadOnly, Domain::range(0, 65535));
    let lut = k.add_buf("lut", BufAccess::LookupTable(8), Domain::range(0, 1023));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());

    let mut mem = k.push(Op::MemInit);
    let c17 = k.push(Op::Const(17));
    let c8 = k.push(Op::Const(8));
    let c251 = k.push(Op::Const(251));
    let c64 = k.push(Op::Const(64));

    for lane in 0..LANES as u8 {
        let ai = k.push(Op::Load { mem, buf: a, lane });
        let bi = k.push(Op::Load { mem, buf: b, lane });
        let a17 = k.push(Op::Mul(ai, c17));
        let b17 = k.push(Op::Mul(bi, c17));
        let sum = k.push(Op::Add(a17, b17));
        let s = k.push(Op::Div(sum, c8));
        let m = k.push(Op::Rem(s, c251));
        let raw_idx = k.push(Op::Rem(m, c8));
        let idx = k.push(Op::BoundsCheck { idx: raw_idx, limit: 8 });
        let lv = k.push(Op::LoadLut { buf: lut, idx });
        mem = k.push(Op::Store { mem, buf: out, lane, val: m });
        let m_reload = k.push(Op::Load { mem, buf: out, lane });
        let a_reload = k.push(Op::Load { mem, buf: a, lane });
        let a_mod = k.push(Op::Rem(a_reload, c64));
        let p1 = k.push(Op::Add(m_reload, lv));
        let res = k.push(Op::Add(p1, a_mod));
        k.add_output(out, lane, res);
    }
    k
}

/// Kernel 2: 8-lane cross-lane radix-2 butterfly with shift fusion + modular congruence zeroing.
fn build_butterfly_8lane() -> Kernel {
    let mut k = Kernel::new("butterfly_8lane");
    let x = k.add_buf("x", BufAccess::ReadOnly, Domain::range(0, 16383));
    let y = k.add_buf("y", BufAccess::ReadOnly, Domain::range(0, 16383));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());

    let mem = k.push(Op::MemInit);
    let c16 = k.push(Op::Const(16));
    let c15 = k.push(Op::Const(15));
    let c8 = k.push(Op::Const(8));
    let c4 = k.push(Op::Const(4));

    for pair in 0..4u8 {
        let lo = pair;
        let hi = pair + 4;
        let xlo = k.push(Op::Load { mem, buf: x, lane: lo });
        let ylo = k.push(Op::Load { mem, buf: y, lane: lo });
        let xhi = k.push(Op::Load { mem, buf: x, lane: hi });
        let yhi = k.push(Op::Load { mem, buf: y, lane: hi });

        let xl16 = k.push(Op::Mul(xlo, c16));
        let yl16 = k.push(Op::Mul(ylo, c16));
        let u_num = k.push(Op::Add(xl16, yl16));
        let u = k.push(Op::Div(u_num, c4));

        let xh15 = k.push(Op::Mul(xhi, c15));
        let yh15 = k.push(Op::Mul(yhi, c15));
        let v = k.push(Op::Sub(xh15, yh15));

        let x8 = k.push(Op::Mul(xlo, c8));
        let zero_rem = k.push(Op::Rem(x8, c8));
        let sum = k.push(Op::Add(u, v));
        let out_lo = k.push(Op::Add(sum, zero_rem));
        let out_hi = k.push(Op::Sub(u, v));
        k.add_output(out, lo, out_lo);
        k.add_output(out, hi, out_hi);
    }
    k
}

/// Kernel 3: 8-lane affine filter + store-forwarding + load-hoisting + Barrett `% 17`.
fn build_affine_filter_8lane() -> Kernel {
    let mut k = Kernel::new("affine_filter_8lane");
    let sig = k.add_buf("sig", BufAccess::ReadOnly, Domain::range(0, 32767));
    let bias = k.add_buf("bias", BufAccess::ReadOnly, Domain::range(0, 4095));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());

    let mut mem = k.push(Op::MemInit);
    let c9 = k.push(Op::Const(9));
    let c7 = k.push(Op::Const(7));
    let c16 = k.push(Op::Const(16));
    let c17 = k.push(Op::Const(17));

    for lane in 0..LANES as u8 {
        let s = k.push(Op::Load { mem, buf: sig, lane });
        let b = k.push(Op::Load { mem, buf: bias, lane });
        let s9 = k.push(Op::Mul(s, c9));
        let b9 = k.push(Op::Mul(b, c9));
        let ab = k.push(Op::Add(s9, b9));
        mem = k.push(Op::Store { mem, buf: out, lane, val: ab });
        let t = k.push(Op::Load { mem, buf: out, lane });
        let s2 = k.push(Op::Load { mem, buf: sig, lane });
        let s7 = k.push(Op::Mul(s2, c7));
        let num = k.push(Op::Add(t, s7));
        let q = k.push(Op::Div(num, c16));
        let r = k.push(Op::Rem(q, c17));
        k.add_output(out, lane, r);
    }
    k
}

/// Kernel 4: 8-lane CRT + quadratic congruence lifting (`x == 2 (mod 4)` -> `(x*x) % 16 == 4`,
/// `((2*x + 1) * 3) % 8 == 7`), eliminating both `Rem` and `BoundsCheck` cleanly.
fn build_quad_crt_8lane(use_congruence: bool) -> Kernel {
    let mut k = Kernel::new("quad_crt_8lane");
    let x_dom = if use_congruence {
        Domain::congruent(0, 1000, 4, 2)
    } else {
        Domain::range(0, 1000)
    };
    let x = k.add_buf("x", BufAccess::ReadOnly, x_dom);
    let lut = k.add_buf("lut", BufAccess::LookupTable(8), Domain::range(0, 1023));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());

    let mem = k.push(Op::MemInit);
    let c2 = k.push(Op::Const(2));
    let c1 = k.push(Op::Const(1));
    let c3 = k.push(Op::Const(3));
    let c8 = k.push(Op::Const(8));
    let c16 = k.push(Op::Const(16));

    for lane in 0..LANES as u8 {
        let xi = k.push(Op::Load { mem, buf: x, lane });
        let sq = k.push(Op::Mul(xi, xi));
        let r16 = k.push(Op::Rem(sq, c16));
        let idx = k.push(Op::BoundsCheck { idx: r16, limit: 8 });
        let lv = k.push(Op::LoadLut { buf: lut, idx });
        let x2 = k.push(Op::Mul(xi, c2));
        let odd = k.push(Op::Add(x2, c1));
        let odd3 = k.push(Op::Mul(odd, c3));
        let r8 = k.push(Op::Rem(odd3, c8));
        let p1 = k.push(Op::Add(lv, r8));
        let res = k.push(Op::Add(p1, odd3));
        k.add_output(out, lane, res);
    }
    k
}

#[test]
fn crt_quadratic_congruence_knockout() {
    // 1. Exact CRT/LCM meet and tightening: x == 2 (mod 3) /\ x == 3 (mod 5) => x == 8 (mod 15).
    let d3 = Domain::congruent(0, 100, 3, 2);
    let d5 = Domain::congruent(0, 100, 5, 3);
    let crt = d3.try_meet(d5).expect("consistent CRT");
    assert_eq!((crt.lo, crt.hi, crt.modulus, crt.rem), (8, 98, 15, 8));
    assert!(crt.contains(8) && crt.contains(23) && !crt.contains(9));

    // Contradictory CRT meet: x == 1 (mod 2) /\ x == 0 (mod 2) => empty bottom.
    let odd = Domain::congruent(0, 100, 2, 1);
    let even = Domain::congruent(0, 100, 2, 0);
    assert_eq!(odd.try_meet(even), None);
    let bot = odd.meet(even);
    assert!(bot.is_empty());
    assert!(!bot.is_nonneg());
    assert!(!bot.contains(1));

    // 2. Quadratic residue + affine congruence lifting: (2x + 1)^2 == 1 (mod 8), (4k + 2)^2 == 4 (mod 32).
    let odd_sq = odd.eval_square();
    assert_eq!(odd_sq.rem_mod(8), Some(1));
    let k_cong = build_quad_crt_8lane(true);
    let k_ablate = build_quad_crt_8lane(false);
    let c_cong = compile_kernel(&k_cong, RegBudget::default()).expect("compile congruent quad_crt");
    let c_ablate = compile_kernel(&k_ablate, RegBudget::default()).expect("compile ablated quad_crt");

    // Counterfactual knockout: congruence eliminates all Rem + BoundsCheck; ablating keeps BoundsCheck.
    assert_eq!(c_cong.optimized_cost.divs_rems, 0);
    assert_eq!(c_cong.optimized_cost.bounds_checks, 0);
    assert_ne!(c_cong.optimized_cost.bounds_checks, c_ablate.optimized_cost.bounds_checks);
    assert_ne!(c_cong.optimized_cost.total_latency, c_ablate.optimized_cost.total_latency);
    assert_eq!(c_ablate.optimized_cost.bounds_checks, LANES);

    // Bit-exact verification across valid x == 2 (mod 4) inputs + OOB fault on ablated kernel for x = 3.
    let lut = [11i64, 23, 37, 53, 79, 97, 131, 199];
    for step in 0..250i64 {
        let mut xs = [0i64; LANES];
        for (l, slot) in xs.iter_mut().enumerate() {
            *slot = ((step * 4 + (l as i64) * 12) % 996) + 2;
        }
        let out_z = [0i64; LANES];
        let refs: &[&[i64]] = &[&xs, &lut, &out_z];
        assert_eq!(k_cong.eval(refs).expect("orig"), c_cong.optimized.eval(refs).expect("opt"));
    }
    let bad_x = [3i64; LANES];
    let out_z = [0i64; LANES];
    assert_eq!(
        c_ablate.optimized.eval(&[&bad_x, &lut, &out_z]).unwrap_err(),
        CompileError::OutOfBounds { idx: 9, limit: 8 }
    );

    // Idempotent optimization fixed-point: optimize_kernel(optimize_kernel(k)) == optimize_kernel(k).
    let reopt = optimize_kernel(&c_cong.optimized).expect("reopt quad_crt");
    assert_eq!(reopt.cost(), c_cong.optimized_cost);
    assert_eq!(reopt.nodes, c_cong.optimized.nodes);
}

#[test]
fn ntf1_airlock_and_recursive_cross_verify() {
    use std::path::Path;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let spec_path = root.join("specs/zero_cc.ntf1");
    let spec_src = std::fs::read_to_string(&spec_path).expect("read specs/zero_cc.ntf1");
    let canon = verify_ntf1_airlock(&spec_src).expect("valid specs/zero_cc.ntf1");
    let reparsed = parse_ntf1(&canon).expect("reparse canonical");
    assert_eq!(emit_ntf1(&reparsed), canon, "0t emit/parse fixed-point");

    // Counterfactual knockouts on NTF1: SVO prose in body and comment, uncovered Fault.*, ungrounded 0t, broken 0t arrow.
    let svo_body = "NTF1 bad\n+t the -> Kernel\n";
    assert_ne!(parse_ntf1(&spec_src).is_ok(), parse_ntf1(svo_body).is_ok());
    assert!(matches!(parse_ntf1(svo_body), Err(Ntf1Error::SvoTax { ref word, .. }) if word == "the"));

    let svo_comment = "NTF1 bad\n# this comment leaks prose\n+t A -> B\n";
    assert!(matches!(parse_ntf1(svo_comment), Err(Ntf1Error::SvoTax { ref word, .. }) if word == "this"));

    let uncovered_fault = "NTF1 bad\n+t IR -> Fault.Orphan\n";
    assert!(matches!(parse_ntf1(uncovered_fault), Err(Ntf1Error::UncoveredFault(ref a)) if a == "Fault.Orphan"));

    let ungrounded_airlock = "NTF1 bad\n0t GhostA >< GhostB\n";
    assert!(matches!(parse_ntf1(ungrounded_airlock), Err(Ntf1Error::UngroundedAirlock(_))));

    let broken_airlock = "NTF1 bad\n+t A -> B\n0t A -> B\n";
    assert_ne!(verify_ntf1_airlock(&canon).is_ok(), verify_ntf1_airlock(broken_airlock).is_ok());

    // Export all 4 compiled kernels to NTF1 and verify airlock fixed-point + optimize idempotence.
    let all_kernels = [
        build_modular_hash_lut(),
        build_butterfly_8lane(),
        build_affine_filter_8lane(),
        build_quad_crt_8lane(true),
    ];
    let mut emitted_rs = String::from("#![forbid(unsafe_code)]\n\n");
    let mut last_k_ntf1 = String::new();
    for k in &all_kernels {
        let compiled = compile_kernel(k, RegBudget::default()).expect("compile kernel");
        let reopt = optimize_kernel(&compiled.optimized).expect("reopt");
        assert_eq!(reopt.nodes, compiled.optimized.nodes);
        let k_ntf1 = export_kernel_ntf1(&compiled);
        let k_canon = verify_ntf1_airlock(&k_ntf1).expect("kernel NTF1 airlock");
        assert_eq!(k_canon, k_ntf1);
        emitted_rs.push_str(&compiled.rust_source);
        emitted_rs.push('\n');
        last_k_ntf1 = k_ntf1;
    }

    // Sibling mutual NTF1 cross-check when ../zero-check exists (zero Cargo.toml dep).
    if let Some(parent) = root.parent() {
        let sibling_spec = parent.join("zero-check/specs/zero_check.ntf1");
        if sibling_spec.exists() {
            let sib_src = std::fs::read_to_string(&sibling_spec).expect("read zero_check.ntf1");
            let sib_canon = verify_ntf1_airlock(&sib_src).expect("zero-cc parses zero_check.ntf1");
            assert_eq!(verify_ntf1_airlock(&sib_canon).expect("idempotent sibling"), sib_canon);
        }
        #[cfg(not(miri))]
        {
            let sibling_bin = parent.join("zero-check/target/release/zero-check");
            if sibling_bin.exists() {
                let tmp = std::env::temp_dir();
                let rs_tmp = tmp.join("zero_cc_emitted_cross.rs");
                let ntf1_tmp = tmp.join("zero_cc_kernel_cross.ntf1");
                std::fs::write(&rs_tmp, &emitted_rs).expect("write emitted rs");
                std::fs::write(&ntf1_tmp, &last_k_ntf1).expect("write kernel ntf1");
                let out = std::process::Command::new(&sibling_bin)
                    .args(["--no-ledger", "--ntf1"])
                    .arg(&spec_path)
                    .arg(&ntf1_tmp)
                    .arg(&rs_tmp)
                    .output()
                    .expect("invoke sibling zero-check");
                let _ = std::fs::remove_file(&rs_tmp);
                let _ = std::fs::remove_file(&ntf1_tmp);
                assert!(
                    out.status.success(),
                    "zero-check rejected zero-cc artifacts:\n{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }
}

#[test]
fn zipfian_equivalence_and_cost_knockout() {
    let kernels = [
        build_modular_hash_lut(),
        build_butterfly_8lane(),
        build_affine_filter_8lane(),
    ];
    let samples_per_kernel = if cfg!(miri) { 128 } else { 4_000 };
    let mut zipf = ZipfSampler::new(1024, 1.2, 0x51a7_c0de_2026_beef);

    for (ki, orig) in kernels.iter().enumerate() {
        let compiled = compile_kernel(orig, RegBudget::default()).expect("compile kernel");
        // Counterfactual knockout: optimization must genuinely slash cost, bounds checks, and div/rem.
        assert_ne!(compiled.original_cost, compiled.optimized_cost);
        assert!(
            compiled.optimized_cost.total_latency < compiled.original_cost.total_latency,
            "kernel {} failed latency reduction: {:?} -> {:?}",
            orig.name, compiled.original_cost, compiled.optimized_cost
        );
        assert_eq!(compiled.optimized_cost.bounds_checks, 0, "kernel {} kept bounds check", orig.name);
        assert_eq!(compiled.optimized_cost.divs_rems, 0, "kernel {} kept div/rem", orig.name);
        if ki == 0 || ki == 2 {
            assert!(
                compiled.optimized_cost.mem_ops < compiled.original_cost.mem_ops,
                "kernel {} did not hoist/forward memory ops",
                orig.name
            );
        }
        assert!(compiled.schedule.peak_gprs <= 14);
        assert!(compiled.schedule.peak_vecs <= 16);
        assert_eq!(compiled.optimized.proven_domains.len(), compiled.optimized.nodes.len());
        let reopt = optimize_kernel(&compiled.optimized).expect("idempotent re-opt");
        assert_eq!(reopt.cost(), compiled.optimized_cost, "optimize_kernel must be idempotent");

        let mut bufs: Vec<Vec<i64>> = orig
            .bufs
            .iter()
            .map(|b| match b.access {
                BufAccess::LookupTable(n) => vec![0i64; n],
                _ => vec![0i64; LANES],
            })
            .collect();

        for iter in 0..samples_per_kernel {
            for (bi, decl) in orig.bufs.iter().enumerate() {
                let (lo, hi) = if decl.domain.lo <= decl.domain.hi && decl.domain.lo > i64::MIN / 2 {
                    (decl.domain.lo, decl.domain.hi)
                } else {
                    (-1000, 1000)
                };
                for slot in &mut bufs[bi] {
                    *slot = match iter {
                        0 => lo,
                        1 => hi,
                        _ => zipf.sample_range(lo, hi),
                    };
                }
            }
            let refs: Vec<&[i64]> = bufs.iter().map(Vec::as_slice).collect();
            let got_orig = orig.eval(&refs).expect("orig eval");
            let got_opt = compiled.optimized.eval(&refs).expect("opt eval");
            assert_eq!(
                got_orig, got_opt,
                "bit mismatch on kernel {} iter {}",
                orig.name, iter
            );
        }
    }
}

#[test]
fn fail_closed_ownership_bounds_sign_and_spill_guards() {
    // 1. Ownership guard: writing to ReadOnly buffer fails closed.
    let mut bad = Kernel::new("bad_write");
    let ro = bad.add_buf("ro", BufAccess::ReadOnly, Domain::range(0, 10));
    let c1 = bad.push(Op::Const(1));
    bad.add_output(ro, 0, c1);
    assert_eq!(compile_kernel(&bad, RegBudget::default()).unwrap_err(), CompileError::WriteToReadOnlyBuf(ro));

    // 2. Bounds guard: unproven index keeps BoundsCheck and fails closed at runtime.
    let mut bchk = Kernel::new("unproven_bounds");
    let a = bchk.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 100));
    let out = bchk.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mem = bchk.push(Op::MemInit);
    let v = bchk.push(Op::Load { mem, buf: a, lane: 0 });
    let chk = bchk.push(Op::BoundsCheck { idx: v, limit: 8 });
    bchk.add_output(out, 0, chk);
    let c = compile_kernel(&bchk, RegBudget::default()).expect("compile unproven bounds");
    assert_eq!(c.optimized_cost.bounds_checks, 1);
    let oob_in: &[&[i64]] = &[&[8, 0, 0, 0, 0, 0, 0, 0], &[0; 8]];
    assert_eq!(c.optimized.eval(oob_in).unwrap_err(), CompileError::OutOfBounds { idx: 8, limit: 8 });

    // 3. Signed modulo guard: negative domain MUST NOT rewrite `% 8` to `& 7`.
    let mut srem = Kernel::new("signed_rem");
    let a = srem.add_buf("a", BufAccess::ReadOnly, Domain::range(-100, 100));
    let out = srem.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mem = srem.push(Op::MemInit);
    let v = srem.push(Op::Load { mem, buf: a, lane: 0 });
    let c8 = srem.push(Op::Const(8));
    let r = srem.push(Op::Rem(v, c8));
    srem.add_output(out, 0, r);
    let opt = optimize_kernel(&srem).expect("opt signed rem");
    assert_eq!(opt.cost().divs_rems, 1);
    let neg_in: &[&[i64]] = &[&[-1, 0, 0, 0, 0, 0, 0, 0], &[0; 8]];
    assert_eq!(opt.eval(neg_in).expect("eval")[1][0], -1);

    // 4. Zero-spill register cliff guard: fails closed when peak registers exceed budget.
    let k = build_butterfly_8lane();
    let err = compile_kernel(&k, RegBudget { max_gprs: 3, max_vecs: 4 }).unwrap_err();
    assert!(matches!(err, CompileError::RegisterSpill { .. }));

    // 5. High-pressure 8x8 all-to-all coupled DAG trips even the default 14-GPR / 16-YMM guard.
    let mut wide = Kernel::new("wide_spill");
    let inp = wide.add_buf("inp", BufAccess::ReadOnly, Domain::range(0, 1000));
    let out = wide.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mem = wide.push(Op::MemInit);
    let mut loaded = [0usize; LANES];
    for (i, slot) in loaded.iter_mut().enumerate() {
        *slot = wide.push(Op::Load { mem, buf: inp, lane: i as u8 });
    }
    // Create 10 distinct partial values that are all mutually consumed by the final lane outputs.
    let mut mid = Vec::new();
    for i in 0..10 {
        let a = loaded[i % LANES];
        let b = loaded[(i + 3) % LANES];
        let c = wide.push(Op::Const((i as i64) + 3));
        let m = wide.push(Op::Mul(a, c));
        mid.push(wide.push(Op::Add(m, b)));
    }
    for lane in 0..LANES {
        let mut acc = mid[lane];
        for (j, &mj) in mid.iter().enumerate() {
            if j != lane {
                acc = wide.push(Op::Add(acc, mj));
            }
        }
        wide.add_output(out, lane as u8, acc);
    }
    let err_default = compile_kernel(&wide, RegBudget::default()).unwrap_err();
    assert!(matches!(err_default, CompileError::RegisterSpill { .. }));
}

#[cfg(not(miri))]
#[test]
fn real_rustc_o3_benchmark_and_objdump_zero_spill() {
    use std::fs;
    use std::process::Command;

    let k1 = compile_kernel(&build_modular_hash_lut(), RegBudget::default()).expect("compile k1");
    let k2 = compile_kernel(&build_butterfly_8lane(), RegBudget::default()).expect("compile k2");
    let k3 = compile_kernel(&build_affine_filter_8lane(), RegBudget::default()).expect("compile k3");
    let k4 = compile_kernel(&build_quad_crt_8lane(true), RegBudget::default()).expect("compile k4");

    let mut harness = String::new();
    harness.push_str("#![forbid(unsafe_code)]\nuse std::hint::black_box;\nuse std::time::Instant;\n\n");
    harness.push_str(&k1.rust_source);
    harness.push('\n');
    harness.push_str(&k2.rust_source);
    harness.push('\n');
    harness.push_str(&k3.rust_source);
    harness.push('\n');
    harness.push_str(&k4.rust_source);
    harness.push_str(
        r#"
#[inline(never)]
pub fn naive_modular_hash_lut(a: &[i64], b: &[i64], lut: &[i64], out: &mut [i64], n: usize) {
    for i in 0..n {
        let s = (a[i].wrapping_mul(17).wrapping_add(b[i].wrapping_mul(17))) / 8;
        let m = s % 251;
        let idx = m % 8;
        assert!(idx >= 0 && idx < 8);
        let lv = lut[idx as usize];
        out[i] = m;
        let m_reload = out[i];
        let a_reload = a[i];
        out[i] = m_reload.wrapping_add(lv).wrapping_add(a_reload % 64);
    }
}

#[inline(never)]
pub fn naive_butterfly_8lane(x: &[i64], y: &[i64], out: &mut [i64], blocks: usize) {
    for blk in 0..blocks {
        let base = blk * 8;
        for pair in 0..4 {
            let lo = base + pair;
            let hi = base + pair + 4;
            let u = (x[lo].wrapping_mul(16).wrapping_add(y[lo].wrapping_mul(16))) / 4;
            let v = x[hi].wrapping_mul(15).wrapping_sub(y[hi].wrapping_mul(15));
            let z = (x[lo].wrapping_mul(8)) % 8;
            out[lo] = u.wrapping_add(v).wrapping_add(z);
            out[hi] = u.wrapping_sub(v);
        }
    }
}

#[inline(never)]
pub fn naive_affine_filter_8lane(sig: &[i64], bias: &[i64], out: &mut [i64], n: usize) {
    for i in 0..n {
        out[i] = sig[i].wrapping_mul(9).wrapping_add(bias[i].wrapping_mul(9));
        let t = out[i];
        let s2 = sig[i];
        let num = t.wrapping_add(s2.wrapping_mul(7));
        out[i] = (num / 16) % 17;
    }
}

fn median_ns(mut f: impl FnMut(), iters: u64) -> f64 {
    let mut runs = [0.0f64; 5];
    for r in &mut runs {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        *r = t0.elapsed().as_nanos() as f64 / iters as f64;
    }
    runs.sort_by(|a, b| a.total_cmp(b));
    runs[2]
}

fn main() {
    const BLOCKS: usize = 4096;
    const N: usize = BLOCKS * 8;
    let mut a_flat = vec![0i64; N];
    let mut b_flat = vec![0i64; N];
    let mut a_blk = vec![[0i64; 8]; BLOCKS];
    let mut b_blk = vec![[0i64; 8]; BLOCKS];
    let lut: [i64; 8] = [13, 29, 47, 71, 101, 211, 509, 997];
    for blk in 0..BLOCKS {
        for lane in 0..8 {
            let i = blk * 8 + lane;
            let av = ((i.wrapping_mul(2654435761) >> 16) % 16000) as i64;
            let bv = ((i.wrapping_mul(1597334677) >> 16) % 4000) as i64;
            a_flat[i] = av;
            b_flat[i] = bv;
            a_blk[blk][lane] = av;
            b_blk[blk][lane] = bv;
        }
    }
    let mut out_naive = vec![0i64; N];
    let mut out_blk = vec![[0i64; 8]; BLOCKS];

    // Bit-exact verification across all 4 compiled kernels before timing.
    naive_modular_hash_lut(&a_flat, &b_flat, &lut, &mut out_naive, N);
    modular_hash_lut_batch(&a_blk, &b_blk, &lut, &mut out_blk);
    for blk in 0..BLOCKS {
        assert_eq!(&out_naive[blk * 8..(blk + 1) * 8], &out_blk[blk][..]);
    }
    naive_butterfly_8lane(&a_flat, &b_flat, &mut out_naive, BLOCKS);
    butterfly_8lane_batch(&a_blk, &b_blk, &mut out_blk);
    for blk in 0..BLOCKS {
        assert_eq!(&out_naive[blk * 8..(blk + 1) * 8], &out_blk[blk][..]);
    }
    naive_affine_filter_8lane(&a_flat, &b_flat, &mut out_naive, N);
    affine_filter_8lane_batch(&a_blk, &b_blk, &mut out_blk);
    for blk in 0..BLOCKS {
        assert_eq!(&out_naive[blk * 8..(blk + 1) * 8], &out_blk[blk][..]);
    }
    let mut q_blk = vec![[2i64; 8]; BLOCKS];
    for blk in 0..BLOCKS {
        for lane in 0..8 {
            q_blk[blk][lane] = (((blk * 8 + lane) % 240) * 4 + 2) as i64;
        }
    }
    quad_crt_8lane_batch(black_box(&q_blk), black_box(&lut), black_box(&mut out_blk));
    assert_eq!(out_blk[0][0], lut[4] + 7 + (2 * 2 + 1) * 3);

    let iters = 300u64;
    let t_n1 = median_ns(|| { naive_modular_hash_lut(black_box(&a_flat), black_box(&b_flat), black_box(&lut), black_box(&mut out_naive), black_box(N)); }, iters);
    let t_b1 = median_ns(|| { modular_hash_lut_batch(black_box(&a_blk), black_box(&b_blk), black_box(&lut), black_box(&mut out_blk)); }, iters);

    let t_n2 = median_ns(|| { naive_butterfly_8lane(black_box(&a_flat), black_box(&b_flat), black_box(&mut out_naive), black_box(BLOCKS)); }, iters);
    let t_b2 = median_ns(|| { butterfly_8lane_batch(black_box(&a_blk), black_box(&b_blk), black_box(&mut out_blk)); }, iters);

    let t_n3 = median_ns(|| { naive_affine_filter_8lane(black_box(&a_flat), black_box(&b_flat), black_box(&mut out_naive), black_box(N)); }, iters);
    let t_b3 = median_ns(|| { affine_filter_8lane_batch(black_box(&a_blk), black_box(&b_blk), black_box(&mut out_blk)); }, iters);

    println!("MODULAR_HASH_LUT naive_ns={t_n1:.1} zero_cc_ns={t_b1:.1} speedup={:.2}", t_n1 / t_b1);
    println!("BUTTERFLY_8LANE naive_ns={t_n2:.1} zero_cc_ns={t_b2:.1} speedup={:.2}", t_n2 / t_b2);
    println!("AFFINE_FILTER_8LANE naive_ns={t_n3:.1} zero_cc_ns={t_b3:.1} speedup={:.2}", t_n3 / t_b3);
}
"#,
    );

    let tmp = std::env::temp_dir();
    let rs_path = tmp.join("zero_cc_live_bench.rs");
    fs::write(&rs_path, &harness).expect("write bench harness");

    let profiles: &[(&str, &[&str], bool)] = &[
        ("rustc -D warnings -C opt-level=3", &["-D", "warnings", "-C", "opt-level=3"], false),
        (
            "rustc -D warnings -C opt-level=3 -C target-cpu=native",
            &["-D", "warnings", "-C", "opt-level=3", "-C", "target-cpu=native"],
            true,
        ),
    ];

    for (idx, (label, flags, require_all_fast)) in profiles.iter().enumerate() {
        let bin_path = tmp.join(format!("zero_cc_live_bench_bin_{idx}"));
        let compile = Command::new("rustc")
            .env_remove("RUSTFLAGS")
            .args(*flags)
            .arg("-o")
            .arg(&bin_path)
            .arg(&rs_path)
            .output()
            .expect("invoke rustc");
        assert!(
            compile.status.success(),
            "{label} failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let run = Command::new(&bin_path).output().expect("run bench bin");
        assert!(
            run.status.success(),
            "{label} run failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let stdout = String::from_utf8_lossy(&run.stdout);
        eprintln!("=== LIVE {label} BENCHMARK ===\n{stdout}");

        for line in stdout.lines() {
            if let Some(sp) = line.split("speedup=").nth(1) {
                let val: f64 = sp.trim().parse().expect("parse speedup");
                if *require_all_fast {
                    assert!(val > 1.5, "expected native speedup > 1.5x, got {line}");
                } else if !line.starts_with("BUTTERFLY_8LANE") {
                    assert!(val > 1.2, "expected -O3 speedup > 1.2x, got {line}");
                }
            }
        }

        // Disassemble compiled binary and verify zero frame allocs, zero spills, zero loop rsp,
        // zero hardware idiv, and zero panic/bounds-check calls!
        let dis = Command::new("objdump")
            .args(["-d", "--demangle"])
            .arg(&bin_path)
            .output()
            .expect("invoke objdump");
        assert!(dis.status.success());
        let asm = String::from_utf8_lossy(&dis.stdout);

        for fn_name in [
            "modular_hash_lut_batch",
            "butterfly_8lane_batch",
            "affine_filter_8lane_batch",
            "quad_crt_8lane_batch",
        ] {
            let stats = inspect_asm_fn(&asm, fn_name);
            eprintln!(
                "OBJDUMP [{label}] {fn_name}: insns={} frame_allocs={} stack_spills={} loop_rsp_ops={} idivs={} calls={} vec_insns={}",
                stats.insns, stats.frame_allocs, stats.stack_spills, stats.loop_rsp_ops,
                stats.idivs, stats.calls, stats.vec_insns
            );
            assert!(stats.insns > 0, "did not find symbol {fn_name} in objdump");
            assert_eq!(stats.frame_allocs, 0, "{fn_name} allocated stack frame!");
            assert_eq!(stats.stack_spills, 0, "{fn_name} had stack spills!");
            assert_eq!(stats.loop_rsp_ops, 0, "{fn_name} touched (%rsp) inside hot loop!");
            assert_eq!(stats.idivs, 0, "{fn_name} had idiv!");
            assert_eq!(stats.calls, 0, "{fn_name} had panic/bounds-check or helper call!");
        }
        for fn_name in [
            "naive_modular_hash_lut",
            "naive_butterfly_8lane",
            "naive_affine_filter_8lane",
        ] {
            let stats = inspect_asm_fn(&asm, fn_name);
            eprintln!(
                "OBJDUMP [{label}] {fn_name}: insns={} frame_allocs={} stack_spills={} loop_rsp_ops={} idivs={} calls={} vec_insns={}",
                stats.insns, stats.frame_allocs, stats.stack_spills, stats.loop_rsp_ops,
                stats.idivs, stats.calls, stats.vec_insns
            );
            assert!(stats.calls > 0, "expected naive {fn_name} to retain bounds-check panic calls");
        }
        let _ = fs::remove_file(&bin_path);
    }
    let _ = fs::remove_file(&rs_path);
}

#[cfg(not(miri))]
struct AsmStats {
    insns: usize,
    frame_allocs: usize,
    stack_spills: usize,
    loop_rsp_ops: usize,
    idivs: usize,
    calls: usize,
    vec_insns: usize,
}

#[cfg(not(miri))]
fn inspect_asm_fn(asm: &str, sym_substr: &str) -> AsmStats {
    let mut in_fn = false;
    let mut parsed: Vec<(u64, String, String)> = Vec::new();
    for line in asm.lines() {
        if line.ends_with(">:") {
            in_fn = line.contains(sym_substr);
            continue;
        }
        if !in_fn || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let Some(addr_part) = parts.next() else { continue; };
        if parts.next().is_none() { continue; }
        let Some(asm_part) = parts.next() else { continue; };
        let addr = u64::from_str_radix(addr_part.trim().trim_end_matches(':'), 16).unwrap_or(0);
        let code = asm_part.split('#').next().unwrap_or("").trim();
        let mut toks = code.split_whitespace();
        let Some(mnem) = toks.next() else { continue; };
        let ops = toks.collect::<Vec<_>>().join(" ");
        parsed.push((addr, mnem.to_string(), ops));
    }

    // Identify hot-loop address intervals via backward conditional/unconditional jumps.
    let mut loop_ranges: Vec<(u64, u64)> = Vec::new();
    for (addr, mnem, ops) in &parsed {
        if mnem.starts_with('j') {
            if let Some(tok) = ops.split_whitespace().next() {
                if let Ok(target) = u64::from_str_radix(tok, 16) {
                    if target < *addr {
                        loop_ranges.push((target, *addr));
                    }
                }
            }
        }
    }

    let mut s = AsmStats {
        insns: 0, frame_allocs: 0, stack_spills: 0, loop_rsp_ops: 0,
        idivs: 0, calls: 0, vec_insns: 0,
    };
    for (addr, mnem, ops) in &parsed {
        s.insns += 1;
        if (mnem == "sub" || mnem == "subq") && ops.ends_with(",%rsp") {
            s.frame_allocs += 1;
        }
        // In AT&T syntax, the destination operand is after the last comma; a store to (%rsp) is a spill.
        if !mnem.starts_with("push") && !mnem.starts_with("pop") && !mnem.starts_with("call") {
            if let Some(dst) = ops.rsplit(',').next() {
                if dst.contains("(%rsp)") {
                    s.stack_spills += 1;
                }
            }
        }
        if ops.contains("(%rsp)") && loop_ranges.iter().any(|&(lo, hi)| *addr >= lo && *addr <= hi) {
            s.loop_rsp_ops += 1;
        }
        if matches!(mnem.as_str(), "idiv" | "idivq" | "idivl" | "div" | "divq" | "divl") {
            s.idivs += 1;
        }
        if mnem.starts_with("call") {
            s.calls += 1;
        }
        if ops.contains("%ymm") || ops.contains("%xmm") {
            s.vec_insns += 1;
        }
    }
    s
}

#[test]
fn rust_source_virtio_ring_roundtrip_and_guards() {
    let virtio_src = r#"
/// @domain desc: 0..=65535
/// @domain pkt: 0..=65535
/// @domain policy: 0..=1023
pub fn virtio_ring_filter(desc: &[i64; 8], pkt: &[i64; 8], policy: &[i64; 16], out: &mut [i64; 8]) {
    for lane in 0..8 {
        let d = desc[lane];
        let p = pkt[lane];
        let ring_mix = (d * 17 + p * 17) / 8;
        let hash = ring_mix % 251;
        let slot = hash % 16;
        assert!(slot >= 0 && slot < 16);
        let action = policy[slot as usize];
        out[lane] = hash;
        let h_reload = out[lane];
        let d_reload = desc[lane];
        out[lane] = h_reload.wrapping_add(action).wrapping_add(d_reload % 64);
    }
}
"#;
    let orig_k = parse_rust_kernel(virtio_src).expect("parse virtio_ring_filter");
    let compiled = compile_rust_source(virtio_src, RegBudget::default()).expect("compile virtio_ring_filter");

    // 1. Counterfactual knockout: naive .rs has divs/rems + bounds checks; optimized has 0 and 0.
    assert!(compiled.original_cost.divs_rems > 0);
    assert!(compiled.original_cost.bounds_checks > 0);
    assert_ne!(compiled.original_cost, compiled.optimized_cost);
    assert_eq!(compiled.optimized_cost.divs_rems, 0);
    assert_eq!(compiled.optimized_cost.bounds_checks, 0);
    assert!(compiled.optimized_cost.mem_ops < compiled.original_cost.mem_ops);
    assert!(compiled.rust_source.contains("/// @domain desc: 0..=65535"));
    assert!(compiled.rust_source.contains("/// @domain pkt: 0..=65535"));
    assert!(compiled.rust_source.contains("/// @domain policy: 0..=1023"));

    // Bit-exact verification across 4,000 Zipfian 8-lane blocks (128 under Miri).
    let samples = if cfg!(miri) { 128 } else { 4_000 };
    let mut zipf = ZipfSampler::new(1024, 1.2, 0x91e1_55aa_2026_7777);
    let mut policy = [0i64; 16];
    for (i, slot) in policy.iter_mut().enumerate() {
        *slot = ((i as i64) * 53 + 17) % 1024;
    }
    for iter in 0..samples {
        let mut desc = [0i64; LANES];
        let mut pkt = [0i64; LANES];
        for lane in 0..LANES {
            desc[lane] = match iter { 0 => 0, 1 => 65535, _ => zipf.sample_range(0, 65535) };
            pkt[lane] = match iter { 0 => 0, 1 => 65535, _ => zipf.sample_range(0, 65535) };
        }
        let out_z = [0i64; LANES];
        let refs: &[&[i64]] = &[&desc, &pkt, &policy, &out_z];
        let got_orig = orig_k.eval(refs).expect("orig eval");
        let got_opt = compiled.optimized.eval(refs).expect("opt eval");
        assert_eq!(got_orig, got_opt, "virtio_ring_filter mismatch at iter {iter}");
    }

    // Also verify #[zero_domain(... mod ... == ...)] attribute syntax strips on emit and eliminates checks.
    let quad_attr_src = r#"
#[zero_domain(x = 2..=962 mod 4 == 2)]
#[zero_domain(lut = 0..=1023)]
pub fn quad_attr_kernel(x: &[i64; 8], lut: &[i64; 8], out: &mut [i64; 8]) {
    for lane in 0..8 {
        let xi = x[lane];
        let r16 = (xi * xi) % 16;
        assert!(r16 >= 0 && r16 < 8);
        let lv = lut[r16 as usize];
        out[lane] = lv + ((xi * 2 + 1) * 3) % 8 + xi;
    }
}
"#;
    let quad_c = compile_rust_source(quad_attr_src, RegBudget::default()).expect("compile quad_attr");
    assert_eq!(quad_c.optimized_cost.divs_rems, 0);
    assert_eq!(quad_c.optimized_cost.bounds_checks, 0);
    assert!(!quad_c.rust_source.contains("#[zero_domain"));
    assert!(quad_c.rust_source.contains("/// @domain x: 2..=962 mod 4 == 2"));

    // 2. Fail-closed knockouts: write to &[i64; 8], >14 GPR spill, >16 loop trips, negative unsigned shift, negative LUT index.
    let bad_ro_write = r#"
pub fn bad_ro(a: &[i64; 8], out: &[i64; 8]) {
    for lane in 0..8 { out[lane] = a[lane] + 1; }
}
"#;
    assert_ne!(compile_rust_source(virtio_src, RegBudget::default()).is_ok(), compile_rust_source(bad_ro_write, RegBudget::default()).is_ok());
    assert_eq!(compile_rust_source(bad_ro_write, RegBudget::default()).unwrap_err(), CompileError::WriteToReadOnlyBuf(1));

    let spill_src = r#"
/// @domain inp: 0..=1000
pub fn spill_14(inp: &[i64; 8], out: &mut [i64; 8]) {
    let m0 = inp[0] * 3 + inp[3]; let m1 = inp[1] * 4 + inp[4]; let m2 = inp[2] * 5 + inp[5];
    let m3 = inp[3] * 6 + inp[6]; let m4 = inp[4] * 7 + inp[7]; let m5 = inp[5] * 8 + inp[0];
    let m6 = inp[6] * 9 + inp[1]; let m7 = inp[7] * 10 + inp[2]; let m8 = inp[0] * 11 + inp[3];
    let m9 = inp[1] * 12 + inp[4];
    let all = m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8 + m9;
    out[0] = all - m0; out[1] = all - m1; out[2] = all - m2; out[3] = all - m3;
    out[4] = all - m4; out[5] = all - m5; out[6] = all - m6; out[7] = all - m7;
}
"#;
    assert_ne!(compile_rust_source(virtio_src, RegBudget::default()).is_ok(), compile_rust_source(spill_src, RegBudget::default()).is_ok());
    assert!(matches!(compile_rust_source(spill_src, RegBudget::default()), Err(CompileError::RegisterSpill { .. })));

    let bad_loop = "pub fn bad_loop(a: &[i64; 8], out: &mut [i64; 8]) { for i in 0..32 { out[0] = a[0] + i; } }";
    assert!(matches!(parse_rust_kernel(bad_loop), Err(CompileError::ParseError(_))));

    let bad_ushr = "/// @domain a: -10..=10\npub fn bad_ushr(a: &[i64; 8], out: &mut [i64; 8]) { out[0] = ((a[0] as u64) >> 2) as i64; }";
    assert!(matches!(parse_rust_kernel(bad_ushr), Err(CompileError::ParseError(_))));

    let neg_lut_src = "/// @domain a: -10..=10\npub fn neg_lut(a: &[i64; 8], lut: &[i64; 16], out: &mut [i64; 8]) { out[0] = lut[a[0] as usize]; }";
    let neg_k = parse_rust_kernel(neg_lut_src).expect("parse neg_lut");
    assert_eq!(neg_k.eval(&[&[-1, 0, 0, 0, 0, 0, 0, 0], &[0; 16], &[0; 8]]).unwrap_err(), CompileError::OutOfBounds { idx: -1, limit: 16 });

    // Precedence check: `a + b * 3` must bind `*` tighter than `+`, and `+` tighter than `&`.
    let prec_src = "pub fn prec(a: &[i64; 8], b: &[i64; 8], out: &mut [i64; 8]) { out[0] = a[0] + b[0] * 3; out[1] = a[0] + b[0] & 7; }";
    let prec_k = parse_rust_kernel(prec_src).expect("parse prec");
    let prec_res = prec_k.eval(&[&[2, 0, 0, 0, 0, 0, 0, 0], &[5, 0, 0, 0, 0, 0, 0, 0], &[0; 8]]).expect("eval prec");
    assert_eq!(prec_res[2][0], 17);
    assert_eq!(prec_res[2][1], 7);

    // 3. Round-trip fixed-point: compile_rust_source(&compiled.rust_source) matches optimized_cost.
    let rt = compile_rust_source(&compiled.rust_source, RegBudget::default()).expect("roundtrip compile_rust_source");
    assert_eq!(rt.optimized_cost, compiled.optimized_cost);
    assert_eq!(rt.optimized.nodes, compiled.optimized.nodes);

    #[cfg(not(miri))]
    {
        use std::path::Path;
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let rs_tmp = tmp.join("zero_cc_virtio_rt.rs");
        let rlib_tmp = tmp.join("libzero_cc_virtio_rt.rlib");
        let full_rs = format!("#![forbid(unsafe_code)]\n\n{}\n{}\n", compiled.rust_source, quad_c.rust_source);
        std::fs::write(&rs_tmp, &full_rs).expect("write virtio emitted rs");

        let rc = Command::new("rustc")
            .env_remove("RUSTFLAGS")
            .args(["--crate-type", "lib", "-D", "warnings", "-C", "opt-level=3", "-C", "target-cpu=native", "-o"])
            .arg(&rlib_tmp)
            .arg(&rs_tmp)
            .output()
            .expect("invoke rustc on emitted virtio");
        assert!(rc.status.success(), "rustc -D warnings failed:\n{}", String::from_utf8_lossy(&rc.stderr));

        let dis = Command::new("objdump").args(["-d", "--demangle"]).arg(&rlib_tmp).output().expect("objdump virtio");
        assert!(dis.status.success());
        let asm = String::from_utf8_lossy(&dis.stdout);
        let stats = inspect_asm_fn(&asm, "virtio_ring_filter_batch");
        assert!(stats.insns > 0, "missing virtio_ring_filter_batch in objdump");
        assert_eq!(stats.frame_allocs, 0);
        assert_eq!(stats.stack_spills, 0);
        assert_eq!(stats.loop_rsp_ops, 0);
        assert_eq!(stats.idivs, 0);
        assert_eq!(stats.calls, 0);

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        if let Some(parent) = root.parent() {
            let sibling_bin = parent.join("zero-check/target/release/zero-check");
            if sibling_bin.exists() {
                let out = Command::new(&sibling_bin)
                    .args(["--no-ledger", "--prove-fns", "--kernel-strict"])
                    .arg(&rs_tmp)
                    .output()
                    .expect("invoke sibling zero-check --prove-fns --kernel-strict");
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stderr.contains("unknown") {
                    assert!(
                        out.status.success(),
                        "zero-check --prove-fns --kernel-strict rejected emitted virtio:\n{}\n{}",
                        String::from_utf8_lossy(&out.stdout),
                        stderr
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&rs_tmp);
        let _ = std::fs::remove_file(&rlib_tmp);
    }
}

#[test]
fn branchless_select_mux_roundtrip_and_zero_spill() {
    let raw_src = r#"
/// @domain sig: 0..=4095
/// @domain thresh: 0..=4095
pub fn branchless_clamp_and_mux(sig: &[i64; 8], thresh: &[i64; 8], out: &mut [i64; 8]) {
    for lane in 0..8 {
        let s = sig[lane];
        let t = thresh[lane];
        let safe_s = if s < 65536 { s } else { 65535 };
        if safe_s > t {
            out[lane] = safe_s - t;
        } else {
            out[lane] = t - safe_s;
        }
    }
}
"#;
    let ablated_src = raw_src.replace("sig: 0..=4095", "sig: 0..=100000");

    let orig_k = parse_rust_kernel(raw_src).expect("parse branchless_clamp_and_mux");
    let compiled = compile_rust_source(raw_src, RegBudget::default()).expect("compile branchless_clamp_and_mux");
    let ablated = compile_rust_source(&ablated_src, RegBudget::default()).expect("compile ablated mux");

    // 1. Counterfactual knockout: domain-proven dead branch `if s < 65536` folds away when sig in 0..=4095.
    assert_ne!(compiled.optimized_cost, ablated.optimized_cost);
    assert_ne!(compiled.optimized.nodes.len(), ablated.optimized.nodes.len());
    assert!(compiled.optimized.nodes.len() < ablated.optimized.nodes.len());
    assert!(compiled.optimized.nodes.iter().any(|op| matches!(op, Op::Select { .. })));

    // Bit-exact verification across 4,000 Zipfian 8-lane blocks (128 under Miri).
    let samples = if cfg!(miri) { 128 } else { 4_000 };
    let mut zipf = ZipfSampler::new(1024, 1.2, 0xcafe_babe_2026_1234);
    for iter in 0..samples {
        let (mut sig, mut thresh, mut expected) = ([0i64; LANES], [0i64; LANES], [0i64; LANES]);
        for lane in 0..LANES {
            sig[lane] = match iter { 0 => 0, 1 => 4095, _ => zipf.sample_range(0, 4095) };
            thresh[lane] = match iter { 0 => 4095, 1 => 0, _ => zipf.sample_range(0, 4095) };
            expected[lane] = (sig[lane] - thresh[lane]).abs();
        }
        let out_z = [0i64; LANES];
        let refs: &[&[i64]] = &[&sig, &thresh, &out_z];
        assert_eq!(orig_k.eval(refs).expect("orig mux")[2], expected);
        assert_eq!(compiled.optimized.eval(refs).expect("opt mux")[2], expected);
    }

    // 2. Exact fixed-point round-trip: recompiled.optimized.nodes == compiled.optimized.nodes!
    let recompiled = compile_rust_source(&compiled.rust_source, RegBudget::default()).expect("roundtrip mux");
    assert_eq!(recompiled.optimized.nodes, compiled.optimized.nodes);
    assert_eq!(recompiled.optimized_cost, compiled.optimized_cost);

    // 3. Real rustc -D warnings -O3 native objdump zero-spill + live zero-check proof.
    #[cfg(not(miri))]
    {
        use std::path::Path;
        use std::process::Command;

        let tmp = std::env::temp_dir();
        let raw_tmp = tmp.join("zero_cc_mux_raw.rs");
        let em_tmp = tmp.join("zero_cc_mux_emitted.rs");
        let rlib_tmp = tmp.join("libzero_cc_mux_emitted.rlib");
        std::fs::write(&raw_tmp, format!("#![forbid(unsafe_code)]\n{raw_src}")).expect("write raw mux");
        std::fs::write(&em_tmp, format!("#![forbid(unsafe_code)]\n\n{}", compiled.rust_source)).expect("write emitted mux");

        let rc = Command::new("rustc")
            .env_remove("RUSTFLAGS")
            .args(["--crate-type", "lib", "-D", "warnings", "-C", "opt-level=3", "-C", "target-cpu=native", "-o"])
            .arg(&rlib_tmp)
            .arg(&em_tmp)
            .output()
            .expect("invoke rustc on emitted mux");
        assert!(rc.status.success(), "rustc -D warnings failed:\n{}", String::from_utf8_lossy(&rc.stderr));

        let dis = Command::new("objdump").args(["-d", "--demangle"]).arg(&rlib_tmp).output().expect("objdump mux");
        assert!(dis.status.success());
        let stats = inspect_asm_fn(&String::from_utf8_lossy(&dis.stdout), "branchless_clamp_and_mux_batch");
        assert!(stats.insns > 0, "missing branchless_clamp_and_mux_batch in objdump");
        assert_eq!(stats.frame_allocs, 0);
        assert_eq!(stats.stack_spills, 0);
        assert_eq!(stats.loop_rsp_ops, 0);
        assert_eq!(stats.idivs, 0);
        assert_eq!(stats.calls, 0);

        if let Some(parent) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            let sibling_bin = parent.join("zero-check/target/release/zero-check");
            if sibling_bin.exists() {
                for (args, path) in [
                    (&["--no-ledger", "--prove-fns", "--kernel-strict", "--require-totality"][..], &raw_tmp),
                    (&["--no-ledger", "--prove-fns", "--kernel-strict"][..], &em_tmp),
                ] {
                    let out = Command::new(&sibling_bin).args(args).arg(path).output().expect("invoke zero-check");
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if !stderr.contains("unknown") {
                        assert!(out.status.success(), "zero-check failed on {}:\n{}\n{}", path.display(), String::from_utf8_lossy(&out.stdout), stderr);
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&raw_tmp);
        let _ = std::fs::remove_file(&em_tmp);
        let _ = std::fs::remove_file(&rlib_tmp);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use zero_cc::barrett_params;

    #[kani::proof]
    fn kani_barrett_and_domain_soundness() {
        let x: i64 = kani::any();
        let y: i64 = kani::any();
        kani::assume((0..=65535).contains(&x));
        kani::assume((0..=65535).contains(&y));

        // Exhaustively verify Barrett reduction for all 5 odd moduli over [0, 65535].
        for d in [3i64, 7, 13, 17, 251] {
            let (m, s) = barrett_params(d, 65535).expect("barrett params");
            let q = ((x as u64).wrapping_mul(m as u64) >> s) as i64;
            let r = x.wrapping_sub(q.wrapping_mul(d));
            assert_eq!(q, x / d);
            assert_eq!(r, x % d);
        }

        // Verify Domain interval & modular transfer functions soundly contain true results.
        let dx = Domain::range(0, 65535);
        let dy = Domain::range(0, 65535);
        assert!(dx.eval_add_sub(dy, false).contains(x.wrapping_add(y)));
        assert!(dx.eval_add_sub(dy, true).contains(x.wrapping_sub(y)));
        assert!(dx.eval_mul(Domain::exact(17)).contains(x.wrapping_mul(17)));
        assert!(dx.eval_div(Domain::exact(8)).contains(x / 8));
        assert!(dx.eval_rem(Domain::exact(251)).contains(x % 251));
        assert!(dx.eval_shl(4).contains(x.wrapping_shl(4)));
        assert!(dx.eval_shr(3).contains(x.wrapping_shr(3)));
        assert!(dx.eval_and(Domain::exact(63)).contains(x & 63));
    }

    #[kani::proof]
    fn kani_egraph_rewrite_equivalence_symbolic() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let lut0: i64 = kani::any();
        kani::assume((0..=65535).contains(&a));
        kani::assume((0..=65535).contains(&b));
        kani::assume((0..=1023).contains(&lut0));

        // 1. Modular hash lane equivalence: original IR semantics vs zero-cc lowered form.
        let s_orig = (a.wrapping_mul(17).wrapping_add(b.wrapping_mul(17))) / 8;
        let m_orig = s_orig % 251;
        let idx_orig = m_orig % 8;
        assert!((0..8).contains(&idx_orig));
        let out_orig = m_orig.wrapping_add(lut0).wrapping_add(a % 64);

        let ab = a.wrapping_add(b);
        let ab17 = ab.wrapping_shl(4).wrapping_add(ab);
        let s_opt = ((ab17 as u64) >> 3) as i64;
        let (m251, sh251) = barrett_params(251, (65535 * 2 * 17) / 8).expect("barrett 251");
        let q251 = (((s_opt as u32 as u64).wrapping_mul(m251 as u32 as u64)) >> sh251) as i64;
        let m_opt = s_opt.wrapping_sub(q251.wrapping_mul(251));
        let idx_opt = m_opt & 7;
        let out_opt = m_opt.wrapping_add(lut0).wrapping_add(a & 63);
        assert_eq!(idx_orig, idx_opt);
        assert_eq!(out_orig, out_opt);

        // 2. Affine filter lane equivalence: original vs zero-cc lowered form (nested Barrett % 17).
        kani::assume(a <= 32767 && b <= 4095);
        let af_orig = ((a.wrapping_mul(9).wrapping_add(b.wrapping_mul(9))).wrapping_add(a.wrapping_mul(7)) / 16) % 17;
        let sum9 = a.wrapping_add(b);
        let t9 = sum9.wrapping_shl(3).wrapping_add(sum9);
        let a7 = a.wrapping_shl(3).wrapping_sub(a);
        let q16 = ((t9.wrapping_add(a7) as u64) >> 4) as i64;
        let (m17, sh17) = barrett_params(17, ((32767 + 4095) * 9 + 32767 * 7) / 16).expect("barrett 17");
        let q17 = (((q16 as u32 as u64).wrapping_mul(m17 as u32 as u64)) >> sh17) as i64;
        let q17_x17 = q17.wrapping_shl(4).wrapping_add(q17);
        let af_opt = q16.wrapping_sub(q17_x17);
        assert_eq!(af_orig, af_opt);
    }

    #[kani::proof]
    fn kani_butterfly_and_domain_meet_soundness() {
        let xlo: i64 = kani::any();
        let ylo: i64 = kani::any();
        let xhi: i64 = kani::any();
        let yhi: i64 = kani::any();
        kani::assume((0..=16383).contains(&xlo));
        kani::assume((0..=16383).contains(&ylo));
        kani::assume((0..=16383).contains(&xhi));
        kani::assume((0..=16383).contains(&yhi));

        // 1. Butterfly radix-2 equivalence: shift fusion + distributive factoring + mod-8 zeroing.
        let u_orig = (xlo.wrapping_mul(16).wrapping_add(ylo.wrapping_mul(16))) / 4;
        let v_orig = xhi.wrapping_mul(15).wrapping_sub(yhi.wrapping_mul(15));
        let z_orig = (xlo.wrapping_mul(8)) % 8;
        let out_lo_orig = u_orig.wrapping_add(v_orig).wrapping_add(z_orig);
        let out_hi_orig = u_orig.wrapping_sub(v_orig);

        let u_opt = xlo.wrapping_add(ylo).wrapping_shl(2);
        let d = xhi.wrapping_sub(yhi);
        let v_opt = d.wrapping_shl(4).wrapping_sub(d);
        let out_lo_opt = u_opt.wrapping_add(v_opt);
        let out_hi_opt = u_opt.wrapping_sub(v_opt);
        assert_eq!(out_lo_orig, out_lo_opt);
        assert_eq!(out_hi_orig, out_hi_opt);

        // 2. Domain::meet soundness and negative signed-modulo guard.
        let d1 = Domain::range(0, 16383).eval_mul(Domain::exact(7));
        let d2 = Domain::range(-16383, 16383 * 8);
        let met = d1.meet(d2);
        let v7 = xlo.wrapping_mul(7);
        assert!(met.contains(v7));
        assert!(met.is_nonneg());

        let neg: i64 = kani::any();
        kani::assume((-100..=100).contains(&neg));
        let rem_dom = Domain::range(-100, 100).eval_rem(Domain::exact(8));
        assert!(rem_dom.contains(neg % 8));
        assert!(!rem_dom.is_nonneg());
    }
}
