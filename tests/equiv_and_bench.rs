#![forbid(unsafe_code)]

use zero_cc::{
    compile_kernel, optimize_kernel, BufAccess, CompileError, Domain, Kernel, Op, RegBudget, LANES,
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

    let mut harness = String::new();
    harness.push_str("#![forbid(unsafe_code)]\nuse std::hint::black_box;\nuse std::time::Instant;\n\n");
    harness.push_str(&k1.rust_source);
    harness.push('\n');
    harness.push_str(&k2.rust_source);
    harness.push('\n');
    harness.push_str(&k3.rust_source);
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

    // Bit-exact verification across all 3 compiled kernels before timing.
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
        let Some(_bytes_part) = parts.next() else { continue; };
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
