#![forbid(unsafe_code)]

use std::fs;
use std::process::Command;

use zero_cc::{
    compile_kernel_direct, compile_rust_to_elf, compile_rust_to_elf_exec,
    compile_standalone_elf_exec, emit_kernel_elf, parse_rust_kernel, verify_elf64_executable,
    verify_elf64_object, BufAccess, CompileError, Domain, ElemTy, Kernel, Op, RegBudget, TargetIsa,
    EM_AARCH64, EM_X86_64, LANES,
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
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi as i128 - lo as i128) as u128;
        if span == 0 { lo } else { (lo as i128 + ((self.next_u64() as u128) % (span + 1)) as i128) as i64 }
    }
}

fn build_modular_hash_lut() -> Kernel {
    let mut k = Kernel::new("modular_hash_lut");
    let a = k.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 65535));
    let b = k.add_buf("b", BufAccess::ReadOnly, Domain::range(0, 65535));
    let lut = k.add_buf("lut", BufAccess::LookupTable(8), Domain::range(0, 1023));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mut mem = k.push(Op::MemInit);
    let (c17, c8, c251, c64) = (k.push(Op::Const(17)), k.push(Op::Const(8)), k.push(Op::Const(251)), k.push(Op::Const(64)));
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

fn build_butterfly_8lane() -> Kernel {
    let mut k = Kernel::new("butterfly_8lane");
    let x = k.add_buf("x", BufAccess::ReadOnly, Domain::range(0, 16383));
    let y = k.add_buf("y", BufAccess::ReadOnly, Domain::range(0, 16383));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mem = k.push(Op::MemInit);
    let (c16, c15, c8, c4) = (k.push(Op::Const(16)), k.push(Op::Const(15)), k.push(Op::Const(8)), k.push(Op::Const(4)));
    for pair in 0..4u8 {
        let (lo, hi) = (pair, pair + 4);
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

fn build_affine_filter_8lane() -> Kernel {
    let mut k = Kernel::new("affine_filter_8lane");
    let sig = k.add_buf("sig", BufAccess::ReadOnly, Domain::range(0, 32767));
    let bias = k.add_buf("bias", BufAccess::ReadOnly, Domain::range(0, 4095));
    let out = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let mut mem = k.push(Op::MemInit);
    let (c9, c7, c16, c17) = (k.push(Op::Const(9)), k.push(Op::Const(7)), k.push(Op::Const(16)), k.push(Op::Const(17)));
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

const SIGNED_SELECT_SRC: &str = r#"
/// @domain a: -100000..=100000
/// @domain b: -100000..=100000
pub fn signed_select_ashr(a: &[i64; 8], b: &[i64; 8], out: &mut [i64; 8]) {
    for lane in 0..8 {
        let x = a[lane];
        let y = b[lane];
        let sar = x >> 3;
        let diff = if x >= y { x - y } else { y - x };
        let pick = if sar < 0 { diff.wrapping_add(sar) } else { diff.wrapping_sub(sar) };
        out[lane] = if x == y { sar } else { pick };
    }
}
"#;

#[cfg(not(miri))]
struct ObjDisasm {
    insns: usize,
    idivs: usize,
    cmovs: usize,
    cond_jumps: usize,
    rsp_mem_ops: usize,
    calls: usize,
}

#[cfg(not(miri))]
fn inspect_intel_symbol(asm: &str, sym: &str) -> ObjDisasm {
    let header = format!("<{sym}>:");
    let mut in_sym = false;
    let mut stats = ObjDisasm { insns: 0, idivs: 0, cmovs: 0, cond_jumps: 0, rsp_mem_ops: 0, calls: 0 };
    for line in asm.lines() {
        if line.ends_with(">:") {
            in_sym = line.contains(&header);
            continue;
        }
        if !in_sym || line.trim().is_empty() { continue; }
        let mut parts = line.split('\t');
        let Some(_) = parts.next() else { continue; };
        if parts.next().is_none() { continue; }
        let Some(insn_txt) = parts.next() else { continue; };
        let clean = insn_txt.split('#').next().unwrap_or("").trim();
        let Some(mnem) = clean.split_whitespace().next() else { continue; };
        if mnem == "nop" { continue; }
        stats.insns += 1;
        if matches!(mnem, "idiv" | "div") { stats.idivs += 1; }
        if mnem.starts_with("cmov") { stats.cmovs += 1; }
        if mnem.starts_with('j') && mnem != "jmp" { stats.cond_jumps += 1; }
        if !mnem.starts_with("push") && !mnem.starts_with("pop") && !mnem.starts_with("call") && !mnem.starts_with("ret")
            && (clean.contains("[rsp") || clean.contains("[rbp"))
        {
            stats.rsp_mem_ops += 1;
        }
        if mnem.starts_with("call") { stats.calls += 1; }
    }
    stats
}

#[test]
fn elf64_airlock_aarch64_isa_and_counterfactual_knockouts() {
    let k_hash = build_modular_hash_lut();
    let k_bfly = build_butterfly_8lane();
    let k_aff = build_affine_filter_8lane();
    let k_sel = parse_rust_kernel(SIGNED_SELECT_SRC).expect("parse signed_select_ashr");

    for k in [&k_hash, &k_bfly, &k_aff, &k_sel] {
        let x86 = compile_kernel_direct(k, RegBudget::default(), TargetIsa::X86_64).expect("x86_64 direct");
        let arm = compile_kernel_direct(k, RegBudget::default(), TargetIsa::Aarch64).expect("aarch64 direct");
        assert!(x86.zero_spills && arm.zero_spills);
        assert!(x86.direct_share_pct >= 80, "{} x86 share {} < 80", k.name, x86.direct_share_pct);
        assert!(arm.direct_share_pct >= 80, "{} arm share {} < 80", k.name, arm.direct_share_pct);

        // Counterfactual ISA knockout: x86_64 vs AArch64 ELF64 bytes & headers differ.
        assert_ne!(x86.elf_bytes, arm.elf_bytes);
        assert_ne!(x86.code_bytes, arm.code_bytes);

        let sx = verify_elf64_object(&x86.elf_bytes).expect("verify x86 elf");
        let sa = verify_elf64_object(&arm.elf_bytes).expect("verify arm elf");
        assert_eq!(sx.machine, EM_X86_64);
        assert_eq!(sa.machine, EM_AARCH64);
        assert!(sx.has_nonexec_gnu_stack && sa.has_nonexec_gnu_stack);
        assert_eq!(sx.symbols.len(), 2);
        assert_eq!(sa.symbols.len(), 2);

        // AArch64 instruction alignment & RET / CSEL word verification.
        assert_eq!(arm.code_bytes.len() % 4, 0, "AArch64 instructions must be 4-byte aligned");
        let words: Vec<u32> = arm.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let ret_count = words.iter().filter(|&&w| w == 0xD65F_03C0).count();
        assert!(ret_count >= 2, "expected RET in both kernel and batch entrypoints");
        if k.name == "signed_select_ashr" {
            let csel_count = words.iter().filter(|&&w| (w & 0xFFE0_FC00) == 0x9A80_1000).count();
            let asr_count = words.iter().filter(|&&w| (w & 0xFFC0_FC00) == 0x9340_FC00).count();
            assert!(csel_count >= LANES * 2, "expected AArch64 CSEL NE words, got {csel_count}");
            assert!(asr_count >= LANES, "expected AArch64 signed ASR words, got {asr_count}");
        }
    }

    // Counterfactual E-Graph optimization knockout on Direct ELF64 emission:
    for k in [&k_hash, &k_aff] {
        let opt_obj = emit_kernel_elf(k, RegBudget::default(), TargetIsa::X86_64, true, false).expect("opt elf");
        let unopt_obj = emit_kernel_elf(k, RegBudget::default(), TargetIsa::X86_64, false, false).expect("unopt elf");
        assert_ne!(opt_obj.code_bytes, unopt_obj.code_bytes);
        assert!(
            opt_obj.code_bytes.len() < unopt_obj.code_bytes.len(),
            "{}: opt len {} not < unopt len {}",
            k.name, opt_obj.code_bytes.len(), unopt_obj.code_bytes.len()
        );
    }

    // Relocation mode check (.rela.text with R_X86_64_PLT32 and R_AARCH64_CALL26).
    let x86_call = emit_kernel_elf(&k_bfly, RegBudget::default(), TargetIsa::X86_64, true, true).expect("x86 call_batch");
    let arm_call = emit_kernel_elf(&k_bfly, RegBudget::default(), TargetIsa::Aarch64, true, true).expect("arm call_batch");
    assert_eq!(verify_elf64_object(&x86_call.elf_bytes).expect("x86 rela").rela_count, 1);
    assert_eq!(verify_elf64_object(&arm_call.elf_bytes).expect("arm rela").rela_count, 1);

    // Fail-closed spill guard on direct ELF64 backend.
    let err = compile_kernel_direct(&k_bfly, RegBudget { max_gprs: 3, max_vecs: 4 }, TargetIsa::X86_64).unwrap_err();
    assert!(matches!(err, CompileError::RegisterSpill { .. }));
}

#[cfg(not(miri))]
#[test]
fn live_hardware_link_differential_fuzz_objdump_and_bench() {
    let tmp = std::env::temp_dir().join(format!("zero_cc_elf64_test_{}", std::process::id()));
    let _ = fs::create_dir_all(&tmp);

    let k_hash = build_modular_hash_lut();
    let k_bfly = build_butterfly_8lane();
    let k_aff = build_affine_filter_8lane();
    let k_sel = parse_rust_kernel(SIGNED_SELECT_SRC).expect("parse signed_select_ashr");

    let mut sdiv_k = Kernel::new("signed_div_rem_edge");
    let sa = sdiv_k.add_buf("a", BufAccess::ReadOnly, Domain::top());
    let sb = sdiv_k.add_buf("b", BufAccess::ReadOnly, Domain::range(-1000, 1000));
    let sout = sdiv_k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let smem = sdiv_k.push(Op::MemInit);
    for lane in 0..LANES as u8 {
        let va = sdiv_k.push(Op::Load { mem: smem, buf: sa, lane });
        let vb = sdiv_k.push(Op::Load { mem: smem, buf: sb, lane });
        let q = sdiv_k.push(Op::Div(va, vb));
        let r = sdiv_k.push(Op::Rem(va, vb));
        let sum = sdiv_k.push(Op::Add(q, r));
        sdiv_k.add_output(sout, lane, sum);
    }

    let o_hash = compile_kernel_direct(&k_hash, RegBudget::default(), TargetIsa::X86_64).expect("hash o");
    let o_bfly = emit_kernel_elf(&k_bfly, RegBudget::default(), TargetIsa::X86_64, true, true).expect("bfly call_batch o");
    let o_bfly_inl = compile_kernel_direct(&k_bfly, RegBudget::default(), TargetIsa::X86_64).expect("bfly inline o");
    let o_aff = compile_kernel_direct(&k_aff, RegBudget::default(), TargetIsa::X86_64).expect("aff o");
    let o_sel = compile_rust_to_elf(SIGNED_SELECT_SRC, RegBudget::default(), TargetIsa::X86_64).expect("sel o");
    let o_sdiv = compile_kernel_direct(&sdiv_k, RegBudget::default(), TargetIsa::X86_64).expect("sdiv o");
    let arm_hash = compile_kernel_direct(&k_hash, RegBudget::default(), TargetIsa::Aarch64).expect("arm hash o");

    let arm_path = tmp.join("arm_hash.o");
    fs::write(&arm_path, &arm_hash.elf_bytes).expect("write arm .o");
    let re_arm = Command::new("readelf").args(["-h", "-S", "-W", "-s"]).arg(&arm_path).output().expect("readelf arm");
    assert!(re_arm.status.success(), "readelf arm failed: {}", String::from_utf8_lossy(&re_arm.stderr));
    let re_arm_txt = String::from_utf8_lossy(&re_arm.stdout);
    assert!(re_arm_txt.contains("AArch64"), "readelf missing AArch64:\n{re_arm_txt}");
    assert!(re_arm_txt.contains(".note.GNU-stack"), "readelf arm missing .note.GNU-stack");
    assert!(re_arm_txt.contains("zero_cc_modular_hash_lut_batch"), "readelf arm missing batch symbol");

    // Verify all x86_64 .o files via readelf -SWs and objdump -d -M intel BEFORE linking!
    let mut o_paths = Vec::new();
    for (label, obj, expect_cmov, allow_idiv, expect_call) in [
        ("hash", &o_hash, false, false, false),
        ("bfly_call", &o_bfly, false, false, true),
        ("aff", &o_aff, false, false, false),
        ("sel", &o_sel, true, false, false),
        ("sdiv", &o_sdiv, false, true, false),
    ] {
        let p = tmp.join(format!("{label}.o"));
        fs::write(&p, &obj.elf_bytes).expect("write x86 .o");

        let re = Command::new("readelf").args(["-S", "-W", "-s"]).arg(&p).output().expect("readelf x86");
        assert!(re.status.success(), "readelf failed on {label}");
        let re_txt = String::from_utf8_lossy(&re.stdout);
        let gnu_line = re_txt.lines().find(|l| l.contains(".note.GNU-stack")).expect("missing .note.GNU-stack");
        assert!(!gnu_line.contains(" X"), ".note.GNU-stack must not have exec flag: {gnu_line}");
        assert!(re_txt.contains(&obj.kernel_sym) && re_txt.contains(&obj.batch_sym));

        let dis = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&p).output().expect("objdump x86 .o");
        assert!(dis.status.success(), "objdump failed on {label}");
        let asm = String::from_utf8_lossy(&dis.stdout);
        let ks = inspect_intel_symbol(&asm, &obj.kernel_sym);
        let bs = inspect_intel_symbol(&asm, &obj.batch_sym);
        eprintln!(
            "DIRECT .O [{label}] {}: insns={} cmovs={} jcc={} idiv={} rsp_mem={} | {}: insns={} calls={} rsp_mem={} share={}%",
            obj.kernel_sym, ks.insns, ks.cmovs, ks.cond_jumps, ks.idivs, ks.rsp_mem_ops,
            obj.batch_sym, bs.insns, bs.calls, bs.rsp_mem_ops, obj.direct_share_pct
        );
        assert!(ks.insns > 0 && bs.insns > 0);
        assert_eq!(ks.rsp_mem_ops, 0, "{label} kernel had stack-pointer memory spill!");
        assert_eq!(bs.rsp_mem_ops, 0, "{label} batch loop had stack-pointer memory spill!");
        if !allow_idiv {
            assert_eq!(ks.idivs, 0, "{label} had unexpected idiv!");
            assert_eq!(bs.idivs, 0, "{label} batch had unexpected idiv!");
            assert_eq!(ks.cond_jumps, 0, "{label} kernel must have 0 conditional jumps on data!");
        }
        if expect_cmov {
            assert!(ks.cmovs >= LANES * 2, "{label} expected cmov lowering, got {}", ks.cmovs);
        }
        if expect_call {
            assert_eq!(bs.calls, 1, "expected 1 direct CALL relocation in batch");
        }
        o_paths.push(p);
    }

    // Also compile ONLY the thin Rust FFI shim to .o to measure exact real-objdump instruction ratio!
    let shim_only_rs = tmp.join("shim_only.rs");
    let shim_only_o = tmp.join("shim_only.o");
    fs::write(&shim_only_rs, &o_hash.rust_ffi_shim).expect("write shim_only.rs");
    let rc_shim = Command::new("rustc")
        .env_remove("RUSTFLAGS")
        .args(["--crate-type", "lib", "--emit=obj", "-C", "opt-level=3", "-o"])
        .arg(&shim_only_o)
        .arg(&shim_only_rs)
        .output()
        .expect("compile shim_only.o");
    assert!(rc_shim.status.success(), "shim_only compile failed: {}", String::from_utf8_lossy(&rc_shim.stderr));
    let dis_shim = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&shim_only_o).output().expect("objdump shim");
    let shim_insns = String::from_utf8_lossy(&dis_shim.stdout)
        .lines()
        .filter(|l| l.split('\t').count() >= 3)
        .count();
    let direct_o_dis = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&o_paths[0]).output().expect("objdump hash.o");
    let direct_o_insns = String::from_utf8_lossy(&direct_o_dis.stdout)
        .lines()
        .filter(|l| l.split('\t').count() >= 3 && !l.contains("\tnop"))
        .count();
    let measured_direct_pct = (direct_o_insns * 100) / (direct_o_insns + shim_insns);
    eprintln!(
        "MEASURED OBJDUMP SHARE [modular_hash_lut]: direct_elf_insns={direct_o_insns} rust_shim_insns={shim_insns} direct_pct={measured_direct_pct}%"
    );
    assert!(measured_direct_pct >= 80, "expected >= 80% direct zero-cc machine instructions, got {measured_direct_pct}%");

    // Generate deterministic test vectors in parent process, runKernel::eval on 2,200+ vectors per kernel,
    // and verify linked native ELF64 binary produces 100% bit-exact outputs!
    let samples = 2_200usize;
    let mut rng = SplitMix64(0x2026_1124_dead_beef);
    let lut: [i64; 8] = [13, 29, 47, 71, 101, 211, 509, 997];

    let mut hash_a = vec![[0i64; 8]; samples];
    let mut hash_b = vec![[0i64; 8]; samples];
    let mut hash_exp = vec![[0i64; 8]; samples];

    let mut bfly_x = vec![[0i64; 8]; samples];
    let mut bfly_y = vec![[0i64; 8]; samples];
    let mut bfly_exp = vec![[0i64; 8]; samples];

    let mut aff_s = vec![[0i64; 8]; samples];
    let mut aff_b = vec![[0i64; 8]; samples];
    let mut aff_exp = vec![[0i64; 8]; samples];

    let mut sel_a = vec![[0i64; 8]; samples];
    let mut sel_b = vec![[0i64; 8]; samples];
    let mut sel_exp = vec![[0i64; 8]; samples];

    let mut div_a = vec![[0i64; 8]; 256];
    let mut div_b = vec![[1i64; 8]; 256];
    let mut div_exp = vec![[0i64; 8]; 256];

    let z8 = [0i64; 8];
    for i in 0..samples {
        for l in 0..LANES {
            hash_a[i][l] = match i { 0 => 0, 1 => 65535, 2 => 251, _ => rng.range(0, 65535) };
            hash_b[i][l] = match i { 0 => 0, 1 => 65535, 2 => 250, _ => rng.range(0, 65535) };
            bfly_x[i][l] = match i { 0 => 0, 1 => 16383, _ => rng.range(0, 16383) };
            bfly_y[i][l] = match i { 0 => 16383, 1 => 0, _ => rng.range(0, 16383) };
            aff_s[i][l] = match i { 0 => 0, 1 => 32767, 2 => 17, _ => rng.range(0, 32767) };
            aff_b[i][l] = match i { 0 => 0, 1 => 4095, 2 => 16, _ => rng.range(0, 4095) };
            sel_a[i][l] = match i { 0 => 0, 1 => -100_000, 2 => 100_000, 3 => -1, _ => rng.range(-100_000, 100_000) };
            sel_b[i][l] = match i { 0 => 0, 1 => 100_000, 2 => -100_000, 3 => -1, _ => rng.range(-100_000, 100_000) };
        }
        hash_exp[i] = k_hash.eval(&[&hash_a[i], &hash_b[i], &lut, &z8]).expect("eval hash")[3];
        bfly_exp[i] = k_bfly.eval(&[&bfly_x[i], &bfly_y[i], &z8]).expect("eval bfly")[2];
        aff_exp[i] = k_aff.eval(&[&aff_s[i], &aff_b[i], &z8]).expect("eval aff")[2];
        sel_exp[i] = k_sel.eval(&[&sel_a[i], &sel_b[i], &z8]).expect("eval sel")[2];
    }
    for i in 0..256 {
        for l in 0..LANES {
            div_a[i][l] = match (i, l) {
                (0, 0) => i64::MIN, (0, 1) => i64::MAX, (0, 2) => 0, (0, 3) => -1,
                _ => rng.range(-1_000_000_000_000, 1_000_000_000_000),
            };
            let mut d = rng.range(-1000, 1000);
            if d == 0 || (div_a[i][l] == i64::MIN && d == -1) { d = 7; }
            div_b[i][l] = d;
        }
        div_exp[i] = sdiv_k.eval(&[&div_a[i], &div_b[i], &z8]).expect("eval sdiv")[2];
    }

    let write_i64_blocks = |path: &std::path::Path, blocks: &[[i64; 8]]| {
        let mut raw = Vec::with_capacity(blocks.len() * 64);
        for blk in blocks {
            for &v in blk { raw.extend_from_slice(&v.to_le_bytes()); }
        }
        fs::write(path, raw).expect("write bin blocks");
    };
    write_i64_blocks(&tmp.join("hash_a.bin"), &hash_a);
    write_i64_blocks(&tmp.join("hash_b.bin"), &hash_b);
    write_i64_blocks(&tmp.join("hash_exp.bin"), &hash_exp);
    write_i64_blocks(&tmp.join("bfly_x.bin"), &bfly_x);
    write_i64_blocks(&tmp.join("bfly_y.bin"), &bfly_y);
    write_i64_blocks(&tmp.join("bfly_exp.bin"), &bfly_exp);
    write_i64_blocks(&tmp.join("aff_s.bin"), &aff_s);
    write_i64_blocks(&tmp.join("aff_b.bin"), &aff_b);
    write_i64_blocks(&tmp.join("aff_exp.bin"), &aff_exp);
    write_i64_blocks(&tmp.join("sel_a.bin"), &sel_a);
    write_i64_blocks(&tmp.join("sel_b.bin"), &sel_b);
    write_i64_blocks(&tmp.join("sel_exp.bin"), &sel_exp);
    write_i64_blocks(&tmp.join("div_a.bin"), &div_a);
    write_i64_blocks(&tmp.join("div_b.bin"), &div_b);
    write_i64_blocks(&tmp.join("div_exp.bin"), &div_exp);

    let bfly_inl_path = tmp.join("bfly_inl.o");
    // Export inline batch under distinct name to benchmark inline vs naive too.
    let mut bfly_fast_k = k_bfly.clone();
    bfly_fast_k.name = String::from("butterfly_fast");
    let o_bfly_fast = compile_kernel_direct(&bfly_fast_k, RegBudget::default(), TargetIsa::X86_64).expect("bfly fast");
    fs::write(&bfly_inl_path, &o_bfly_fast.elf_bytes).expect("write bfly_inl.o");
    let _ = o_bfly_inl;

    let mut runner_src = String::new();
    runner_src.push_str("use std::fs;\nuse std::hint::black_box;\nuse std::time::Instant;\n\n");
    for obj in [&o_hash, &o_bfly, &o_bfly_fast, &o_aff, &o_sel, &o_sdiv] {
        runner_src.push_str(&obj.rust_ffi_shim);
        runner_src.push('\n');
    }
    runner_src.push_str(
        r#"
fn load_blocks(dir: &str, name: &str) -> Vec<[i64; 8]> {
    let bytes = fs::read(format!("{dir}/{name}")).expect("read bin");
    bytes.chunks_exact(64).map(| b | {
        let mut out = [0i64; 8];
        for (i, c) in b.chunks_exact(8).enumerate() {
            out[i] = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        }
        out
    }).collect()
}

#[inline(never)]
fn naive_modular_hash_lut(a: &[[i64; 8]], b: &[[i64; 8]], lut: &[i64; 8], out: &mut [[i64; 8]]) {
    let n = a.len().min(b.len()).min(out.len());
    for blk in 0..n {
        for l in 0..8 {
            let s = (a[blk][l].wrapping_mul(17).wrapping_add(b[blk][l].wrapping_mul(17))) / 8;
            let m = s % 251;
            let idx = m % 8;
            assert!(idx >= 0 && idx < 8);
            out[blk][l] = m.wrapping_add(lut[idx as usize]).wrapping_add(a[blk][l] % 64);
        }
    }
}

#[inline(never)]
fn naive_butterfly_8lane(x: &[[i64; 8]], y: &[[i64; 8]], out: &mut [[i64; 8]]) {
    let n = x.len().min(y.len()).min(out.len());
    for blk in 0..n {
        for pair in 0..4 {
            let (lo, hi) = (pair, pair + 4);
            let u = (x[blk][lo].wrapping_mul(16).wrapping_add(y[blk][lo].wrapping_mul(16))) / 4;
            let v = x[blk][hi].wrapping_mul(15).wrapping_sub(y[blk][hi].wrapping_mul(15));
            let z = (x[blk][lo].wrapping_mul(8)) % 8;
            out[blk][lo] = u.wrapping_add(v).wrapping_add(z);
            out[blk][hi] = u.wrapping_sub(v);
        }
    }
}

#[inline(never)]
fn naive_affine_filter(sig: &[[i64; 8]], bias: &[[i64; 8]], out: &mut [[i64; 8]]) {
    let n = sig.len().min(bias.len()).min(out.len());
    for blk in 0..n {
        for l in 0..8 {
            let t = sig[blk][l].wrapping_mul(9).wrapping_add(bias[blk][l].wrapping_mul(9));
            let num = t.wrapping_add(sig[blk][l].wrapping_mul(7));
            out[blk][l] = (num / 16) % 17;
        }
    }
}

fn median_ns(mut f: impl FnMut(), iters: u64) -> f64 {
    let mut runs = [0.0f64; 5];
    for r in &mut runs {
        let t0 = Instant::now();
        for _ in 0..iters { f(); }
        *r = t0.elapsed().as_nanos() as f64 / iters as f64;
    }
    runs.sort_by(|a, b| a.total_cmp(b));
    runs[2]
}

fn main() {
    let dir = std::env::args().nth(1).expect("dir arg");
    let lut: [i64; 8] = [13, 29, 47, 71, 101, 211, 509, 997];

    let (ha, hb, hexp) = (load_blocks(&dir, "hash_a.bin"), load_blocks(&dir, "hash_b.bin"), load_blocks(&dir, "hash_exp.bin"));
    let (bx, by, bexp) = (load_blocks(&dir, "bfly_x.bin"), load_blocks(&dir, "bfly_y.bin"), load_blocks(&dir, "bfly_exp.bin"));
    let (asig, abias, aexp) = (load_blocks(&dir, "aff_s.bin"), load_blocks(&dir, "aff_b.bin"), load_blocks(&dir, "aff_exp.bin"));
    let (sa, sb, sexp) = (load_blocks(&dir, "sel_a.bin"), load_blocks(&dir, "sel_b.bin"), load_blocks(&dir, "sel_exp.bin"));
    let (da, db, dexp) = (load_blocks(&dir, "div_a.bin"), load_blocks(&dir, "div_b.bin"), load_blocks(&dir, "div_exp.bin"));

    // 1. Single-block <name> differential check across all 2,200 vectors!
    for i in 0..ha.len() {
        let mut got = [0i64; 8];
        modular_hash_lut(&ha[i], &hb[i], &lut, &mut got);
        assert_eq!(got, hexp[i], "single hash mismatch at {i}");

        let mut got_b = [0i64; 8];
        butterfly_8lane(&bx[i], &by[i], &mut got_b);
        assert_eq!(got_b, bexp[i], "single bfly mismatch at {i}");

        let mut got_a = [0i64; 8];
        affine_filter_8lane(&asig[i], &abias[i], &mut got_a);
        assert_eq!(got_a, aexp[i], "single aff mismatch at {i}");

        let mut got_s = [0i64; 8];
        signed_select_ashr(&sa[i], &sb[i], &mut got_s);
        assert_eq!(got_s, sexp[i], "single sel mismatch at {i}");
    }
    for i in 0..da.len() {
        let mut got_d = [0i64; 8];
        signed_div_rem_edge(&da[i], &db[i], &mut got_d);
        assert_eq!(got_d, dexp[i], "single sdiv mismatch at {i}");
    }

    // 2. Multi-block <name>_batch (including zero-block edge case + CALL relocation!) differential check!
    let mut empty: [[i64; 8]; 0] = [];
    modular_hash_lut_batch(&[], &[], &lut, &mut empty);
    butterfly_8lane_batch(&[], &[], &mut empty);

    let mut batch_got = vec![[0i64; 8]; ha.len()];
    modular_hash_lut_batch(&ha, &hb, &lut, &mut batch_got);
    assert_eq!(batch_got, hexp, "batch hash mismatch");

    butterfly_8lane_batch(&bx, &by, &mut batch_got);
    assert_eq!(batch_got, bexp, "batch call_reloc bfly mismatch");

    butterfly_fast_batch(&bx, &by, &mut batch_got);
    assert_eq!(batch_got, bexp, "batch inline bfly mismatch");

    affine_filter_8lane_batch(&asig, &abias, &mut batch_got);
    assert_eq!(batch_got, aexp, "batch aff mismatch");

    signed_select_ashr_batch(&sa, &sb, &mut batch_got);
    assert_eq!(batch_got, sexp, "batch sel mismatch");

    let mut div_got = vec![[0i64; 8]; da.len()];
    signed_div_rem_edge_batch(&da, &db, &mut div_got);
    assert_eq!(div_got, dexp, "batch sdiv mismatch");

    // 3. Live throughput measurement: Direct ELF64 .o vs rustc -O3 naive safe Rust!
    let mut out_n = vec![[0i64; 8]; ha.len()];
    let iters = 400u64;
    let t_nh = median_ns(|| naive_modular_hash_lut(black_box(&ha), black_box(&hb), black_box(&lut), black_box(&mut out_n)), iters);
    let t_dh = median_ns(|| modular_hash_lut_batch(black_box(&ha), black_box(&hb), black_box(&lut), black_box(&mut batch_got)), iters);

    let t_nb = median_ns(|| naive_butterfly_8lane(black_box(&bx), black_box(&by), black_box(&mut out_n)), iters);
    let t_db = median_ns(|| butterfly_fast_batch(black_box(&bx), black_box(&by), black_box(&mut batch_got)), iters);

    let t_na = median_ns(|| naive_affine_filter(black_box(&asig), black_box(&abias), black_box(&mut out_n)), iters);
    let t_da = median_ns(|| affine_filter_8lane_batch(black_box(&asig), black_box(&abias), black_box(&mut batch_got)), iters);

    println!("DIRECT_ELF_BENCH modular_hash_lut: naive_rustc_o3_ns={t_nh:.1} direct_elf64_ns={t_dh:.1} speedup={:.2}x", t_nh / t_dh);
    println!("DIRECT_ELF_BENCH butterfly_8lane: naive_rustc_o3_ns={t_nb:.1} direct_elf64_ns={t_db:.1} speedup={:.2}x", t_nb / t_db);
    println!("DIRECT_ELF_BENCH affine_filter_8lane: naive_rustc_o3_ns={t_na:.1} direct_elf64_ns={t_da:.1} speedup={:.2}x", t_na / t_da);
}
"#,
    );

    let runner_rs = tmp.join("runner.rs");
    let runner_bin = tmp.join("runner_bin");
    fs::write(&runner_rs, &runner_src).expect("write runner.rs");

    let mut cmd = Command::new("rustc");
    cmd.env_remove("RUSTFLAGS")
        .args(["-C", "opt-level=3", "-o"])
        .arg(&runner_bin)
        .arg(&runner_rs);
    for p in &o_paths {
        cmd.arg(format!("-Clink-arg={}", p.display()));
    }
    cmd.arg(format!("-Clink-arg={}", bfly_inl_path.display()));
    let link_out = cmd.output().expect("link .o files with rustc");
    assert!(
        link_out.status.success(),
        "linking zero-cc .o failed:\n{}",
        String::from_utf8_lossy(&link_out.stderr)
    );

    let run_out = Command::new(&runner_bin).arg(&tmp).output().expect("execute linked ELF64 binary");
    assert!(
        run_out.status.success(),
        "live hardware differential execution failed:\n{}\n{}",
        String::from_utf8_lossy(&run_out.stdout),
        String::from_utf8_lossy(&run_out.stderr)
    );
    let bench_stdout = String::from_utf8_lossy(&run_out.stdout);
    eprintln!("=== LIVE DIRECT ELF64 EXECUTION ===\n{bench_stdout}");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn judge_f1_to_f5_adversarial_knockout_regressions() {
    use zero_cc::{emit_rust, schedule_and_verify};

    // F5b: Kernel::validate() must reject BoundsCheck with limit <= 0.
    for bad_limit in [0i64, -1, -99] {
        let mut bad_bc = Kernel::new("bad_bc");
        let a = bad_bc.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 10));
        let out = bad_bc.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let mem = bad_bc.push(Op::MemInit);
        let v = bad_bc.push(Op::Load { mem, buf: a, lane: 0 });
        let chk = bad_bc.push(Op::BoundsCheck { idx: v, limit: bad_limit });
        bad_bc.add_output(out, 0, chk);
        assert_eq!(bad_bc.validate().unwrap_err(), CompileError::OutOfBounds { idx: 0, limit: bad_limit });
    }

    // F5a: AArch64 Op::Div / Op::Rem must emit i64::MIN / -1 trap (cmn + movz 0x8000 lsl 48 + cmp + brk) when in domain,
    // and elide it when proven impossible.
    let mut arm_div_full = Kernel::new("arm_div_full");
    let da = arm_div_full.add_buf("a", BufAccess::ReadOnly, Domain::top());
    let db = arm_div_full.add_buf("b", BufAccess::ReadOnly, Domain::top());
    let dout = arm_div_full.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let dmem = arm_div_full.push(Op::MemInit);
    let va = arm_div_full.push(Op::Load { mem: dmem, buf: da, lane: 0 });
    let vb = arm_div_full.push(Op::Load { mem: dmem, buf: db, lane: 0 });
    let q = arm_div_full.push(Op::Div(va, vb));
    arm_div_full.add_output(dout, 0, q);

    let mut arm_div_safe = Kernel::new("arm_div_safe");
    let sa = arm_div_safe.add_buf("a", BufAccess::ReadOnly, Domain::range(-100, 100));
    let sb = arm_div_safe.add_buf("b", BufAccess::ReadOnly, Domain::range(1, 10));
    let sout = arm_div_safe.add_buf("out", BufAccess::MutNoAlias, Domain::top());
    let smem = arm_div_safe.push(Op::MemInit);
    let sva = arm_div_safe.push(Op::Load { mem: smem, buf: sa, lane: 0 });
    let svb = arm_div_safe.push(Op::Load { mem: smem, buf: sb, lane: 0 });
    let sq = arm_div_safe.push(Op::Div(sva, svb));
    arm_div_safe.add_output(sout, 0, sq);

    let o_arm_full = emit_kernel_elf(&arm_div_full, RegBudget::default(), TargetIsa::Aarch64, false, false).expect("arm_div_full");
    let o_arm_safe = emit_kernel_elf(&arm_div_safe, RegBudget::default(), TargetIsa::Aarch64, false, false).expect("arm_div_safe");
    let words_full: Vec<u32> = o_arm_full.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let words_safe: Vec<u32> = o_arm_safe.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let brk_full = words_full.iter().filter(|&&w| w == 0xD420_0000).count();
    let brk_safe = words_safe.iter().filter(|&&w| w == 0xD420_0000).count();
    let min_movz = words_full.iter().filter(|&&w| w == (0xD2E0_0000 | (0x8000 << 5) | 17)).count();
    assert!(brk_full >= 4, "expected cbz + MIN/-1 brk #0 in kernel & batch, got {brk_full}");
    assert!(min_movz >= 2, "expected i64::MIN movz guard in AArch64 div, got {min_movz}");
    assert_eq!(brk_safe, 0, "domain-proven safe div must emit 0 brk traps");
    assert_ne!(brk_full, brk_safe);

    // F4: Proven vs unproven LoadLut bounds traps on BOTH x86_64 and AArch64 (power-of-two 8 AND non-power-of-two 10).
    for lut_len in [8usize, 10usize] {
        let mk_lut = |dom: Domain, name: &str| {
            let mut k = Kernel::new(name);
            let idx_b = k.add_buf("idx", BufAccess::ReadOnly, dom);
            let lut_b = k.add_buf("lut", BufAccess::LookupTable(lut_len), Domain::range(0, 999));
            let out_b = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
            let mem = k.push(Op::MemInit);
            for lane in 0..LANES as u8 {
                let ix = k.push(Op::Load { mem, buf: idx_b, lane });
                let lv = k.push(Op::LoadLut { buf: lut_b, idx: ix });
                k.add_output(out_b, lane, lv);
            }
            k
        };
        let k_proven = mk_lut(Domain::range(0, (lut_len - 1) as i64), "lut_proven");
        let k_unproven = mk_lut(Domain::range(-5, 50), "lut_unproven");

        for isa in [TargetIsa::X86_64, TargetIsa::Aarch64] {
            let o_prov = compile_kernel_direct(&k_proven, RegBudget::default(), isa).expect("proven lut");
            let o_unpr = compile_kernel_direct(&k_unproven, RegBudget::default(), isa).expect("unproven lut");
            assert!(o_prov.code_bytes.len() < o_unpr.code_bytes.len());
            assert_ne!(o_prov.code_bytes, o_unpr.code_bytes);
            if isa == TargetIsa::Aarch64 {
                let w_p: Vec<u32> = o_prov.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let w_u: Vec<u32> = o_unpr.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                assert_eq!(w_p.iter().filter(|&&w| w == 0xD420_0000).count(), 0, "proven AArch64 LUT must have 0 brk");
                assert!(w_u.iter().filter(|&&w| w == 0xD420_0000).count() >= LANES * 2, "unproven AArch64 LUT missing brk");
            } else {
                let ud2_p = o_prov.code_bytes.windows(2).filter(|w| *w == [0x0F, 0x0B]).count();
                let ud2_u = o_unpr.code_bytes.windows(2).filter(|w| *w == [0x0F, 0x0B]).count();
                assert_eq!(ud2_p, 0, "proven x86_64 LUT must have 0 ud2");
                assert!(ud2_u >= LANES * 2, "unproven x86_64 LUT (len={lut_len}) missing ud2");
            }
        }
    }

    // F1 + F2 + F3 + F4 live hardware link & execution!
    #[cfg(not(miri))]
    {
        let tmp = std::env::temp_dir().join(format!("zero_cc_f1_f5_{}", std::process::id()));
        let _ = fs::create_dir_all(&tmp);

        let mut o_files = Vec::new();
        let mut shims = String::new();

        // F1: nb = 1..=5 with call_batch = true (verifies no OOB panic on nb=5 AND 16-byte stack alignment before CALL!).
        for nb in 1..=5usize {
            let mut k = Kernel::new(format!("call_nb{nb}"));
            let mut in_bufs = Vec::new();
            for bi in 0..(nb - 1) {
                in_bufs.push(k.add_buf(format!("b{bi}"), BufAccess::ReadOnly, Domain::range(0, 1000)));
            }
            let out_b = k.add_buf("out", BufAccess::MutNoAlias, Domain::top());
            let mem = k.push(Op::MemInit);
            let c3 = k.push(Op::Const(3));
            for lane in 0..LANES as u8 {
                let mut acc = c3;
                for &ib in &in_bufs {
                    let v = k.push(Op::Load { mem, buf: ib, lane });
                    acc = k.push(Op::Add(acc, v));
                }
                k.add_output(out_b, lane, acc);
            }
            let obj = emit_kernel_elf(&k, RegBudget::default(), TargetIsa::X86_64, true, true).expect("emit call_batch nb");
            let p = tmp.join(format!("call_nb{nb}.o"));
            fs::write(&p, &obj.elf_bytes).expect("write call_nb.o");
            o_files.push(p);
            shims.push_str(&obj.rust_ffi_shim);
            shims.push('\n');
        }

        // F2: optimize = false with unforwarded Store(out, a + 5); Load(out) + 5 (out pre-filled with 999!).
        let mut k_f2 = Kernel::new("unopt_store_reload");
        let f2_a = k_f2.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 1000));
        let f2_out = k_f2.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let mut f2_mem = k_f2.push(Op::MemInit);
        let c5 = k_f2.push(Op::Const(5));
        for lane in 0..LANES as u8 {
            let av = k_f2.push(Op::Load { mem: f2_mem, buf: f2_a, lane });
            let a5 = k_f2.push(Op::Add(av, c5));
            f2_mem = k_f2.push(Op::Store { mem: f2_mem, buf: f2_out, lane, val: a5 });
            let reload = k_f2.push(Op::Load { mem: f2_mem, buf: f2_out, lane });
            let res = k_f2.push(Op::Add(reload, c5));
            k_f2.add_output(f2_out, lane, res);
        }
        let f2_sched = schedule_and_verify(&k_f2, RegBudget::default()).expect("schedule f2");
        let f2_rs = emit_rust(&k_f2, &f2_sched).expect("emit_rust f2");
        assert!(f2_rs.contains("out[0] = "), "emit_rust must emit intermediate Op::Store");
        let o_f2 = emit_kernel_elf(&k_f2, RegBudget::default(), TargetIsa::X86_64, false, false).expect("emit f2 x86");
        let _ = emit_kernel_elf(&k_f2, RegBudget::default(), TargetIsa::Aarch64, false, false).expect("emit f2 arm");
        let p_f2 = tmp.join("f2.o");
        fs::write(&p_f2, &o_f2.elf_bytes).expect("write f2.o");
        o_files.push(p_f2);
        shims.push_str(&o_f2.rust_ffi_shim);
        shims.push('\n');

        // F2-WAR: Transitive cross-lane write-after-read overwrite (`optimize = false`).
        // s1 = Store(mem0, out, 0, a0 + 100); y0 = Load(s1, out, 0);
        // s2 = Store(s1, out, 1, 200); s3 = Store(s2, out, 0, 999);
        // r0 = Load(s3, out, 0); out[0] = y0 + r0 (must be (a0 + 100) + 999, NOT 999 + 999 = 1998!).
        let mut k_war = Kernel::new("f2_war_cross_lane");
        let war_a = k_war.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 1000));
        let war_out = k_war.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let war_m0 = k_war.push(Op::MemInit);
        let (c100, c200, c999) = (k_war.push(Op::Const(100)), k_war.push(Op::Const(200)), k_war.push(Op::Const(999)));
        let war_a0 = k_war.push(Op::Load { mem: war_m0, buf: war_a, lane: 0 });
        let war_v1 = k_war.push(Op::Add(war_a0, c100));
        let war_s1 = k_war.push(Op::Store { mem: war_m0, buf: war_out, lane: 0, val: war_v1 });
        let war_y0 = k_war.push(Op::Load { mem: war_s1, buf: war_out, lane: 0 });
        let war_s2 = k_war.push(Op::Store { mem: war_s1, buf: war_out, lane: 1, val: c200 });
        let war_s3 = k_war.push(Op::Store { mem: war_s2, buf: war_out, lane: 0, val: c999 });
        let war_r0 = k_war.push(Op::Load { mem: war_s3, buf: war_out, lane: 0 });
        let war_r1 = k_war.push(Op::Load { mem: war_s3, buf: war_out, lane: 1 });
        let war_sum = k_war.push(Op::Add(war_y0, war_r0));
        k_war.add_output(war_out, 0, war_sum);
        k_war.add_output(war_out, 1, war_r1);
        let war_sched = schedule_and_verify(&k_war, RegBudget::default()).expect("schedule f2_war");
        let war_rs = emit_rust(&k_war, &war_sched).expect("emit_rust f2_war");
        let o_war = emit_kernel_elf(&k_war, RegBudget::default(), TargetIsa::X86_64, false, false).expect("emit f2_war x86");
        let _ = emit_kernel_elf(&k_war, RegBudget::default(), TargetIsa::Aarch64, false, false).expect("emit f2_war arm");
        let p_war = tmp.join("f2_war.o");
        fs::write(&p_war, &o_war.elf_bytes).expect("write f2_war.o");
        o_files.push(p_war);
        shims.push_str(&o_war.rust_ffi_shim);
        shims.push('\n');

        // F2-WAW: Early output_order drain must NOT precede an unscheduled Op::Store to the same (buf, lane)!
        // a0 = Load(mem0, a, 0) (10); s1 = Store(mem0, out, 0, 50); s2 = Store(s1, out, 1, 888);
        // r1 = Load(s2, out, 1); outputs = [(out, 0, a0), (out, 1, r1)] -> must yield [10, 888], NOT [50, 888]!
        let mut k_waw = Kernel::new("f2_waw_output_drain");
        let waw_a = k_waw.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 1000));
        let waw_out = k_waw.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let waw_m0 = k_waw.push(Op::MemInit);
        let (c50, c888) = (k_waw.push(Op::Const(50)), k_waw.push(Op::Const(888)));
        let waw_a0 = k_waw.push(Op::Load { mem: waw_m0, buf: waw_a, lane: 0 });
        let waw_s1 = k_waw.push(Op::Store { mem: waw_m0, buf: waw_out, lane: 0, val: c50 });
        let waw_s2 = k_waw.push(Op::Store { mem: waw_s1, buf: waw_out, lane: 1, val: c888 });
        let waw_r1 = k_waw.push(Op::Load { mem: waw_s2, buf: waw_out, lane: 1 });
        k_waw.add_output(waw_out, 0, waw_a0);
        k_waw.add_output(waw_out, 1, waw_r1);
        let waw_sched = schedule_and_verify(&k_waw, RegBudget::default()).expect("schedule f2_waw");
        let waw_rs = emit_rust(&k_waw, &waw_sched).expect("emit_rust f2_waw");
        let o_waw = emit_kernel_elf(&k_waw, RegBudget::default(), TargetIsa::X86_64, false, false).expect("emit f2_waw x86");
        let _ = emit_kernel_elf(&k_waw, RegBudget::default(), TargetIsa::Aarch64, false, false).expect("emit f2_waw arm");
        let p_waw = tmp.join("f2_waw.o");
        fs::write(&p_waw, &o_waw.elf_bytes).expect("write f2_waw.o");
        o_files.push(p_waw);
        shims.push_str(&o_waw.rust_ffi_shim);
        shims.push('\n');

        // F3: Dead unreachable Op::Div + 3 buffers + optimize = false on AVX2-isomorphic lanes!
        let mut k_f3 = Kernel::new("dead_div_avx2_3buf");
        let f3_a = k_f3.add_buf("a", BufAccess::ReadOnly, Domain::range(0, 1000));
        let f3_b = k_f3.add_buf("b", BufAccess::ReadOnly, Domain::range(1, 1000));
        let f3_out = k_f3.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let f3_mem = k_f3.push(Op::MemInit);
        let dead_l = k_f3.push(Op::Load { mem: f3_mem, buf: f3_a, lane: 0 });
        let dead_r = k_f3.push(Op::Load { mem: f3_mem, buf: f3_b, lane: 0 });
        let _ = k_f3.push(Op::Div(dead_l, dead_r));
        for lane in 0..LANES as u8 {
            let va = k_f3.push(Op::Load { mem: f3_mem, buf: f3_a, lane });
            let vb = k_f3.push(Op::Load { mem: f3_mem, buf: f3_b, lane });
            let sum = k_f3.push(Op::Add(va, vb));
            k_f3.add_output(f3_out, lane, sum);
        }
        let o_f3 = emit_kernel_elf(&k_f3, RegBudget::default(), TargetIsa::X86_64, false, false).expect("emit f3");
        let p_f3 = tmp.join("f3.o");
        fs::write(&p_f3, &o_f3.elf_bytes).expect("write f3.o");
        o_files.push(p_f3);
        shims.push_str(&o_f3.rust_ffi_shim);
        shims.push('\n');

        // F4 live OOB trap check: unproven LoadLut with OOB negative index (-1) must trap (non-zero exit)!
        let mut k_f4 = Kernel::new("oob_lut_trap");
        let f4_idx = k_f4.add_buf("idx", BufAccess::ReadOnly, Domain::range(-10, 30));
        let f4_lut = k_f4.add_buf("lut", BufAccess::LookupTable(8), Domain::range(0, 100));
        let f4_out = k_f4.add_buf("out", BufAccess::MutNoAlias, Domain::top());
        let f4_mem = k_f4.push(Op::MemInit);
        for lane in 0..LANES as u8 {
            let ix = k_f4.push(Op::Load { mem: f4_mem, buf: f4_idx, lane });
            let lv = k_f4.push(Op::LoadLut { buf: f4_lut, idx: ix });
            k_f4.add_output(f4_out, lane, lv);
        }
        let o_f4 = compile_kernel_direct(&k_f4, RegBudget::default(), TargetIsa::X86_64).expect("emit f4");
        let p_f4 = tmp.join("f4.o");
        fs::write(&p_f4, &o_f4.elf_bytes).expect("write f4.o");
        o_files.push(p_f4);
        shims.push_str(&o_f4.rust_ffi_shim);
        shims.push('\n');

        let mut drv = String::new();
        drv.push_str(&shims);
        drv.push_str("\nmod rust_f2 {\n#![forbid(unsafe_code)]\n");
        drv.push_str(&f2_rs);
        drv.push('\n');
        drv.push_str(&war_rs);
        drv.push('\n');
        drv.push_str(&waw_rs);
        drv.push_str("\n}\n");
        drv.push_str(
            r#"
fn main() {
    if std::env::args().any(|a| a == "--trap-lut") {
        let bad_idx = [-1i64, 0, 0, 0, 0, 0, 0, 0];
        let lut = [10i64, 20, 30, 40, 50, 60, 70, 80];
        let mut out = [0i64; 8];
        oob_lut_trap(&bad_idx, &lut, &mut out);
        return;
    }

    // F1: Execute call_nb1..call_nb5 across multiple blocks!
    let b0 = [[1i64; 8]; 4];
    let b1 = [[2i64; 8]; 4];
    let b2 = [[4i64; 8]; 4];
    let b3 = [[8i64; 8]; 4];
    let mut out = [[999i64; 8]; 4];

    call_nb1_batch(&mut out);
    assert_eq!(out, [[3i64; 8]; 4]);
    call_nb2_batch(&b0, &mut out);
    assert_eq!(out, [[4i64; 8]; 4]);
    call_nb3_batch(&b0, &b1, &mut out);
    assert_eq!(out, [[6i64; 8]; 4]);
    call_nb4_batch(&b0, &b1, &b2, &mut out);
    assert_eq!(out, [[10i64; 8]; 4]);
    call_nb5_batch(&b0, &b1, &b2, &b3, &mut out);
    assert_eq!(out, [[18i64; 8]; 4]);

    // F2: Pre-fill out with 999; Store(out, a + 5); Load(out) + 5 must yield a + 10 (NOT 1004!).
    let a_in = [7i64, 11, 13, 17, 19, 23, 29, 31];
    let mut out_s = [999i64; 8];
    unopt_store_reload(&a_in, &mut out_s);
    for l in 0..8 { assert_eq!(out_s[l], a_in[l] + 10, "F2 single mismatch lane {l}"); }
    let mut out_rs = [999i64; 8];
    rust_f2::unopt_store_reload(&a_in, &mut out_rs);
    assert_eq!(out_rs, out_s);

    let mut out_b = [[999i64; 8]; 3];
    unopt_store_reload_batch(&[a_in; 3], &mut out_b);
    for blk in &out_b { assert_eq!(blk, &out_s); }

    // F2-WAR: (a0=5 => y0=105, r0=999 => out[0]=1104, out[1]=200; NOT 1998!)
    let war_in = [5i64, 0, 0, 0, 0, 0, 0, 0];
    let mut war_elf = [777i64; 8];
    let mut war_rust = [777i64; 8];
    f2_war_cross_lane(&war_in, &mut war_elf);
    rust_f2::f2_war_cross_lane(&war_in, &mut war_rust);
    assert_eq!(war_elf[0], 1104, "F2-WAR ELF64 overwrote out[0] before y0 read!");
    assert_eq!(war_elf[1], 200);
    assert_eq!(war_elf, war_rust);

    // F2-WAW: (a0=10, s1 writes 50 to lane 0, final output writes a0=10 to lane 0 => [10, 888], NOT [50, 888]!)
    let waw_in = [10i64, 0, 0, 0, 0, 0, 0, 0];
    let mut waw_elf = [777i64; 8];
    let mut waw_rust = [777i64; 8];
    f2_waw_output_drain(&waw_in, &mut waw_elf);
    rust_f2::f2_waw_output_drain(&waw_in, &mut waw_rust);
    assert_eq!(waw_elf[0], 10, "F2-WAW s1 overwrote early-drained output out[0]!");
    assert_eq!(waw_elf[1], 888);
    assert_eq!(waw_elf, waw_rust);

    // F3: Dead Div + 3 buffers + AVX2 must write cleanly to out without segfaulting on r9!
    let x_in = [10i64, 20, 30, 40, 50, 60, 70, 80];
    let y_in = [1i64, 2, 3, 4, 5, 6, 7, 8];
    let mut out_f3 = [0i64; 8];
    dead_div_avx2_3buf(&x_in, &y_in, &mut out_f3);
    assert_eq!(out_f3, [11, 22, 33, 44, 55, 66, 77, 88]);
    let mut out_f3_b = [[0i64; 8]; 3];
    dead_div_avx2_3buf_batch(&[x_in; 3], &[y_in; 3], &mut out_f3_b);
    for blk in &out_f3_b { assert_eq!(blk, &out_f3); }

    // F4 valid LUT read:
    let good_idx = [7i64, 6, 5, 4, 3, 2, 1, 0];
    let lut = [10i64, 20, 30, 40, 50, 60, 70, 80];
    let mut out_f4 = [0i64; 8];
    oob_lut_trap(&good_idx, &lut, &mut out_f4);
    assert_eq!(out_f4, [80, 70, 60, 50, 40, 30, 20, 10]);
}
"#,
        );

        let drv_rs = tmp.join("drv.rs");
        let drv_bin = tmp.join("drv_bin");
        fs::write(&drv_rs, &drv).expect("write drv.rs");
        let mut cmd = Command::new("rustc");
        cmd.env_remove("RUSTFLAGS").args(["-C", "opt-level=3", "-o"]).arg(&drv_bin).arg(&drv_rs);
        for p in &o_files { cmd.arg(format!("-Clink-arg={}", p.display())); }
        let linked = cmd.output().expect("link F1-F5 driver");
        assert!(linked.status.success(), "link F1-F5 failed:\n{}", String::from_utf8_lossy(&linked.stderr));

        // 1. Valid execution of F1, F2, F3, F4-in-bounds must succeed!
        let ok_run = Command::new(&drv_bin).output().expect("run F1-F5 valid");
        assert!(ok_run.status.success(), "F1-F5 valid run failed:\n{}", String::from_utf8_lossy(&ok_run.stderr));

        // 2. OOB negative LUT index (-1 on power-of-two table!) must trap (non-zero exit), matching Kernel::eval Err!
        assert!(k_f4.eval(&[&[-1, 0, 0, 0, 0, 0, 0, 0], &[10, 20, 30, 40, 50, 60, 70, 80], &[0; 8]]).is_err());
        let trap_run = Command::new(&drv_bin).arg("--trap-lut").output().expect("run F4 trap");
        assert!(!trap_run.status.success(), "F4 OOB LoadLut(-1) failed to trap!");
        let f2_eval = k_f2.eval(&[&[7, 11, 13, 17, 19, 23, 29, 31], &[999; 8]]).expect("eval f2");
        assert_eq!(f2_eval[1], [17, 21, 23, 27, 29, 33, 39, 41]);
        let war_eval = k_war.eval(&[&[5, 0, 0, 0, 0, 0, 0, 0], &[777; 8]]).expect("eval f2_war");
        assert_eq!((war_eval[1][0], war_eval[1][1]), (1104, 200));
        let waw_eval = k_waw.eval(&[&[10, 0, 0, 0, 0, 0, 0, 0], &[777; 8]]).expect("eval f2_waw");
        assert_eq!((waw_eval[1][0], waw_eval[1][1]), (10, 888));
        let f3_eval = k_f3.eval(&[&[10, 20, 30, 40, 50, 60, 70, 80], &[1, 2, 3, 4, 5, 6, 7, 8], &[0; 8]]).expect("eval f3");
        assert_eq!(f3_eval[2], [11, 22, 33, 44, 55, 66, 77, 88]);

        let _ = fs::remove_dir_all(&tmp);
    }
}

fn expected_multiblock_output_bytes(k: &Kernel, inputs: &[&[i64]], blocks: usize) -> Vec<u8> {
    let out_idx = k
        .bufs
        .iter()
        .position(|b| b.access == BufAccess::MutNoAlias)
        .expect("kernel must have a primary MutNoAlias output buffer");
    let out_decl = &k.bufs[out_idx];
    let mut bytes = Vec::with_capacity(blocks * out_decl.block_bytes());
    for blk in 0..blocks {
        let mut slices: Vec<&[i64]> = Vec::with_capacity(k.bufs.len());
        for (idx, decl) in k.bufs.iter().enumerate() {
            match decl.access {
                BufAccess::LookupTable(_) => slices.push(inputs[idx]),
                _ => {
                    let lanes = usize::from(decl.lanes);
                    let start = blk * lanes;
                    slices.push(&inputs[idx][start..start + lanes]);
                }
            }
        }
        let res = k.eval(&slices).expect("Kernel::eval block parity");
        for &val in res[out_idx].iter().take(usize::from(out_decl.lanes)) {
            out_decl.elem_ty.push_le_elem(&mut bytes, val);
        }
    }
    bytes
}

#[cfg(not(miri))]
fn write_exec_file(path: &std::path::Path, bytes: &[u8]) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, bytes).expect("write ET_EXEC binary");
    let mut perms = fs::metadata(path).expect("stat ET_EXEC").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).expect("chmod 0755 ET_EXEC");
}

#[cfg(not(miri))]
fn verify_live_et_exec_readelf_objdump_and_run(
    tmp: &std::path::Path,
    tag: &str,
    exec: &zero_cc::StandaloneExec,
    expected: &[u8],
) {
    let bin_path = tmp.join(tag);
    write_exec_file(&bin_path, &exec.elf_bytes);

    let re = Command::new("readelf")
        .args(["-h", "-l", "-S", "-s", "-W"])
        .arg(&bin_path)
        .output()
        .expect("run readelf on ET_EXEC");
    assert!(
        re.status.success(),
        "readelf failed on {tag}:\n{}",
        String::from_utf8_lossy(&re.stderr)
    );
    let re_txt = String::from_utf8_lossy(&re.stdout);
    assert!(!re_txt.contains("RWE"), "{tag} readelf reported W+X RWE segment:\n{re_txt}");
    assert!(re_txt.contains("EXEC"), "{tag} readelf missing EXEC type:\n{re_txt}");
    assert!(re_txt.contains("GNU_STACK"), "{tag} readelf missing PT_GNU_STACK:\n{re_txt}");
    assert!(re_txt.contains("_start"), "{tag} readelf missing _start symbol:\n{re_txt}");

    let od = Command::new("objdump")
        .args(["-d", "-M", "intel"])
        .arg(&bin_path)
        .output()
        .expect("run objdump on ET_EXEC");
    assert!(
        od.status.success(),
        "objdump failed on {tag}:\n{}",
        String::from_utf8_lossy(&od.stderr)
    );
    let asm = String::from_utf8_lossy(&od.stdout);
    let sym_headers: Vec<&str> = asm.lines().filter(|l| l.ends_with(">:")).collect();
    assert_eq!(
        sym_headers.len(),
        3,
        "{tag} expected exactly 3 direct symbols (100% zero-cc, zero libc/crt glue), got: {sym_headers:?}"
    );
    for sym in [&exec.kernel_sym, &exec.batch_sym, &exec.entry_sym] {
        let stats = inspect_intel_symbol(&asm, sym);
        assert!(stats.insns > 0, "{tag} empty disassembly for {sym}");
        assert_eq!(stats.rsp_mem_ops, 0, "{tag} unexpected stack spill in {sym}:\n{asm}");
    }

    let run = Command::new(&bin_path).output().expect("execute standalone ET_EXEC directly");
    assert_eq!(
        run.stdout.as_slice(),
        expected,
        "{tag} live ET_EXEC stdout mismatch against Kernel::eval"
    );
    assert_eq!(expected.len(), exec.output_bytes, "{tag} output_bytes metadata mismatch");
    let expected_exit = (expected.iter().fold(0u64, |acc, &b| acc + u64::from(b)) & 0x7F) as i32;
    assert_eq!(
        run.status.code(),
        Some(expected_exit),
        "{tag} live ET_EXEC exit code mismatch (sum(stdout) & 0x7F)"
    );
}

#[test]
fn standalone_et_exec_100pct_direct_linux_kernel_and_wx_airlock() {
    let k_hash = build_modular_hash_lut();
    let budget = RegBudget::default();

    for (isa, mach) in [(TargetIsa::X86_64, EM_X86_64), (TargetIsa::Aarch64, EM_AARCH64)] {
        let def = compile_standalone_elf_exec(&k_hash, budget, isa, &[])
            .expect("empty default 1-block ET_EXEC");
        assert_eq!(def.shim_glue_insns, 0);
        assert_eq!(def.direct_share_pct, 100);
        assert_eq!(def.blocks, 1);
        assert_eq!(def.entry_sym, "_start");
        assert!(def.zero_spills);
        let summary = verify_elf64_executable(&def.elf_bytes).expect("verify default ET_EXEC");
        assert_eq!(summary.machine, mach);
        assert_eq!(summary.entry_vaddr, def.entry_vaddr);
        assert_eq!(summary.data_vaddr, def.data_vaddr);
        assert_eq!(summary.text_size, def.code_bytes.len() as u64);
    }

    let base_x86 = compile_standalone_elf_exec(&k_hash, budget, TargetIsa::X86_64, &[])
        .expect("x86_64 ET_EXEC for W^X knockouts");
    let mut bad_wx = base_x86.elf_bytes.clone();
    bad_wx[64 + 4] |= 2;
    assert!(verify_elf64_executable(&bad_wx).is_err(), "RX+W PT_LOAD must be rejected");

    let mut bad_stack = base_x86.elf_bytes.clone();
    bad_stack[64 + 2 * 56 + 4] |= 1;
    assert!(verify_elf64_executable(&bad_stack).is_err(), "executable PT_GNU_STACK must be rejected");

    let mut bad_align = base_x86.elf_bytes.clone();
    bad_align[64 + 56 + 16] = bad_align[64 + 56 + 16].wrapping_add(1);
    assert!(verify_elf64_executable(&bad_align).is_err(), "broken page congruence must be rejected");

    let mut k_narrow = Kernel::new("narrow_i32_8_to_i64_4");
    let a = k_narrow.add_typed_buf("a", BufAccess::ReadOnly, ElemTy::I32, 8, ElemTy::I32.natural_domain());
    let out = k_narrow.add_typed_buf("out", BufAccess::MutNoAlias, ElemTy::I64, 4, Domain::top());
    let mem = k_narrow.push(Op::MemInit);
    for lane in 0..4u8 {
        let lo = k_narrow.push(Op::Load { mem, buf: a, lane: lane * 2 });
        let hi = k_narrow.push(Op::Load { mem, buf: a, lane: lane * 2 + 1 });
        let sum = k_narrow.push(Op::Add(lo, hi));
        k_narrow.add_output(out, lane, sum);
    }

    #[cfg(not(miri))]
    {
        let tmp = std::env::temp_dir().join(format!("zero_cc_et_exec_wx_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create et_exec tmp dir");

        let mut rng = SplitMix64(0xE1F6_4001_5EED_CAFE);
        let hash_blocks = 4usize;
        let hash_a: Vec<i64> = (0..hash_blocks * LANES).map(|_| rng.range(0, 65535)).collect();
        let hash_b: Vec<i64> = (0..hash_blocks * LANES).map(|_| rng.range(0, 65535)).collect();
        let hash_lut: Vec<i64> = (0..8).map(|_| rng.range(0, 1023)).collect();
        let hash_out = vec![0i64; hash_blocks * LANES];
        let hash_inputs: [&[i64]; 4] = [&hash_a, &hash_b, &hash_lut, &hash_out];
        let hash_exec = compile_standalone_elf_exec(&k_hash, budget, TargetIsa::X86_64, &hash_inputs)
            .expect("compile 4-block modular_hash_lut ET_EXEC");
        assert_eq!(hash_exec.blocks, hash_blocks);
        assert_eq!(hash_exec.shim_glue_insns, 0);
        assert_eq!(hash_exec.direct_share_pct, 100);
        let hash_exp = expected_multiblock_output_bytes(&k_hash, &hash_inputs, hash_blocks);
        verify_live_et_exec_readelf_objdump_and_run(&tmp, "modular_hash_lut_4blk", &hash_exec, &hash_exp);

        let sel_blocks = 3usize;
        let mut sel_a: Vec<i64> = (0..sel_blocks * LANES).map(|_| rng.range(-100_000, 100_000)).collect();
        let mut sel_b: Vec<i64> = (0..sel_blocks * LANES).map(|_| rng.range(-100_000, 100_000)).collect();
        sel_a[0] = -99_999;
        sel_b[0] = -99_999;
        sel_a[1] = -8_191;
        sel_b[1] = 42_000;
        let sel_out = vec![0i64; sel_blocks * LANES];
        let sel_inputs: [&[i64]; 3] = [&sel_a, &sel_b, &sel_out];
        let k_sel = parse_rust_kernel(SIGNED_SELECT_SRC).expect("parse signed_select_ashr");
        let sel_exec = compile_rust_to_elf_exec(SIGNED_SELECT_SRC, budget, TargetIsa::X86_64, &sel_inputs)
            .expect("compile 3-block signed_select_ashr ET_EXEC");
        assert_eq!(sel_exec.blocks, sel_blocks);
        assert_eq!(sel_exec.shim_glue_insns, 0);
        assert_eq!(sel_exec.direct_share_pct, 100);
        let sel_exp = expected_multiblock_output_bytes(&k_sel, &sel_inputs, sel_blocks);
        verify_live_et_exec_readelf_objdump_and_run(&tmp, "signed_select_ashr_3blk", &sel_exec, &sel_exp);

        let nar_blocks = 4usize;
        let mut nar_a: Vec<i64> = (0..nar_blocks * 8)
            .map(|i| match i % 6 {
                0 => i64::from(i32::MIN),
                1 => i64::from(i32::MAX),
                2 => -1,
                3 => 0x7FFF_FFFF,
                4 => -1_234_567,
                _ => rng.range(i64::from(i32::MIN), i64::from(i32::MAX)),
            })
            .collect();
        nar_a[5] = 0;
        let nar_out = vec![0i64; nar_blocks * 4];
        let nar_inputs: [&[i64]; 2] = [&nar_a, &nar_out];
        let nar_exec = compile_standalone_elf_exec(&k_narrow, budget, TargetIsa::X86_64, &nar_inputs)
            .expect("compile 4-block narrow [i32; 8] -> [i64; 4] ET_EXEC");
        assert_eq!(nar_exec.blocks, nar_blocks);
        assert_eq!(nar_exec.shim_glue_insns, 0);
        assert_eq!(nar_exec.direct_share_pct, 100);
        let nar_exp = expected_multiblock_output_bytes(&k_narrow, &nar_inputs, nar_blocks);
        verify_live_et_exec_readelf_objdump_and_run(&tmp, "narrow_i32_8_to_i64_4_4blk", &nar_exec, &nar_exp);

        let arm_exec = compile_standalone_elf_exec(&k_hash, budget, TargetIsa::Aarch64, &hash_inputs)
            .expect("compile 4-block AArch64 ET_EXEC");
        assert_eq!(arm_exec.shim_glue_insns, 0);
        assert_eq!(arm_exec.direct_share_pct, 100);
        let arm_path = tmp.join("modular_hash_lut_arm64");
        write_exec_file(&arm_path, &arm_exec.elf_bytes);
        let re_arm = Command::new("readelf")
            .args(["-h", "-l", "-S", "-s", "-W"])
            .arg(&arm_path)
            .output()
            .expect("readelf on AArch64 ET_EXEC");
        assert!(re_arm.status.success());
        let re_arm_txt = String::from_utf8_lossy(&re_arm.stdout);
        assert!(!re_arm_txt.contains("RWE"), "AArch64 ET_EXEC has RWE:\n{re_arm_txt}");
        assert!(re_arm_txt.contains("EXEC"));
        assert!(re_arm_txt.contains("AArch64"));
        assert!(re_arm_txt.contains("GNU_STACK"));
        assert!(re_arm_txt.contains("_start"));

        let _ = fs::remove_dir_all(&tmp);
    }
}

#[test]
fn narrow_elem_ty_all_six_widths_variable_lanes_and_u8_rex_pressure() {
    let budget = RegBudget::default();
    let all_tys = [ElemTy::I64, ElemTy::I32, ElemTy::I16, ElemTy::U32, ElemTy::U16, ElemTy::U8];
    let edge_vals: [i64; 16] = [
        -1,
        0,
        1,
        7,
        -7,
        0x7F,
        0x80,
        0xFE,
        0xFF,
        0x7FFF,
        0x8000,
        0xFFFF,
        -32_768,
        0x7FFF_FFFF,
        -2_147_483_648,
        i64::MIN,
    ];

    #[cfg(not(miri))]
    let tmp = {
        let dir = std::env::temp_dir().join(format!("zero_cc_narrow_elem_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create narrow_elem tmp dir");
        dir
    };

    for &ty in &all_tys {
        let rty = ty.rust_name();
        let (exp_ldr, exp_str, exp_lut) = match ty {
            ElemTy::I64 => (0xF940_0000u32, 0xF900_0000u32, 0xF860_7800u32),
            ElemTy::I32 => (0xB980_0000u32, 0xB900_0000u32, 0xB8A0_7800u32),
            ElemTy::I16 => (0x7980_0000u32, 0x7900_0000u32, 0x78A0_7800u32),
            ElemTy::U32 => (0xB940_0000u32, 0xB900_0000u32, 0xB860_7800u32),
            ElemTy::U16 => (0x7940_0000u32, 0x7900_0000u32, 0x7860_7800u32),
            ElemTy::U8 => (0x3940_0000u32, 0x3900_0000u32, 0x3860_6800u32),
        };

        for lanes in [4u8, 8u8] {
            let mut k = Kernel::new(format!("narrow_{rty}_{lanes}"));
            let a = k.add_typed_buf("a", BufAccess::ReadOnly, ty, lanes, ty.natural_domain());
            let lut = k.add_typed_buf("lut", BufAccess::LookupTable(8), ty, lanes, ty.natural_domain());
            let out = k.add_typed_buf("out", BufAccess::MutNoAlias, ty, lanes, ty.natural_domain());
            let mem = k.push(Op::MemInit);
            let c7 = k.push(Op::Const(7));
            for lane in 0..lanes {
                let v = k.push(Op::Load { mem, buf: a, lane });
                let raw_idx = k.push(Op::BitAnd(v, c7));
                let idx = k.push(Op::BoundsCheck { idx: raw_idx, limit: 8 });
                let lv = k.push(Op::LoadLut { buf: lut, idx });
                let sum = k.push(Op::Add(v, lv));
                k.add_output(out, lane, sum);
            }

            let obj_x86 = compile_kernel_direct(&k, budget, TargetIsa::X86_64)
                .expect("compile narrow x86_64 object");
            assert!(obj_x86.rust_ffi_shim.contains(&format!("a: &[{rty}; {lanes}]")));
            assert!(obj_x86.rust_ffi_shim.contains(&format!("lut: &[{rty}; 8]")));
            assert!(obj_x86.rust_ffi_shim.contains(&format!("out: &mut [{rty}; {lanes}]")));
            assert!(obj_x86.rust_ffi_shim.contains(&format!("a: &[[{rty}; {lanes}]]")));
            assert!(obj_x86.rust_ffi_shim.contains(&format!("out: &mut [[{rty}; {lanes}]]")));

            let obj_arm = compile_kernel_direct(&k, budget, TargetIsa::Aarch64)
                .expect("compile narrow AArch64 object");
            assert_eq!(obj_arm.code_bytes.len() % 4, 0);
            let words: Vec<u32> = obj_arm
                .code_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert!(
                words.iter().any(|&w| (w & 0xFFC0_0000) == exp_ldr),
                "AArch64 narrow_{rty}_{lanes} missing load opcode {exp_ldr:#010X}"
            );
            assert!(
                words.iter().any(|&w| (w & 0xFFC0_0000) == exp_str),
                "AArch64 narrow_{rty}_{lanes} missing store opcode {exp_str:#010X}"
            );
            assert!(
                words.iter().any(|&w| (w & 0xFFE0_FC00) == exp_lut),
                "AArch64 narrow_{rty}_{lanes} missing LUT opcode {exp_lut:#010X}"
            );

            let blocks = 3usize;
            let n_elems = blocks * usize::from(lanes);
            let a_in: Vec<i64> = (0..n_elems)
                .map(|i| ty.cast_i64(edge_vals[(i + usize::from(lanes)) % edge_vals.len()]))
                .collect();
            let lut_in: Vec<i64> = (0..8)
                .map(|i| ty.cast_i64(edge_vals[(i * 2 + 1) % edge_vals.len()]))
                .collect();
            let out_init = vec![0i64; n_elems];
            let inputs: [&[i64]; 3] = [&a_in, &lut_in, &out_init];
            let expected = expected_multiblock_output_bytes(&k, &inputs, blocks);

            #[cfg(not(miri))]
            {
                let exec = compile_standalone_elf_exec(&k, budget, TargetIsa::X86_64, &inputs)
                    .expect("compile narrow ET_EXEC");
                assert_eq!(exec.shim_glue_insns, 0);
                assert_eq!(exec.direct_share_pct, 100);
                let tag = format!("narrow_{rty}_{lanes}_3blk");
                verify_live_et_exec_readelf_objdump_and_run(&tmp, &tag, &exec, &expected);
            }
            #[cfg(miri)]
            {
                assert_eq!(expected.len(), blocks * usize::from(lanes) * usize::from(ty.bytes()));
            }
        }
    }

    let mut k_u8 = Kernel::new("u8_rex_sil_dil_pressure");
    let b0 = k_u8.add_typed_buf("b0", BufAccess::ReadOnly, ElemTy::U8, 8, ElemTy::U8.natural_domain());
    let b1 = k_u8.add_typed_buf("b1", BufAccess::ReadOnly, ElemTy::U8, 8, ElemTy::U8.natural_domain());
    let b2 = k_u8.add_typed_buf("b2", BufAccess::ReadOnly, ElemTy::U8, 8, ElemTy::U8.natural_domain());
    let b3 = k_u8.add_typed_buf("b3", BufAccess::ReadOnly, ElemTy::U8, 8, ElemTy::U8.natural_domain());
    let out = k_u8.add_typed_buf("out", BufAccess::MutNoAlias, ElemTy::U8, 8, ElemTy::U8.natural_domain());
    let mem = k_u8.push(Op::MemInit);
    let mut ring = [0usize; 2];
    for i in 0..2u8 {
        let v0 = k_u8.push(Op::Load { mem, buf: b0, lane: i });
        let v1 = k_u8.push(Op::Load { mem, buf: b1, lane: (i + 1) & 7 });
        ring[i as usize] = k_u8.push(Op::Add(v0, v1));
    }
    for lane in 0..8u8 {
        let v0 = k_u8.push(Op::Load { mem, buf: b0, lane });
        let v1 = k_u8.push(Op::Load { mem, buf: b1, lane: (lane + 1) & 7 });
        let v2 = k_u8.push(Op::Load { mem, buf: b2, lane: (lane + 2) & 7 });
        let v3 = k_u8.push(Op::Load { mem, buf: b3, lane: (lane + 3) & 7 });
        let s01 = k_u8.push(Op::Add(v0, v1));
        let s23 = k_u8.push(Op::Sub(v2, v3));
        let mix = k_u8.push(Op::Add(s01, s23));
        let fin = k_u8.push(Op::Add(mix, ring[(lane & 1) as usize]));
        k_u8.add_output(out, lane, fin);
    }

    let u8_blocks = 4usize;
    let mut rng = SplitMix64(0x0808_511D_11F0_0001);
    let mk_u8 = |r: &mut SplitMix64| -> Vec<i64> {
        (0..u8_blocks * 8)
            .map(|i| match i % 7 {
                0 => 0x00,
                1 => 0x7F,
                2 => 0x80,
                3 => 0xFE,
                4 => 0xFF,
                _ => r.range(0, 255),
            })
            .collect()
    };
    let d0 = mk_u8(&mut rng);
    let d1 = mk_u8(&mut rng);
    let d2 = mk_u8(&mut rng);
    let d3 = mk_u8(&mut rng);
    let dout = vec![0i64; u8_blocks * 8];
    let u8_inputs: [&[i64]; 5] = [&d0, &d1, &d2, &d3, &dout];
    let u8_exp = expected_multiblock_output_bytes(&k_u8, &u8_inputs, u8_blocks);

    #[cfg(not(miri))]
    {
        let u8_exec = compile_standalone_elf_exec(&k_u8, budget, TargetIsa::X86_64, &u8_inputs)
            .expect("compile 4-block u8_rex_sil_dil_pressure ET_EXEC");
        verify_live_et_exec_readelf_objdump_and_run(&tmp, "u8_rex_sil_dil_pressure_4blk", &u8_exec, &u8_exp);

        let bin_path = tmp.join("u8_rex_sil_dil_pressure_4blk");
        let od = Command::new("objdump")
            .args(["-d", "-M", "intel"])
            .arg(&bin_path)
            .output()
            .expect("objdump u8_rex_sil_dil_pressure");
        assert!(od.status.success());
        let asm = String::from_utf8_lossy(&od.stdout);
        for line in asm.lines() {
            let mut parts = line.split('\t');
            let Some(_) = parts.next() else { continue; };
            if parts.next().is_none() { continue; }
            let Some(insn_txt) = parts.next() else { continue; };
            let clean = insn_txt.split('#').next().unwrap_or("");
            for tok in clean.split(|c: char| !c.is_ascii_alphanumeric()) {
                assert!(
                    !matches!(tok, "ah" | "ch" | "dh" | "bh"),
                    "u8_rex_sil_dil_pressure emitted illegal high-8-bit register {tok} (missing REX prefix!) in:\n{line}"
                );
            }
        }

        let _ = fs::remove_dir_all(&tmp);
    }
    #[cfg(miri)]
    {
        assert_eq!(u8_exp.len(), u8_blocks * 8);
    }
}
