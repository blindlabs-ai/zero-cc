#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;
use std::process::Command;

use zero_cc::{
    bench_tile_blocks, build_bench_inputs, compile_bench_elf_exec, compile_bench_elf_exec_for_profile,
    compile_bench_elf_exec_with_pf, compile_kernel_direct, compile_kernel_direct_for_profile,
    compile_standalone_elf_exec, default_cpu_specs, detect_host_profile, emit_kernel_elf_with_pf,
    eval_primary_output_checksum, find_cpu_profile, fnv1a64, load_cpu_specs_from_path, parse_cpu_specs,
    parse_proc_cpuinfo, parse_rust_kernel, plan_gemm_tiles_for_profile, resolve_cpu_profile,
    select_profile_for_host, verify_elf64_executable, verify_elf64_object, BufAccess, RegBudget, SimdTier,
    TargetIsa, X86VectorMode, DEFAULT_AMD_ZEN_EPYC_SPEC,
};

const PAETH_PRED8_SRC: &str = r#"
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
pub fn paeth_pred8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let d_n = n[k] - nw[k];
        let pa = d_n.max(-d_n);
        let d_w = w[k] - nw[k];
        let pb = d_w.max(-d_w);
        let d_c = d_w + d_n;
        let pc = d_c.max(-d_c);
        let m_ab = (pa - pb - 1) >> 31;
        let m_ac = (pa - pc - 1) >> 31;
        let m_bc = (pb - pc - 1) >> 31;
        let sel_w = m_ab & m_ac;
        let n_or_nw = nw[k] ^ ((n[k] ^ nw[k]) & m_bc);
        out[k] = n_or_nw ^ ((w[k] ^ n_or_nw) & sel_w);
    }
}
"#;

const ZIGZAG_U8_8_SRC: &str = r#"
/// @domain err: -128..=127
pub fn zigzag_u8_8(err: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        out[k] = ((err[k] << 1) ^ (err[k] >> 7)) & 255;
    }
}
"#;

const GRAD_SOCKET8_SRC: &str = r#"
/// @domain w: 0..=255, n: 0..=255, nw: 0..=255
pub fn grad_socket8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let dh = w[k] - nw[k];
        let qh = -((-24 - dh) >> 31) - ((-7 - dh) >> 31) - ((-1 - dh) >> 31) - ((-dh) >> 31)
            - ((2 - dh) >> 31) - ((8 - dh) >> 31) - ((24 - dh) >> 31);
        let dv = n[k] - nw[k];
        let qv = -((-24 - dv) >> 31) - ((-7 - dv) >> 31) - ((-1 - dv) >> 31) - ((-dv) >> 31)
            - ((2 - dv) >> 31) - ((8 - dv) >> 31) - ((24 - dv) >> 31);
        out[k] = (qh << 3) | qv;
    }
}
"#;

const SIGNED_SHR_I32X8_SRC: &str = r#"
/// @domain x: -2147483648..=2147483647
pub fn signed_shr_mix8(x: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let v = x[k];
        let a = v >> 0;
        let b = v >> 1;
        let c = v >> 7;
        let d = v >> 15;
        let e = v >> 31;
        out[k] = (a ^ b) + (c ^ d) + e;
    }
}
"#;

const SIGNED_SHR_BATCH8_I64_SRC: &str = r#"
pub fn signed_shr_batch8_i64(x: &[i64; 1], out: &mut [i64; 1]) {
    let v = x[0];
    let s0 = v >> 0;
    let s1 = v >> 1;
    let s7 = v >> 7;
    let s31 = v >> 31;
    let s32 = v >> 32;
    let s47 = v >> 47;
    let s63 = v >> 63;
    out[0] = (s0 ^ s1) ^ (s7 ^ s31) ^ (s32 ^ s47) ^ s63;
}
"#;

const SIGNED_SHR_BATCH8_I32_SRC: &str = r#"
/// @domain x: -2147483648..=2147483647
pub fn signed_shr_batch8_i32(x: &[i32; 1], out: &mut [i64; 1]) {
    let v = x[0] as i64;
    let a = v >> 1;
    let b = v >> 7;
    let c = v >> 31;
    let d = v >> 32;
    out[0] = (a ^ b) + (c ^ d);
}
"#;

const NEG_I32_MIN_SHR_SRC: &str = r#"
/// @domain x: -2147483648..=2147483647
pub fn neg_i32_min_shr(x: &[i32; 8], out: &mut [i64; 8]) {
    for k in 0..8usize {
        let neg = 0i64 - (x[k] as i64);
        out[k] = neg >> 1;
    }
}
"#;

#[cfg(unix)]
fn set_exec_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod 0755");
}

#[cfg(not(unix))]
fn set_exec_mode(_path: &Path) {}

fn run_standalone_bytes(tag: &str, elf_bytes: &[u8]) -> Vec<u8> {
    let p = std::env::temp_dir().join(format!("zero_cc_test_{tag}_{}", std::process::id()));
    fs::write(&p, elf_bytes).expect("write elf");
    set_exec_mode(&p);
    let out = Command::new(&p).output().expect("exec elf");
    let _ = fs::remove_file(&p);
    let expected_exit = i32::from(out.stdout.iter().fold(0u8, |a, &b| a.wrapping_add(b)) & 0x7F);
    assert_eq!(out.status.code(), Some(expected_exit), "exec {tag} exit mismatch: {:?}", out.status);
    out.stdout
}

fn run_bench_bytes(tag: &str, elf_bytes: &[u8]) -> Vec<u8> {
    let p = std::env::temp_dir().join(format!("zero_cc_test_{tag}_{}", std::process::id()));
    fs::write(&p, elf_bytes).expect("write elf");
    set_exec_mode(&p);
    let out = Command::new(&p).output().expect("exec elf");
    let _ = fs::remove_file(&p);
    assert!(out.status.success(), "exec {tag} failed: {:?}", out.status);
    out.stdout
}

#[test]
fn amd_zen_epyc_spec_all_six_profiles_and_host_detection() {
    let specs = default_cpu_specs().expect("parse embedded amd_zen_epyc.spec");
    assert_eq!(specs.len(), 6, "expected 6 Zen/EPYC profiles");
    let file_specs = load_cpu_specs_from_path(Path::new("specs/amd_zen_epyc.spec"))
        .expect("load specs/amd_zen_epyc.spec from disk");
    assert_eq!(specs, file_specs);
    assert!(DEFAULT_AMD_ZEN_EPYC_SPEC.is_ascii());

    let expected = [
        ("zen1_naples", 23u16, 32usize, 8usize, 512usize, 8usize, 0usize, 4usize, 64usize, SimdTier::Avx2Split128, 16usize, 256usize, 128usize, 16_384usize),
        ("zen2_rome", 23, 32, 8, 512, 16, 0, 4, 64, SimdTier::Avx2, 16, 256, 256, 16_384),
        ("zen3_milan", 25, 32, 8, 512, 32, 96, 8, 64, SimdTier::Avx2, 16, 256, 256, 16_384),
        ("zen4_genoa", 25, 32, 8, 1024, 32, 96, 8, 64, SimdTier::Avx512Split256, 32, 512, 256, 16_384),
        ("zen4c_bergamo", 25, 32, 8, 1024, 16, 0, 8, 64, SimdTier::Avx512Split256, 32, 512, 256, 16_384),
        ("zen5_turin", 26, 48, 12, 1024, 32, 96, 8, 64, SimdTier::Avx512Native512, 32, 512, 512, 24_576),
    ];

    for (name, fam, l1d, ways, l2, l3, l3v, ccx, line, simd, vregs, vbits, pipe, tile) in expected {
        let p = find_cpu_profile(&specs, name).expect("profile exists");
        assert_eq!(p.family, fam);
        assert_eq!(p.l1d_kib, l1d);
        assert_eq!(p.l1d_ways, ways);
        assert_eq!(p.l2_kib, l2);
        assert_eq!(p.l3_ccx_mib, l3);
        assert_eq!(p.l3_3dv_mib, l3v);
        assert_eq!(p.ccx_cores, ccx);
        assert_eq!(p.line_bytes, line);
        assert_eq!(p.simd, simd);
        assert_eq!(p.vec_regs, vregs);
        assert_eq!(p.vec_bits, vbits);
        assert_eq!(p.native_pipe_bits, pipe);
        assert_eq!(p.l1_bench_tile_bytes(), tile);
        assert_eq!(p.reg_budget(), RegBudget { max_gprs: 14, max_vecs: vregs });
        let plan = plan_gemm_tiles_for_profile(p).expect("plan_gemm_tiles_for_profile");
        assert_eq!(plan.spills, 0);
        assert!(plan.peak_vecs <= vregs);
    }

    let zen2 = find_cpu_profile(&specs, "zen2_rome").expect("zen2_rome");
    let zen4 = find_cpu_profile(&specs, "zen4_genoa").expect("zen4_genoa");
    let zen5 = find_cpu_profile(&specs, "zen5_turin").expect("zen5_turin");
    assert_ne!(zen5.l1_bench_tile_bytes(), zen2.l1_bench_tile_bytes());
    assert_ne!(zen4.reg_budget().max_vecs, zen2.reg_budget().max_vecs);
    assert_ne!(
        plan_gemm_tiles_for_profile(zen5).expect("zen5 plan").kc,
        plan_gemm_tiles_for_profile(zen4).expect("zen4 plan").kc
    );

    let synth_cases = [
        ("AuthenticAMD", 23, 1, "avx2", "zen1_naples"),
        ("AuthenticAMD", 23, 15, "avx2", "zen1_naples"),
        ("AuthenticAMD", 23, 49, "avx2", "zen2_rome"),
        ("AuthenticAMD", 23, 113, "avx2", "zen2_rome"),
        ("AuthenticAMD", 25, 1, "avx2", "zen3_milan"),
        ("AuthenticAMD", 25, 17, "avx2 avx512f", "zen4_genoa"),
        ("AuthenticAMD", 25, 24, "avx2 avx512f", "zen4c_bergamo"),
        ("AuthenticAMD", 26, 2, "avx2 avx512f", "zen5_turin"),
        ("GenuineIntel", 6, 143, "avx2 avx512f", "generic_avx512"),
    ];
    for (vendor, fam, model, flags, want) in synth_cases {
        let txt = format!("vendor_id\t: {vendor}\ncpu family\t: {fam}\nmodel\t\t: {model}\nflags\t\t: {flags}\n");
        let info = parse_proc_cpuinfo(&txt).expect("parse cpuinfo");
        let sel = select_profile_for_host(&specs, &info);
        assert_eq!(sel.name, want, "family={fam} model={model}");
    }

    let host = detect_host_profile(&specs);
    assert!(host.l1d_kib >= 16 && host.vec_regs >= 16);
    assert_eq!(resolve_cpu_profile(Some("zen5_turin"), None).expect("resolve").name, "zen5_turin");
    assert!(resolve_cpu_profile(Some("nonexistent_cpu"), None).is_err());
    assert!(parse_cpu_specs("not_valid_section").is_err());
}

#[test]
fn signed_shr_horizontal_and_batch8_adversarial_parity() {
    let budget = RegBudget::default();

    // 1. paeth_pred8 and zigzag_u8_8 must compile as HorizontalI32x8 with tight insn budgets
    let paeth_k = parse_rust_kernel(PAETH_PRED8_SRC).expect("parse paeth_pred8");
    let paeth_obj = compile_kernel_direct(&paeth_k, budget, TargetIsa::X86_64).expect("compile paeth_pred8");
    verify_elf64_object(&paeth_obj.elf_bytes).expect("verify paeth_pred8 elf");
    assert_eq!(paeth_obj.vector_mode, X86VectorMode::HorizontalI32x8);
    assert!(paeth_obj.zero_spills);
    assert!(paeth_obj.direct_insns < 120, "paeth_pred8 insns={} expected < 120", paeth_obj.direct_insns);

    let zz_k = parse_rust_kernel(ZIGZAG_U8_8_SRC).expect("parse zigzag_u8_8");
    let zz_obj = compile_kernel_direct(&zz_k, budget, TargetIsa::X86_64).expect("compile zigzag_u8_8");
    verify_elf64_object(&zz_obj.elf_bytes).expect("verify zigzag_u8_8 elf");
    assert_eq!(zz_obj.vector_mode, X86VectorMode::HorizontalI32x8);
    assert!(zz_obj.zero_spills);
    assert!(zz_obj.direct_insns < 60, "zigzag_u8_8 insns={} expected < 60", zz_obj.direct_insns);

    // 2. Adversarial signed Op::Shr in HorizontalI32x8
    let shr8_k = parse_rust_kernel(SIGNED_SHR_I32X8_SRC).expect("parse signed_shr_mix8");
    let shr8_obj = compile_kernel_direct(&shr8_k, budget, TargetIsa::X86_64).expect("compile signed_shr_mix8");
    assert_eq!(shr8_obj.vector_mode, X86VectorMode::HorizontalI32x8);
    assert!(shr8_obj.zero_spills);

    // 3. Adversarial signed Op::Shr in 1-lane Batch8U32x8 (both full i64 and i32->i64)
    let b8_i64_k = parse_rust_kernel(SIGNED_SHR_BATCH8_I64_SRC).expect("parse signed_shr_batch8_i64");
    let b8_i64_obj = compile_kernel_direct(&b8_i64_k, budget, TargetIsa::X86_64).expect("compile b8 i64");
    assert_eq!(b8_i64_obj.vector_mode, X86VectorMode::Batch8U32x8);
    assert!(b8_i64_obj.zero_spills);

    let b8_i32_k = parse_rust_kernel(SIGNED_SHR_BATCH8_I32_SRC).expect("parse signed_shr_batch8_i32");
    let b8_i32_obj = compile_kernel_direct(&b8_i32_k, budget, TargetIsa::X86_64).expect("compile b8 i32");
    assert_eq!(b8_i32_obj.vector_mode, X86VectorMode::Batch8U32x8);
    assert!(b8_i32_obj.zero_spills);

    // 4. Out-of-i32-range (0 - i32::MIN) >> 1 must NOT miscompile in i32x8
    let neg_min_k = parse_rust_kernel(NEG_I32_MIN_SHR_SRC).expect("parse neg_i32_min_shr");
    let neg_min_obj = compile_kernel_direct(&neg_min_k, budget, TargetIsa::X86_64).expect("compile neg_i32_min_shr");
    assert_ne!(neg_min_obj.vector_mode, X86VectorMode::HorizontalI32x8);
    assert!(neg_min_obj.zero_spills);

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // Exhaustive -128..=127 parity for zigzag_u8_8 (32 blocks * 8 lanes = 256 values)
        let zz_in: Vec<i64> = (-128i64..=127).collect();
        let zz_out = vec![0i64; 256];
        let zz_exec = compile_standalone_elf_exec(&zz_k, budget, TargetIsa::X86_64, &[&zz_in, &zz_out])
            .expect("standalone zigzag_u8_8");
        verify_elf64_executable(&zz_exec.elf_bytes).expect("verify zigzag exec");
        let got = run_standalone_bytes("zigzag_u8_8", &zz_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&zz_k, &[zz_in, zz_out], 32).expect("ref zigzag");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);

        // Adversarial paeth_pred8 parity
        let p_in = build_bench_inputs(&paeth_k, 64, 0x1234_5678);
        let p_refs: Vec<&[i64]> = p_in.iter().map(Vec::as_slice).collect();
        let p_exec = compile_standalone_elf_exec(&paeth_k, budget, TargetIsa::X86_64, &p_refs)
            .expect("standalone paeth_pred8");
        let got = run_standalone_bytes("paeth_pred8", &p_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&paeth_k, &p_in, 64).expect("ref paeth");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);

        // Adversarial i32 corners for signed_shr_mix8
        let corners_i32 = [
            i32::MIN as i64, -1_000_000, -32_768, -255, -1, 0, 1, 255,
            32_767, 1_000_000, i32::MAX as i64, -2, -128, -129, 127, 128,
        ];
        let mut x8 = Vec::with_capacity(64);
        for i in 0..64usize { x8.push(corners_i32[i % corners_i32.len()]); }
        let o8 = vec![0i64; 64];
        let s8_exec = compile_standalone_elf_exec(&shr8_k, budget, TargetIsa::X86_64, &[&x8, &o8])
            .expect("standalone signed_shr_mix8");
        let got = run_standalone_bytes("signed_shr_mix8", &s8_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&shr8_k, &[x8.clone(), o8.clone()], 8).expect("ref s8");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);

        // Adversarial i64 corners for Batch8U32x8 signed Op::Shr (including 16+ blocks for 2x unrolled batch8 loop)
        let corners_i64 = [
            i64::MIN, i64::MIN + 1, -(1i64 << 40), i32::MIN as i64 - 1, i32::MIN as i64,
            -32_768, -255, -1, 0, 1, 255, 32_767, i32::MAX as i64, (1i64 << 40), i64::MAX - 1, i64::MAX,
        ];
        let mut x64 = Vec::with_capacity(32);
        for i in 0..32usize { x64.push(corners_i64[i % corners_i64.len()]); }
        let o64 = vec![0i64; 32];
        let b8_exec = compile_standalone_elf_exec(&b8_i64_k, budget, TargetIsa::X86_64, &[&x64, &o64])
            .expect("standalone b8 i64");
        let got = run_standalone_bytes("b8_i64", &b8_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&b8_i64_k, &[x64, o64], 32).expect("ref b8 i64");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);

        // Adversarial i32->i64 Batch8U32x8 signed Op::Shr
        let mut x32 = Vec::with_capacity(32);
        for i in 0..32usize { x32.push(corners_i32[i % corners_i32.len()]); }
        let o32 = vec![0i64; 32];
        let b8_32_exec = compile_standalone_elf_exec(&b8_i32_k, budget, TargetIsa::X86_64, &[&x32, &o32])
            .expect("standalone b8 i32");
        let got = run_standalone_bytes("b8_i32", &b8_32_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&b8_i32_k, &[x32, o32], 32).expect("ref b8 i32");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);

        // Out-of-i32-range (0 - i32::MIN) >> 1 must return +1073741824, NOT -1073741824
        let neg_exec = compile_standalone_elf_exec(&neg_min_k, budget, TargetIsa::X86_64, &[&x8, &o8])
            .expect("standalone neg_i32_min_shr");
        let got = run_standalone_bytes("neg_i32_min_shr", &neg_exec.elf_bytes);
        let (exp_ck, exp_len) = eval_primary_output_checksum(&neg_min_k, &[x8, o8], 8).expect("ref neg_min");
        assert_eq!(got.len(), exp_len);
        assert_eq!(fnv1a64(&got), exp_ck);
        let lane0 = i64::from_le_bytes(got[0..8].try_into().expect("8 bytes"));
        assert_eq!(lane0, 1_073_741_824i64, "0 - i32::MIN >> 1 must be +2^30");
    }
}

#[test]
fn l1_tiled_bench_elf_exec_live_parity_and_timing() {
    let specs = default_cpu_specs().expect("default specs");
    let host = detect_host_profile(&specs);
    let tile_bytes = host.l1_bench_tile_bytes();
    let budget = host.reg_budget();

    for (name, src) in [
        ("paeth_pred8", PAETH_PRED8_SRC),
        ("zigzag_u8_8", ZIGZAG_U8_8_SRC),
        ("grad_socket8", GRAD_SOCKET8_SRC),
    ] {
        let k = parse_rust_kernel(src).expect("parse bench kernel");
        let (tile_blocks, bpb) = bench_tile_blocks(&k, tile_bytes);
        assert!(tile_blocks >= 256 && tile_blocks % 8 == 0);
        assert!(bpb > 0);
        let inputs = build_bench_inputs(&k, tile_blocks, 0x9e37_79b9);
        let refs: Vec<&[i64]> = inputs.iter().map(Vec::as_slice).collect();
        let be = compile_bench_elf_exec(&k, budget, &refs, tile_bytes, 32, 512)
            .expect("compile_bench_elf_exec");
        assert_eq!(be.exec.vector_mode, X86VectorMode::HorizontalI32x8);
        assert!(be.exec.zero_spills);
        assert_eq!(be.total_blocks, (512 * tile_blocks) as u64);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let stdout = run_bench_bytes(&format!("bench_{name}"), &be.exec.elf_bytes);
            assert_eq!(stdout.len(), 24);
            let elapsed_ns = u64::from_le_bytes(stdout[0..8].try_into().expect("u64"));
            let total_blocks = u64::from_le_bytes(stdout[8..16].try_into().expect("u64"));
            let got_ck = u64::from_le_bytes(stdout[16..24].try_into().expect("u64"));
            assert!(elapsed_ns > 0, "{name}: elapsed_ns must be > 0");
            assert_eq!(total_blocks, be.total_blocks);
            let (exp_ck, _) = eval_primary_output_checksum(&k, &inputs, tile_blocks).expect("ref ck");
            assert_eq!(got_ck, exp_ck, "{name}: bench checksum mismatch");

            // Counterfactual knockout: flipping an input byte changes output checksum
            let mut alt_inputs = inputs.clone();
            if let Some(ro_idx) = k.bufs.iter().position(|b| b.access == BufAccess::ReadOnly) {
                let d = k.bufs[ro_idx].domain;
                let cur = alt_inputs[ro_idx][0];
                alt_inputs[ro_idx][0] = if cur == d.lo { d.hi } else { d.lo };
                let (alt_ck, _) = eval_primary_output_checksum(&k, &alt_inputs, tile_blocks).expect("alt ck");
                assert_ne!(got_ck, alt_ck, "{name}: counterfactual input flip must change checksum");
            }
        }
    }
}

const NC_HASH_O2_8_SRC: &str = r#"
/// @domain p2: 0..=255, p1: 0..=255
pub fn nc_hash_o2_8(p2: &[i32; 8], p1: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        out[k] = (p2[k] * 137 + (p2[k] >> 3) + p1[k] * 3) & 255;
    }
}
"#;

const NC_HASH4_8_SRC: &str = r#"
/// @domain b0: 0..=255, b1: 0..=255, b2: 0..=255, b3: 0..=255
pub fn nc_hash4_8(b0: &[i32; 8], b1: &[i32; 8], b2: &[i32; 8], b3: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let x = b0[k] * 131 + b1[k] * 4099 + b2[k] * 65537 + b3[k] * 1_048_583;
        out[k] = (x + (x >> 11) + (x >> 5)) & 131_071;
    }
}
"#;

const NC_MIX3_PROB12_8_SRC: &str = r#"
/// @domain p_o0: 1..=4095, p_o1: 1..=4095, p_o2: 1..=4095, w_o2: 0..=24
pub fn nc_mix3_prob12_8(p_o0: &[i32; 8], p_o1: &[i32; 8], p_o2: &[i32; 8], w_o2: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        let base = (p_o0[k] + p_o1[k] * 3 + 2) >> 2;
        let w = w_o2[k];
        let m = (w * p_o2[k] + (32 - w) * base + 16) >> 5;
        out[k] = m.max(1).min(4095);
    }
}
"#;

#[test]
fn rtos_prefetch_spec_and_l1_vs_ram_roofline() {
    let specs = default_cpu_specs().expect("default specs");
    for p in &specs {
        assert_eq!(p.prefetch_distance_bytes(), 4 * p.line_bytes);
        assert_eq!(p.prefetch_distance_bytes(), 256);
        assert_eq!(p.streaming_ram_tile_bytes(), 4 * 1024 * 1024);
    }
    let host = detect_host_profile(&specs);
    let mut wide_line = host.clone();
    wide_line.line_bytes = 128;
    assert_eq!(wide_line.prefetch_distance_bytes(), 512);
    assert_ne!(wide_line.prefetch_distance_bytes(), host.prefetch_distance_bytes());

    let paeth_k = parse_rust_kernel(PAETH_PRED8_SRC).expect("parse paeth_pred8");
    let obj_pf = compile_kernel_direct_for_profile(&paeth_k, &host, TargetIsa::X86_64).expect("compile pf");
    let obj_nopf = emit_kernel_elf_with_pf(&paeth_k, host.reg_budget(), TargetIsa::X86_64, true, false, 0)
        .expect("compile nopf");
    assert_eq!(obj_pf.vector_mode, X86VectorMode::HorizontalI32x8);
    assert!(obj_pf.zero_spills && obj_nopf.zero_spills);

    let p_pf = std::env::temp_dir().join(format!("zero_cc_pf_{}.o", std::process::id()));
    let p_no = std::env::temp_dir().join(format!("zero_cc_nopf_{}.o", std::process::id()));
    fs::write(&p_pf, &obj_pf.elf_bytes).expect("write pf.o");
    fs::write(&p_no, &obj_nopf.elf_bytes).expect("write nopf.o");
    let d_pf = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&p_pf).output().expect("objdump pf");
    let d_no = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&p_no).output().expect("objdump nopf");
    let _ = fs::remove_file(&p_pf);
    let _ = fs::remove_file(&p_no);
    let asm_pf = String::from_utf8_lossy(&d_pf.stdout);
    let asm_no = String::from_utf8_lossy(&d_no.stdout);
    assert!(asm_pf.contains("prefetcht0") && asm_pf.contains("+0x100]"), "expected prefetcht0 +0x100:\n{asm_pf}");
    assert!(!asm_no.contains("prefetcht0"), "pf=0 must emit 0 prefetcht0:\n{asm_no}");

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        let l1_bytes = host.l1_bench_tile_bytes();
        let ram_bytes = host.streaming_ram_tile_bytes();
        for (name, src) in [
            ("paeth_pred8", PAETH_PRED8_SRC),
            ("zigzag_u8_8", ZIGZAG_U8_8_SRC),
            ("nc_hash_o2_8", NC_HASH_O2_8_SRC),
            ("nc_hash4_8", NC_HASH4_8_SRC),
            ("nc_mix3_prob12_8", NC_MIX3_PROB12_8_SRC),
        ] {
            let k = parse_rust_kernel(src).expect("parse kernel");
            let (l1_blks, bpb) = bench_tile_blocks(&k, l1_bytes);
            let l1_in = build_bench_inputs(&k, l1_blks, 0x243f_6a88_85a3_08d3);
            let l1_refs: Vec<&[i64]> = l1_in.iter().map(Vec::as_slice).collect();
            let (l1_exp, _) = eval_primary_output_checksum(&k, &l1_in, l1_blks).expect("l1 ref");

            let (ram_blks, _) = bench_tile_blocks(&k, ram_bytes);
            let ram_in = build_bench_inputs(&k, ram_blks, 0x243f_6a88_85a3_08d3);
            let ram_refs: Vec<&[i64]> = ram_in.iter().map(Vec::as_slice).collect();
            let (ram_exp, _) = eval_primary_output_checksum(&k, &ram_in, ram_blks).expect("ram ref");

            let run_one = |tag: &str, be: &zero_cc::BenchExec, exp_ck: u64| -> (f64, f64) {
                let out = run_bench_bytes(tag, &be.exec.elf_bytes);
                assert_eq!(out.len(), 24);
                let ns = u64::from_le_bytes(out[0..8].try_into().expect("u64"));
                let blks = u64::from_le_bytes(out[8..16].try_into().expect("u64"));
                let ck = u64::from_le_bytes(out[16..24].try_into().expect("u64"));
                assert!(ns > 0 && blks == be.total_blocks);
                assert_eq!(ck, exp_ck, "{tag}: checksum mismatch");
                let ns_blk = (ns as f64) / (blks as f64);
                let gbs = ((blks as f64) * (bpb as f64)) / (ns as f64);
                (ns_blk, gbs)
            };

            let be_l1_no = compile_bench_elf_exec_with_pf(&k, host.reg_budget(), &l1_refs, l1_bytes, 64, 2048, 0).expect("l1 no");
            let be_l1_pf = compile_bench_elf_exec_for_profile(&k, &host, &l1_refs, l1_bytes, 64, 2048).expect("l1 pf");
            let be_ram_no = compile_bench_elf_exec_with_pf(&k, host.reg_budget(), &ram_refs, ram_bytes, 4, 64, 0).expect("ram no");
            let be_ram_pf = compile_bench_elf_exec_for_profile(&k, &host, &ram_refs, ram_bytes, 4, 64).expect("ram pf");

            let (l1_ns_no, l1_gb_no) = run_one(&format!("l1_no_{name}"), &be_l1_no, l1_exp);
            let (l1_ns_pf, l1_gb_pf) = run_one(&format!("l1_pf_{name}"), &be_l1_pf, l1_exp);
            let (ram_ns_no, ram_gb_no) = run_one(&format!("ram_no_{name}"), &be_ram_no, ram_exp);
            let (ram_ns_pf, ram_gb_pf) = run_one(&format!("ram_pf_{name}"), &be_ram_pf, ram_exp);
            println!(
                "ROOFLINE|{name}|L1_pf0={l1_ns_no:.3}ns({l1_gb_no:.2}GB/s)|L1_pf256={l1_ns_pf:.3}ns({l1_gb_pf:.2}GB/s)|RAM4M_pf0={ram_ns_no:.3}ns({ram_gb_no:.2}GB/s)|RAM4M_pf256={ram_ns_pf:.3}ns({ram_gb_pf:.2}GB/s)"
            );
        }
    }
}
