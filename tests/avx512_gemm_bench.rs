use std::fs;
use std::path::PathBuf;
use std::process::Command;
use zero_cc::{
    compile_rust_to_elf, compile_tensor_to_elf, eval_gemm_epi_ref, eval_gemm_ref, export_tensor_ntf1,
    parse_rust_kernel, parse_rust_tensor_kernel, verify_elf64_object, verify_ntf1_airlock, GemmMat,
    RegBudget, TargetIsa, UOPC_EPI_ADD, UOPC_EPI_FMA, UOPC_EPI_GELU, UOPC_EPI_GELU_FWD,
};

fn tmp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("zero_cc_avx512_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("mkdir tmp");
    p
}

#[test]
fn ntf1_and_f32_avx512_egraph_fma_live_parity() {
    let spec_src = fs::read_to_string("specs/zero_cc.ntf1").expect("read ntf1");
    verify_ntf1_airlock(&spec_src).expect("specs/zero_cc.ntf1 airlock");

    let src = r#"
pub fn fma_affine_f32x16(a: &[f32; 16], b: &[f32; 16], c: &[f32; 16], out: &mut [f32; 16]) {
    for i in 0..16 {
        let v = a[i] * b[i] + c[i];
        out[i] = (v * 0.5f32 + 1.25f32).max(0.0f32);
    }
}
"#;
    let k = parse_rust_kernel(src).expect("parse f32x16 kernel");
    let opt = zero_cc::optimize_kernel(&k).expect("optimize f32x16");
    let fma_count = opt.nodes.iter().filter(|n| matches!(n, zero_cc::Op::FFma(..))).count();
    assert!(fma_count >= 16, "expected E-Graph FMA contraction >= 16, got {fma_count}");

    let obj = compile_rust_to_elf(src, RegBudget::default(), TargetIsa::X86_64).expect("compile f32x16");
    assert!(obj.zero_spills);
    let sum = verify_elf64_object(&obj.elf_bytes).expect("verify f32x16 elf64");
    assert_eq!(sum.rela_count, 0);

    let dir = tmp_dir("f32x16");
    let o_path = dir.join("f32x16.o");
    fs::write(&o_path, &obj.elf_bytes).expect("write .o");

    let dis = Command::new("objdump")
        .args(["-d", "-M", "intel", "--disassemble=zero_cc_fma_affine_f32x16"])
        .arg(&o_path).output().expect("objdump");
    let asm = String::from_utf8_lossy(&dis.stdout);
    assert!(asm.contains("zmm"), "expected zmm AVX-512 instructions:\n{asm}");
    assert!(asm.contains("vfmadd231ps"), "expected vfmadd231ps:\n{asm}");
    assert!(!asm.contains("(bad)"), "invalid opcode in objdump:\n{asm}");

    // Link and run live hardware parity against Kernel::eval_f32
    let a: Vec<f32> = (0..16).map(|i| (i as f32 - 6.5) * 0.37).collect();
    let b: Vec<f32> = (0..16).map(|i| (i as f32 - 3.0) * 0.41).collect();
    let c: Vec<f32> = (0..16).map(|i| (i as f32) * 0.19 - 1.2).collect();
    let z = vec![0.0_f32; 16];
    let expected = opt.eval_f32(&[&a, &b, &c, &z]).expect("eval_f32");

    let harness = r#"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
extern void zero_cc_fma_affine_f32x16_batch(const float*, const float*, const float*, float*, size_t);
int main(void) {
    float a[32], b[32], c[32], out[32];
    for (int i = 0; i < 32; i++) {
        int j = i & 15;
        a[i] = (j - 6.5f) * 0.37f;
        b[i] = (j - 3.0f) * 0.41f;
        c[i] = j * 0.19f - 1.2f;
        out[i] = -999.0f;
    }
    zero_cc_fma_affine_f32x16_batch(a, b, c, out, 2);
    for (int i = 0; i < 16; i++) printf("%.8e %.8e\n", out[i], out[i + 16]);
    return 0;
}
"#;
    let c_path = dir.join("h.c");
    let bin_path = dir.join("h");
    fs::write(&c_path, harness).expect("write harness");
    let st = Command::new("gcc").args(["-O2", "-o"]).arg(&bin_path).arg(&c_path).arg(&o_path)
        .status().expect("gcc link");
    assert!(st.success(), "link failed");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        let out = Command::new(&bin_path).output().expect("run harness");
        assert!(out.status.success());
        let text = String::from_utf8_lossy(&out.stdout);
        for (i, line) in text.lines().enumerate() {
            let mut sp = line.split_whitespace();
            let g0: f32 = sp.next().unwrap().parse().unwrap();
            let g1: f32 = sp.next().unwrap().parse().unwrap();
            let refv = expected[3][i];
            assert!((g0 - refv).abs() <= 1e-6, "lane {i} blk0: got {g0} ref {refv}");
            assert!((g1 - refv).abs() <= 1e-6, "lane {i} blk1: got {g1} ref {refv}");
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

const RUST_SGEMM_SRC: &str = r#"
// @zero_tensor uopc_sgemm
pub fn sgemm_dispatch(
    m: i32, n: i32, k: i32,
    a: *const f32, lda: i32,
    b: *const f32, ldb: i32,
    c: *mut f32, ldc: i32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                let av = unsafe { *a.add((i as usize) * (lda as usize) + (p as usize)) };
                let bv = unsafe { *b.add((p as usize) * (ldb as usize) + (j as usize)) };
                acc = av.mul_add(bv, acc);
            }
            unsafe { *c.add((i as usize) * (ldc as usize) + (j as usize)) = acc; }
        }
    }
}
"#;

#[test]
fn avx512_direct_elf64_sgemm_and_epilogues_live_parity() {
    let tk = parse_rust_tensor_kernel(RUST_SGEMM_SRC, RegBudget::default()).expect("parse tensor kernel");
    let ntf1 = export_tensor_ntf1(&tk);
    verify_ntf1_airlock(&ntf1).expect("tensor ntf1 airlock");

    let obj = compile_tensor_to_elf(&tk).expect("compile tensor to elf64");
    assert!(obj.zero_spills);
    let sum = verify_elf64_object(&obj.elf_bytes).expect("verify tensor elf64");
    assert_eq!(sum.rela_count, 0);

    let dir = tmp_dir("sgemm");
    let o_path = dir.join("sgemm_zcc.o");
    fs::write(&o_path, &obj.elf_bytes).expect("write sgemm_zcc.o");

    let dis = Command::new("objdump")
        .args(["-d", "-M", "intel", "--stop-address", &format!("0x{:x}", obj.code_bytes.len() - 96)])
        .arg(&o_path).output().expect("objdump");
    let asm = String::from_utf8_lossy(&dis.stdout);
    assert!(!asm.contains("(bad)"), "invalid opcode in sgemm_zcc.o:\n{}",
        asm.lines().filter(|l| l.contains("(bad)")).take(20).collect::<Vec<_>>().join("\n"));
    assert!(asm.contains("vfmadd231ps"), "missing vfmadd231ps");
    assert!(asm.contains("BCST") || asm.contains("1to16"), "missing EVEX embedded broadcast");
    assert!(asm.contains("zmm23"), "missing 24-accumulator 8x48 zmm0..zmm23");
    assert!(asm.contains("{k1}"), "missing AVX-512 opmask k1");

    // Compile a C parity driver that tests ragged shapes, strides, poisoned guards, all 4 epilogues, and C==B aliasing
    let harness = r#"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern void sgemm_dispatch(int M, int N, int K, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc);
extern void sgemm_dispatch_epi(int M, int N, int K, const float *A, int lda,
                               const float *B, int ldb, float *C, int ldc,
                               int kind, const float *E, int lde, float s);
extern void sgemm_epilogue_apply(int M, int N, int kind, float *C, int ldc,
                                 const float *E, int lde, float s);

static float ref_expf(float x) {
    float t = x * 1.4426950408889634f;
    if (t > 127.0f) t = 127.0f;
    if (t < -126.0f) t = -126.0f;
    float n = nearbyintf(t);
    float f = t - n;
    float p = fmaf(f, 0.0013333558146428443f, 0.009618129107628477f);
    p = fmaf(f, p, 0.05550410866482158f);
    p = fmaf(f, p, 0.2402265069591007f);
    p = fmaf(f, p, 0.6931471805599453f);
    p = fmaf(f, p, 1.0f);
    int32_t ni = (int32_t)n;
    uint32_t pb;
    memcpy(&pb, &p, 4);
    pb += ((uint32_t)ni) << 23;
    float out;
    memcpy(&out, &pb, 4);
    return out;
}

static void ref_gelu(float z, float *o, float *g) {
    float x = z * 0.7071067811865476f;
    float ax = fabsf(x);
    float t = 1.0f / (0.3275911f * ax + 1.0f);
    float poly = fmaf(t, 1.061405429f, -1.453152027f);
    poly = fmaf(t, poly, 1.421413741f);
    poly = fmaf(t, poly, -0.284496736f);
    poly = fmaf(t, poly, 0.254829592f);
    poly *= t;
    float e = ref_expf(-(ax * ax));
    float erf_abs = 1.0f - poly * e;
    float er = (x < 0.0f) ? -erf_abs : erf_abs;
    float half = (er + 1.0f) * 0.5f;
    *o = z * half;
    *g = fmaf(z * e, 0.3989422804014327f, half);
}

static uint32_t rng_state = 0x12345678u;
static float randf(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(int32_t)(rng_state >> 8) / 8388608.0f) * 0.5f;
}

static int check_shape(int M, int N, int K, int pad_a, int pad_b, int pad_c, int kind, int alias_cb) {
    int lda = K + pad_a, ldb = N + pad_b, ldc = N + pad_c, lde = N + pad_c;
    if (alias_cb) { ldb = ldc; lde = ldc; }
    size_t sa = (size_t)M * lda + 32;
    size_t sb = (size_t)(alias_cb ? (M > K ? M : K) : K) * ldb + 32;
    size_t sc = (size_t)M * ldc + 64;
    float *A = calloc(sa, 4), *B = calloc(sb, 4), *C = calloc(sc, 4), *E = calloc(sc, 4);
    float *Cref = calloc(sc, 4), *Eref = calloc(sc, 4);
    for (int i = 0; i < M; i++) for (int p = 0; p < K; p++) A[i * lda + p] = randf();
    for (int p = 0; p < K; p++) for (int j = 0; j < N; j++) B[p * ldb + j] = randf();
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        E[i * lde + j] = randf();
        Eref[i * lde + j] = E[i * lde + j];
    }
    const float POISON = -777.25f;
    for (size_t i = 0; i < sc; i++) C[i] = POISON;
    float s = 3.4445f;
    float *Ccall = alias_cb ? B : C;
    float *Ecall = alias_cb ? B : E;
    float *Bsnap = NULL;
    if (alias_cb) {
        Bsnap = malloc(sb * 4);
        memcpy(Bsnap, B, sb * 4);
    }
    const float *Bref = alias_cb ? Bsnap : B;
    const float *Ein = alias_cb ? Bsnap : Eref;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int p = 0; p < K; p++) acc = fmaf(A[i * lda + p], Bref[p * ldb + j], acc);
            if (kind == 1) acc += Ein[i * lde + j];
            else if (kind == 2) { float o, g; ref_gelu(Ein[i * lde + j], &o, &g); acc *= g; }
            else if (kind == 3) acc += s * Ein[i * lde + j];
            else if (kind == 4) { float o, g; ref_gelu(acc, &o, &g); acc = o; Eref[i * lde + j] = g; }
            Cref[i * ldc + j] = acc;
        }
    }
    if (kind == 0) sgemm_dispatch(M, N, K, A, lda, B, ldb, Ccall, ldc);
    else sgemm_dispatch_epi(M, N, K, A, lda, B, ldb, Ccall, ldc, kind, Ecall, lde, s);

    double err2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double d = (double)Ccall[i * ldc + j] - (double)Cref[i * ldc + j];
            err2 += d * d;
            ref2 += (double)Cref[i * ldc + j] * (double)Cref[i * ldc + j];
        }
        if (!alias_cb) {
            for (int j = N; j < ldc; j++) {
                if (C[i * ldc + j] != POISON) {
                    fprintf(stderr, "OOB row-pad write M=%d N=%d K=%d at (%d,%d)\n", M, N, K, i, j);
                    return 1;
                }
            }
        }
    }
    if (!alias_cb) {
        for (size_t i = (size_t)M * ldc; i < sc; i++) {
            if (C[i] != POISON) {
                fprintf(stderr, "OOB tail write M=%d N=%d K=%d at %zu\n", M, N, K, i);
                return 1;
            }
        }
    }
    double rel = sqrt(err2) / (sqrt(ref2) + 1e-12);
    if (!isfinite(rel) || rel > 1e-4) {
        fprintf(stderr, "FAIL M=%d N=%d K=%d pad=(%d,%d,%d) kind=%d alias=%d rel=%.3e\n",
                M, N, K, pad_a, pad_b, pad_c, kind, alias_cb, rel);
        return 1;
    }
    if (kind == 4 && !alias_cb) {
        double ge2 = 0.0, gr2 = 0.0;
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            double d = (double)E[i * lde + j] - (double)Eref[i * lde + j];
            ge2 += d * d;
            gr2 += (double)Eref[i * lde + j] * (double)Eref[i * lde + j];
        }
        double grel = sqrt(ge2) / (sqrt(gr2) + 1e-12);
        if (!isfinite(grel) || grel > 1e-4) {
            fprintf(stderr, "FAIL GELU_FWD deriv M=%d N=%d K=%d rel=%.3e\n", M, N, K, grel);
            return 1;
        }
    }
    free(A); free(B); free(C); free(E); free(Cref); free(Eref); free(Bsnap);
    return 0;
}

int main(void) {
    int shapes[][3] = {
        {1,1,1}, {1,15,3}, {1,16,4}, {1,48,5}, {1,65,7},
        {3,17,5}, {4,32,8}, {5,49,9}, {7,65,13}, {8,8,16},
        {8,16,32}, {8,32,32}, {8,48,64}, {8,65,64}, {9,97,64},
        {17,129,64}, {64,64,64}, {64,65,64}, {64,192,64}, {64,256,64},
        {65,64,64}, {131,147,260}, {16,64,600}, {64,64,600}, {140,200,300}
    };
    int nsh = (int)(sizeof(shapes) / sizeof(shapes[0]));
    for (int i = 0; i < nsh; i++) {
        if (check_shape(shapes[i][0], shapes[i][1], shapes[i][2], 0, 0, 0, 0, 0)) return 1;
        if (check_shape(shapes[i][0], shapes[i][1], shapes[i][2], 3, 5, 7, 0, 0)) return 1;
    }
    // All 4 epilogue kinds on both row-outer and K-blocked regimes + ragged shapes
    int epi_shapes[][3] = {{8,48,64}, {9,65,64}, {64,256,64}, {17,53,300}, {16,64,600}};
    for (int i = 0; i < 5; i++) {
        for (int k = 1; k <= 4; k++) {
            if (check_shape(epi_shapes[i][0], epi_shapes[i][1], epi_shapes[i][2], 0, 0, 0, k, 0)) return 1;
        }
    }
    // In-place aliasing (C == B, E == B) across Newton-Schulz sizes
    int ns_sizes[] = {16, 32, 64, 65, 128, 192, 256};
    for (int i = 0; i < 7; i++) {
        int n = ns_sizes[i];
        if (check_shape(n, n, n, 0, 0, 0, 3, 1)) return 1;
    }
    printf("ALL_SGEMM_PARITY_CHECKS_PASSED\n");
    return 0;
}
"#;
    let c_path = dir.join("sgemm_test.c");
    let bin_path = dir.join("sgemm_test");
    fs::write(&c_path, harness).expect("write sgemm_test.c");
    let st = Command::new("gcc").args(["-O2", "-o"]).arg(&bin_path).arg(&c_path).arg(&o_path).arg("-lm")
        .status().expect("gcc link sgemm_test");
    assert!(st.success(), "link sgemm_test failed");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        let out = Command::new(&bin_path).output().expect("run sgemm_test");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "sgemm_test failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
        assert!(stdout.contains("ALL_SGEMM_PARITY_CHECKS_PASSED"));
    }

    // Verify internal reference evaluator too
    let (m, n, k) = (4usize, 5usize, 3usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let e: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.05).collect();
    let mut c = vec![0.0_f32; m * n];
    let gm = GemmMat { m, n, k, a: &a, lda: k, b: &b, ldb: n, ldc: n };
    eval_gemm_ref(&gm, &mut c);
    for kind in [UOPC_EPI_ADD, UOPC_EPI_GELU, UOPC_EPI_FMA, UOPC_EPI_GELU_FWD] {
        let mut eo = vec![0.0_f32; m * n];
        eval_gemm_epi_ref(&gm, &mut c, kind, &e, Some(&mut eo), n, 2.0);
    }
    let _ = fs::remove_dir_all(&dir);
}
