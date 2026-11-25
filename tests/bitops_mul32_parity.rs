#![forbid(unsafe_code)]

use std::fs;
use std::process::Command;

use zero_cc::{
    compile_kernel_direct, compile_kernel_wasm, parse_rust_pipeline, verify_elf64_object,
    verify_wasm_module, RegBudget, TargetIsa,
};

const POPCNT_TZCNT_64_SRC: &str = r#"
pub fn bitops_popcnt_tzcnt_64(
    in_x: &[u64; 1],
    in_mask: &[u64; 1],
    out_res: &mut [u64; 2],
) {
    let masked = in_x[0] & in_mask[0];
    out_res[0] = masked.count_ones() as u64;
    out_res[1] = masked.trailing_zeros() as u64;
}
"#;

const POPCNT_TZCNT_32_SRC: &str = r#"
pub fn bitops_popcnt_tzcnt_32(
    in_x: &[u32; 1],
    out_res: &mut [u32; 2],
) {
    let x = in_x[0];
    out_res[0] = x.count_ones();
    out_res[1] = x.trailing_zeros();
}
"#;

const MUL32_SAT_SUB_SRC: &str = r#"
pub fn mul32_const64_satsub(
    in_a: &[u64; 1],
    in_b: &[u64; 1],
    out_res: &mut [u64; 2],
) {
    let a = in_a[0] & 0xFFFF;
    let b = in_b[0] & 0xFFFF;
    let lo_prod = a * b + (a.count_ones() as u64);
    let scaled = (a * 1000 + 2) / 3;
    let diff = scaled.saturating_sub(b * 65);
    out_res[0] = lo_prod;
    out_res[1] = diff;
}
"#;

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

#[test]
fn bitops_popcnt_tzcnt_mul32_codegen_objdump_and_live_parity() {
    let budget = RegBudget { max_gprs: 14, max_vecs: 32 };

    let k64 = parse_rust_pipeline(POPCNT_TZCNT_64_SRC, "bitops_popcnt_tzcnt_64").expect("parse k64");
    let k32 = parse_rust_pipeline(POPCNT_TZCNT_32_SRC, "bitops_popcnt_tzcnt_32").expect("parse k32");
    let kmul = parse_rust_pipeline(MUL32_SAT_SUB_SRC, "mul32_const64_satsub").expect("parse kmul");

    // 1. Compile all 3 kernels to x86_64 ELF64, AArch64 ELF64, and WASM32.
    let x86_64 = compile_kernel_direct(&k64, budget, TargetIsa::X86_64).expect("x86 k64");
    let x86_32 = compile_kernel_direct(&k32, budget, TargetIsa::X86_64).expect("x86 k32");
    let x86_mul = compile_kernel_direct(&kmul, budget, TargetIsa::X86_64).expect("x86 kmul");

    for (name, obj) in [("k64", &x86_64), ("k32", &x86_32), ("kmul", &x86_mul)] {
        verify_elf64_object(&obj.elf_bytes).unwrap_or_else(|e| panic!("verify x86 {name}: {e:?}"));
        assert!(obj.zero_spills, "{name} x86_64 must have zero spills");
    }

    for (name, k) in [("k64", &k64), ("k32", &k32), ("kmul", &kmul)] {
        let a64 = compile_kernel_direct(k, budget, TargetIsa::Aarch64)
            .unwrap_or_else(|e| panic!("a64 {name}: {e:?}"));
        verify_elf64_object(&a64.elf_bytes).unwrap_or_else(|e| panic!("verify a64 {name}: {e:?}"));
        assert!(a64.zero_spills, "{name} aarch64 must have zero spills");

        let wasm = compile_kernel_wasm(k, budget)
            .unwrap_or_else(|e| panic!("wasm {name}: {e:?}"));
        verify_wasm_module(&wasm.wasm_bytes).unwrap_or_else(|e| panic!("verify wasm {name}: {e:?}"));
    }

    // Verify Kernel::eval exactness on edge cases including 0 -> 64 / 0 -> 32.
    for &x in &[0u64, 1, 2, 1 << 31, 1 << 32, 1 << 63, u64::MAX, 0xA5A5_5A5A_F00D_CAFE] {
        let m = u64::MAX;
        let out = k64.eval(&[&[x as i64], &[m as i64], &[0, 0]]).expect("eval k64");
        assert_eq!(out[2][0] as u64, u64::from(x.count_ones()));
        assert_eq!(out[2][1] as u64, u64::from(x.trailing_zeros()));
    }
    for &x in &[0u32, 1, 2, 1 << 15, 1 << 31, u32::MAX, 0xDEAD_BEEF] {
        let out = k32.eval(&[&[i64::from(x)], &[0, 0]]).expect("eval k32");
        assert_eq!(out[1][0] as u32, x.count_ones());
        assert_eq!(out[1][1] as u32, x.trailing_zeros());
    }

    // 2. Disassemble x86_64 ELF64 objects with objdump to verify exact AVX2 & scalar ISA emission.
    #[cfg(not(miri))]
    {
        let dir = std::env::temp_dir().join(format!("zero_cc_bitops_parity_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir temp");

        let p64 = dir.join("k64.o");
        let p32 = dir.join("k32.o");
        let pmul = dir.join("kmul.o");
        fs::write(&p64, &x86_64.elf_bytes).expect("write k64.o");
        fs::write(&p32, &x86_32.elf_bytes).expect("write k32.o");
        fs::write(&pmul, &x86_mul.elf_bytes).expect("write kmul.o");

        let dis64 = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&p64).output().expect("objdump k64");
        assert!(dis64.status.success());
        let asm64 = String::from_utf8_lossy(&dis64.stdout);
        assert!(asm64.contains("vpshufb"), "expected AVX2 Mula-Kurz-Lemire vpshufb in k64:\n{asm64}");
        assert!(asm64.contains("vpmulld"), "expected AVX2 vpmulld in k64 popcnt reduction:\n{asm64}");
        assert!(asm64.contains("popcnt"), "expected scalar popcnt in k64:\n{asm64}");
        assert!(
            asm64.contains("tzcnt") && (asm64.contains("0f bc") && asm64.contains("f3 4")),
            "expected scalar tzcnt (f3 REX.W 0f bc) in k64:\n{asm64}"
        );

        let dismul = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&pmul).output().expect("objdump kmul");
        assert!(dismul.status.success());
        let asmmul = String::from_utf8_lossy(&dismul.stdout);
        assert!(asmmul.contains("vpmulld"), "expected AVX2 vpmulld in kmul:\n{asmmul}");

        // 3. Link & execute live x86_64 native harness testing BOTH 8-wide AVX2 loop and scalar tail.
        let mut xs64 = vec![
            0u64, 1, 2, 3, 7, 8, 0x8000_0000, 0xFFFF_FFFF, 0x1_0000_0000,
            0xFFFF_FFFF_0000_0000, 1u64 << 63, u64::MAX - 1, u64::MAX,
        ];
        for bit in 0..64u32 {
            xs64.push(1u64 << bit);
            xs64.push(!0u64 << bit);
            xs64.push(!0u64 >> bit);
        }
        let mut rng = SplitMix64(0x50AA_B170_C0DE_2026);
        while xs64.len() < 1037 {
            xs64.push(rng.next_u64());
        }
        let mut masks64 = Vec::with_capacity(xs64.len());
        for i in 0..xs64.len() {
            masks64.push(if i < 205 { u64::MAX } else { rng.next_u64() });
        }

        let mut xs32 = vec![0u32, 1, 2, 0x8000, 0x8000_0000, u32::MAX - 1, u32::MAX];
        for bit in 0..32u32 {
            xs32.push(1u32 << bit);
        }
        while xs32.len() < 261 {
            xs32.push(rng.next_u64() as u32);
        }

        let mut as64 = Vec::with_capacity(261);
        let mut bs64 = Vec::with_capacity(261);
        for i in 0..261usize {
            let a = match i {
                0 => 0,
                1 => 1,
                2 => u32::MAX as u64,
                3 => 0x0000_00FF_FFFF_FFFF,
                _ => rng.next_u64() & 0x0000_FFFF_FFFF_FFFF,
            };
            let b = match i {
                0 => 0,
                1 => 1000,
                2 => 0xFFFF,
                3 => 0x0000_00FF_FFFF_FFFF,
                _ => rng.next_u64() & 0x0000_FFFF_FFFF_FFFF,
            };
            as64.push(a);
            bs64.push(b);
        }

        // Write binary input files and compile small C verifier linked directly against k64.o, k32.o, kmul.o.
        let write_u64s = |p: &std::path::Path, v: &[u64]| {
            let mut b = Vec::with_capacity(v.len() * 8);
            for &x in v { b.extend_from_slice(&x.to_le_bytes()); }
            fs::write(p, b).expect("write u64 bin");
        };
        let write_u32s = |p: &std::path::Path, v: &[u32]| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for &x in v { b.extend_from_slice(&x.to_le_bytes()); }
            fs::write(p, b).expect("write u32 bin");
        };
        write_u64s(&dir.join("xs64.bin"), &xs64);
        write_u64s(&dir.join("masks64.bin"), &masks64);
        write_u32s(&dir.join("xs32.bin"), &xs32);
        write_u64s(&dir.join("as64.bin"), &as64);
        write_u64s(&dir.join("bs64.bin"), &bs64);

        let harness_c = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void zero_cc_bitops_popcnt_tzcnt_64_batch(const uint64_t *x, const uint64_t *m, uint64_t *out, uint64_t cnt);
extern void zero_cc_bitops_popcnt_tzcnt_32_batch(const uint32_t *x, uint32_t *out, uint64_t cnt);
extern void zero_cc_mul32_const64_satsub_batch(const uint64_t *a, const uint64_t *b, uint64_t *out, uint64_t cnt);

static void load_bin(const char *path, void *dst, size_t nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f || fread(dst, 1, nbytes, f) != nbytes) { fprintf(stderr, "read fail %s\n", path); exit(2); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    char path[512];
    const size_t n64 = 1037, n32 = 261, nmul = 261;
    uint64_t xs64[1037], masks64[1037], out64[1037 * 2];
    uint32_t xs32[261], out32[261 * 2];
    uint64_t as64[261], bs64[261], outm[261 * 2];

    snprintf(path, sizeof(path), "%s/xs64.bin", argv[1]); load_bin(path, xs64, sizeof(xs64));
    snprintf(path, sizeof(path), "%s/masks64.bin", argv[1]); load_bin(path, masks64, sizeof(masks64));
    snprintf(path, sizeof(path), "%s/xs32.bin", argv[1]); load_bin(path, xs32, sizeof(xs32));
    snprintf(path, sizeof(path), "%s/as64.bin", argv[1]); load_bin(path, as64, sizeof(as64));
    snprintf(path, sizeof(path), "%s/bs64.bin", argv[1]); load_bin(path, bs64, sizeof(bs64));

    zero_cc_bitops_popcnt_tzcnt_64_batch(xs64, masks64, out64, n64);
    for (size_t i = 0; i < n64; i++) {
        uint64_t v = xs64[i] & masks64[i];
        uint64_t ep = (uint64_t)__builtin_popcountll(v);
        uint64_t et = (v == 0) ? 64ULL : (uint64_t)__builtin_ctzll(v);
        if (out64[2*i] != ep || out64[2*i+1] != et) {
            fprintf(stderr, "k64 mismatch i=%zu v=0x%016llx got=(%llu,%llu) exp=(%llu,%llu)\n",
                i, (unsigned long long)v, (unsigned long long)out64[2*i], (unsigned long long)out64[2*i+1],
                (unsigned long long)ep, (unsigned long long)et);
            return 3;
        }
    }

    zero_cc_bitops_popcnt_tzcnt_32_batch(xs32, out32, n32);
    for (size_t i = 0; i < n32; i++) {
        uint32_t v = xs32[i];
        uint32_t ep = (uint32_t)__builtin_popcount(v);
        uint32_t et = (v == 0) ? 32U : (uint32_t)__builtin_ctz(v);
        if (out32[2*i] != ep || out32[2*i+1] != et) {
            fprintf(stderr, "k32 mismatch i=%zu v=0x%08x got=(%u,%u) exp=(%u,%u)\n",
                i, v, out32[2*i], out32[2*i+1], ep, et);
            return 4;
        }
    }

    zero_cc_mul32_const64_satsub_batch(as64, bs64, outm, nmul);
    for (size_t i = 0; i < nmul; i++) {
        uint64_t a = as64[i] & 0xFFFFULL, b = bs64[i] & 0xFFFFULL;
        uint64_t e0 = a * b + (uint64_t)__builtin_popcountll(a);
        uint64_t s = (a * 1000ULL + 2ULL) / 3ULL;
        uint64_t t = b * 65ULL;
        uint64_t e1 = (s >= t) ? (s - t) : 0ULL;
        if (outm[2*i] != e0 || outm[2*i+1] != e1) {
            fprintf(stderr, "kmul mismatch i=%zu a=%llu b=%llu got=(%llu,%llu) exp=(%llu,%llu)\n",
                i, (unsigned long long)a, (unsigned long long)b,
                (unsigned long long)outm[2*i], (unsigned long long)outm[2*i+1],
                (unsigned long long)e0, (unsigned long long)e1);
            return 5;
        }
    }
    return 0;
}
"#;
        let c_path = dir.join("harness.c");
        let bin_path = dir.join("harness");
        fs::write(&c_path, harness_c).expect("write harness.c");
        let cc = Command::new("cc")
            .args(["-O2", "-o"])
            .arg(&bin_path)
            .arg(&c_path)
            .arg(&p64)
            .arg(&p32)
            .arg(&pmul)
            .output()
            .expect("invoke cc");
        assert!(cc.status.success(), "cc failed:\n{}", String::from_utf8_lossy(&cc.stderr));

        let run = Command::new(&bin_path).arg(&dir).output().expect("run harness");
        assert!(
            run.status.success(),
            "live bitops_mul32_parity harness failed:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

const SPLIT64_SHRVAR_CMP_SRC: &str = r#"
pub fn split64_shrvar_cmp(
    in_a: &[u64; 1],
    in_b: &[u64; 1],
    in_s: &[u64; 1],
    out_res: &mut [u64; 2],
) {
    let a = in_a[0];
    let b = in_b[0];
    let s = in_s[0] & 63;
    let shifted = a >> s;
    let lo_b = b & 0xFFFF_FFFF;
    let c_eq = if a == b { 1 } else { 0 };
    let c_ne = if a != b { 2 } else { 0 };
    let c_lt = if a < b { 4 } else { 0 };
    let c_le = if a <= b { 8 } else { 0 };
    let c_gt = if a > b { 16 } else { 0 };
    let c_ge = if a >= b { 32 } else { 0 };
    let c_lo = if a <= lo_b { 64 } else { 0 };
    out_res[0] = shifted;
    out_res[1] = c_eq | c_ne | c_lt | c_le | c_gt | c_ge | c_lo;
}
"#;

#[test]
fn split64_shrvar_and_cmp_batch8_live_parity() {
    let budget = RegBudget { max_gprs: 14, max_vecs: 32 };
    let k = parse_rust_pipeline(SPLIT64_SHRVAR_CMP_SRC, "split64_shrvar_cmp").expect("parse");
    let obj = compile_kernel_direct(&k, budget, TargetIsa::X86_64).expect("compile");
    verify_elf64_object(&obj.elf_bytes).expect("verify elf");
    assert!(obj.zero_spills, "split64_shrvar_cmp must have 0 spills");

    let dir = std::env::temp_dir().join(format!("zero_cc_split64_cmp_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    let obj_path = dir.join("k.o");
    fs::write(&obj_path, &obj.elf_bytes).expect("write k.o");

    if cfg!(all(target_os = "linux", target_arch = "x86_64"))
        && std::arch::is_x86_feature_detected!("avx2")
    {
        if let Ok(od) = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&obj_path).output() {
            let dis = String::from_utf8_lossy(&od.stdout);
            assert!(dis.contains("vpsrlvd") && dis.contains("vpsllvd"), "expected AVX2 Batch8 vpsrlvd/vpsllvd:\n{dis}");
            assert!(!dis.contains("[rsp") && !dis.contains("[rbp"), "unexpected stack spill:\n{dis}");
            assert!(!dis.contains("\tdiv ") && !dis.contains("\tidiv "), "unexpected div:\n{dis}");
        }

        let mut rng = SplitMix64(0x91A2_B3C4_D5E6_F708);
        let n = 512usize;
        let mut a_in = Vec::with_capacity(n);
        let mut b_in = Vec::with_capacity(n);
        let mut s_in = Vec::with_capacity(n);
        let edge_vals = [
            0u64, 1, 0xFFFF_FFFF, 0x1_0000_0000, 0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000, u64::MAX, 0xDEAD_BEEF_CAFE_BABE,
        ];
        for i in 0..n {
            let a = if i < 64 { edge_vals[i % edge_vals.len()] } else { rng.next_u64() };
            let b = if i < 64 {
                edge_vals[(i / edge_vals.len()) % edge_vals.len()]
            } else if i % 5 == 0 {
                a
            } else if i % 5 == 1 {
                a ^ 1
            } else if i % 5 == 2 {
                a ^ (1u64 << 32)
            } else {
                rng.next_u64()
            };
            let s = (i as u64) % 64;
            a_in.push(a);
            b_in.push(b);
            s_in.push(s);
        }
        let to_bytes = |v: &[u64]| -> Vec<u8> {
            let mut b = Vec::with_capacity(v.len() * 8);
            for &x in v { b.extend_from_slice(&x.to_le_bytes()); }
            b
        };
        fs::write(dir.join("a.bin"), to_bytes(&a_in)).expect("write a");
        fs::write(dir.join("b.bin"), to_bytes(&b_in)).expect("write b");
        fs::write(dir.join("s.bin"), to_bytes(&s_in)).expect("write s");

        let c_src = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void zero_cc_split64_shrvar_cmp_batch(const uint64_t*, const uint64_t*, const uint64_t*, uint64_t*, uint64_t);
static void load_bin(const char *p, void *dst, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f || fread(dst, 1, n, f) != n) exit(2);
    fclose(f);
}
int main(int argc, char **argv) {
    enum { N = 512 };
    uint64_t a[N], b[N], s[N], out[N * 2];
    char p[512];
    snprintf(p, sizeof(p), "%s/a.bin", argv[1]); load_bin(p, a, sizeof(a));
    snprintf(p, sizeof(p), "%s/b.bin", argv[1]); load_bin(p, b, sizeof(b));
    snprintf(p, sizeof(p), "%s/s.bin", argv[1]); load_bin(p, s, sizeof(s));
    zero_cc_split64_shrvar_cmp_batch(a, b, s, out, N);
    for (size_t i = 0; i < N; i++) {
        uint64_t sh = s[i] & 63ULL;
        uint64_t e0 = a[i] >> sh;
        uint64_t lo_b = b[i] & 0xFFFFFFFFULL;
        uint64_t e1 = (a[i] == b[i] ? 1ULL : 0ULL)
            | (a[i] != b[i] ? 2ULL : 0ULL)
            | (a[i] < b[i] ? 4ULL : 0ULL)
            | (a[i] <= b[i] ? 8ULL : 0ULL)
            | (a[i] > b[i] ? 16ULL : 0ULL)
            | (a[i] >= b[i] ? 32ULL : 0ULL)
            | (a[i] <= lo_b ? 64ULL : 0ULL);
        if (out[2 * i] != e0 || out[2 * i + 1] != e1) {
            fprintf(stderr, "mismatch i=%zu a=0x%016llx b=0x%016llx s=%llu got=(0x%llx,0x%llx) exp=(0x%llx,0x%llx)\n",
                i, (unsigned long long)a[i], (unsigned long long)b[i], (unsigned long long)sh,
                (unsigned long long)out[2 * i], (unsigned long long)out[2 * i + 1],
                (unsigned long long)e0, (unsigned long long)e1);
            return 3;
        }
    }
    return 0;
}
"#;
        let c_path = dir.join("h.c");
        let bin_path = dir.join("h");
        fs::write(&c_path, c_src).expect("write h.c");
        let cc = Command::new("cc").args(["-O2", "-o"]).arg(&bin_path).arg(&c_path).arg(&obj_path).output().expect("cc");
        assert!(cc.status.success(), "cc failed:\n{}", String::from_utf8_lossy(&cc.stderr));
        let run = Command::new(&bin_path).arg(&dir).output().expect("run");
        assert!(run.status.success(), "harness failed:\n{}", String::from_utf8_lossy(&run.stderr));
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_batch8_split64_shl_imm_1_to_31_parity() {
    let src = r#"
pub fn split64_shl_imm(a: &[u64; 1], b: &[u64; 1], out: &mut [u64; 2]) {
    let x = a[0];
    let y = b[0];
    let s1 = x << 1;
    let s7 = y << 7;
    let s16 = x << 16;
    let s20 = y << 20;
    let s31 = x << 31;
    out[0] = (s1 ^ s16) + s31;
    out[1] = (s7 ^ s20) + (y << 13);
}
"#;
    let budget = RegBudget { max_gprs: 14, max_vecs: 32 };
    let k = parse_rust_pipeline(src, "split64_shl_imm").expect("parse split64_shl_imm");
    let obj = compile_kernel_direct(&k, budget, TargetIsa::X86_64).expect("compile split64_shl_imm");
    verify_elf64_object(&obj.elf_bytes).expect("verify elf");
    assert!(obj.zero_spills, "split64_shl_imm must have 0 spills");

    let dir = std::env::temp_dir().join(format!("zero_cc_shl_imm_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    let obj_path = dir.join("kshl.o");
    fs::write(&obj_path, &obj.elf_bytes).expect("write kshl.o");

    if cfg!(all(target_arch = "x86_64", target_os = "linux")) && std::arch::is_x86_feature_detected!("avx2") {
        if let Ok(od) = Command::new("objdump").args(["-d", "-M", "intel"]).arg(&obj_path).output() {
            let dis = String::from_utf8_lossy(&od.stdout);
            assert!(dis.contains("vpslld") && dis.contains("vpsrld"), "expected AVX2 Batch8 vpslld/vpsrld:\n{dis}");
            assert!(!dis.contains("[rsp") && !dis.contains("[rbp"), "unexpected stack spill:\n{dis}");
            assert!(!dis.contains("\tdiv ") && !dis.contains("\tidiv "), "unexpected div:\n{dis}");
        }
        let mut rng = SplitMix64(0x511F_7640_A1B2_C3D4);
        let n = 521usize;
        let mut a_in = Vec::with_capacity(n);
        let mut b_in = Vec::with_capacity(n);
        let edges: [u64; 8] = [
            0,
            1,
            u32::MAX as u64,
            1u64 << 31,
            1u64 << 32,
            0x8000_0000_8000_0001,
            0xFFFF_FFFF_0000_0001,
            u64::MAX,
        ];
        for i in 0..n {
            let a = if i < 64 { edges[i & 7] } else { rng.next_u64() };
            let b = if i < 64 { edges[(i >> 3) & 7] } else { rng.next_u64() };
            a_in.push(a);
            b_in.push(b);
        }
        let to_bytes = |v: &[u64]| -> Vec<u8> {
            let mut buf = Vec::with_capacity(v.len() * 8);
            for &x in v { buf.extend_from_slice(&x.to_le_bytes()); }
            buf
        };
        fs::write(dir.join("a.bin"), to_bytes(&a_in)).expect("write a");
        fs::write(dir.join("b.bin"), to_bytes(&b_in)).expect("write b");

        let c_src = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void zero_cc_split64_shl_imm_batch(const uint64_t*, const uint64_t*, uint64_t*, uint64_t);
static void load_bin(const char *p, void *dst, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f || fread(dst, 1, n, f) != n) exit(2);
    fclose(f);
}
int main(int argc, char **argv) {
    enum { N = 521 };
    uint64_t a[N], b[N], out[N * 2];
    char p[512];
    snprintf(p, sizeof(p), "%s/a.bin", argv[1]); load_bin(p, a, sizeof(a));
    snprintf(p, sizeof(p), "%s/b.bin", argv[1]); load_bin(p, b, sizeof(b));
    zero_cc_split64_shl_imm_batch(a, b, out, N);
    for (size_t i = 0; i < N; i++) {
        uint64_t x = a[i], y = b[i];
        uint64_t e0 = ((x << 1) ^ (x << 16)) + (x << 31);
        uint64_t e1 = ((y << 7) ^ (y << 20)) + (y << 13);
        if (out[2 * i] != e0 || out[2 * i + 1] != e1) {
            fprintf(stderr, "shl mismatch i=%zu x=0x%016llx y=0x%016llx got=(0x%llx,0x%llx) exp=(0x%llx,0x%llx)\n",
                i, (unsigned long long)x, (unsigned long long)y,
                (unsigned long long)out[2 * i], (unsigned long long)out[2 * i + 1],
                (unsigned long long)e0, (unsigned long long)e1);
            return 3;
        }
    }
    return 0;
}
"#;
        let c_path = dir.join("h_shl.c");
        let bin_path = dir.join("h_shl");
        fs::write(&c_path, c_src).expect("write h_shl.c");
        let cc = Command::new("cc").args(["-O2", "-o"]).arg(&bin_path).arg(&c_path).arg(&obj_path).output().expect("cc");
        assert!(cc.status.success(), "cc failed:\n{}", String::from_utf8_lossy(&cc.stderr));
        let run = Command::new(&bin_path).arg(&dir).output().expect("run");
        assert!(run.status.success(), "shl harness failed:\n{}", String::from_utf8_lossy(&run.stderr));
    }
    let _ = fs::remove_dir_all(&dir);
}
