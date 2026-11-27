#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use zero_cc::rust_fe::{kernel_reg_budget, parse_rust_file_kernels};
use zero_cc::{
    compile_kernel_direct, compile_kernel_wasm, compile_standalone_elf_exec, optimize_kernel,
    verify_elf64_executable, verify_elf64_object, verify_wasm_module, BufAccess, CompiledWasm,
    Domain, ElemTy, Kernel, RegBudget, TargetIsa,
};

struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (i128::from(hi) - i128::from(lo) + 1) as u128;
        let v = (u128::from(self.next_u64()) % span) as i128;
        (i128::from(lo) + v) as i64
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zero_cc_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[cfg(unix)]
fn set_exec_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod 0755");
}

#[cfg(not(unix))]
fn set_exec_mode(_path: &Path) {}

fn audit_objdump_x86(path: &Path, expected_syms: Option<&[&str]>) -> (usize, usize, usize) {
    let out = Command::new("objdump")
        .args(["-d", "-M", "intel"])
        .arg(path)
        .output()
        .expect("run objdump");
    assert!(out.status.success(), "objdump failed on {}", path.display());
    let text = String::from_utf8_lossy(&out.stdout);
    let mut insns = 0usize;
    let mut ymm_insns = 0usize;
    let mut syms_seen: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.ends_with(">:") {
            if let Some(s) = trimmed.split('<').nth(1).and_then(|r| r.strip_suffix(">:")) {
                syms_seen.push(s.to_string());
            }
            continue;
        }
        let Some((_, rest)) = trimmed.split_once(":\t") else { continue; };
        let Some((_, asm)) = rest.split_once('\t') else { continue; };
        let asm = asm.trim();
        if asm.is_empty() || asm.starts_with("nop") || asm.starts_with("data16") {
            continue;
        }
        insns += 1;
        if asm.contains("ymm") {
            ymm_insns += 1;
        }
        let mnemonic = asm.split_whitespace().next().unwrap_or("");
        assert_ne!(mnemonic, "idiv", "forbidden idiv in {}: {asm}", path.display());
        assert_ne!(mnemonic, "div", "forbidden div in {}: {asm}", path.display());
        assert!(
            !asm.contains("[rsp") && !asm.contains("[rbp"),
            "forbidden stack spill in {}: {asm}",
            path.display()
        );
    }
    if let Some(want) = expected_syms {
        assert_eq!(
            syms_seen.len(),
            want.len(),
            "ET_EXEC symbol count mismatch in {}: got {:?}, expected {:?}",
            path.display(),
            syms_seen,
            want
        );
        for s in &syms_seen {
            assert!(
                want.contains(&s.as_str()),
                "unexpected non-direct symbol '{s}' in ET_EXEC {}",
                path.display()
            );
        }
    }
    (insns, syms_seen.len(), ymm_insns)
}

fn corner_values_for(name: &str, dom: Domain) -> Vec<i64> {
    let raw_corners: &[i64] = match name {
        "raw" => &[
            -262_144, -262_143, -290, -289, -288, -30, -29, -28, -1, 0, 1, 28, 29, 30, 288, 289,
            290, 262_143, 262_144,
        ],
        "x" => &[-32_768, -32_767, -289, -29, -1, 0, 1, 29, 289, 32_766, 32_767],
        "even" | "odd" => &[
            -1_000_000_000, -70_000_000, -289, -29, -1, 0, 1, 29, 289, 70_000_000, 1_000_000_000,
        ],
        "next" => &[
            0, 1, 32_767, 32_768, 32_769, 65_535, 65_536, 2_147_483_647, 2_147_483_648,
            4_294_967_295,
        ],
        "word" => &[0, 1, 255, 256, 32_767, 32_768, 0x55AA, 65_535],
        _ => &[dom.lo, -1, 0, 1, dom.hi],
    };
    let mut out = Vec::new();
    for &v in raw_corners {
        if v >= dom.lo && v <= dom.hi {
            out.push(v);
        }
    }
    if out.is_empty() {
        out.push(dom.lo);
        out.push(dom.hi);
    }
    out
}

fn build_exec_inputs(kernel: &Kernel, blocks: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut rng = XorShift64(seed);
    let mut bufs = Vec::with_capacity(kernel.bufs.len());
    for (bidx, decl) in kernel.bufs.iter().enumerate() {
        let lanes = usize::from(decl.lanes);
        let mut data = vec![0i64; blocks * lanes];
        if matches!(decl.access, BufAccess::ReadOnly) {
            let corners = corner_values_for(&decl.name, decl.domain);
            for blk in 0..blocks {
                for l in 0..lanes {
                    let val = if blk < corners.len() {
                        corners[(blk + l + bidx) % corners.len()]
                    } else if blk == corners.len() {
                        if l % 2 == 0 { decl.domain.lo } else { decl.domain.hi }
                    } else {
                        rng.range_i64(decl.domain.lo, decl.domain.hi)
                    };
                    data[blk * lanes + l] = decl.elem_ty.cast_i64(val);
                }
            }
        }
        bufs.push(data);
    }
    bufs
}

fn eval_expected_stdout(kernel: &Kernel, inputs: &[Vec<i64>], blocks: usize) -> Vec<u8> {
    let primary_idx = kernel
        .bufs
        .iter()
        .position(|b| matches!(b.access, BufAccess::MutNoAlias))
        .expect("kernel must have MutNoAlias output buffer");
    let out_decl = &kernel.bufs[primary_idx];
    let out_lanes = usize::from(out_decl.lanes);
    let mut bytes = Vec::with_capacity(blocks * out_decl.block_bytes());
    let mut block_slices: Vec<&[i64]> = Vec::with_capacity(kernel.bufs.len());
    for blk in 0..blocks {
        block_slices.clear();
        for (idx, decl) in kernel.bufs.iter().enumerate() {
            let lanes = usize::from(decl.lanes);
            block_slices.push(&inputs[idx][blk * lanes..(blk + 1) * lanes]);
        }
        let res = kernel.eval(&block_slices).expect("Kernel::eval");
        for &v in res[primary_idx].iter().take(out_lanes) {
            out_decl.elem_ty.push_le_elem(&mut bytes, v);
        }
    }
    bytes
}

fn generate_bench_harness(shims: &BTreeMap<String, String>) -> String {
    let mut src = String::with_capacity(16_384);
    src.push_str(
        r#"use std::hint::black_box;
use std::time::Instant;

#[inline]
const fn saturate_i32_from_i64(v: i64) -> i32 {
    let hi = if v > 2_147_483_647 { 2_147_483_647 } else { v };
    if hi < -2_147_483_648 { -2_147_483_648 } else { hi as i32 }
}

#[inline]
fn ref_dct8_1d_even(x: &[i32; 8], out: &mut [i32; 8]) {
    let a0 = (x[0] as i64) + (x[7] as i64);
    let a3 = (x[3] as i64) + (x[4] as i64);
    let a1 = (x[1] as i64) + (x[6] as i64);
    let a2 = (x[2] as i64) + (x[5] as i64);
    let e0 = a0 + a3;
    let e1 = a1 + a2;
    out[0] = saturate_i32_from_i64((e0 + e1) * 8);
    out[4] = saturate_i32_from_i64((e0 - e1) * 8);
    let e2 = a0 - a3;
    let e3 = a1 - a2;
    out[2] = saturate_i32_from_i64(10 * e2 + 4 * e3);
    out[6] = saturate_i32_from_i64(4 * e2 - 10 * e3);
}

#[inline]
fn ref_dct8_1d_odd(x: &[i32; 8], out: &mut [i32; 8]) {
    let b0 = (x[0] as i64) - (x[7] as i64);
    let b1 = (x[1] as i64) - (x[6] as i64);
    let b2 = (x[2] as i64) - (x[5] as i64);
    let b3 = (x[3] as i64) - (x[4] as i64);
    out[1] = saturate_i32_from_i64(12 * b0 + 10 * b1 + 6 * b2 + 3 * b3);
    out[3] = saturate_i32_from_i64(10 * b0 - 3 * b1 - 12 * b2 - 6 * b3);
    out[5] = saturate_i32_from_i64(6 * b0 - 12 * b1 + 3 * b2 + 10 * b3);
    out[7] = saturate_i32_from_i64(3 * b0 - 6 * b1 + 10 * b2 - 12 * b3);
}

#[inline]
fn ref_idct8_1d_even(raw: &[i32; 8], even: &mut [i64; 4]) {
    let y0 = raw[0] as i64;
    let y4 = raw[4] as i64;
    let y2 = ((raw[2] as i64) * 32 + 14) / 29;
    let y6 = ((raw[6] as i64) * 32 + 14) / 29;
    let e0 = 8 * (y0 + y4);
    let e2 = 10 * y2 + 4 * y6;
    even[0] = e0 + e2;
    even[3] = e0 - e2;
    let e1 = 8 * (y0 - y4);
    let e3 = 4 * y2 - 10 * y6;
    even[1] = e1 + e3;
    even[2] = e1 - e3;
}

#[inline]
fn ref_idct8_1d_odd(raw: &[i32; 8], odd: &mut [i64; 4]) {
    let y1 = ((raw[1] as i64) * 256 + 144) / 289;
    let y3 = ((raw[3] as i64) * 256 + 144) / 289;
    let y5 = ((raw[5] as i64) * 256 + 144) / 289;
    let y7 = ((raw[7] as i64) * 256 + 144) / 289;
    odd[0] = 12 * y1 + 10 * y3 + 6 * y5 + 3 * y7;
    odd[1] = 10 * y1 - 3 * y3 - 12 * y5 - 6 * y7;
    odd[2] = 6 * y1 - 12 * y3 + 3 * y5 + 10 * y7;
    odd[3] = 3 * y1 - 6 * y3 + 10 * y5 - 12 * y7;
}

#[inline]
fn ref_idct8_1d_combine(even: &[i64; 4], odd: &[i64; 4], out: &mut [i32; 8]) {
    for k in 0..4usize {
        let a = even[k];
        let b = odd[k];
        if a >= -70_000_000_000 && b >= -70_000_000_000 && a <= 70_000_000_000 && b <= 70_000_000_000 {
            out[k] = saturate_i32_from_i64(a + b);
            out[7 - k] = saturate_i32_from_i64(a - b);
        } else {
            out[k] = 0;
            out[7 - k] = 0;
        }
    }
}

#[inline(always)]
const fn ref_rans_refill_branchless(next: u32, word: u16) -> (u32, usize) {
    let need = if next < 32_768 { 1u32 } else { 0u32 };
    let mask = 0u32.wrapping_sub(need);
    let refill = (next << 16) | (word as u32);
    let state = next ^ ((refill ^ next) & mask);
    (state, (need as usize) << 1)
}

#[inline(never)]
fn ref_idct8_1d_even_batch(raw: &[[i32; 8]], even: &mut [[i64; 4]]) {
    for (r, e) in raw.iter().zip(even.iter_mut()) { ref_idct8_1d_even(r, e); }
}

#[inline(never)]
fn ref_idct8_1d_odd_batch(raw: &[[i32; 8]], odd: &mut [[i64; 4]]) {
    for (r, o) in raw.iter().zip(odd.iter_mut()) { ref_idct8_1d_odd(r, o); }
}

#[inline(never)]
fn ref_dct8_1d_even_batch(x: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for (xi, o) in x.iter().zip(out.iter_mut()) { ref_dct8_1d_even(xi, o); }
}

#[inline(never)]
fn ref_dct8_1d_odd_batch(x: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for (xi, o) in x.iter().zip(out.iter_mut()) { ref_dct8_1d_odd(xi, o); }
}

#[inline(never)]
fn ref_idct8_1d_combine_batch(even: &[[i64; 4]], odd: &[[i64; 4]], out: &mut [[i32; 8]]) {
    for ((e, o), dst) in even.iter().zip(odd.iter()).zip(out.iter_mut()) {
        ref_idct8_1d_combine(e, o, dst);
    }
}

#[inline(never)]
fn ref_rans_refill_branchless_batch(
    next: &[[u32; 8]],
    word: &[[u16; 8]],
    out0: &mut [[u32; 8]],
    out1: &mut [[i64; 8]],
) {
    let n = next.len().min(word.len()).min(out0.len()).min(out1.len());
    for b in 0..n {
        for l in 0..8 {
            let (st, adv) = ref_rans_refill_branchless(next[b][l], word[b][l]);
            out0[b][l] = st;
            out1[b][l] = adv as i64;
        }
    }
}
"#,
    );

    for (name, shim) in shims {
        src.push_str(&format!("\nmod zcc_{name} {{\n{shim}}}\n"));
    }

    src.push_str(
        r#"
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (i128::from(hi) - i128::from(lo) + 1) as u128;
        (i128::from(lo) + (u128::from(self.next()) % span) as i128) as i64
    }
}

fn bench_pair<F1: FnMut(), F2: FnMut()>(name: &str, calls_per_trial: usize, mut f_ref: F1, mut f_zcc: F2) {
    for _ in 0..3 { f_ref(); f_zcc(); }
    let mut ref_ns = [0.0f64; 9];
    let mut zcc_ns = [0.0f64; 9];
    for t in 0..9 {
        if t % 2 == 0 {
            let t0 = Instant::now();
            f_ref();
            ref_ns[t] = t0.elapsed().as_nanos() as f64 / calls_per_trial as f64;
            let t1 = Instant::now();
            f_zcc();
            zcc_ns[t] = t1.elapsed().as_nanos() as f64 / calls_per_trial as f64;
        } else {
            let t1 = Instant::now();
            f_zcc();
            zcc_ns[t] = t1.elapsed().as_nanos() as f64 / calls_per_trial as f64;
            let t0 = Instant::now();
            f_ref();
            ref_ns[t] = t0.elapsed().as_nanos() as f64 / calls_per_trial as f64;
        }
    }
    ref_ns.sort_by(|a, b| a.total_cmp(b));
    zcc_ns.sort_by(|a, b| a.total_cmp(b));
    let r_med = ref_ns[4];
    let z_med = zcc_ns[4];
    let speedup = r_med / z_med;
    println!("BENCH|{name}|{r_med:.3}|{z_med:.3}|{speedup:.3}");
}

fn main() {
    const PARITY_BLOCKS: usize = 4096;
    const BENCH_BLOCKS: usize = 512;
    const ITERS: usize = 256;
    const CALLS: usize = BENCH_BLOCKS * ITERS;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // 1. idct8_1d_even & idct8_1d_odd (raw in -262144..=262144)
    let raw_corners: [i32; 19] = [
        -262_144, -262_143, -290, -289, -288, -30, -29, -28, -1, 0,
        1, 28, 29, 30, 288, 289, 290, 262_143, 262_144,
    ];
    let mut raw = vec![[0i32; 8]; PARITY_BLOCKS];
    for (b, row) in raw.iter_mut().enumerate() {
        for l in 0..8 {
            row[l] = if b < raw_corners.len() {
                raw_corners[(b + l) % raw_corners.len()]
            } else {
                rng.range_i64(-262_144, 262_144) as i32
            };
        }
    }
    let mut even_ref = vec![[0i64; 4]; PARITY_BLOCKS];
    let mut even_zcc = vec![[0i64; 4]; PARITY_BLOCKS];
    ref_idct8_1d_even_batch(&raw, &mut even_ref);
    zcc_idct8_1d_even::idct8_1d_even_batch(&raw, &mut even_zcc);
    assert_eq!(even_ref, even_zcc, "idct8_1d_even 4096-block parity mismatch");

    let mut odd_ref = vec![[0i64; 4]; PARITY_BLOCKS];
    let mut odd_zcc = vec![[0i64; 4]; PARITY_BLOCKS];
    ref_idct8_1d_odd_batch(&raw, &mut odd_ref);
    zcc_idct8_1d_odd::idct8_1d_odd_batch(&raw, &mut odd_zcc);
    assert_eq!(odd_ref, odd_zcc, "idct8_1d_odd 4096-block parity mismatch");

    // 2. dct8_1d_even & dct8_1d_odd (x in -32768..=32767)
    let x_corners: [i32; 11] = [-32_768, -32_767, -289, -29, -1, 0, 1, 29, 289, 32_766, 32_767];
    let mut xbuf = vec![[0i32; 8]; PARITY_BLOCKS];
    for (b, row) in xbuf.iter_mut().enumerate() {
        for l in 0..8 {
            row[l] = if b < x_corners.len() {
                x_corners[(b + l) % x_corners.len()]
            } else {
                rng.range_i64(-32_768, 32_767) as i32
            };
        }
    }
    let mut de_ref = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut de_zcc = vec![[0i32; 8]; PARITY_BLOCKS];
    ref_dct8_1d_even_batch(&xbuf, &mut de_ref);
    zcc_dct8_1d_even::dct8_1d_even_batch(&xbuf, &mut de_zcc);
    assert_eq!(de_ref, de_zcc, "dct8_1d_even 4096-block parity mismatch");

    let mut do_ref = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut do_zcc = vec![[0i32; 8]; PARITY_BLOCKS];
    ref_dct8_1d_odd_batch(&xbuf, &mut do_ref);
    zcc_dct8_1d_odd::dct8_1d_odd_batch(&xbuf, &mut do_zcc);
    assert_eq!(do_ref, do_zcc, "dct8_1d_odd 4096-block parity mismatch");

    // 3. idct8_1d_combine (even, odd in -1_000_000_000..=1_000_000_000)
    let eo_corners: [i64; 11] = [
        -1_000_000_000, -70_000_000, -289, -29, -1, 0, 1, 29, 289, 70_000_000, 1_000_000_000,
    ];
    let mut ceven = vec![[0i64; 4]; PARITY_BLOCKS];
    let mut codd = vec![[0i64; 4]; PARITY_BLOCKS];
    for b in 0..PARITY_BLOCKS {
        for l in 0..4 {
            if b < eo_corners.len() {
                ceven[b][l] = eo_corners[(b + l) % eo_corners.len()];
                codd[b][l] = eo_corners[(b + l + 3) % eo_corners.len()];
            } else {
                ceven[b][l] = rng.range_i64(-1_000_000_000, 1_000_000_000);
                codd[b][l] = rng.range_i64(-1_000_000_000, 1_000_000_000);
            }
        }
    }
    let mut comb_ref = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut comb_zcc = vec![[0i32; 8]; PARITY_BLOCKS];
    ref_idct8_1d_combine_batch(&ceven, &codd, &mut comb_ref);
    zcc_idct8_1d_combine::idct8_1d_combine_batch(&ceven, &codd, &mut comb_zcc);
    assert_eq!(comb_ref, comb_zcc, "idct8_1d_combine 4096-block parity mismatch");

    // 4. rans_refill_branchless (full u32 next in 0..=u32::MAX, word in 0..=65535)
    let next_corners: [u32; 10] = [
        0, 1, 32_767, 32_768, 32_769, 65_535, 65_536, 2_147_483_647, 2_147_483_648, u32::MAX,
    ];
    let word_corners: [u16; 8] = [0, 1, 255, 256, 32_767, 32_768, 0x55AA, 65_535];
    let mut rnext = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut rword = vec![[0u16; 8]; PARITY_BLOCKS];
    for b in 0..PARITY_BLOCKS {
        for l in 0..8 {
            if b < next_corners.len() {
                rnext[b][l] = next_corners[(b + l) % next_corners.len()];
                rword[b][l] = word_corners[(b + l * 3) % word_corners.len()];
            } else {
                rnext[b][l] = rng.range_i64(0, i64::from(u32::MAX)) as u32;
                rword[b][l] = rng.range_i64(0, 65_535) as u16;
            }
        }
    }
    let mut r0_ref = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut r1_ref = vec![[0i64; 8]; PARITY_BLOCKS];
    let mut r0_zcc = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut r1_zcc = vec![[0i64; 8]; PARITY_BLOCKS];
    ref_rans_refill_branchless_batch(&rnext, &rword, &mut r0_ref, &mut r1_ref);
    zcc_rans_refill_branchless::rans_refill_branchless_batch(&rnext, &rword, &mut r0_zcc, &mut r1_zcc);
    assert_eq!(r0_ref, r0_zcc, "rans_refill_branchless out0 4096-block parity mismatch");
    assert_eq!(r1_ref, r1_zcc, "rans_refill_branchless out1 4096-block parity mismatch");

    // 5. Non-multiple-of-4 scalar tail parity (blocks = 1, 3, 7) across all 6 kernels
    for &tail_n in &[1usize, 3, 7] {
        let mut te = vec![[0i64; 4]; tail_n];
        zcc_idct8_1d_even::idct8_1d_even_batch(&raw[..tail_n], &mut te);
        assert_eq!(&even_ref[..tail_n], &te[..], "idct8_1d_even tail_n={tail_n} mismatch");

        let mut to = vec![[0i64; 4]; tail_n];
        zcc_idct8_1d_odd::idct8_1d_odd_batch(&raw[..tail_n], &mut to);
        assert_eq!(&odd_ref[..tail_n], &to[..], "idct8_1d_odd tail_n={tail_n} mismatch");

        let mut tde = vec![[0i32; 8]; tail_n];
        zcc_dct8_1d_even::dct8_1d_even_batch(&xbuf[..tail_n], &mut tde);
        assert_eq!(&de_ref[..tail_n], &tde[..], "dct8_1d_even tail_n={tail_n} mismatch");

        let mut tdo = vec![[0i32; 8]; tail_n];
        zcc_dct8_1d_odd::dct8_1d_odd_batch(&xbuf[..tail_n], &mut tdo);
        assert_eq!(&do_ref[..tail_n], &tdo[..], "dct8_1d_odd tail_n={tail_n} mismatch");

        let mut tc = vec![[0i32; 8]; tail_n];
        zcc_idct8_1d_combine::idct8_1d_combine_batch(&ceven[..tail_n], &codd[..tail_n], &mut tc);
        assert_eq!(&comb_ref[..tail_n], &tc[..], "idct8_1d_combine tail_n={tail_n} mismatch");

        let mut tr0 = vec![[0u32; 8]; tail_n];
        let mut tr1 = vec![[0i64; 8]; tail_n];
        zcc_rans_refill_branchless::rans_refill_branchless_batch(
            &rnext[..tail_n],
            &rword[..tail_n],
            &mut tr0,
            &mut tr1,
        );
        assert_eq!(&r0_ref[..tail_n], &tr0[..], "rans_refill_branchless out0 tail_n={tail_n} mismatch");
        assert_eq!(&r1_ref[..tail_n], &tr1[..], "rans_refill_branchless out1 tail_n={tail_n} mismatch");
    }

    // Live hardware timing (512 blocks x 256 iters = 131,072 calls/trial, median of 9)
    let raw_b = &raw[..BENCH_BLOCKS];
    let x_b = &xbuf[..BENCH_BLOCKS];
    let ce_b = &ceven[..BENCH_BLOCKS];
    let co_b = &codd[..BENCH_BLOCKS];
    let rn_b = &rnext[..BENCH_BLOCKS];
    let rw_b = &rword[..BENCH_BLOCKS];

    bench_pair(
        "idct8_1d_even",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_idct8_1d_even_batch(black_box(raw_b), black_box(&mut even_ref[..BENCH_BLOCKS]));
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_idct8_1d_even::idct8_1d_even_batch(black_box(raw_b), black_box(&mut even_zcc[..BENCH_BLOCKS]));
            }
        },
    );
    bench_pair(
        "idct8_1d_odd",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_idct8_1d_odd_batch(black_box(raw_b), black_box(&mut odd_ref[..BENCH_BLOCKS]));
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_idct8_1d_odd::idct8_1d_odd_batch(black_box(raw_b), black_box(&mut odd_zcc[..BENCH_BLOCKS]));
            }
        },
    );
    bench_pair(
        "dct8_1d_even",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_dct8_1d_even_batch(black_box(x_b), black_box(&mut de_ref[..BENCH_BLOCKS]));
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_dct8_1d_even::dct8_1d_even_batch(black_box(x_b), black_box(&mut de_zcc[..BENCH_BLOCKS]));
            }
        },
    );
    bench_pair(
        "dct8_1d_odd",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_dct8_1d_odd_batch(black_box(x_b), black_box(&mut do_ref[..BENCH_BLOCKS]));
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_dct8_1d_odd::dct8_1d_odd_batch(black_box(x_b), black_box(&mut do_zcc[..BENCH_BLOCKS]));
            }
        },
    );
    bench_pair(
        "idct8_1d_combine",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_idct8_1d_combine_batch(black_box(ce_b), black_box(co_b), black_box(&mut comb_ref[..BENCH_BLOCKS]));
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_idct8_1d_combine::idct8_1d_combine_batch(black_box(ce_b), black_box(co_b), black_box(&mut comb_zcc[..BENCH_BLOCKS]));
            }
        },
    );
    bench_pair(
        "rans_refill_branchless",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_rans_refill_branchless_batch(
                    black_box(rn_b),
                    black_box(rw_b),
                    black_box(&mut r0_ref[..BENCH_BLOCKS]),
                    black_box(&mut r1_ref[..BENCH_BLOCKS]),
                );
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_rans_refill_branchless::rans_refill_branchless_batch(
                    black_box(rn_b),
                    black_box(rw_b),
                    black_box(&mut r0_zcc[..BENCH_BLOCKS]),
                    black_box(&mut r1_zcc[..BENCH_BLOCKS]),
                );
            }
        },
    );
}
"#,
    );
    src
}

#[test]
fn zero_rans_six_kernels_bit_exact_et_exec_objdump_and_live_bench() {
    let rans_src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../zero-rans/src");
    assert!(
        rans_src_dir.is_dir(),
        "sibling zero-rans/src directory missing at {}",
        rans_src_dir.display()
    );

    let mut doms = BTreeMap::new();
    doms.insert("raw".to_string(), Domain::range(-262_144, 262_144));
    doms.insert("x".to_string(), Domain::range(-32_768, 32_767));
    doms.insert("even".to_string(), Domain::range(-1_000_000_000, 1_000_000_000));
    doms.insert("odd".to_string(), Domain::range(-1_000_000_000, 1_000_000_000));
    doms.insert("next".to_string(), Domain::range(0, i64::from(u32::MAX)));
    let mut overrides = BTreeMap::new();
    overrides.insert("*".to_string(), doms);

    let residual_src = fs::read_to_string(rans_src_dir.join("residual.rs")).expect("read residual.rs");
    const RANS_REFILL_SRC: &str = r#"
    pub const fn rans_refill_branchless(next: u32, word: u16) -> (u32, usize) {
        let need = if next < 32_768 { 1u32 } else { 0u32 };
        let mask = 0u32.wrapping_sub(need);
        let refill = (next << 16) | (word as u32);
        let state = next ^ ((refill ^ next) & mask);
        (state, (need as usize) << 1)
    }
    "#;
    let mut extracted = BTreeMap::new();
    for k in parse_rust_file_kernels(&residual_src, &overrides) {
        extracted.insert(k.name.clone(), k);
    }
    for k in parse_rust_file_kernels(RANS_REFILL_SRC, &overrides) {
        extracted.insert(k.name.clone(), k);
    }

    let targets = [
        "idct8_1d_even",
        "idct8_1d_odd",
        "dct8_1d_even",
        "dct8_1d_odd",
        "idct8_1d_combine",
        "rans_refill_branchless",
    ];
    let dir = temp_dir("zero_rans_bench");
    let mut obj_paths: Vec<PathBuf> = Vec::new();
    let mut shims: BTreeMap<String, String> = BTreeMap::new();

    for (idx, &name) in targets.iter().enumerate() {
        let kernel = extracted
            .get(name)
            .unwrap_or_else(|| panic!("expected kernel '{name}' from zero-rans"));
        let orig_cost = kernel.cost();
        let opt = optimize_kernel(kernel).expect("optimize_kernel");
        let opt_cost = opt.cost();
        assert_eq!(opt_cost.divs_rems, 0, "{name} must have 0 remaining divs/rems");

        let budget = kernel_reg_budget(kernel, RegBudget::default());
        let obj = compile_kernel_direct(kernel, budget, TargetIsa::X86_64).expect("compile ET_REL");
        verify_elf64_object(&obj.elf_bytes).expect("verify ET_REL");
        assert!(obj.zero_spills, "{name} ET_REL must have 0 spills");

        let obj_path = dir.join(format!("{name}.o"));
        fs::write(&obj_path, &obj.elf_bytes).expect("write .o");
        let (rel_insns, _, rel_ymm) = audit_objdump_x86(&obj_path, None);
        assert!(rel_insns > 0, "{name} ET_REL must disassemble non-empty");
        assert!(rel_ymm > 0, "{name} ET_REL must emit packed AVX2 ymm instructions");
        obj_paths.push(obj_path);
        shims.insert(name.to_string(), obj.rust_ffi_shim.clone());

        // Standalone 100% direct Linux ET_EXEC with 512 real non-zero blocks (corners + random)
        const EXEC_BLOCKS: usize = 512;
        let exec_inputs = build_exec_inputs(kernel, EXEC_BLOCKS, 0xA5A5_1234_5678_0001u64 + idx as u64);
        let input_refs: Vec<&[i64]> = exec_inputs.iter().map(Vec::as_slice).collect();
        let exec = compile_standalone_elf_exec(kernel, budget, TargetIsa::X86_64, &input_refs)
            .expect("compile ET_EXEC");
        let esum = verify_elf64_executable(&exec.elf_bytes).expect("verify ET_EXEC");
        assert_eq!(exec.direct_share_pct, 100, "{name} ET_EXEC must be 100% direct");
        assert_eq!(exec.shim_glue_insns, 0, "{name} ET_EXEC must have 0 shim glue");
        assert!(exec.zero_spills, "{name} ET_EXEC must have 0 spills");
        assert_eq!(exec.blocks, EXEC_BLOCKS);

        let exec_path = dir.join(format!("{name}_exec"));
        fs::write(&exec_path, &exec.elf_bytes).expect("write ET_EXEC");
        set_exec_mode(&exec_path);

        let want_syms = [exec.kernel_sym.as_str(), exec.batch_sym.as_str(), exec.entry_sym.as_str()];
        let (exec_insns, sym_cnt, exec_ymm) = audit_objdump_x86(&exec_path, Some(&want_syms));
        assert_eq!(sym_cnt, 3, "{name} ET_EXEC must contain only 3 direct zero-cc symbols");
        assert!(exec_ymm > 0, "{name} ET_EXEC must emit packed AVX2 ymm instructions");

        let expected_stdout = eval_expected_stdout(kernel, &exec_inputs, EXEC_BLOCKS);
        let expected_exit = i32::from(expected_stdout.iter().fold(0u8, |a, &b| a.wrapping_add(b)) & 0x7F);
        let run_out = Command::new(&exec_path).output().expect("execute standalone ET_EXEC");
        assert_eq!(
            run_out.stdout, expected_stdout,
            "{name} standalone ET_EXEC stdout bytes mismatch Kernel::eval"
        );
        assert_eq!(
            run_out.status.code(),
            Some(expected_exit),
            "{name} standalone ET_EXEC exit code mismatch"
        );

        println!(
            "KERNEL|{name}|ops={}->{}|divs={}->{}|rel_insns={rel_insns}|rel_ymm={rel_ymm}|exec_insns={exec_insns}|exec_ymm={exec_ymm}|rel_B={}|exec_B={}|entry=0x{:x}|stdout_B={}|exit={expected_exit}",
            orig_cost.total_nodes,
            opt_cost.total_nodes,
            orig_cost.divs_rems,
            opt_cost.divs_rems,
            obj.elf_bytes.len(),
            exec.elf_bytes.len(),
            esum.entry_vaddr,
            expected_stdout.len()
        );
    }

    // Build and execute native rustc -O3 vs zero-cc Direct ELF64 parity + timing harness
    let harness_rs = dir.join("bench_harness.rs");
    let harness_bin = dir.join("bench_harness");
    fs::write(&harness_rs, generate_bench_harness(&shims)).expect("write harness");

    let mut cmd = Command::new("rustc");
    cmd.env_remove("RUSTFLAGS")
        .args(["-C", "opt-level=3", "--edition", "2021", "-o"])
        .arg(&harness_bin)
        .arg(&harness_rs);
    for obj in &obj_paths {
        cmd.arg("-C").arg(format!("link-arg={}", obj.display()));
    }
    let build = cmd.output().expect("invoke rustc");
    assert!(
        build.status.success(),
        "rustc bench harness build failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&harness_bin).output().expect("run bench harness");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "bench harness failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    println!("{stdout}");

    let bench_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("BENCH|")).collect();
    assert_eq!(bench_lines.len(), targets.len(), "expected 6 BENCH lines");
    for line in &bench_lines {
        let sp_str = line
            .split('|')
            .nth(4)
            .unwrap_or_else(|| panic!("missing speedup in {line}"));
        let sp: f64 = sp_str.parse().unwrap_or_else(|_| panic!("bad speedup in {line}"));
        assert!(sp > 1.05, "expected >1.05x AVX2 speedup over rustc -O3, got {sp:.2}x in {line}");
    }

    let _ = fs::remove_dir_all(&dir);
}

fn pack_buf_bytes(ty: ElemTy, vals: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * usize::from(ty.bytes()));
    for &v in vals {
        ty.push_le_elem(&mut out, v);
    }
    out
}

fn eval_expected_outputs(kernel: &Kernel, inputs: &[Vec<i64>], blocks: usize) -> Vec<(usize, Vec<u8>)> {
    let mut mut_outs: Vec<(usize, Vec<u8>)> = kernel
        .bufs
        .iter()
        .enumerate()
        .filter(|(_, b)| b.access == BufAccess::MutNoAlias)
        .map(|(idx, b)| (idx, Vec::with_capacity(blocks * b.block_bytes())))
        .collect();
    let mut block_slices: Vec<&[i64]> = Vec::with_capacity(kernel.bufs.len());
    for blk in 0..blocks {
        block_slices.clear();
        for (idx, decl) in kernel.bufs.iter().enumerate() {
            let lanes = usize::from(decl.lanes);
            block_slices.push(&inputs[idx][blk * lanes..(blk + 1) * lanes]);
        }
        let res = kernel.eval(&block_slices).expect("Kernel::eval");
        for (bidx, bytes) in &mut mut_outs {
            let decl = &kernel.bufs[*bidx];
            for &v in res[*bidx].iter().take(usize::from(decl.lanes)) {
                decl.elem_ty.push_le_elem(bytes, v);
            }
        }
    }
    mut_outs
}

fn run_node_wasm_socket_case(
    dir: &Path,
    tag: &str,
    cw: &CompiledWasm,
    kernel: &Kernel,
    inputs: &[Vec<i64>],
    block_counts: &[usize],
    bench_blocks: usize,
) -> f64 {
    let wasm_path = dir.join(format!("{tag}.wasm"));
    fs::write(&wasm_path, &cw.wasm_bytes).expect("write wasm");
    let mut buf_meta = Vec::with_capacity(kernel.bufs.len());
    for (idx, decl) in kernel.bufs.iter().enumerate() {
        let p = dir.join(format!("{tag}_buf{idx}.bin"));
        fs::write(&p, pack_buf_bytes(decl.elem_ty, &inputs[idx])).expect("write wasm buf");
        buf_meta.push(format!(
            "{{\"idx\":{},\"blockBytes\":{},\"isMut\":{},\"path\":{:?}}}",
            idx,
            decl.block_bytes(),
            decl.access == BufAccess::MutNoAlias,
            p.to_string_lossy()
        ));
    }
    let mut case_checks = Vec::new();
    for &n_blocks in block_counts {
        let expected = eval_expected_outputs(kernel, inputs, n_blocks);
        let mut exp_entries = Vec::new();
        for (bidx, exp_bytes) in &expected {
            let ep = dir.join(format!("{tag}_exp_{n_blocks}_{bidx}.bin"));
            fs::write(&ep, exp_bytes).expect("write expected wasm out");
            exp_entries.push(format!("{{\"idx\":{bidx},\"path\":{:?}}}", ep.to_string_lossy()));
        }
        case_checks.push(format!("{{\"blocks\":{n_blocks},\"expected\":[{}]}}", exp_entries.join(",")));
    }

    let script = format!(
        r#"const fs = require('fs');
const wasmBytes = fs.readFileSync({wasm_path:?});
if (!WebAssembly.validate(wasmBytes)) {{
  console.error('WebAssembly.validate FAILED for {tag}');
  process.exit(2);
}}
const bufs = [{buf_meta}];
const cases = [{cases}];
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(wasmBytes, {{}});
  const mem = instance.exports.memory;
  if (!(mem instanceof WebAssembly.Memory)) {{
    throw new Error('missing exported memory');
  }}
  const singleFn = instance.exports[{kernel_sym:?}];
  const batchFn = instance.exports[{batch_sym:?}];
  const rawBufs = bufs.map(b => fs.readFileSync(b.path));
  let totalBytes = 64;
  const ptrs = [];
  for (let i = 0; i < bufs.length; i++) {{
    totalBytes = (totalBytes + 15) & ~15;
    ptrs.push(totalBytes);
    totalBytes += rawBufs[i].length + 64;
  }}
  const needPages = Math.ceil(totalBytes / 65536);
  const curPages = mem.buffer.byteLength / 65536;
  if (needPages > curPages) mem.grow(needPages - curPages);
  function loadInputs(nBlocks) {{
    const view = new Uint8Array(mem.buffer);
    for (let i = 0; i < bufs.length; i++) {{
      const len = nBlocks * bufs[i].blockBytes;
      if (bufs[i].isMut) view.fill(0xAA, ptrs[i], ptrs[i] + len);
      view.set(rawBufs[i].subarray(0, len), ptrs[i]);
    }}
  }}
  function checkOutputs(expected, label) {{
    const view = new Uint8Array(mem.buffer);
    for (const ex of expected) {{
      const want = fs.readFileSync(ex.path);
      const got = Buffer.from(view.subarray(ptrs[ex.idx], ptrs[ex.idx] + want.length));
      if (!got.equals(want)) {{
        throw new Error(`MISMATCH {tag} ${{label}} buf=${{ex.idx}}`);
      }}
    }}
  }}
  for (const c of cases) {{
    loadInputs(c.blocks);
    batchFn(...ptrs, c.blocks);
    checkOutputs(c.expected, `batch(${{c.blocks}})`);
    if (c.blocks === 1) {{
      loadInputs(1);
      singleFn(...ptrs);
      checkOutputs(c.expected, 'single(1)');
    }}
  }}
  loadInputs({bench_blocks});
  for (let w = 0; w < 8; w++) batchFn(...ptrs, {bench_blocks});
  const iters = 64;
  const t0 = process.hrtime.bigint();
  for (let r = 0; r < iters; r++) batchFn(...ptrs, {bench_blocks});
  const dtNs = Number(process.hrtime.bigint() - t0);
  console.log('NS_PER_CALL=' + (dtNs / (iters * {bench_blocks})).toFixed(3));
}})().catch(err => {{
  console.error(err);
  process.exit(1);
}});
"#,
        wasm_path = wasm_path.to_string_lossy(),
        tag = tag,
        buf_meta = buf_meta.join(","),
        cases = case_checks.join(","),
        kernel_sym = cw.kernel_sym.as_str(),
        batch_sym = cw.batch_sym.as_str(),
        bench_blocks = bench_blocks,
    );

    let js_path = dir.join(format!("{tag}.js"));
    fs::write(&js_path, &script).expect("write node script");
    let out = Command::new("node").arg(&js_path).output().expect("run node");
    assert!(
        out.status.success(),
        "node WASM parity failed for {tag}:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("NS_PER_CALL=").and_then(|v| v.parse::<f64>().ok()))
        .expect("missing NS_PER_CALL from node WASM run")
}

fn generate_socket_bench_harness(shims: &BTreeMap<String, String>) -> String {
    let mut src = String::with_capacity(12_288);
    src.push_str(
        r#"use std::hint::black_box;
use std::time::Instant;

#[inline]
fn ref_zcc_paeth_pred8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
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

#[inline]
fn ref_zcc_grad_socket8(w: &[i32; 8], n: &[i32; 8], nw: &[i32; 8], out: &mut [i32; 8]) {
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

#[inline]
fn ref_zcc_sub_mod256_8(cur: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        out[k] = (cur[k] - ref_b[k]) & 255;
    }
}

#[inline]
fn ref_zcc_add_mod256_8(res: &[i32; 8], ref_b: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        out[k] = (res[k] + ref_b[k]) & 255;
    }
}

#[inline]
fn ref_zcc_zigzag_u8_8(err: &[i32; 8], out: &mut [i32; 8]) {
    for k in 0..8usize {
        out[k] = ((err[k] << 1) ^ (err[k] >> 7)) & 255;
    }
}

#[inline(never)]
fn ref_paeth_batch(w: &[[i32; 8]], n: &[[i32; 8]], nw: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for b in 0..out.len() { ref_zcc_paeth_pred8(&w[b], &n[b], &nw[b], &mut out[b]); }
}
#[inline(never)]
fn ref_grad_batch(w: &[[i32; 8]], n: &[[i32; 8]], nw: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for b in 0..out.len() { ref_zcc_grad_socket8(&w[b], &n[b], &nw[b], &mut out[b]); }
}
#[inline(never)]
fn ref_sub_batch(a: &[[i32; 8]], b: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for i in 0..out.len() { ref_zcc_sub_mod256_8(&a[i], &b[i], &mut out[i]); }
}
#[inline(never)]
fn ref_add_batch(a: &[[i32; 8]], b: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for i in 0..out.len() { ref_zcc_add_mod256_8(&a[i], &b[i], &mut out[i]); }
}
#[inline(never)]
fn ref_zz_batch(a: &[[i32; 8]], out: &mut [[i32; 8]]) {
    for i in 0..out.len() { ref_zcc_zigzag_u8_8(&a[i], &mut out[i]); }
}
"#,
    );

    for (name, shim) in shims {
        src.push_str(&format!("\nmod {name}_mod {{\n{shim}}}\n"));
    }

    src.push_str(
        r#"
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        (i64::from(lo) + (self.next() % span) as i64) as i32
    }
}

fn bench_pair<F1: FnMut(), F2: FnMut()>(name: &str, calls: usize, mut f_ref: F1, mut f_zcc: F2) {
    for _ in 0..3 { f_ref(); f_zcc(); }
    let mut r_ns = [0.0f64; 9];
    let mut z_ns = [0.0f64; 9];
    for t in 0..9 {
        if t % 2 == 0 {
            let t0 = Instant::now();
            f_ref();
            r_ns[t] = t0.elapsed().as_nanos() as f64 / calls as f64;
            let t1 = Instant::now();
            f_zcc();
            z_ns[t] = t1.elapsed().as_nanos() as f64 / calls as f64;
        } else {
            let t1 = Instant::now();
            f_zcc();
            z_ns[t] = t1.elapsed().as_nanos() as f64 / calls as f64;
            let t0 = Instant::now();
            f_ref();
            r_ns[t] = t0.elapsed().as_nanos() as f64 / calls as f64;
        }
    }
    r_ns.sort_by(|a, b| a.total_cmp(b));
    z_ns.sort_by(|a, b| a.total_cmp(b));
    let rm = r_ns[4];
    let zm = z_ns[4];
    println!("SOCKET_BENCH|{name}|{rm:.3}|{zm:.3}|{:.3}", rm / zm);
}

fn main() {
    const PARITY_BLOCKS: usize = 4096;
    const BENCH_BLOCKS: usize = 512;
    const ITERS: usize = 256;
    const CALLS: usize = BENCH_BLOCKS * ITERS;
    let mut rng = Rng(0x517C_C1B7_2722_0A95);
    let u8_corners: [i32; 11] = [0, 1, 2, 7, 8, 24, 25, 127, 128, 254, 255];
    let i8_corners: [i32; 9] = [-128, -127, -2, -1, 0, 1, 2, 126, 127];

    let mut w = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut n = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut nw = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut err = vec![[0i32; 8]; PARITY_BLOCKS];
    for b in 0..PARITY_BLOCKS {
        for l in 0..8 {
            if b < u8_corners.len() {
                w[b][l] = u8_corners[(b + l) % u8_corners.len()];
                n[b][l] = u8_corners[(b + l * 3 + 1) % u8_corners.len()];
                nw[b][l] = u8_corners[(b + l * 5 + 2) % u8_corners.len()];
            } else {
                w[b][l] = rng.range_i32(0, 255);
                n[b][l] = rng.range_i32(0, 255);
                nw[b][l] = rng.range_i32(0, 255);
            }
            err[b][l] = if b < i8_corners.len() {
                i8_corners[(b + l) % i8_corners.len()]
            } else {
                rng.range_i32(-128, 127)
            };
        }
    }

    let mut out_ref = vec![[0i32; 8]; PARITY_BLOCKS];
    let mut out_zcc = vec![[0i32; 8]; PARITY_BLOCKS];

    for &len in &[1usize, 3, 7, PARITY_BLOCKS] {
        ref_paeth_batch(&w[..len], &n[..len], &nw[..len], &mut out_ref[..len]);
        zcc_paeth_pred8_mod::zcc_paeth_pred8_batch(&w[..len], &n[..len], &nw[..len], &mut out_zcc[..len]);
        assert_eq!(&out_ref[..len], &out_zcc[..len], "zcc_paeth_pred8 mismatch len={len}");

        ref_grad_batch(&w[..len], &n[..len], &nw[..len], &mut out_ref[..len]);
        zcc_grad_socket8_mod::zcc_grad_socket8_batch(&w[..len], &n[..len], &nw[..len], &mut out_zcc[..len]);
        assert_eq!(&out_ref[..len], &out_zcc[..len], "zcc_grad_socket8 mismatch len={len}");

        ref_sub_batch(&w[..len], &n[..len], &mut out_ref[..len]);
        zcc_sub_mod256_8_mod::zcc_sub_mod256_8_batch(&w[..len], &n[..len], &mut out_zcc[..len]);
        assert_eq!(&out_ref[..len], &out_zcc[..len], "zcc_sub_mod256_8 mismatch len={len}");

        ref_add_batch(&w[..len], &n[..len], &mut out_ref[..len]);
        zcc_add_mod256_8_mod::zcc_add_mod256_8_batch(&w[..len], &n[..len], &mut out_zcc[..len]);
        assert_eq!(&out_ref[..len], &out_zcc[..len], "zcc_add_mod256_8 mismatch len={len}");

        ref_zz_batch(&err[..len], &mut out_ref[..len]);
        zcc_zigzag_u8_8_mod::zcc_zigzag_u8_8_batch(&err[..len], &mut out_zcc[..len]);
        assert_eq!(&out_ref[..len], &out_zcc[..len], "zcc_zigzag_u8_8 mismatch len={len}");
    }

    // Counterfactual perturbation knockout: perturbing lane 0 must change output
    let mut w_alt = vec![[100i32; 8]; 1];
    let n_alt = vec![[100i32; 8]; 1];
    let nw_alt = vec![[100i32; 8]; 1];
    let mut k0 = vec![[0i32; 8]; 1];
    let mut k1 = vec![[0i32; 8]; 1];
    zcc_paeth_pred8_mod::zcc_paeth_pred8_batch(&w_alt, &n_alt, &nw_alt, &mut k0);
    w_alt[0][0] = 180;
    zcc_paeth_pred8_mod::zcc_paeth_pred8_batch(&w_alt, &n_alt, &nw_alt, &mut k1);
    assert_ne!(k0[0][0], k1[0][0], "paeth perturbation knockout");

    let (wb, nb, nwb, eb) = (&w[..BENCH_BLOCKS], &n[..BENCH_BLOCKS], &nw[..BENCH_BLOCKS], &err[..BENCH_BLOCKS]);
    bench_pair("zcc_paeth_pred8", CALLS,
        || { for _ in 0..ITERS { ref_paeth_batch(black_box(wb), black_box(nb), black_box(nwb), black_box(&mut out_ref[..BENCH_BLOCKS])); } },
        || { for _ in 0..ITERS { zcc_paeth_pred8_mod::zcc_paeth_pred8_batch(black_box(wb), black_box(nb), black_box(nwb), black_box(&mut out_zcc[..BENCH_BLOCKS])); } },
    );
    bench_pair("zcc_grad_socket8", CALLS,
        || { for _ in 0..ITERS { ref_grad_batch(black_box(wb), black_box(nb), black_box(nwb), black_box(&mut out_ref[..BENCH_BLOCKS])); } },
        || { for _ in 0..ITERS { zcc_grad_socket8_mod::zcc_grad_socket8_batch(black_box(wb), black_box(nb), black_box(nwb), black_box(&mut out_zcc[..BENCH_BLOCKS])); } },
    );
    bench_pair("zcc_sub_mod256_8", CALLS,
        || { for _ in 0..ITERS { ref_sub_batch(black_box(wb), black_box(nb), black_box(&mut out_ref[..BENCH_BLOCKS])); } },
        || { for _ in 0..ITERS { zcc_sub_mod256_8_mod::zcc_sub_mod256_8_batch(black_box(wb), black_box(nb), black_box(&mut out_zcc[..BENCH_BLOCKS])); } },
    );
    bench_pair("zcc_add_mod256_8", CALLS,
        || { for _ in 0..ITERS { ref_add_batch(black_box(wb), black_box(nb), black_box(&mut out_ref[..BENCH_BLOCKS])); } },
        || { for _ in 0..ITERS { zcc_add_mod256_8_mod::zcc_add_mod256_8_batch(black_box(wb), black_box(nb), black_box(&mut out_zcc[..BENCH_BLOCKS])); } },
    );
    bench_pair("zcc_zigzag_u8_8", CALLS,
        || { for _ in 0..ITERS { ref_zz_batch(black_box(eb), black_box(&mut out_ref[..BENCH_BLOCKS])); } },
        || { for _ in 0..ITERS { zcc_zigzag_u8_8_mod::zcc_zigzag_u8_8_batch(black_box(eb), black_box(&mut out_zcc[..BENCH_BLOCKS])); } },
    );
}
"#,
    );
    src
}

#[test]
fn zero_rans_socket_five_kernels_elf64_wasm32_parity_and_bench() {
    let sock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../zero-rans/src/socket_kernels.rs");
    let sock_src = fs::read_to_string(&sock_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", sock_path.display()));
    let overrides: BTreeMap<String, BTreeMap<String, Domain>> = BTreeMap::new();
    let mut extracted = BTreeMap::new();
    for k in parse_rust_file_kernels(&sock_src, &overrides) {
        extracted.insert(k.name.clone(), k);
    }

    let targets = [
        "zcc_paeth_pred8",
        "zcc_grad_socket8",
        "zcc_sub_mod256_8",
        "zcc_add_mod256_8",
        "zcc_zigzag_u8_8",
    ];
    let dir = temp_dir("zero_rans_socket_kernels");
    let mut obj_paths: Vec<PathBuf> = Vec::new();
    let mut shims: BTreeMap<String, String> = BTreeMap::new();

    for (idx, &name) in targets.iter().enumerate() {
        let kernel = extracted
            .get(name)
            .unwrap_or_else(|| panic!("missing socket kernel '{name}'"));
        let orig_cost = kernel.cost();
        let opt = optimize_kernel(kernel).expect("optimize_kernel");
        let opt_cost = opt.cost();
        assert_eq!(opt_cost.divs_rems, 0, "{name} must have 0 remaining divs/rems");

        let budget = kernel_reg_budget(kernel, RegBudget::default());
        let obj = compile_kernel_direct(kernel, budget, TargetIsa::X86_64).expect("compile ET_REL");
        verify_elf64_object(&obj.elf_bytes).expect("verify ET_REL");
        assert!(obj.zero_spills, "{name} ET_REL must have 0 spills");

        let obj_path = dir.join(format!("{name}.o"));
        fs::write(&obj_path, &obj.elf_bytes).expect("write .o");
        let (rel_insns, _, rel_ymm) = audit_objdump_x86(&obj_path, None);
        assert!(rel_insns > 0 && rel_ymm > 0, "{name} ET_REL must emit packed AVX2 ymm insns");
        obj_paths.push(obj_path);
        shims.insert(name.to_string(), obj.rust_ffi_shim.clone());

        const EXEC_BLOCKS: usize = 512;
        let exec_inputs = build_exec_inputs(kernel, EXEC_BLOCKS, 0xD1B5_4A32_D192_ED03u64 + idx as u64);
        let input_refs: Vec<&[i64]> = exec_inputs.iter().map(Vec::as_slice).collect();
        let exec = compile_standalone_elf_exec(kernel, budget, TargetIsa::X86_64, &input_refs)
            .expect("compile ET_EXEC");
        let esum = verify_elf64_executable(&exec.elf_bytes).expect("verify ET_EXEC");
        assert_eq!(exec.direct_share_pct, 100, "{name} ET_EXEC must be 100% direct");
        assert_eq!(exec.shim_glue_insns, 0, "{name} ET_EXEC must have 0 shim glue");
        assert!(exec.zero_spills, "{name} ET_EXEC must have 0 spills");

        let exec_path = dir.join(format!("{name}_exec"));
        fs::write(&exec_path, &exec.elf_bytes).expect("write ET_EXEC");
        set_exec_mode(&exec_path);
        let want_syms = [exec.kernel_sym.as_str(), exec.batch_sym.as_str(), exec.entry_sym.as_str()];
        let (exec_insns, sym_cnt, exec_ymm) = audit_objdump_x86(&exec_path, Some(&want_syms));
        assert_eq!(sym_cnt, 3, "{name} ET_EXEC must expose exactly 3 direct symbols");
        assert!(exec_ymm > 0, "{name} ET_EXEC must emit packed AVX2 ymm insns");

        let expected_stdout = eval_expected_stdout(kernel, &exec_inputs, EXEC_BLOCKS);
        let expected_exit = i32::from(expected_stdout.iter().fold(0u8, |a, &b| a.wrapping_add(b)) & 0x7F);
        let run_out = Command::new(&exec_path).output().expect("execute standalone ET_EXEC");
        assert_eq!(run_out.stdout, expected_stdout, "{name} ET_EXEC stdout mismatch");
        assert_eq!(run_out.status.code(), Some(expected_exit), "{name} ET_EXEC exit mismatch");

        // Standalone 0-import WASM32 + SIMD128 verification & live Node v22 parity
        let cw = compile_kernel_wasm(kernel, budget).expect("compile WASM32");
        let wsum = verify_wasm_module(&cw.wasm_bytes).expect("verify WASM32");
        assert!(!wsum.imports_memory, "{name} WASM32 must have 0 imports");
        assert_eq!(wsum.div_rem_insns, 0, "{name} WASM32 must have 0 div/rem opcodes");
        assert!(wsum.simd_insns > 0, "{name} WASM32 must emit SIMD128 instructions");
        let node_ns = run_node_wasm_socket_case(
            &dir,
            name,
            &cw,
            kernel,
            &exec_inputs,
            &[1, 3, 7, EXEC_BLOCKS],
            EXEC_BLOCKS,
        );

        println!(
            "SOCKET_KERNEL|{name}|ops={}->{}|divs={}->{}|rel_insns={rel_insns}|rel_ymm={rel_ymm}|exec_insns={exec_insns}|exec_ymm={exec_ymm}|rel_B={}|exec_B={}|entry=0x{:x}",
            orig_cost.total_nodes,
            opt_cost.total_nodes,
            orig_cost.divs_rems,
            opt_cost.divs_rems,
            obj.elf_bytes.len(),
            exec.elf_bytes.len(),
            esum.entry_vaddr
        );
        println!(
            "SOCKET_WASM|{name}|mode={:?}|wasm_B={}|insns={}|simd={}|wasm_divs={}|node_ns={node_ns:.3}",
            cw.vector_mode,
            cw.wasm_bytes.len(),
            wsum.total_insns,
            wsum.simd_insns,
            wsum.div_rem_insns,
        );
    }

    let harness_rs = dir.join("socket_bench_harness.rs");
    let harness_bin = dir.join("socket_bench_harness");
    fs::write(&harness_rs, generate_socket_bench_harness(&shims)).expect("write socket harness");

    let mut cmd = Command::new("rustc");
    cmd.env_remove("RUSTFLAGS")
        .args(["-C", "opt-level=3", "--edition", "2021", "-o"])
        .arg(&harness_bin)
        .arg(&harness_rs);
    for obj in &obj_paths {
        cmd.arg("-C").arg(format!("link-arg={}", obj.display()));
    }
    let build = cmd.output().expect("invoke rustc");
    assert!(
        build.status.success(),
        "rustc socket bench harness failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&harness_bin).output().expect("run socket bench harness");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "socket bench harness failed:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    println!("{stdout}");
    let bench_lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("SOCKET_BENCH|")).collect();
    assert_eq!(bench_lines.len(), targets.len(), "expected 5 SOCKET_BENCH lines");

    let _ = fs::remove_dir_all(&dir);
}
