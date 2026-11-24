#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use zero_cc::rust_fe::{kernel_reg_budget, parse_rust_file_kernels};
use zero_cc::{
    compile_kernel_direct, compile_standalone_elf_exec, optimize_kernel, verify_elf64_executable,
    verify_elf64_object, BufAccess, Domain, Kernel, RegBudget, TargetIsa,
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

fn audit_objdump_x86(path: &Path, expected_syms: Option<&[&str]>) -> (usize, usize) {
    let out = Command::new("objdump")
        .args(["-d", "-M", "intel"])
        .arg(path)
        .output()
        .expect("run objdump");
    assert!(out.status.success(), "objdump failed on {}", path.display());
    let text = String::from_utf8_lossy(&out.stdout);
    let mut insns = 0usize;
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
    (insns, syms_seen.len())
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
        "next" => &[0, 1, 32_767, 32_768, 32_769, 65_535, 65_536, 2_147_483_647],
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
const fn ref_rans_renorm_branchless(next: u32, word: u16) -> (u32, usize) {
    let need = if next < 32_768 { 1u32 } else { 0u32 };
    let mask = 0u32.wrapping_sub(need);
    let renorm = (next << 16) | (word as u32);
    let state = next ^ ((renorm ^ next) & mask);
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
fn ref_rans_renorm_branchless_batch(
    next: &[[u32; 8]],
    word: &[[u16; 8]],
    out0: &mut [[u32; 8]],
    out1: &mut [[i64; 8]],
) {
    let n = next.len().min(word.len()).min(out0.len()).min(out1.len());
    for b in 0..n {
        for l in 0..8 {
            let (st, adv) = ref_rans_renorm_branchless(next[b][l], word[b][l]);
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

    // 4. rans_renorm_branchless (next in 0..=2147483647, word in 0..=65535)
    let next_corners: [u32; 8] = [0, 1, 32_767, 32_768, 32_769, 65_535, 65_536, 2_147_483_647];
    let word_corners: [u16; 8] = [0, 1, 255, 256, 32_767, 32_768, 0x55AA, 65_535];
    let mut rnext = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut rword = vec![[0u16; 8]; PARITY_BLOCKS];
    for b in 0..PARITY_BLOCKS {
        for l in 0..8 {
            if b < next_corners.len() {
                rnext[b][l] = next_corners[(b + l) % next_corners.len()];
                rword[b][l] = word_corners[(b + l * 3) % word_corners.len()];
            } else {
                rnext[b][l] = rng.range_i64(0, 2_147_483_647) as u32;
                rword[b][l] = rng.range_i64(0, 65_535) as u16;
            }
        }
    }
    let mut r0_ref = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut r1_ref = vec![[0i64; 8]; PARITY_BLOCKS];
    let mut r0_zcc = vec![[0u32; 8]; PARITY_BLOCKS];
    let mut r1_zcc = vec![[0i64; 8]; PARITY_BLOCKS];
    ref_rans_renorm_branchless_batch(&rnext, &rword, &mut r0_ref, &mut r1_ref);
    zcc_rans_renorm_branchless::rans_renorm_branchless_batch(&rnext, &rword, &mut r0_zcc, &mut r1_zcc);
    assert_eq!(r0_ref, r0_zcc, "rans_renorm_branchless out0 4096-block parity mismatch");
    assert_eq!(r1_ref, r1_zcc, "rans_renorm_branchless out1 4096-block parity mismatch");

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
        "rans_renorm_branchless",
        CALLS,
        || {
            for _ in 0..ITERS {
                ref_rans_renorm_branchless_batch(
                    black_box(rn_b),
                    black_box(rw_b),
                    black_box(&mut r0_ref[..BENCH_BLOCKS]),
                    black_box(&mut r1_ref[..BENCH_BLOCKS]),
                );
            }
        },
        || {
            for _ in 0..ITERS {
                zcc_rans_renorm_branchless::rans_renorm_branchless_batch(
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
    let mut overrides = BTreeMap::new();
    overrides.insert("*".to_string(), doms);

    let residual_src = fs::read_to_string(rans_src_dir.join("residual.rs")).expect("read residual.rs");
    let rans8_src = fs::read_to_string(rans_src_dir.join("rans8.rs")).expect("read rans8.rs");
    let mut extracted = BTreeMap::new();
    for k in parse_rust_file_kernels(&residual_src, &overrides) {
        extracted.insert(k.name.clone(), k);
    }
    for k in parse_rust_file_kernels(&rans8_src, &overrides) {
        extracted.insert(k.name.clone(), k);
    }

    let targets = [
        "idct8_1d_even",
        "idct8_1d_odd",
        "dct8_1d_even",
        "dct8_1d_odd",
        "idct8_1d_combine",
        "rans_renorm_branchless",
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
        let (rel_insns, _) = audit_objdump_x86(&obj_path, None);
        assert!(rel_insns > 0, "{name} ET_REL must disassemble non-empty");
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
        let (exec_insns, sym_cnt) = audit_objdump_x86(&exec_path, Some(&want_syms));
        assert_eq!(sym_cnt, 3, "{name} ET_EXEC must contain only 3 direct zero-cc symbols");

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
            "KERNEL|{name}|ops={}->{}|divs={}->{}|rel_insns={rel_insns}|exec_insns={exec_insns}|rel_B={}|exec_B={}|entry=0x{:x}|stdout_B={}|exit={expected_exit}",
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

    let _ = fs::remove_dir_all(&dir);
}
