#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use zero_cc::rust_fe::{kernel_reg_budget, parse_rust_file_kernels};
use zero_cc::{
    compile_kernel_wasm, compile_kernel_wasm_opts, compile_kernels_wasm, compile_rust_to_wasm,
    verify_wasm_module, BufAccess, CmpKind, CompiledWasm, Domain, ElemTy, Kernel, Op, RegBudget,
    WasmVectorMode,
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
    let dir = std::env::temp_dir().join(format!("zero_cc_wasm_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
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
            if matches!(decl.access, BufAccess::LookupTable(_)) {
                block_slices.push(&inputs[idx]);
            } else {
                let lanes = usize::from(decl.lanes);
                block_slices.push(&inputs[idx][blk * lanes..(blk + 1) * lanes]);
            }
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

fn run_node_wasm_case(
    dir: &Path,
    tag: &str,
    cw: &CompiledWasm,
    kernel: &Kernel,
    inputs: &[Vec<i64>],
    block_counts: &[usize],
    bench_blocks: usize,
) -> Option<f64> {
    let wasm_path = dir.join(format!("{tag}.wasm"));
    fs::write(&wasm_path, &cw.wasm_bytes).expect("write wasm");
    let kernel_sym = cw.kernel_sym.as_str();
    let batch_sym = cw.batch_sym.as_str();

    let mut buf_paths = Vec::with_capacity(kernel.bufs.len());
    let mut buf_meta = Vec::with_capacity(kernel.bufs.len());
    for (idx, decl) in kernel.bufs.iter().enumerate() {
        let p = dir.join(format!("{tag}_buf{idx}.bin"));
        let raw = pack_buf_bytes(decl.elem_ty, &inputs[idx]);
        fs::write(&p, &raw).expect("write input buf");
        buf_paths.push(p);
        buf_meta.push(format!(
            "{{\"idx\":{},\"blockBytes\":{},\"isLut\":{},\"isMut\":{},\"path\":{:?}}}",
            idx,
            decl.block_bytes(),
            matches!(decl.access, BufAccess::LookupTable(_)),
            decl.access == BufAccess::MutNoAlias,
            buf_paths[idx].to_string_lossy()
        ));
    }

    let mut case_checks = Vec::new();
    for &n_blocks in block_counts {
        let expected = eval_expected_outputs(kernel, inputs, n_blocks);
        let mut exp_entries = Vec::new();
        for (bidx, exp_bytes) in &expected {
            let ep = dir.join(format!("{tag}_exp_{n_blocks}_{bidx}.bin"));
            fs::write(&ep, exp_bytes).expect("write expected");
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
  if (typeof singleFn !== 'function' || typeof batchFn !== 'function') {{
    throw new Error('missing exported kernel or batch function');
  }}
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
  if (needPages > curPages) {{
    mem.grow(needPages - curPages);
  }}
  function loadInputs(nBlocks) {{
    const view = new Uint8Array(mem.buffer);
    for (let i = 0; i < bufs.length; i++) {{
      const len = bufs[i].isLut ? rawBufs[i].length : nBlocks * bufs[i].blockBytes;
      if (bufs[i].isMut) {{
        view.fill(0xAA, ptrs[i], ptrs[i] + len);
      }}
      view.set(rawBufs[i].subarray(0, len), ptrs[i]);
    }}
  }}
  function checkOutputs(nBlocks, expected, label) {{
    const view = new Uint8Array(mem.buffer);
    for (const ex of expected) {{
      const want = fs.readFileSync(ex.path);
      const got = Buffer.from(view.subarray(ptrs[ex.idx], ptrs[ex.idx] + want.length));
      if (!got.equals(want)) {{
        for (let k = 0; k < want.length; k++) {{
          if (got[k] !== want[k]) {{
            throw new Error(`MISMATCH {tag} ${{label}} buf=${{ex.idx}} byte=${{k}} got=${{got[k]}} want=${{want[k]}}`);
          }}
        }}
        throw new Error(`MISMATCH {tag} ${{label}} len got=${{got.length}} want=${{want.length}}`);
      }}
    }}
  }}
  for (const c of cases) {{
    loadInputs(c.blocks);
    batchFn(...ptrs, c.blocks);
    checkOutputs(c.blocks, c.expected, `batch(${{c.blocks}})`);
    if (c.blocks === 1) {{
      loadInputs(1);
      singleFn(...ptrs);
      checkOutputs(1, c.expected, 'single(1)');
    }}
  }}
  if ({bench_blocks} > 0) {{
    loadInputs({bench_blocks});
    for (let w = 0; w < 8; w++) batchFn(...ptrs, {bench_blocks});
    const iters = 64;
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < iters; r++) batchFn(...ptrs, {bench_blocks});
    const dtNs = Number(process.hrtime.bigint() - t0);
    const nsPerCall = dtNs / (iters * {bench_blocks});
    console.log('NS_PER_CALL=' + nsPerCall.toFixed(3));
  }}
}})().catch(err => {{
  console.error(err);
  process.exit(1);
}});
"#,
        wasm_path = wasm_path.to_string_lossy(),
        tag = tag,
        buf_meta = buf_meta.join(","),
        cases = case_checks.join(","),
        kernel_sym = kernel_sym,
        batch_sym = batch_sym,
        bench_blocks = bench_blocks,
    );

    let js_path = dir.join(format!("{tag}.js"));
    fs::write(&js_path, &script).expect("write node script");
    let out = Command::new("node").arg(&js_path).output().expect("run node v22");
    assert!(
        out.status.success(),
        "node v22 failed for {tag}:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("NS_PER_CALL=").and_then(|v| v.parse::<f64>().ok()))
}

fn build_width_kernel(ty: ElemTy, name: &str) -> Kernel {
    let mut k = Kernel::new(name);
    let dom = ty.natural_domain();
    let a = k.add_typed_buf("a", BufAccess::ReadOnly, ty, 4, dom);
    let b = k.add_typed_buf("b", BufAccess::ReadOnly, ty, 4, dom);
    let out = k.add_typed_buf("out", BufAccess::MutNoAlias, ty, 4, ty.natural_domain());
    let m0 = k.push(Op::MemInit);
    let c63 = k.push(Op::Const(63));
    let c64 = k.push(Op::Const(64));
    let cn64 = k.push(Op::Const(-64));
    let cn65 = k.push(Op::Const(-65));
    let c127 = k.push(Op::Const(127));
    let c128 = k.push(Op::Const(128));
    let cn128 = k.push(Op::Const(-128));
    for lane in 0..4u8 {
        let la = k.push(Op::Load { mem: m0, buf: a, lane });
        let lb = k.push(Op::Load { mem: m0, buf: b, lane });
        let sum = k.push(Op::Add(la, lb));
        let shifted = k.push(Op::Shr(sum, 1));
        let cmp = k.push(Op::Cmp(CmpKind::Gt, la, lb));
        let bias = match lane {
            0 => c63,
            1 => c64,
            2 => cn64,
            _ => cn65,
        };
        let sub_c = match lane {
            0 => c127,
            1 => c128,
            2 => cn128,
            _ => c127,
        };
        let t_val = k.push(Op::Add(shifted, bias));
        let f_val = k.push(Op::Sub(sum, sub_c));
        let sel = k.push(Op::Select { cond: cmp, t_val, f_val });
        k.add_output(out, lane, sel);
    }
    k
}

#[test]
fn wasm32_all_six_elem_ty_widths_node_v22_bit_exact() {
    let dir = temp_dir("elem_ty");
    let widths = [
        (ElemTy::I64, "w_i64"),
        (ElemTy::I32, "w_i32"),
        (ElemTy::I16, "w_i16"),
        (ElemTy::U32, "w_u32"),
        (ElemTy::U16, "w_u16"),
        (ElemTy::U8, "w_u8"),
    ];
    let blocks = 4096usize;
    let mut rng = XorShift64(0x5A17_9921_CAFE_BEEF);

    for (ty, name) in widths {
        let kernel = build_width_kernel(ty, name);
        let cw = compile_kernel_wasm(&kernel, RegBudget::default()).expect("compile wasm");
        assert_eq!(cw.summary.div_rem_insns, 0);
        assert!(cw.summary.has_simd128, "expected SIMD128 for {name}");
        assert_eq!(cw.vector_mode, WasmVectorMode::VerticalI64x2);

        let corners: Vec<i64> = match ty {
            ElemTy::I64 | ElemTy::U64 => vec![
                i64::MIN, -1_000_000_000, -128, -65, -64, -1, 0, 1, 63, 64, 127, 128,
                1_000_000_000, i64::MAX,
            ],
            ElemTy::I32 => vec![i32::MIN as i64, -128, -65, -64, -1, 0, 1, 63, 64, 127, 128, i32::MAX as i64],
            ElemTy::I16 => vec![i16::MIN as i64, -12345, -128, -65, -64, -1, 0, 1, 63, 64, 127, 128, i16::MAX as i64],
            ElemTy::U32 => vec![0, 1, 63, 64, 127, 128, 65_535, 0x7FFF_FFFF, 0x8000_0000, u32::MAX as i64],
            ElemTy::U16 => vec![0, 1, 63, 64, 127, 128, 32_767, 32_768, 65_535],
            ElemTy::U8 => vec![0, 1, 63, 64, 127, 128, 254, 255],
        };
        let mut inputs = Vec::new();
        for (bidx, decl) in kernel.bufs.iter().enumerate() {
            let mut data = vec![0i64; blocks * usize::from(decl.lanes)];
            if decl.access == BufAccess::ReadOnly {
                for blk in 0..blocks {
                    for l in 0..usize::from(decl.lanes) {
                        let v = if blk < corners.len() {
                            corners[(blk + l + bidx) % corners.len()]
                        } else {
                            rng.range_i64(decl.domain.lo, decl.domain.hi)
                        };
                        data[blk * usize::from(decl.lanes) + l] = decl.elem_ty.cast_i64(v);
                    }
                }
            }
            inputs.push(data);
        }
        run_node_wasm_case(
            &dir,
            name,
            &cw,
            &kernel,
            &inputs,
            &[0, 1, 2, 3, 7, blocks],
            0,
        );
    }

    // Scalar LoadLut + Barrett division-elimination kernel (tests scalar path + LUT pinning in batch)
    let mut lut_k = Kernel::new("lut_barrett_u16");
    let x = lut_k.add_typed_buf("x", BufAccess::ReadOnly, ElemTy::U16, 4, Domain::range(0, 65_535));
    let lut = lut_k.add_typed_buf("lut", BufAccess::LookupTable(16), ElemTy::I32, 16, Domain::range(-1000, 1000));
    let out = lut_k.add_typed_buf("out", BufAccess::MutNoAlias, ElemTy::I32, 4, Domain::range(-1000, 10_000));
    let m0 = lut_k.push(Op::MemInit);
    let c29 = lut_k.push(Op::Const(29));
    let c15 = lut_k.push(Op::Const(15));
    for lane in 0..4u8 {
        let lx = lut_k.push(Op::Load { mem: m0, buf: x, lane });
        let q = lut_k.push(Op::Div(lx, c29));
        let idx = lut_k.push(Op::BitAnd(q, c15));
        let chk = lut_k.push(Op::BoundsCheck { idx, limit: 16 });
        let lv = lut_k.push(Op::LoadLut { buf: lut, idx: chk });
        let res = lut_k.push(Op::Add(q, lv));
        lut_k.add_output(out, lane, res);
    }
    let cw_lut = compile_kernel_wasm(&lut_k, RegBudget::default()).expect("compile lut wasm");
    assert_eq!(cw_lut.vector_mode, WasmVectorMode::ScalarI64);
    assert_eq!(cw_lut.summary.div_rem_insns, 0, "Barrett must eliminate / 29 in WASM");
    let mut lut_inputs = vec![vec![0i64; 64 * 4], (0..16i64).map(|i| i * 37 - 250).collect(), vec![0i64; 64 * 4]];
    for (i, slot) in lut_inputs[0].iter_mut().enumerate() {
        *slot = ((i * 1021) % 65_536) as i64;
    }
    run_node_wasm_case(
        &dir,
        "lut_barrett_u16",
        &cw_lut,
        &lut_k,
        &lut_inputs,
        &[0, 1, 3, 7, 64],
        0,
    );
    let _ = fs::remove_dir_all(&dir);
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

fn build_zero_rans_inputs(kernel: &Kernel, blocks: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut rng = XorShift64(seed);
    let mut bufs = Vec::with_capacity(kernel.bufs.len());
    for (bidx, decl) in kernel.bufs.iter().enumerate() {
        let lanes = usize::from(decl.lanes);
        let mut data = vec![0i64; blocks * lanes];
        if decl.access == BufAccess::ReadOnly {
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

#[test]
fn wasm32_all_six_zero_rans_kernels_4096_blocks_bit_exact() {
    let dir = temp_dir("zero_rans");
    let zr_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../zero-rans/src");
    let mut doms = BTreeMap::new();
    doms.insert("raw".to_string(), Domain::range(-262_144, 262_144));
    doms.insert("x".to_string(), Domain::range(-32_768, 32_767));
    doms.insert("even".to_string(), Domain::range(-1_000_000_000, 1_000_000_000));
    doms.insert("odd".to_string(), Domain::range(-1_000_000_000, 1_000_000_000));
    doms.insert("next".to_string(), Domain::range(0, 4_294_967_295));

    let mut ov = BTreeMap::new();
    ov.insert("*".to_string(), doms);
    let res_src = fs::read_to_string(zr_root.join("residual.rs")).expect("read residual.rs");
    const RANS_REFILL_SRC: &str = r#"
    pub const fn rans_refill_branchless(next: u32, word: u16) -> (u32, usize) {
        let need = if next < 32_768 { 1u32 } else { 0u32 };
        let mask = 0u32.wrapping_sub(need);
        let refill = (next << 16) | (word as u32);
        let state = next ^ ((refill ^ next) & mask);
        (state, (need as usize) << 1)
    }
    "#;
    let res_kernels = parse_rust_file_kernels(&res_src, &ov);
    let rans_kernels = parse_rust_file_kernels(RANS_REFILL_SRC, &ov);
    let all_extracted: Vec<Kernel> = res_kernels.into_iter().chain(rans_kernels).collect();

    let targets = [
        ("idct8_1d_combine", WasmVectorMode::VerticalI64x2),
        ("dct8_1d_even", WasmVectorMode::VerticalI64x2),
        ("dct8_1d_odd", WasmVectorMode::VerticalI64x2),
        ("idct8_1d_even", WasmVectorMode::VerticalI64x2),
        ("idct8_1d_odd", WasmVectorMode::VerticalI64x2),
        ("rans_refill_branchless", WasmVectorMode::HorizontalI32x4),
    ];

    let mut bundle_kernels = Vec::new();
    let base_budget = RegBudget::default();
    let mut max_budget = base_budget;
    for (idx, &(name, want_mode)) in targets.iter().enumerate() {
        let ek = all_extracted
            .iter()
            .find(|k| k.name == name)
            .unwrap_or_else(|| panic!("missing zero-rans kernel {name}"));
        let kb = kernel_reg_budget(ek, base_budget);
        max_budget = RegBudget {
            max_gprs: max_budget.max_gprs.max(kb.max_gprs),
            max_vecs: max_budget.max_vecs.max(kb.max_vecs),
        };
        let t0 = Instant::now();
        let cw = compile_kernel_wasm(ek, kb).expect("compile zero-rans wasm");
        let compile_us = t0.elapsed().as_micros();
        assert_eq!(cw.vector_mode, want_mode, "vector mode mismatch on {name}");
        assert_eq!(cw.summary.div_rem_insns, 0, "zero WASM div/rem opcodes required on {name}");
        assert!(cw.summary.has_simd128 && cw.summary.simd_insns > 0, "SIMD128 required on {name}");

        let inputs = build_zero_rans_inputs(ek, 4096, 0xC0DE_0001 + idx as u64);
        let ns_per_call = run_node_wasm_case(
            &dir,
            name,
            &cw,
            ek,
            &inputs,
            &[0, 1, 3, 7, 4096],
            4096,
        )
        .unwrap_or(0.0);
        println!(
            "WASM32 zero-rans kernel={name:22} mode={:?} bytes={:4} insns={:3} simd={:3} divs={} compile_us={:4} node_v22={:.2} ns/call",
            cw.vector_mode, cw.wasm_bytes.len(), cw.summary.total_insns, cw.summary.simd_insns,
            cw.summary.div_rem_insns, compile_us, ns_per_call
        );
        bundle_kernels.push(ek.clone());
    }

    // Verify multi-kernel bundle contains all 12 exported functions + memory and passes Node v22 validate+instantiate
    let (bundle_bytes, bsum) = compile_kernels_wasm(&bundle_kernels, max_budget).expect("compile multi-kernel bundle");
    assert_eq!(bsum.func_count, 12);
    assert_eq!(bsum.div_rem_insns, 0);
    assert!(bsum.has_simd128);
    let bpath = dir.join("zero_rans_bundle.wasm");
    fs::write(&bpath, &bundle_bytes).expect("write bundle");
    let check_out = Command::new("node")
        .args([
            "-e",
            &format!(
                "const b=require('fs').readFileSync({:?}); if(!WebAssembly.validate(b)) process.exit(2); WebAssembly.instantiate(b,{{}}).then(r=>{{ if(Object.keys(r.instance.exports).length!==13) process.exit(3); }}).catch(()=>process.exit(4));",
                bpath.to_string_lossy()
            ),
        ])
        .output()
        .expect("validate bundle in node");
    assert!(check_out.status.success(), "bundle failed Node v22 validation/instantiation");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn wasm32_verify_module_counterfactual_tamper_knockouts() {
    let src = r#"
        /// @domain x: 0..=65535
        pub fn div_barrett_probe(x: &[u32; 4], out: &mut [u32; 4]) {
            for i in 0..4 {
                out[i] = x[i] / 29;
            }
        }
    "#;
    let cw = compile_rust_to_wasm(src, RegBudget::default()).expect("compile probe");
    assert!(verify_wasm_module(&cw.wasm_bytes).is_ok());
    assert_eq!(cw.summary.div_rem_insns, 0, "optimized Barrett kernel must have 0 WASM div opcodes");

    // Counterfactual: unoptimized division kernel must report div_rem_insns > 0
    let k_unopt = zero_cc::parse_rust_kernel(src).expect("parse");
    let cw_unopt = compile_kernel_wasm_opts(&k_unopt, RegBudget::default(), false).expect("compile unopt");
    assert!(cw_unopt.summary.div_rem_insns > 0, "unoptimized div kernel must report div_rem_insns > 0");

    // Knockout 1: corrupted magic
    let mut bad_magic = cw.wasm_bytes.clone();
    bad_magic[0] ^= 0xFF;
    assert!(verify_wasm_module(&bad_magic).is_err());

    // Knockout 2: corrupted version
    let mut bad_ver = cw.wasm_bytes.clone();
    bad_ver[4] = 2;
    assert!(verify_wasm_module(&bad_ver).is_err());

    // Knockout 3: out-of-order section ID (change Type id 1 -> 4 so 4 > Function id 3)
    let mut bad_order = cw.wasm_bytes.clone();
    bad_order[8] = 4;
    assert!(verify_wasm_module(&bad_order).is_err());

    // Knockout 4: LEB128 section length mismatch
    let mut bad_len = cw.wasm_bytes.clone();
    bad_len[9] = bad_len[9].wrapping_add(1);
    assert!(verify_wasm_module(&bad_len).is_err());

    // Knockout 5: truncated module
    let trunc = &cw.wasm_bytes[..cw.wasm_bytes.len() - 3];
    assert!(verify_wasm_module(trunc).is_err());

    // Knockout 6: trailing garbage byte
    let mut trailing = cw.wasm_bytes.clone();
    trailing.push(0x00);
    assert!(verify_wasm_module(&trailing).is_err());

    // Knockout 7: unbalanced end (replace closing 0x0B of last code body with 0x01 nop)
    let mut bad_end = cw.wasm_bytes.clone();
    let last_idx = bad_end.len() - 1;
    assert_eq!(bad_end[last_idx], 0x0B);
    bad_end[last_idx] = 0x01;
    assert!(verify_wasm_module(&bad_end).is_err());
}

#[test]
fn wasm32_cli_emit_wasm_zero_rans_dir_and_single() {
    let dir = temp_dir("cli");
    let bin = env!("CARGO_BIN_EXE_zero-cc");
    let zr_orig = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../zero-rans/src");
    let zr_src = dir.join("in_src");
    fs::create_dir_all(&zr_src).expect("create in_src");
    fs::copy(zr_orig.join("residual.rs"), zr_src.join("residual.rs")).expect("copy residual.rs");
    const RANS_REFILL_SRC: &str = r#"
    pub const fn rans_refill_branchless(next: u32, word: u16) -> (u32, usize) {
        let need = if next < 32_768 { 1u32 } else { 0u32 };
        let mask = 0u32.wrapping_sub(need);
        let refill = (next << 16) | (word as u32);
        let state = next ^ ((refill ^ next) & mask);
        (state, (need as usize) << 1)
    }
    "#;
    fs::write(zr_src.join("rans_refill.rs"), RANS_REFILL_SRC).expect("write rans_refill.rs");
    let out_dir = dir.join("crate_wasm");

    let t0 = Instant::now();
    let out = Command::new(bin)
        .arg(&zr_src)
        .args([
            "--domain",
            "raw: -262144..=262144, x: -32768..=32767, even: -1000000000..=1000000000, odd: -1000000000..=1000000000, next: 0..=4294967295",
            "--emit-wasm",
        ])
        .arg(&out_dir)
        .output()
        .expect("run zero-cc --emit-wasm dir");
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        out.status.success(),
        "CLI --emit-wasm dir failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("target=Wasm32"), "missing target=Wasm32 in stdout: {stdout}");
    println!("CLI --emit-wasm ../zero-rans/src finished in {wall_ms:.2} ms");

    for kname in [
        "idct8_1d_combine",
        "dct8_1d_even",
        "dct8_1d_odd",
        "idct8_1d_even",
        "idct8_1d_odd",
        "rans_refill_branchless",
    ] {
        let p = out_dir.join(format!("{kname}.wasm"));
        let bytes = fs::read(&p).unwrap_or_else(|e| panic!("missing {}: {e}", p.display()));
        let sum = verify_wasm_module(&bytes).expect("verify CLI wasm");
        assert_eq!(sum.div_rem_insns, 0);
        assert!(sum.has_simd128);
    }

    // Single-kernel CLI emit + Node v22 WebAssembly.validate check
    let single_path = dir.join("idct8_1d_odd.wasm");
    let out2 = Command::new(bin)
        .arg(zr_src.join("residual.rs"))
        .args([
            "--kernel",
            "idct8_1d_odd",
            "--domain",
            "raw: -262144..=262144",
            "--target",
            "wasm32",
            "--emit-wasm",
        ])
        .arg(&single_path)
        .output()
        .expect("run zero-cc single --emit-wasm");
    assert!(
        out2.status.success(),
        "CLI single --emit-wasm failed:\n{}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let node_ok = Command::new("node")
        .args([
            "-e",
            &format!(
                "const b=require('fs').readFileSync({:?}); if(!WebAssembly.validate(b)) process.exit(2);",
                single_path.to_string_lossy()
            ),
        ])
        .output()
        .expect("node validate single");
    assert!(node_ok.status.success(), "Node v22 rejected single CLI .wasm");

    // Multi-kernel CLI bundle emit (--emit-wasm <bundle.wasm>) + Node v22 validate/instantiate
    let bundle_path = dir.join("zero_rans_cli_bundle.wasm");
    let out3 = Command::new(bin)
        .arg(&zr_src)
        .args([
            "--domain",
            "raw: -262144..=262144, x: -32768..=32767, even: -1000000000..=1000000000, odd: -1000000000..=1000000000, next: 0..=4294967295",
            "--emit-wasm",
        ])
        .arg(&bundle_path)
        .output()
        .expect("run zero-cc --emit-wasm bundle.wasm");
    assert!(
        out3.status.success(),
        "CLI --emit-wasm bundle.wasm failed:\n{}",
        String::from_utf8_lossy(&out3.stderr)
    );
    let bbytes = fs::read(&bundle_path).expect("read cli bundle");
    let bsum = verify_wasm_module(&bbytes).expect("verify cli bundle");
    assert!(bsum.func_count >= 12 && bsum.has_simd128);
    let bcheck = Command::new("node")
        .args([
            "-e",
            &format!(
                "const b=require('fs').readFileSync({:?}); if(!WebAssembly.validate(b)) process.exit(2); WebAssembly.instantiate(b,{{}}).then(r=>{{ const e=r.instance.exports; if(typeof e.zero_cc_idct8_1d_odd_batch!=='function'||typeof e.zero_cc_rans_refill_branchless_batch!=='function') process.exit(3); }}).catch(()=>process.exit(4));",
                bundle_path.to_string_lossy()
            ),
        ])
        .output()
        .expect("node validate cli bundle");
    assert!(bcheck.status.success(), "Node v22 rejected CLI bundle .wasm");

    // Single-file 1-kernel CLI emit without --kernel writes foo.wasm directly
    let one_rs = dir.join("one_kernel.rs");
    fs::write(
        &one_rs,
        "/// @domain x: 0..=65535\npub fn div29_u32(x: &[u32; 4], out: &mut [u32; 4]) {\n    for i in 0..4 { out[i] = x[i] / 29; }\n}\n",
    )
    .expect("write one_kernel.rs");
    let one_wasm = dir.join("foo.wasm");
    let out4 = Command::new(bin)
        .arg(&one_rs)
        .arg("--emit-wasm")
        .arg(&one_wasm)
        .output()
        .expect("run zero-cc 1-kernel --emit-wasm foo.wasm");
    assert!(
        out4.status.success(),
        "CLI 1-kernel --emit-wasm foo.wasm failed:\n{}",
        String::from_utf8_lossy(&out4.stderr)
    );
    let one_bytes = fs::read(&one_wasm).expect("read foo.wasm");
    let one_sum = verify_wasm_module(&one_bytes).expect("verify foo.wasm");
    assert_eq!(one_sum.div_rem_insns, 0);
    assert_eq!(one_sum.func_count, 2);
    assert!(!one_sum.imports_memory);

    // Imported-memory multi-kernel CLI bundle (--wasm-import-memory + comma-separated --kernel)
    let imp_path = dir.join("zero_cc_imported.wasm");
    let out5 = Command::new(bin)
        .arg(&zr_src)
        .args([
            "--domain",
            "raw: -262144..=262144, x: -2097152..=2097152, even: -1000000000..=1000000000, odd: -1000000000..=1000000000, next: 0..=4294967295",
            "--kernel",
            "dct8_1d_even,dct8_1d_odd,idct8_1d_even,idct8_1d_odd,idct8_1d_combine,rans_refill_branchless",
            "--wasm-import-memory",
            "--emit-wasm",
        ])
        .arg(&imp_path)
        .output()
        .expect("run zero-cc --wasm-import-memory");
    assert!(
        out5.status.success(),
        "CLI --wasm-import-memory failed:\n{}",
        String::from_utf8_lossy(&out5.stderr)
    );
    let imp_bytes = fs::read(&imp_path).expect("read imported wasm");
    let imp_sum = verify_wasm_module(&imp_bytes).expect("verify imported wasm");
    assert!(imp_sum.imports_memory);
    assert_eq!(imp_sum.func_count, 12);
    assert_eq!(imp_sum.div_rem_insns, 0);
    assert!(imp_sum.has_simd128);
    let imp_check = Command::new("node")
        .args([
            "-e",
            &format!(
                "const b=require('fs').readFileSync({:?}); \
                 const m=new WebAssembly.Module(b); \
                 const imps=WebAssembly.Module.imports(m); \
                 if(imps.length!==1||imps[0].module!=='env'||imps[0].name!=='memory'||imps[0].kind!=='memory') process.exit(2); \
                 const mem=new WebAssembly.Memory({{initial:4,maximum:16}}); \
                 const inst=new WebAssembly.Instance(m,{{env:{{memory:mem}}}}); \
                 const e=inst.exports; \
                 if('memory' in e) process.exit(3); \
                 const i32=new Int32Array(mem.buffer,0,16); \
                 for(let i=0;i<8;i++) i32[i]=i*10-35; \
                 e.zero_cc_dct8_1d_even_batch(0,64,2); \
                 e.zero_cc_dct8_1d_odd_batch(0,64,2); \
                 const o32=new Int32Array(mem.buffer,64,8); \
                 if(o32[0]!==0||o32[1]===0) process.exit(4);",
                imp_path.to_string_lossy()
            ),
        ])
        .output()
        .expect("node validate imported memory bundle");
    assert!(
        imp_check.status.success(),
        "Node v22 failed imported-memory bundle:\n{}",
        String::from_utf8_lossy(&imp_check.stderr)
    );

    let _ = fs::remove_dir_all(&dir);
}
