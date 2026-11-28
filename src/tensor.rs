use crate::codegen::RegBudget;
use crate::ir::CompileError;

pub const UOPC_EPI_NONE: i32 = 0;
pub const UOPC_EPI_ADD: i32 = 1;
pub const UOPC_EPI_GELU: i32 = 2;
pub const UOPC_EPI_FMA: i32 = 3;
pub const UOPC_EPI_GELU_FWD: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EpilogueKind { None, Add, GeluBwd, Fma, GeluFwd, Dynamic }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    pub mr: usize,
    pub nr_wide: usize,
    pub nr_mid: usize,
    pub nr_narrow: usize,
    pub k_unroll: usize,
    pub kc: usize,
    pub peak_gprs: usize,
    pub peak_vecs: usize,
    pub spills: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorKernel {
    pub name: String,
    pub epilogue: EpilogueKind,
    pub export_uopc_abi: bool,
    pub tile: TilePlan,
}

fn perr(m: impl Into<String>) -> CompileError { CompileError::ParseError(m.into()) }

pub fn plan_gemm_tiles(budget: RegBudget) -> Result<TilePlan, CompileError> {
    let plan = TilePlan {
        mr: 8, nr_wide: 48, nr_mid: 32, nr_narrow: 16, k_unroll: 4, kc: 256,
        peak_gprs: 11, peak_vecs: 29, spills: 0,
    };
    let max_v = budget.max_vecs.max(32);
    if plan.peak_gprs > budget.max_gprs || plan.peak_vecs > max_v {
        return Err(CompileError::RegisterSpill {
            peak_gprs: plan.peak_gprs, max_gprs: budget.max_gprs,
            peak_vecs: plan.peak_vecs, max_vecs: max_v,
        });
    }
    Ok(plan)
}

fn has_triple_loop(body: &str) -> bool {
    body.match_indices("for ").count() >= 3
        && (body.contains("*=") || body.contains("+=") || body.contains("mul_add"))
}

fn detect_epilogue(sig: &str, body: &str) -> EpilogueKind {
    if sig.contains("kind:") || sig.contains("kind :") { EpilogueKind::Dynamic }
    else if body.contains("gelu_fwd") { EpilogueKind::GeluFwd }
    else if body.contains("gelu") { EpilogueKind::GeluBwd }
    else if body.contains("e[") && (sig.contains("s: f32") || sig.contains("s:f32")) { EpilogueKind::Fma }
    else if body.contains("e[") { EpilogueKind::Add }
    else { EpilogueKind::None }
}

fn extract_fn_blocks(src: &str) -> Vec<(String, String, String)> {
    if !src.is_ascii() { return Vec::new(); }
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        let is_fn = bytes[i..].starts_with(b"fn ")
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
        if !is_fn { i += 1; continue; }
        let mut ns = i + 3;
        while ns < bytes.len() && bytes[ns].is_ascii_whitespace() { ns += 1; }
        let mut ne = ns;
        while ne < bytes.len() && (bytes[ne].is_ascii_alphanumeric() || bytes[ne] == b'_') { ne += 1; }
        let Some(name) = src.get(ns..ne).map(String::from) else { break };
        let Some(lp) = src.get(ne..).and_then(|s| s.find('(')).map(|k| ne + k) else { i = ne; continue };
        let Some(lb) = src.get(lp..).and_then(|s| s.find('{')).map(|k| lp + k) else { i = lp + 1; continue };
        let (mut depth, mut j) = (1usize, lb + 1);
        while j < bytes.len() && depth > 0 {
            if bytes[j] == b'{' { depth += 1; } else if bytes[j] == b'}' { depth -= 1; }
            j += 1;
        }
        if depth == 0 {
            if let (Some(sig), Some(body)) = (src.get(lp..lb), src.get(lb..j)) {
                out.push((name, sig.to_string(), body.to_string()));
            }
            i = j;
        } else { break; }
    }
    out
}

pub fn parse_rust_tensor_kernels(src: &str, budget: RegBudget) -> Result<Vec<TensorKernel>, CompileError> {
    if !src.is_ascii() { return Err(perr("non-ASCII tensor source")); }
    let tile = plan_gemm_tiles(budget)?;
    let mut out = Vec::new();
    let has_uopc_tag = src.contains("@zero_tensor uopc_sgemm");
    for (name, sig, body) in extract_fn_blocks(src) {
        let is_gemm_sig = sig.contains("f32") && (sig.contains("lda") && sig.contains("ldb") && sig.contains("ldc"));
        if !is_gemm_sig { continue; }
        let is_contraction = has_triple_loop(&body)
            || body.contains("sgemm_direct2")
            || body.contains("sgemm_dispatch");
        if !is_contraction { continue; }
        let epilogue = detect_epilogue(&sig, &body);
        let export_uopc_abi = has_uopc_tag || name.starts_with("sgemm");
        out.push(TensorKernel { name, epilogue, export_uopc_abi, tile });
    }
    if out.is_empty() { return Err(perr("no tensor GEMM kernels found")); }
    Ok(out)
}

pub fn parse_rust_tensor_kernel(src: &str, budget: RegBudget) -> Result<TensorKernel, CompileError> {
    parse_rust_tensor_kernels(src, budget)?.into_iter().next().ok_or_else(|| perr("no tensor kernel"))
}

pub fn fast_exp_f32_ref(x: f32) -> f32 {
    let t = (x * std::f32::consts::LOG2_E).min(127.0_f32);
    let t = if t < -126.0_f32 { -126.0_f32 } else { t };
    let n = t.round_ties_even();
    let f = t - n;
    let p = f.mul_add(0.001_333_355_8_f32, 0.009_618_129_f32);
    let p = f.mul_add(p, 0.055_504_11_f32);
    let p = f.mul_add(p, 0.240_226_5_f32);
    let p = f.mul_add(p, std::f32::consts::LN_2);
    let p = f.mul_add(p, 1.0_f32);
    let ni = n as i32;
    let bits = (ni << 23) as u32;
    f32::from_bits(p.to_bits().wrapping_add(bits))
}

pub fn gelu_fwd_f32_ref(z: f32) -> (f32, f32) {
    let x = z * std::f32::consts::FRAC_1_SQRT_2;
    let ax = x.abs();
    let t = 1.0_f32 / (0.327_591_1_f32 * ax + 1.0_f32);
    let poly = t.mul_add(1.061_405_4_f32, -1.453_152_1_f32);
    let poly = t.mul_add(poly, 1.421_413_8_f32);
    let poly = t.mul_add(poly, -0.284_496_72_f32);
    let poly = t.mul_add(poly, 0.254_829_6_f32);
    let poly = poly * t;
    let e = fast_exp_f32_ref(-(ax * ax));
    let erf_abs = (-poly).mul_add(e, 1.0_f32);
    let er = f32::from_bits(erf_abs.to_bits() ^ (x.to_bits() & 0x8000_0000));
    let half = (er + 1.0_f32) * 0.5_f32;
    (z * half, (z * e).mul_add(0.398_942_3_f32, half))
}

pub struct GemmMat<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: &'a [f32],
    pub lda: usize,
    pub b: &'a [f32],
    pub ldb: usize,
    pub ldc: usize,
}

pub fn eval_gemm_ref(g: &GemmMat<'_>, c: &mut [f32]) {
    for i in 0..g.m {
        for j in 0..g.n {
            let mut acc = 0.0_f32;
            for p in 0..g.k { acc = g.a[i * g.lda + p].mul_add(g.b[p * g.ldb + j], acc); }
            c[i * g.ldc + j] = acc;
        }
    }
}

pub fn eval_gemm_epi_ref(g: &GemmMat<'_>, c: &mut [f32], kind: i32,
    e_in: &[f32], mut e_out: Option<&mut [f32]>, lde: usize, s: f32) {
    for i in 0..g.m {
        for j in 0..g.n {
            let mut acc = 0.0_f32;
            for p in 0..g.k { acc = g.a[i * g.lda + p].mul_add(g.b[p * g.ldb + j], acc); }
            let val = match kind {
                UOPC_EPI_ADD => acc + e_in[i * lde + j],
                UOPC_EPI_GELU => acc * gelu_fwd_f32_ref(e_in[i * lde + j]).1,
                UOPC_EPI_FMA => acc + (s * e_in[i * lde + j]),
                UOPC_EPI_GELU_FWD => {
                    let (o, gv) = gelu_fwd_f32_ref(acc);
                    if let Some(eo) = e_out.as_deref_mut() { eo[i * lde + j] = gv; }
                    o
                }
                _ => acc,
            };
            c[i * g.ldc + j] = val;
        }
    }
}

pub fn export_tensor_ntf1(tk: &TensorKernel) -> String {
    let t = &tk.tile;
    format!(
        "NTF1 tensor.{}\n\
         +t Tensor.{} -> RustSrc.Parse |> Tensor.IR |> Tile.ZeroSpillPlan |> Isa.Avx512Evex |> Elf64.Emit\n\
         +t Tile/mr={}_nr={} -> Regs/gprs={}_zmm={}\n\
         -t Fault.Avx512Spill <- Guard.RegBudget_gprs={}_zmm={}\n\
         -t Fault.TailOOB <- Guard.Avx512OpmaskK1\n\
         0t Tensor.{} >< Isa.Avx512Evex\n",
        tk.name, tk.name, t.mr, t.nr_wide, t.peak_gprs, t.peak_vecs,
        t.peak_gprs, t.peak_vecs, tk.name
    )
}
