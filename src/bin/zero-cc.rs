#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use zero_cc::rust_fe::{
    extract_and_compile_crate, extract_crate_kernels, kernel_reg_budget, parse_domain_spec,
    parse_rust_kernel_with_domains, ExtractedKernel,
};
use zero_cc::{
    compile_kernel_direct, compile_kernel_wasm, compile_kernels_wasm, compile_standalone_elf_exec,
    verify_elf64_executable, verify_elf64_object, verify_wasm_module, CompileError, Domain,
    RegBudget, TargetArch, TargetIsa,
};

struct CliArgs {
    input: PathBuf,
    emit_elf: Option<PathBuf>,
    emit_exec: Option<PathBuf>,
    emit_wasm: Option<PathBuf>,
    target: TargetArch,
    kernel_filter: Option<String>,
    domains: BTreeMap<String, Domain>,
    bench: bool,
}

fn usage() -> &'static str {
    "Usage: zero-cc <file_or_dir> [--emit-elf <out.o>] [--emit-exec <out_bin>] [--emit-wasm <out.wasm|dir>] [--target x86_64|aarch64|wasm32] [--isa x86_64|aarch64|wasm32] [--kernel <name>] [--domain <spec>] [--bench]"
}

fn parse_target_name(v: &str) -> Result<TargetArch, String> {
    match v {
        "x86_64" => Ok(TargetArch::X86_64),
        "aarch64" => Ok(TargetArch::Aarch64),
        "wasm32" => Ok(TargetArch::Wasm32),
        _ => Err(format!("unsupported target '{v}' (expected x86_64, aarch64, or wasm32)")),
    }
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut emit_elf: Option<PathBuf> = None;
    let mut emit_exec: Option<PathBuf> = None;
    let mut emit_wasm: Option<PathBuf> = None;
    let mut target = TargetArch::X86_64;
    let mut target_explicit = false;
    let mut kernel_filter: Option<String> = None;
    let mut domains: BTreeMap<String, Domain> = BTreeMap::new();
    let mut bench = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(usage().to_string()),
            "--emit-elf" => {
                let v = args.next().ok_or_else(|| "missing path after --emit-elf".to_string())?;
                emit_elf = Some(PathBuf::from(v));
            }
            "--emit-exec" => {
                let v = args.next().ok_or_else(|| "missing path after --emit-exec".to_string())?;
                emit_exec = Some(PathBuf::from(v));
            }
            "--emit-wasm" => {
                let v = args.next().ok_or_else(|| "missing path after --emit-wasm".to_string())?;
                emit_wasm = Some(PathBuf::from(v));
            }
            "--isa" | "--target" => {
                let v = args.next().ok_or_else(|| format!("missing value after {arg}"))?;
                target = parse_target_name(&v)?;
                target_explicit = true;
            }
            "--kernel" => {
                let v = args.next().ok_or_else(|| "missing name after --kernel".to_string())?;
                kernel_filter = Some(v);
            }
            "--domain" => {
                let v = args.next().ok_or_else(|| "missing spec after --domain".to_string())?;
                let parsed = parse_domain_spec(&v).map_err(|e| format!("bad --domain '{v}': {e}"))?;
                for (k, d) in parsed { domains.insert(k, d); }
            }
            "--bench" => { bench = true; }
            other if other.starts_with('-') => return Err(format!("unknown flag '{other}'\n{}", usage())),
            path => {
                if input.is_some() { return Err(format!("unexpected extra argument '{path}'")); }
                input = Some(PathBuf::from(path));
            }
        }
    }
    let Some(input) = input else { return Err(usage().to_string()); };
    if emit_wasm.is_some() && !target_explicit && emit_elf.is_none() && emit_exec.is_none() {
        target = TargetArch::Wasm32;
    }
    Ok(CliArgs { input, emit_elf, emit_exec, emit_wasm, target, kernel_filter, domains, bench })
}

fn load_selected_kernel(
    path: &Path,
    filter: Option<&str>,
    domains: &BTreeMap<String, Domain>,
) -> Result<ExtractedKernel, CompileError> {
    let mut ov = BTreeMap::new();
    if !domains.is_empty() { ov.insert("*".to_string(), domains.clone()); }
    if path.is_file() {
        let src = fs::read_to_string(path).map_err(|e| CompileError::ParseError(format!("read '{}': {e}", path.display())))?;
        if filter.is_none() {
            if let Ok(k) = parse_rust_kernel_with_domains(&src, domains) {
                return Ok(ExtractedKernel { file: path.display().to_string(), name: k.name.clone(), kernel: k });
            }
        }
    }
    let (_, _, kernels) = extract_crate_kernels(path, &ov)?;
    if let Some(want) = filter {
        kernels.into_iter().find(|k| k.name == want)
            .ok_or_else(|| CompileError::ParseError(format!("kernel '{want}' not found in '{}'", path.display())))
    } else {
        kernels.into_iter().next()
            .ok_or_else(|| CompileError::ParseError(format!("no pure kernels found in '{}'", path.display())))
    }
}

#[cfg(unix)]
fn set_exec_mode(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let perm = fs::Permissions::from_mode(0o755);
    fs::set_permissions(path, perm).map_err(|e| format!("chmod 0755 '{}': {e}", path.display()))
}

#[cfg(not(unix))]
fn set_exec_mode(_path: &Path) -> Result<(), String> { Ok(()) }

fn emit_wasm_single(ek: &ExtractedKernel, budget: RegBudget, out_path: Option<&Path>, t0: Instant) -> Result<(), String> {
    let orig = ek.kernel.cost();
    let opt = zero_cc::optimize_kernel(&ek.kernel).map_err(|e| e.to_string())?.cost();
    let k_budget = kernel_reg_budget(&ek.kernel, budget);
    let wasm = compile_kernel_wasm(&ek.kernel, k_budget).map_err(|e| e.to_string())?;
    let wsum = verify_wasm_module(&wasm.wasm_bytes).map_err(|e| e.to_string())?;
    if let Some(p) = out_path {
        let dest = if p.is_dir() { p.join(format!("{}.wasm", ek.name)) } else { p.to_path_buf() };
        fs::write(&dest, &wasm.wasm_bytes).map_err(|e| format!("write '{}': {e}", dest.display()))?;
    }
    let elapsed_us = t0.elapsed().as_micros();
    println!(
        "zero-cc kernel={} file={} target=Wasm32 mode={:?} ops={}->{} divs={}->{} wasm_bytes={} insns={} simd_insns={} wasm_divs={} compile_us={}",
        ek.name, ek.file, wasm.vector_mode, orig.total_nodes, opt.total_nodes, orig.divs_rems,
        opt.divs_rems, wasm.wasm_bytes.len(), wsum.total_insns, wsum.simd_insns, wsum.div_rem_insns, elapsed_us
    );
    Ok(())
}

fn run_wasm_crate_scan(cfg: &CliArgs, budget: RegBudget, t0: Instant) -> Result<(), String> {
    let mut ov = BTreeMap::new();
    if !cfg.domains.is_empty() { ov.insert("*".to_string(), cfg.domains.clone()); }
    let (files_scanned, fns_seen, kernels) = extract_crate_kernels(&cfg.input, &ov).map_err(|e| e.to_string())?;
    let mut compiled_list = Vec::new();
    let mut skipped = 0usize;
    for ek in &kernels {
        let kb = kernel_reg_budget(&ek.kernel, budget);
        match compile_kernel_wasm(&ek.kernel, kb) {
            Ok(cw) => {
                let orig = ek.kernel.cost();
                let opt = zero_cc::optimize_kernel(&ek.kernel).map_err(|e| e.to_string())?.cost();
                compiled_list.push((ek, cw, orig, opt));
            }
            Err(_) => { skipped += 1; }
        }
    }
    if let Some(out_p) = &cfg.emit_wasm {
        let is_bundle = out_p.extension().and_then(|s| s.to_str()) == Some("wasm");
        if is_bundle {
            let ks: Vec<zero_cc::Kernel> = compiled_list.iter().map(|(ek, _, _, _)| ek.kernel.clone()).collect();
            let max_b = ks.iter().fold(budget, |acc, k| {
                let kb = kernel_reg_budget(k, budget);
                RegBudget { max_gprs: acc.max_gprs.max(kb.max_gprs), max_vecs: acc.max_vecs.max(kb.max_vecs) }
            });
            let (bytes, _) = compile_kernels_wasm(&ks, max_b).map_err(|e| e.to_string())?;
            fs::write(out_p, &bytes).map_err(|e| format!("write '{}': {e}", out_p.display()))?;
        } else {
            fs::create_dir_all(out_p).map_err(|e| format!("mkdir '{}': {e}", out_p.display()))?;
            for (ek, cw, _, _) in &compiled_list {
                let dest = out_p.join(format!("{}.wasm", ek.name));
                fs::write(&dest, &cw.wasm_bytes).map_err(|e| format!("write '{}': {e}", dest.display()))?;
            }
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "zero-cc scan root={} target=Wasm32 files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
        cfg.input.display(), files_scanned, fns_seen, kernels.len(), compiled_list.len(), skipped, elapsed_ms
    );
    for (ek, cw, orig, opt) in &compiled_list {
        println!(
            "  kernel={} file={} ops={}->{} divs={}->{} mode={:?} wasm={}B insns={} simd={} wasm_divs={}",
            ek.name, ek.file, orig.total_nodes, opt.total_nodes, orig.divs_rems, opt.divs_rems,
            cw.vector_mode, cw.wasm_bytes.len(), cw.summary.total_insns, cw.summary.simd_insns, cw.summary.div_rem_insns
        );
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let cfg = parse_args()?;
    let t0 = Instant::now();
    let budget = RegBudget::default();

    if cfg.target == TargetArch::Wasm32 && cfg.emit_elf.is_none() && cfg.emit_exec.is_none() {
        if cfg.kernel_filter.is_some() || cfg.input.is_file() {
            let ek = load_selected_kernel(&cfg.input, cfg.kernel_filter.as_deref(), &cfg.domains)
                .map_err(|e| e.to_string())?;
            return emit_wasm_single(&ek, budget, cfg.emit_wasm.as_deref(), t0);
        }
        return run_wasm_crate_scan(&cfg, budget, t0);
    }

    let isa = match cfg.target {
        TargetArch::Aarch64 => TargetIsa::Aarch64,
        TargetArch::X86_64 | TargetArch::Wasm32 => TargetIsa::X86_64,
    };

    if cfg.emit_elf.is_some() || cfg.emit_exec.is_some() || cfg.kernel_filter.is_some() {
        let ek = load_selected_kernel(&cfg.input, cfg.kernel_filter.as_deref(), &cfg.domains)
            .map_err(|e| e.to_string())?;
        let orig = ek.kernel.cost();
        let opt = zero_cc::optimize_kernel(&ek.kernel).map_err(|e| e.to_string())?.cost();
        let k_budget = kernel_reg_budget(&ek.kernel, budget);
        let obj = compile_kernel_direct(&ek.kernel, k_budget, isa).map_err(|e| e.to_string())?;
        verify_elf64_object(&obj.elf_bytes).map_err(|e| e.to_string())?;
        let exec = compile_standalone_elf_exec(&ek.kernel, k_budget, isa, &[]).map_err(|e| e.to_string())?;
        let esum = verify_elf64_executable(&exec.elf_bytes).map_err(|e| e.to_string())?;

        if let Some(out_o) = &cfg.emit_elf {
            fs::write(out_o, &obj.elf_bytes).map_err(|e| format!("write '{}': {e}", out_o.display()))?;
        }
        if let Some(out_bin) = &cfg.emit_exec {
            fs::write(out_bin, &exec.elf_bytes).map_err(|e| format!("write '{}': {e}", out_bin.display()))?;
            set_exec_mode(out_bin)?;
        }
        if let Some(out_w) = &cfg.emit_wasm {
            let wasm = compile_kernel_wasm(&ek.kernel, k_budget).map_err(|e| e.to_string())?;
            verify_wasm_module(&wasm.wasm_bytes).map_err(|e| e.to_string())?;
            let dest = if out_w.is_dir() { out_w.join(format!("{}.wasm", ek.name)) } else { out_w.clone() };
            fs::write(&dest, &wasm.wasm_bytes).map_err(|e| format!("write '{}': {e}", dest.display()))?;
        }

        let elapsed_us = t0.elapsed().as_micros();
        println!(
            "zero-cc kernel={} file={} isa={:?} ops={}->{} divs={}->{} direct_insns={} rel_share={}% exec_share={}% zero_spills={} rel_bytes={} exec_bytes={} entry=0x{:x} compile_us={}",
            ek.name, ek.file, isa, orig.total_nodes, opt.total_nodes, orig.divs_rems, opt.divs_rems,
            obj.direct_insns, obj.direct_share_pct, exec.direct_share_pct,
            obj.zero_spills && exec.zero_spills, obj.elf_bytes.len(), exec.elf_bytes.len(),
            esum.entry_vaddr, elapsed_us
        );

        if cfg.bench {
            if let (Some(bin_path), TargetIsa::X86_64) = (&cfg.emit_exec, isa) {
                let bt0 = Instant::now();
                let out = Command::new(bin_path).output().map_err(|e| format!("exec '{}': {e}", bin_path.display()))?;
                let exec_us = bt0.elapsed().as_micros();
                println!(
                    "zero-cc standalone_exec_run bytes={} exit_code={} wall_us={}",
                    out.stdout.len(), out.status.code().unwrap_or(-1), exec_us
                );
            }
        }
        return Ok(());
    }

    let mut ov = BTreeMap::new();
    if !cfg.domains.is_empty() { ov.insert("*".to_string(), cfg.domains.clone()); }
    let report = extract_and_compile_crate(&cfg.input, isa, budget, &ov).map_err(|e| e.to_string())?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "zero-cc scan root={} isa={:?} files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
        report.root, isa, report.files_scanned, report.functions_seen,
        report.kernels_extracted, report.compiled.len(), report.skipped, elapsed_ms
    );
    for k in &report.compiled {
        println!(
            "  kernel={} file={} ops={}->{} divs={}->{} insns={} rel={}B({}%) exec={}B({}%) spills={}",
            k.name, k.file, k.original_cost.total_nodes, k.optimized_cost.total_nodes,
            k.original_cost.divs_rems, k.optimized_cost.divs_rems, k.direct_insns,
            k.obj_bytes, k.direct_share_pct, k.exec_bytes, k.exec_share_pct,
            if k.zero_spills { 0 } else { 1 }
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
