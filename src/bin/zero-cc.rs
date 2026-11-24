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
    compile_kernel_direct, compile_standalone_elf_exec, verify_elf64_executable,
    verify_elf64_object, CompileError, Domain, RegBudget, TargetIsa,
};

struct CliArgs {
    input: PathBuf,
    emit_elf: Option<PathBuf>,
    emit_exec: Option<PathBuf>,
    isa: TargetIsa,
    kernel_filter: Option<String>,
    domains: BTreeMap<String, Domain>,
    bench: bool,
}

fn usage() -> &'static str {
    "Usage: zero-cc <file_or_dir> [--emit-elf <out.o>] [--emit-exec <out_bin>] [--isa x86_64|aarch64] [--kernel <name>] [--domain <spec>] [--bench]"
}

fn parse_args() -> Result<CliArgs, String> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut emit_elf: Option<PathBuf> = None;
    let mut emit_exec: Option<PathBuf> = None;
    let mut isa = TargetIsa::X86_64;
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
            "--isa" => {
                let v = args.next().ok_or_else(|| "missing value after --isa".to_string())?;
                isa = match v.as_str() {
                    "x86_64" => TargetIsa::X86_64,
                    "aarch64" => TargetIsa::Aarch64,
                    _ => return Err(format!("unsupported ISA '{v}' (expected x86_64 or aarch64)")),
                };
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
    Ok(CliArgs { input, emit_elf, emit_exec, isa, kernel_filter, domains, bench })
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

fn run() -> Result<(), String> {
    let cfg = parse_args()?;
    let t0 = Instant::now();
    let budget = RegBudget::default();

    if cfg.emit_elf.is_some() || cfg.emit_exec.is_some() || cfg.kernel_filter.is_some() {
        let ek = load_selected_kernel(&cfg.input, cfg.kernel_filter.as_deref(), &cfg.domains)
            .map_err(|e| e.to_string())?;
        let orig = ek.kernel.cost();
        let opt = zero_cc::optimize_kernel(&ek.kernel).map_err(|e| e.to_string())?.cost();
        let k_budget = kernel_reg_budget(&ek.kernel, budget);
        let obj = compile_kernel_direct(&ek.kernel, k_budget, cfg.isa).map_err(|e| e.to_string())?;
        verify_elf64_object(&obj.elf_bytes).map_err(|e| e.to_string())?;
        let exec = compile_standalone_elf_exec(&ek.kernel, k_budget, cfg.isa, &[]).map_err(|e| e.to_string())?;
        let esum = verify_elf64_executable(&exec.elf_bytes).map_err(|e| e.to_string())?;

        if let Some(out_o) = &cfg.emit_elf {
            fs::write(out_o, &obj.elf_bytes).map_err(|e| format!("write '{}': {e}", out_o.display()))?;
        }
        if let Some(out_bin) = &cfg.emit_exec {
            fs::write(out_bin, &exec.elf_bytes).map_err(|e| format!("write '{}': {e}", out_bin.display()))?;
            set_exec_mode(out_bin)?;
        }

        let elapsed_us = t0.elapsed().as_micros();
        println!(
            "zero-cc kernel={} file={} isa={:?} ops={}->{} divs={}->{} direct_insns={} rel_share={}% exec_share={}% zero_spills={} rel_bytes={} exec_bytes={} entry=0x{:x} compile_us={}",
            ek.name, ek.file, cfg.isa, orig.total_nodes, opt.total_nodes, orig.divs_rems, opt.divs_rems,
            obj.direct_insns, obj.direct_share_pct, exec.direct_share_pct,
            obj.zero_spills && exec.zero_spills, obj.elf_bytes.len(), exec.elf_bytes.len(),
            esum.entry_vaddr, elapsed_us
        );

        if cfg.bench {
            if let (Some(bin_path), TargetIsa::X86_64) = (&cfg.emit_exec, cfg.isa) {
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
    let report = extract_and_compile_crate(&cfg.input, cfg.isa, budget, &ov).map_err(|e| e.to_string())?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "zero-cc scan root={} isa={:?} files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
        report.root, cfg.isa, report.files_scanned, report.functions_seen,
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
