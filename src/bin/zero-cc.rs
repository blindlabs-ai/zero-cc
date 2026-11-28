#![forbid(unsafe_code)]
use std::{collections::{BTreeMap, BTreeSet}, env, fs, path::{Path, PathBuf}};
use std::{process::{Command, ExitCode}, time::Instant};
use zero_cc::rust_fe::{extract_and_compile_crate, extract_crate_kernels, kernel_reg_budget, parse_domain_spec,
    parse_rust_kernel_with_domains, ExtractedKernel};
use zero_cc::{compile_kernel_direct, compile_kernel_wasm, compile_kernels_wasm_with_memory,
    compile_standalone_elf_exec, compile_tensor_to_elf, parse_rust_tensor_kernel, verify_elf64_executable,
    verify_elf64_object, verify_wasm_module, Domain, Kernel, KernelCost, RegBudget, TargetArch, TargetIsa,
    WasmSummary};
struct CliArgs {
    input: PathBuf, emit_elf: Option<PathBuf>, emit_exec: Option<PathBuf>, emit_wasm: Option<PathBuf>,
    wasm_import_memory: bool, target: TargetArch, kernel_filter: Option<String>,
    domains: BTreeMap<String, Domain>, bench: bool,
}
const USAGE: &str = "Usage: zero-cc <file_or_dir> [--emit-elf <out.o>] [--emit-exec <out_bin>] \
    [--emit-wasm <out.wasm|dir>] [--wasm-import-memory] [--target x86_64|aarch64|wasm32] \
    [--isa x86_64|aarch64|wasm32] [--kernel <name[,name...]>] [--domain <spec>] [--bench]";
fn to_err<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, String> { r.map_err(|e| e.to_string()) }
fn next_val(a: &mut impl Iterator<Item = String>, f: &str) -> Result<String, String> {
    a.next().ok_or_else(|| format!("missing value after {f}"))
}
fn write_out(p: &Path, b: &[u8]) -> Result<(), String> {
    fs::write(p, b).map_err(|e| format!("write '{}': {e}", p.display()))
}
fn domain_overrides(d: &BTreeMap<String, Domain>) -> BTreeMap<String, BTreeMap<String, Domain>> {
    if d.is_empty() { BTreeMap::new() } else { BTreeMap::from([("*".into(), d.clone())]) }
}
fn wasm_dest(p: &Path, n: &str) -> PathBuf { if p.is_dir() { p.join(format!("{n}.wasm")) } else { p.into() } }
fn single_wasm(k: &Kernel, b: RegBudget, imp: bool, fb: &[u8]) -> Result<(Vec<u8>, WasmSummary), String> {
    if imp { to_err(compile_kernels_wasm_with_memory(std::slice::from_ref(k), kernel_reg_budget(k, b), true)) }
    else { Ok((fb.to_vec(), to_err(verify_wasm_module(fb))?)) }
}
fn parse_args() -> Result<CliArgs, String> {
    let (mut args, mut input, mut target) = (env::args().skip(1), None, None);
    let mut cfg = CliArgs { input: PathBuf::new(), emit_elf: None, emit_exec: None, emit_wasm: None,
        wasm_import_memory: false, target: TargetArch::X86_64, kernel_filter: None,
        domains: BTreeMap::new(), bench: false };
    while let Some(arg) = args.next() { match arg.as_str() {
        "-h" | "--help" => return Err(USAGE.into()),
        "--emit-elf" => cfg.emit_elf = Some(PathBuf::from(next_val(&mut args, "--emit-elf")?)),
        "--emit-exec" => cfg.emit_exec = Some(PathBuf::from(next_val(&mut args, "--emit-exec")?)),
        "--emit-wasm" => cfg.emit_wasm = Some(PathBuf::from(next_val(&mut args, "--emit-wasm")?)),
        "--wasm-import-memory" => cfg.wasm_import_memory = true,
        "--isa" | "--target" => target = Some(match next_val(&mut args, &arg)?.as_str() {
            "x86_64" => TargetArch::X86_64,
            "aarch64" => TargetArch::Aarch64,
            "wasm32" => TargetArch::Wasm32,
            v => return Err(format!("unsupported target '{v}' (expected x86_64, aarch64, or wasm32)")),
        }),
        "--kernel" => cfg.kernel_filter = Some(next_val(&mut args, "--kernel")?),
        "--domain" => {
            let v = next_val(&mut args, "--domain")?;
            cfg.domains.extend(parse_domain_spec(&v).map_err(|e| format!("bad --domain '{v}': {e}"))?);
        }
        "--bench" => cfg.bench = true,
        other if other.starts_with('-') => return Err(format!("unknown flag '{other}'\n{USAGE}")),
        path if input.replace(PathBuf::from(path)).is_some() => return Err(format!("extra argument '{path}'")),
        _ => {}
    }}
    cfg.input = input.ok_or_else(|| USAGE.to_string())?;
    cfg.target = target.unwrap_or(if cfg.emit_wasm.is_some() && cfg.emit_elf.is_none() && cfg.emit_exec.is_none() {
        TargetArch::Wasm32
    } else { TargetArch::X86_64 });
    Ok(cfg)
}
fn load_selected(cfg: &CliArgs) -> Result<(ExtractedKernel, KernelCost, KernelCost), String> {
    let (path, filter, domains) = (&cfg.input, cfg.kernel_filter.as_deref(), &cfg.domains);
    let direct = if path.is_file() && filter.is_none() {
        let src = fs::read_to_string(path).map_err(|e| format!("read '{}': {e}", path.display()))?;
        parse_rust_kernel_with_domains(&src, domains).ok()
            .map(|k| ExtractedKernel { file: path.display().to_string(), name: k.name.clone(), kernel: k })
    } else { None };
    let ek = if let Some(ek) = direct { ek } else {
        let (_, _, ks) = to_err(extract_crate_kernels(path, &domain_overrides(domains)))?;
        match filter {
            Some(w) => ks.into_iter().find(|k| k.name == w)
                .ok_or_else(|| format!("kernel '{w}' not found in '{}'", path.display()))?,
            None => ks.into_iter().next()
                .ok_or_else(|| format!("no pure kernels found in '{}'", path.display()))?,
        }
    };
    let (orig, opt) = (ek.kernel.cost(), to_err(zero_cc::optimize_kernel(&ek.kernel))?.cost());
    Ok((ek, orig, opt))
}
#[cfg(unix)] fn set_exec_mode(p: &Path) -> Result<(), String> {
    fs::set_permissions(p, <fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o755))
        .map_err(|e| format!("chmod 0755 '{}': {e}", p.display()))
}
#[cfg(not(unix))] fn set_exec_mode(_: &Path) -> Result<(), String> { Ok(()) }
fn run_wasm_crate_scan(cfg: &CliArgs, budget: RegBudget, t0: Instant) -> Result<(), String> {
    let (files, fns, kernels) = to_err(extract_crate_kernels(&cfg.input, &domain_overrides(&cfg.domains)))?;
    let selected: Vec<&ExtractedKernel> = match &cfg.kernel_filter {
        Some(f) => f.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|w| kernels.iter().find(|k| k.name == w)
            .ok_or_else(|| format!("kernel '{w}' not found in '{}'", cfg.input.display()))).collect::<Result<_, _>>()?,
        None => kernels.iter().collect(),
    };
    let (mut compiled, mut skipped) = (Vec::new(), 0usize);
    for ek in &selected { match compile_kernel_wasm(&ek.kernel, kernel_reg_budget(&ek.kernel, budget)) {
        Ok(cw) => compiled.push((*ek, cw, ek.kernel.cost(), to_err(zero_cc::optimize_kernel(&ek.kernel))?.cost())),
        Err(e) if cfg.kernel_filter.is_some() => return Err(format!("kernel '{}' wasm failed: {e}", ek.name)),
        Err(_) => skipped += 1,
    }}
    if let Some(out_p) = &cfg.emit_wasm {
        if out_p.extension().and_then(|s| s.to_str()) == Some("wasm") {
            let mut seen = BTreeSet::new();
            let ks: Vec<Kernel> = compiled.iter().map(|(ek, _, _, _)| {
                let mut k = ek.kernel.clone();
                if !seen.insert(k.name.clone()) {
                    let st = Path::new(&ek.file).file_stem().and_then(|s| s.to_str()).unwrap_or("dup");
                    let it = std::iter::once(format!("{st}_{}", k.name))
                        .chain((2..).map(|n| format!("{st}_{}_{n}", k.name)));
                    if let Some(c) = it.into_iter().find(|c| seen.insert(c.clone())) { k.name = c; }
                }
                k
            }).collect();
            let max_b = ks.iter().map(|k| kernel_reg_budget(k, budget)).fold(budget, |a, b|
                RegBudget { max_gprs: a.max_gprs.max(b.max_gprs), max_vecs: a.max_vecs.max(b.max_vecs) });
            write_out(out_p, &to_err(compile_kernels_wasm_with_memory(&ks, max_b, cfg.wasm_import_memory))?.0)?;
        } else {
            fs::create_dir_all(out_p).map_err(|e| format!("mkdir '{}': {e}", out_p.display()))?;
            for (ek, cw, _, _) in &compiled {
                let (bytes, _) = single_wasm(&ek.kernel, budget, cfg.wasm_import_memory, &cw.wasm_bytes)?;
                write_out(&out_p.join(format!("{}.wasm", ek.name)), &bytes)?;
            }
        }
    }
    println!("zero-cc scan root={} target=Wasm32 files={files} fns={fns} extracted={} compiled={} \
        skipped={skipped} import_mem={} elapsed_ms={:.2}", cfg.input.display(), kernels.len(),
        compiled.len(), cfg.wasm_import_memory, t0.elapsed().as_secs_f64() * 1000.0);
    for (ek, cw, o, opt) in &compiled {
        let s = &cw.summary;
        println!("  kernel={} file={} ops={}->{} divs={}->{} mode={:?} wasm={}B insns={} simd={} wasm_divs={}",
            ek.name, ek.file, o.total_nodes, opt.total_nodes, o.divs_rems, opt.divs_rems,
            cw.vector_mode, cw.wasm_bytes.len(), s.total_insns, s.simd_insns, s.div_rem_insns);
    }
    Ok(())
}
fn run() -> Result<(), String> {
    let (cfg, t0, budget) = (parse_args()?, Instant::now(), RegBudget::default());
    if cfg.target == TargetArch::Wasm32 && cfg.emit_elf.is_none() && cfg.emit_exec.is_none() {
        let multi = cfg.kernel_filter.as_deref().is_some_and(|f| f.contains(','));
        if !multi && (cfg.kernel_filter.is_some() || cfg.input.is_file()) {
            let (ek, o, opt) = load_selected(&cfg)?;
            let wasm = to_err(compile_kernel_wasm(&ek.kernel, kernel_reg_budget(&ek.kernel, budget)))?;
            let (bytes, s) = single_wasm(&ek.kernel, budget, cfg.wasm_import_memory, &wasm.wasm_bytes)?;
            if let Some(p) = &cfg.emit_wasm { write_out(&wasm_dest(p, &ek.name), &bytes)?; }
            println!("zero-cc kernel={} file={} target=Wasm32 mode={:?} ops={}->{} divs={}->{} \
                wasm_bytes={} insns={} simd_insns={} wasm_divs={} import_mem={} compile_us={}",
                ek.name, ek.file, wasm.vector_mode, o.total_nodes, opt.total_nodes, o.divs_rems, opt.divs_rems,
                bytes.len(), s.total_insns, s.simd_insns, s.div_rem_insns, s.imports_memory, t0.elapsed().as_micros());
            return Ok(());
        }
        return run_wasm_crate_scan(&cfg, budget, t0);
    }
    let isa = if cfg.target == TargetArch::Aarch64 { TargetIsa::Aarch64 } else { TargetIsa::X86_64 };
    if cfg.emit_elf.is_some() || cfg.emit_exec.is_some() || cfg.kernel_filter.is_some() {
        if cfg.input.is_file() && cfg.emit_exec.is_none() && isa == TargetIsa::X86_64 {
            if let Ok(src) = fs::read_to_string(&cfg.input) {
                if let Ok(tk) = parse_rust_tensor_kernel(&src, budget) {
                    let obj = to_err(compile_tensor_to_elf(&tk))?;
                    to_err(verify_elf64_object(&obj.elf_bytes))?;
                    if let Some(p) = &cfg.emit_elf { write_out(p, &obj.elf_bytes)?; }
                    println!("zero-cc tensor={} file={} isa=X86_64_AVX512 tile={}x{} insns={} zero_spills={} \
                        rel_bytes={} syms={} compile_us={}", tk.name, cfg.input.display(), tk.tile.mr,
                        tk.tile.nr_wide, obj.direct_insns, obj.zero_spills, obj.elf_bytes.len(),
                        obj.symbols.join(","), t0.elapsed().as_micros());
                    return Ok(());
                }
            }
        }
        let (ek, o, opt) = load_selected(&cfg)?;
        let kb = kernel_reg_budget(&ek.kernel, budget);
        let obj = to_err(compile_kernel_direct(&ek.kernel, kb, isa))?;
        to_err(verify_elf64_object(&obj.elf_bytes))?;
        let exec = to_err(compile_standalone_elf_exec(&ek.kernel, kb, isa, &[]))?;
        let esum = to_err(verify_elf64_executable(&exec.elf_bytes))?;
        if let Some(p) = &cfg.emit_elf { write_out(p, &obj.elf_bytes)?; }
        if let Some(p) = &cfg.emit_exec {
            write_out(p, &exec.elf_bytes)?;
            set_exec_mode(p)?;
        }
        if let Some(p) = &cfg.emit_wasm {
            let wasm = to_err(compile_kernel_wasm(&ek.kernel, kb))?;
            to_err(verify_wasm_module(&wasm.wasm_bytes))?;
            write_out(&wasm_dest(p, &ek.name), &wasm.wasm_bytes)?;
        }
        println!("zero-cc kernel={} file={} isa={isa:?} ops={}->{} divs={}->{} direct_insns={} \
            rel_share={}% exec_share={}% zero_spills={} rel_bytes={} exec_bytes={} entry=0x{:x} compile_us={}",
            ek.name, ek.file, o.total_nodes, opt.total_nodes, o.divs_rems, opt.divs_rems, obj.direct_insns,
            obj.direct_share_pct, exec.direct_share_pct, obj.zero_spills && exec.zero_spills, obj.elf_bytes.len(),
            exec.elf_bytes.len(), esum.entry_vaddr, t0.elapsed().as_micros());
        if let (true, Some(bin), TargetIsa::X86_64) = (cfg.bench, &cfg.emit_exec, isa) {
            let bt0 = Instant::now();
            let out = Command::new(bin).output().map_err(|e| format!("exec '{}': {e}", bin.display()))?;
            println!("zero-cc standalone_exec_run bytes={} exit_code={} wall_us={}",
                out.stdout.len(), out.status.code().unwrap_or(-1), bt0.elapsed().as_micros());
        }
        return Ok(());
    }
    let r = to_err(extract_and_compile_crate(&cfg.input, isa, budget, &domain_overrides(&cfg.domains)))?;
    println!("zero-cc scan root={} isa={isa:?} files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
        r.root, r.files_scanned, r.functions_seen, r.kernels_extracted, r.compiled.len(), r.skipped,
        t0.elapsed().as_secs_f64() * 1000.0);
    for k in &r.compiled {
        let (o, p) = (&k.original_cost, &k.optimized_cost);
        println!("  kernel={} file={} ops={}->{} divs={}->{} insns={} rel={}B({}%) exec={}B({}%) spills={}",
            k.name, k.file, o.total_nodes, p.total_nodes, o.divs_rems, p.divs_rems, k.direct_insns,
            k.obj_bytes, k.direct_share_pct, k.exec_bytes, k.exec_share_pct, usize::from(!k.zero_spills));
    }
    Ok(())
}
fn main() -> ExitCode {
    if let Err(msg) = run() {
        eprintln!("error: {msg}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
