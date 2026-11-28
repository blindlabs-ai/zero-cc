#![forbid(unsafe_code)]
use std::{collections::{BTreeMap, BTreeSet}, env, fs, path::{Path, PathBuf}};
use std::{process::{Command, ExitCode}, sync::atomic::{AtomicU64, Ordering}, time::Instant};
use zero_cc::rust_fe::{extract_and_compile_crate, extract_crate_kernels, kernel_reg_budget, parse_domain_spec,
    parse_rust_kernel_with_domains, ExtractedKernel};
use zero_cc::{bench_tile_blocks, build_bench_inputs, compile_bench_elf_exec_with_pf, compile_kernel_direct,
    compile_kernel_wasm, compile_kernels_wasm_with_memory, compile_standalone_elf_exec_with_pf,
    compile_tensor_to_elf, emit_kernel_elf_with_pf, eval_primary_output_checksum, parse_rust_tensor_kernel,
    resolve_cpu_profile, verify_elf64_executable, verify_elf64_object, verify_wasm_module, BufAccess, CpuProfile,
    Domain, Kernel, KernelCost, Op, RegBudget, TargetArch, TargetIsa, WasmSummary, X86VectorMode};
struct CliArgs {
    input: PathBuf, emit_elf: Option<PathBuf>, emit_exec: Option<PathBuf>, emit_wasm: Option<PathBuf>,
    wasm_import_memory: bool, target: TargetArch, kernel_filter: Option<String>,
    domains: BTreeMap<String, Domain>, bench: bool, bench_ram: bool, cpu: Option<String>, cpu_spec: Option<PathBuf>,
}
const USAGE: &str = "Usage: zero-cc <file_or_dir> [--emit-elf <out.o>] [--emit-exec <out_bin>] \
    [--emit-wasm <out.wasm|dir>] [--wasm-import-memory] [--target x86_64|aarch64|wasm32] \
    [--isa x86_64|aarch64|wasm32] [--cpu <profile|auto>] [--cpu-spec <path>] \
    [--kernel <name[,name...]>] [--domain <spec>] [--bench] [--bench-ram]";
const BENCH_WARMUP_ITERS: usize = 64;
const BENCH_TIMED_ITERS: usize = 2048;
const BENCH_RAM_WARMUP_ITERS: usize = 4;
const BENCH_RAM_TIMED_ITERS: usize = 64;
static BENCH_TMP_SEQ: AtomicU64 = AtomicU64::new(0);
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
        domains: BTreeMap::new(), bench: false, bench_ram: false, cpu: None, cpu_spec: None };
    while let Some(arg) = args.next() { match arg.as_str() {
        "-h" | "--help" => return Err(USAGE.into()),
        "--emit-elf" => cfg.emit_elf = Some(PathBuf::from(next_val(&mut args, "--emit-elf")?)),
        "--emit-exec" => cfg.emit_exec = Some(PathBuf::from(next_val(&mut args, "--emit-exec")?)),
        "--emit-wasm" => cfg.emit_wasm = Some(PathBuf::from(next_val(&mut args, "--emit-wasm")?)),
        "--wasm-import-memory" => cfg.wasm_import_memory = true,
        "--cpu" => cfg.cpu = Some(next_val(&mut args, "--cpu")?),
        "--cpu-spec" => cfg.cpu_spec = Some(PathBuf::from(next_val(&mut args, "--cpu-spec")?)),
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
        "--bench-ram" => cfg.bench_ram = true,
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
fn load_selected_many(cfg: &CliArgs) -> Result<Vec<(ExtractedKernel, KernelCost, KernelCost)>, String> {
    let (path, filter, domains) = (&cfg.input, cfg.kernel_filter.as_deref(), &cfg.domains);
    let direct = if path.is_file() && filter.is_none() {
        let src = fs::read_to_string(path).map_err(|e| format!("read '{}': {e}", path.display()))?;
        parse_rust_kernel_with_domains(&src, domains).ok()
            .map(|k| ExtractedKernel { file: path.display().to_string(), name: k.name.clone(), kernel: k })
    } else { None };
    let selected: Vec<ExtractedKernel> = if let Some(ek) = direct { vec![ek] } else {
        let (_, _, ks) = to_err(extract_crate_kernels(path, &domain_overrides(domains)))?;
        match filter {
            Some(f) => f.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|w| {
                ks.iter().find(|k| k.name == w).cloned()
                    .ok_or_else(|| format!("kernel '{w}' not found in '{}'", path.display()))
            }).collect::<Result<_, _>>()?,
            None => vec![ks.into_iter().next()
                .ok_or_else(|| format!("no pure kernels found in '{}'", path.display()))?],
        }
    };
    selected.into_iter().map(|ek| {
        let orig = ek.kernel.cost();
        let opt = to_err(zero_cc::optimize_kernel(&ek.kernel))?.cost();
        Ok((ek, orig, opt))
    }).collect()
}
#[cfg(unix)] fn set_exec_mode(p: &Path) -> Result<(), String> {
    fs::set_permissions(p, <fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o755))
        .map_err(|e| format!("chmod 0755 '{}': {e}", p.display()))
}
#[cfg(not(unix))] fn set_exec_mode(_: &Path) -> Result<(), String> { Ok(()) }
fn is_in_place_kernel(k: &Kernel) -> bool {
    k.nodes.iter().any(|op| {
        matches!(op, Op::Load { mem: 0, buf, .. }
            if k.bufs.get(*buf as usize).is_some_and(|b| b.access == BufAccess::MutNoAlias))
    })
}
fn run_tiled_bench(k: &Kernel, budget: RegBudget, cpu: &CpuProfile, tile_bytes: usize,
    warmup: usize, iters: usize, label: &str) -> Result<(), String> {
    let pf_bytes = cpu.prefetch_distance_bytes();
    let (tile_blocks, _) = bench_tile_blocks(k, tile_bytes);
    let inputs = build_bench_inputs(k, tile_blocks, 0x243f_6a88_85a3_08d3);
    let refs: Vec<&[i64]> = inputs.iter().map(Vec::as_slice).collect();
    let be = to_err(compile_bench_elf_exec_with_pf(k, budget, &refs, tile_bytes, warmup, iters, pf_bytes))?;
    let seq = BENCH_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let bin_path = env::temp_dir().join(format!("zero_cc_bench_{}_{}_{seq}", k.name, std::process::id()));
    write_out(&bin_path, &be.exec.elf_bytes)?;
    set_exec_mode(&bin_path)?;
    let out_res = Command::new(&bin_path).output()
        .map_err(|e| format!("exec bench '{}': {e}", bin_path.display()));
    let _ = fs::remove_file(&bin_path);
    let out = out_res?;
    if !out.status.success() {
        return Err(format!("bench '{}' exited with status {:?}", k.name, out.status.code()));
    }
    if out.stdout.len() != 24 {
        return Err(format!("bench '{}' returned {} bytes (expected 24)", k.name, out.stdout.len()));
    }
    let elapsed_ns = u64::from_le_bytes(out.stdout[0..8].try_into().unwrap_or([0; 8]));
    let total_blocks = u64::from_le_bytes(out.stdout[8..16].try_into().unwrap_or([0; 8]));
    let got_ck = u64::from_le_bytes(out.stdout[16..24].try_into().unwrap_or([0; 8]));
    if total_blocks != be.total_blocks {
        return Err(format!("bench '{}' block count mismatch: got {total_blocks}, expected {}",
            k.name, be.total_blocks));
    }
    let parity = if is_in_place_kernel(k) {
        "IN_PLACE"
    } else {
        let (exp_ck, _) = to_err(eval_primary_output_checksum(k, &inputs, tile_blocks))?;
        if got_ck != exp_ck {
            return Err(format!("bench '{}' checksum mismatch: got 0x{got_ck:016x}, expected 0x{exp_ck:016x}",
                k.name));
        }
        "EXACT"
    };
    let ns_per_blk = if total_blocks > 0 { (elapsed_ns as f64) / (total_blocks as f64) } else { 0.0 };
    let total_bytes = (total_blocks as f64) * (be.bytes_per_block as f64);
    let gb_per_s = if elapsed_ns > 0 { total_bytes / (elapsed_ns as f64) } else { 0.0 };
    println!("zero-cc {label} kernel={} cpu={} mode={} pf_bytes={} tile_bytes={} tile_blocks={} total_blocks={} \
        elapsed_ns={} ns/blk={:.3} GB/s={:.2} checksum=0x{:016x} parity={parity}",
        k.name, cpu.name, be.exec.vector_mode, pf_bytes, tile_bytes, be.tile_blocks, total_blocks,
        elapsed_ns, ns_per_blk, gb_per_s, got_ck);
    Ok(())
}
fn run_l1_bench(k: &Kernel, budget: RegBudget, cpu: &CpuProfile) -> Result<(), String> {
    run_tiled_bench(k, budget, cpu, cpu.l1_bench_tile_bytes(), BENCH_WARMUP_ITERS, BENCH_TIMED_ITERS, "bench")
}
fn run_ram_bench(k: &Kernel, budget: RegBudget, cpu: &CpuProfile) -> Result<(), String> {
    run_tiled_bench(k, budget, cpu, cpu.streaming_ram_tile_bytes(), BENCH_RAM_WARMUP_ITERS, BENCH_RAM_TIMED_ITERS, "bench-ram")
}
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
    let (cfg, t0) = (parse_args()?, Instant::now());
    let cpu = to_err(resolve_cpu_profile(cfg.cpu.as_deref(), cfg.cpu_spec.as_deref()))?;
    let budget = cpu.reg_budget();
    if cfg.target == TargetArch::Wasm32 && cfg.emit_elf.is_none() && cfg.emit_exec.is_none() {
        let multi = cfg.kernel_filter.as_deref().is_some_and(|f| f.contains(','));
        if !multi && (cfg.kernel_filter.is_some() || cfg.input.is_file()) {
            let (ek, o, opt) = load_selected_many(&cfg)?.into_iter().next()
                .ok_or_else(|| "no kernel selected".to_string())?;
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
        if cfg.input.is_file() && cfg.emit_exec.is_none() && isa == TargetIsa::X86_64 && cfg.kernel_filter.is_none() {
            if let Ok(src) = fs::read_to_string(&cfg.input) {
                if let Ok(tk) = parse_rust_tensor_kernel(&src, budget) {
                    let obj = to_err(compile_tensor_to_elf(&tk))?;
                    to_err(verify_elf64_object(&obj.elf_bytes))?;
                    if let Some(p) = &cfg.emit_elf { write_out(p, &obj.elf_bytes)?; }
                    println!("zero-cc tensor={} file={} cpu={} isa=X86_64_AVX512 mode=Avx512F32 tile={}x{} \
                        insns={} zero_spills={} rel_bytes={} syms={} compile_us={}", tk.name,
                        cfg.input.display(), cpu.name, tk.tile.mr, tk.tile.nr_wide, obj.direct_insns,
                        obj.zero_spills, obj.elf_bytes.len(), obj.symbols.join(","), t0.elapsed().as_micros());
                    return Ok(());
                }
            }
        }
        let selected = load_selected_many(&cfg)?;
        let mut fallback_8 = 0usize;
        let mut total_8 = 0usize;
        for (ek, o, opt) in &selected {
            let kb = kernel_reg_budget(&ek.kernel, budget);
            let pf = cpu.prefetch_distance_bytes();
            let obj = to_err(emit_kernel_elf_with_pf(&ek.kernel, kb, isa, true, false, pf))?;
            to_err(verify_elf64_object(&obj.elf_bytes))?;
            let exec = to_err(compile_standalone_elf_exec_with_pf(&ek.kernel, kb, isa, &[], pf))?;
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
            if isa == TargetIsa::X86_64 {
                let is8 = ek.kernel.bufs.iter().any(|b| b.lanes == 8);
                if is8 {
                    total_8 += 1;
                    if obj.vector_mode == X86VectorMode::ScalarFallback { fallback_8 += 1; }
                }
                println!("zero-cc kernel={} file={} isa={isa:?} cpu={} mode={} ops={}->{} divs={}->{} \
                    direct_insns={} rel_share={}% exec_share={}% zero_spills={} rel_bytes={} exec_bytes={} \
                    entry=0x{:x} compile_us={}", ek.name, ek.file, cpu.name, obj.vector_mode, o.total_nodes,
                    opt.total_nodes, o.divs_rems, opt.divs_rems, obj.direct_insns, obj.direct_share_pct,
                    exec.direct_share_pct, obj.zero_spills && exec.zero_spills, obj.elf_bytes.len(),
                    exec.elf_bytes.len(), esum.entry_vaddr, t0.elapsed().as_micros());
                if cfg.bench {
                    run_l1_bench(&ek.kernel, kb, &cpu)?;
                }
                if cfg.bench_ram {
                    run_ram_bench(&ek.kernel, kb, &cpu)?;
                }
            } else {
                println!("zero-cc kernel={} file={} isa={isa:?} ops={}->{} divs={}->{} direct_insns={} \
                    rel_share={}% exec_share={}% zero_spills={} rel_bytes={} exec_bytes={} entry=0x{:x} compile_us={}",
                    ek.name, ek.file, o.total_nodes, opt.total_nodes, o.divs_rems, opt.divs_rems, obj.direct_insns,
                    obj.direct_share_pct, exec.direct_share_pct, obj.zero_spills && exec.zero_spills,
                    obj.elf_bytes.len(), exec.elf_bytes.len(), esum.entry_vaddr, t0.elapsed().as_micros());
            }
        }
        if isa == TargetIsa::X86_64 && fallback_8 > 0 {
            eprintln!("warning: {fallback_8}/{total_8} 8-lane kernel(s) fell back to ScalarFallback");
        }
        return Ok(());
    }
    let r = to_err(extract_and_compile_crate(&cfg.input, isa, budget, &domain_overrides(&cfg.domains)))?;
    if isa == TargetIsa::X86_64 {
        println!("zero-cc scan root={} isa={isa:?} cpu={} files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
            r.root, cpu.name, r.files_scanned, r.functions_seen, r.kernels_extracted, r.compiled.len(), r.skipped,
            t0.elapsed().as_secs_f64() * 1000.0);
    } else {
        println!("zero-cc scan root={} isa={isa:?} files={} fns={} extracted={} compiled={} skipped={} elapsed_ms={:.2}",
            r.root, r.files_scanned, r.functions_seen, r.kernels_extracted, r.compiled.len(), r.skipped,
            t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (mut fallback_8, mut total_8) = (0usize, 0usize);
    for k in &r.compiled {
        let (o, p) = (&k.original_cost, &k.optimized_cost);
        if isa == TargetIsa::X86_64 {
            if k.max_lanes == 8 {
                total_8 += 1;
                if k.vector_mode == X86VectorMode::ScalarFallback { fallback_8 += 1; }
            }
            println!("  kernel={} file={} cpu={} mode={} ops={}->{} divs={}->{} insns={} rel={}B({}%) exec={}B({}%) spills={}",
                k.name, k.file, cpu.name, k.vector_mode, o.total_nodes, p.total_nodes, o.divs_rems, p.divs_rems,
                k.direct_insns, k.obj_bytes, k.direct_share_pct, k.exec_bytes, k.exec_share_pct,
                usize::from(!k.zero_spills));
        } else {
            println!("  kernel={} file={} ops={}->{} divs={}->{} insns={} rel={}B({}%) exec={}B({}%) spills={}",
                k.name, k.file, o.total_nodes, p.total_nodes, o.divs_rems, p.divs_rems, k.direct_insns,
                k.obj_bytes, k.direct_share_pct, k.exec_bytes, k.exec_share_pct, usize::from(!k.zero_spills));
        }
    }
    if isa == TargetIsa::X86_64 && fallback_8 > 0 {
        eprintln!("warning: {fallback_8}/{total_8} 8-lane kernel(s) fell back to ScalarFallback");
    }
    if (cfg.bench || cfg.bench_ram) && isa == TargetIsa::X86_64 {
        let (_, _, ks) = to_err(extract_crate_kernels(&cfg.input, &domain_overrides(&cfg.domains)))?;
        for ek in &ks {
            let kb = kernel_reg_budget(&ek.kernel, budget);
            if compile_kernel_direct(&ek.kernel, kb, isa).is_ok() {
                if cfg.bench { run_l1_bench(&ek.kernel, kb, &cpu)?; }
                if cfg.bench_ram { run_ram_bench(&ek.kernel, kb, &cpu)?; }
            }
        }
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
