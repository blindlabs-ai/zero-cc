#![forbid(unsafe_code)]
macro_rules! define_errors { ($name:ident, $(
    $v:ident $(($($tt:ty),*))? $({$($sf:ident: $st:ty),*})? | $pat:pat => $fmt:literal;
)*) => {
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum $name { $($v $(($($tt),*))? $({$($sf: $st),*})?,)* }
    impl std::fmt::Display for $name { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { $($pat => write!(f, $fmt),)* }
    }}
    impl std::error::Error for $name {}
};}
macro_rules! define_ops { ($f:ident, $(
    $v:ident $(($($tt:ty),*))? $({$($sf:ident: $st:ty),*})? | $pat:pat => $me:expr;
)*) => {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Op<Id> { $($v $(($($tt),*))? $({$($sf: $st),*})?,)* }
    impl<Id: Copy> Op<Id> {
        pub fn children(self) -> ([Id; 3], usize) where Id: Default {
            let (mut out, mut n) = ([Id::default(); 3], 0usize);
            let _ = self.map_ids(|id| if let Some(s) = out.get_mut(n) { (*s, n) = (id, n + 1) });
            (out, n)
        }
        pub fn map_ids<Out>(self, mut $f: impl FnMut(Id) -> Out) -> Op<Out> { match self { $($pat => $me,)* } }
    }
};}
pub mod avx512;
pub(crate) mod batch8;
pub mod codegen;
pub mod cpu_spec;
pub mod egraph;
pub mod elf64;
pub mod ir;
pub mod isa;
pub mod ntf1;
pub mod rust_fe;
pub mod tensor;
pub mod wasm32;
pub use crate::{
    avx512::{compile_tensor_to_elf, compile_uopc_sgemm_elf, x86_evex4, CompiledTensorObject},
    codegen::{emit_rust, schedule_and_verify, RegBudget, ScheduleReport},
    cpu_spec::{default_cpu_specs, detect_host_profile, find_cpu_profile, load_cpu_specs_from_path,
        overlay_sysfs_cache, parse_cpu_specs, parse_proc_cpuinfo, parse_sysfs_cache_size_kib,
        resolve_cpu_profile, select_profile_for_host, CpuProfile, HostCpuInfo, ModelRange, SimdTier,
        DEFAULT_AMD_ZEN_EPYC_SPEC},
    egraph::optimize_kernel,
    elf64::{elf64_exec_data_vaddr, emit_elf64_executable, emit_elf64_object, verify_elf64_executable,
        verify_elf64_object, ElfExecSummary, ElfReloc, ElfSummary, ElfSymbol, TargetIsa, ELF_EXEC_BASE,
        ELF_EXEC_TEXT_OFF, ELF_EXEC_TEXT_VADDR, EM_AARCH64, EM_X86_64, ET_EXEC, ET_REL},
    ir::{barrett_params, fuse_kernels, BufAccess, BufDecl, CmpKind, CompileError, Domain, ElemTy, IrOp, Kernel,
        KernelCost, NodeId, Op, OutputWrite, AVX512_F32_LANES, LANES},
    isa::{allocate_registers, bench_tile_blocks, build_bench_inputs, compile_bench_elf_exec,
        compile_bench_elf_exec_for_profile, compile_bench_elf_exec_with_pf, compile_kernel_direct,
        compile_kernel_direct_for_profile, compile_rust_to_elf, compile_rust_to_elf_exec,
        compile_standalone_elf_exec, compile_standalone_elf_exec_with_pf, emit_kernel_elf,
        emit_kernel_elf_with_pf, eval_primary_output_checksum, fnv1a64, BenchExec, CompiledObject, RegAlloc,
        StandaloneExec, X86VectorMode},
    ntf1::{emit_ntf1, export_kernel_ntf1, parse_ntf1, verify_ntf1_airlock, Ntf1Error, Ntf1Spec, Ntf1Stmt},
    rust_fe::{compile_rust_source, parse_rust_kernel, parse_rust_pipeline},
    tensor::{eval_gemm_epi_ref, eval_gemm_ref, export_tensor_ntf1, fast_exp_f32_ref, gelu_fwd_f32_ref,
        parse_rust_tensor_kernel, parse_rust_tensor_kernels, plan_gemm_tiles, plan_gemm_tiles_for_profile,
        EpilogueKind, GemmMat, TensorKernel, TilePlan, UOPC_EPI_ADD, UOPC_EPI_FMA, UOPC_EPI_GELU,
        UOPC_EPI_GELU_FWD, UOPC_EPI_NONE},
    wasm32::{compile_kernel_wasm, compile_kernel_wasm_opts, compile_kernels_wasm, compile_kernels_wasm_with_memory,
        compile_rust_to_wasm, verify_wasm_module, verify_wasm_module_with_opts, CompiledWasm, TargetArch, WasmExport,
        WasmSummary, WasmVectorMode, WASM_DEFAULT_MEM_PAGES, WASM_MAGIC_VERSION},
};
#[derive(Clone, Debug)] pub struct CompiledKernel {
    pub optimized: Kernel, pub original_cost: KernelCost, pub optimized_cost: KernelCost,
    pub schedule: ScheduleReport, pub rust_source: String,
}
pub fn compile_kernel(kernel: &Kernel, budget: RegBudget) -> Result<CompiledKernel, CompileError> {
    kernel.validate()?;
    let (original_cost, optimized) = (kernel.cost(), optimize_kernel(kernel)?);
    let (optimized_cost, schedule) = (optimized.cost(), schedule_and_verify(&optimized, budget)?);
    Ok(CompiledKernel { rust_source: emit_rust(&optimized, &schedule)?,
        optimized, original_cost, optimized_cost, schedule })
}
