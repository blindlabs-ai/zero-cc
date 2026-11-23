#![forbid(unsafe_code)]

pub mod codegen;
pub mod egraph;
pub mod ir;
pub mod ntf1;

pub use codegen::{emit_rust, schedule_and_verify, RegBudget, ScheduleReport};
pub use egraph::optimize_kernel;
pub use ir::{
    barrett_params, BufAccess, BufDecl, CompileError, Domain, IrOp, Kernel, KernelCost, NodeId,
    Op, OutputWrite, LANES,
};
pub use ntf1::{
    emit_ntf1, export_kernel_ntf1, parse_ntf1, verify_ntf1_airlock, Ntf1Error, Ntf1Spec, Ntf1Stmt,
};

#[derive(Clone, Debug)]
pub struct CompiledKernel {
    pub optimized: Kernel,
    pub original_cost: KernelCost,
    pub optimized_cost: KernelCost,
    pub schedule: ScheduleReport,
    pub rust_source: String,
}

/// Full compilation pipeline: validate -> E-Graph -> zero-spill schedule -> Rust codegen.
pub fn compile_kernel(kernel: &Kernel, budget: RegBudget) -> Result<CompiledKernel, CompileError> {
    kernel.validate()?;
    let original_cost = kernel.cost();
    let optimized = optimize_kernel(kernel)?;
    let optimized_cost = optimized.cost();
    let schedule = schedule_and_verify(&optimized, budget)?;
    let rust_source = emit_rust(&optimized, &schedule)?;
    Ok(CompiledKernel { optimized, original_cost, optimized_cost, schedule, rust_source })
}
