#![forbid(unsafe_code)]

macro_rules! define_errors {
    ($name:ident; $( $v:ident $( ( $($tt:ty),* ) )? $( { $($sf:ident : $st:ty),* } )? | $pat:pat => $fmt:literal ; )*) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum $name { $( $v $( ( $($tt),* ) )? $( { $($sf : $st),* } )?, )* }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self { $( $pat => write!(f, $fmt), )* } }
        }
        impl std::error::Error for $name {}
    };
}

macro_rules! define_ops {
    ($f:ident; $( $v:ident $( ( $($tt:ty),* ) )? $( { $($sf:ident : $st:ty),* } )? | $pat:pat => $me:expr ; )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Op<Id> { $( $v $( ( $($tt),* ) )? $( { $($sf : $st),* } )?, )* }
        impl<Id: Copy> Op<Id> {
            pub fn children(self) -> ([Id; 3], usize) where Id: Default {
                let (mut out, mut n) = ([Id::default(); 3], 0usize);
                let _ = self.map_ids(|id| { out[n] = id; n += 1; }); (out, n)
            }
            pub fn map_ids<Out>(self, mut $f: impl FnMut(Id) -> Out) -> Op<Out> { match self { $( $pat => $me, )* } }
        }
    };
}

pub mod codegen;
pub mod egraph;
pub mod ir;
pub mod ntf1;
pub mod rust_fe;

pub use codegen::{emit_rust, schedule_and_verify, RegBudget, ScheduleReport};
pub use egraph::optimize_kernel;
pub use ir::{barrett_params, BufAccess, BufDecl, CmpKind, CompileError, Domain, IrOp, Kernel, KernelCost, NodeId, Op, OutputWrite, LANES};
pub use ntf1::{emit_ntf1, export_kernel_ntf1, parse_ntf1, verify_ntf1_airlock, Ntf1Error, Ntf1Spec, Ntf1Stmt};
pub use rust_fe::{compile_rust_source, parse_rust_kernel};

#[derive(Clone, Debug)]
pub struct CompiledKernel { pub optimized: Kernel, pub original_cost: KernelCost, pub optimized_cost: KernelCost, pub schedule: ScheduleReport, pub rust_source: String }

/// Full compilation pipeline: validate -> E-Graph -> zero-spill schedule -> Rust codegen.
pub fn compile_kernel(kernel: &Kernel, budget: RegBudget) -> Result<CompiledKernel, CompileError> {
    kernel.validate()?;
    let (original_cost, optimized) = (kernel.cost(), optimize_kernel(kernel)?);
    let (optimized_cost, schedule) = (optimized.cost(), schedule_and_verify(&optimized, budget)?);
    let rust_source = emit_rust(&optimized, &schedule)?;
    Ok(CompiledKernel { optimized, original_cost, optimized_cost, schedule, rust_source })
}
