use crate::codegen::{schedule_and_verify, RegBudget, ScheduleReport};
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op, LANES};
use crate::optimize_kernel;
use crate::rust_fe::parse_rust_kernel;

pub const WASM_MAGIC_VERSION: [u8; 8] = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
pub const WASM_DEFAULT_MEM_PAGES: u32 = 4;

const SEC_TYPE: u8 = 1;
const SEC_IMPORT: u8 = 2;
const SEC_FUNC: u8 = 3;
const SEC_MEM: u8 = 5;
const SEC_EXPORT: u8 = 7;
const SEC_CODE: u8 = 10;

const VAL_I32: u8 = 0x7F;
const VAL_I64: u8 = 0x7E;
const VAL_V128: u8 = 0x7B;

const EXPORT_FUNC: u8 = 0x00;
const EXPORT_MEM: u8 = 0x02;

const SIMD_V128_LOAD: u32 = 0x00;
const SIMD_V128_LOAD16X4_S: u32 = 0x03;
const SIMD_V128_LOAD16X4_U: u32 = 0x04;
const SIMD_V128_LOAD64_SPLAT: u32 = 0x0A;
const SIMD_V128_STORE: u32 = 0x0B;
const SIMD_V128_CONST: u32 = 0x0C;
const SIMD_I64X2_SPLAT: u32 = 0x12;
const SIMD_I64X2_EXTRACT_LANE: u32 = 0x1D;
const SIMD_I64X2_REPLACE_LANE: u32 = 0x1E;
const SIMD_I32X4_EQ: u32 = 0x37;
const SIMD_I32X4_NE: u32 = 0x38;
const SIMD_I32X4_LT_S: u32 = 0x39;
const SIMD_I32X4_LT_U: u32 = 0x3A;
const SIMD_I32X4_GT_S: u32 = 0x3B;
const SIMD_I32X4_GT_U: u32 = 0x3C;
const SIMD_I32X4_LE_S: u32 = 0x3D;
const SIMD_I32X4_LE_U: u32 = 0x3E;
const SIMD_I32X4_GE_S: u32 = 0x3F;
const SIMD_I32X4_GE_U: u32 = 0x40;
const SIMD_V128_AND: u32 = 0x4E;
const SIMD_V128_OR: u32 = 0x50;
const SIMD_V128_XOR: u32 = 0x51;
const SIMD_V128_BITSELECT: u32 = 0x52;
const SIMD_V128_LOAD64_LANE: u32 = 0x57;
const SIMD_V128_STORE64_LANE: u32 = 0x5B;
const SIMD_I32X4_SHL: u32 = 0xAB;
const SIMD_I32X4_SHR_S: u32 = 0xAC;
const SIMD_I32X4_SHR_U: u32 = 0xAD;
const SIMD_I32X4_ADD: u32 = 0xAE;
const SIMD_I32X4_SUB: u32 = 0xB1;
const SIMD_I32X4_MUL: u32 = 0xB5;
const SIMD_I64X2_EXTEND_LOW_I32X4_S: u32 = 0xC7;
const SIMD_I64X2_EXTEND_HIGH_I32X4_S: u32 = 0xC8;
const SIMD_I64X2_EXTEND_LOW_I32X4_U: u32 = 0xC9;
const SIMD_I64X2_EXTEND_HIGH_I32X4_U: u32 = 0xCA;
const SIMD_I64X2_SHL: u32 = 0xCB;
const SIMD_I64X2_SHR_S: u32 = 0xCC;
const SIMD_I64X2_SHR_U: u32 = 0xCD;
const SIMD_I64X2_ADD: u32 = 0xCE;
const SIMD_I64X2_SUB: u32 = 0xD1;
const SIMD_I64X2_MUL: u32 = 0xD5;
const SIMD_I64X2_EQ: u32 = 0xD6;
const SIMD_I64X2_NE: u32 = 0xD7;
const SIMD_I64X2_LT_S: u32 = 0xD8;
const SIMD_I64X2_GT_S: u32 = 0xD9;
const SIMD_I64X2_LE_S: u32 = 0xDA;
const SIMD_I64X2_GE_S: u32 = 0xDB;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetArch {
    X86_64,
    Aarch64,
    Wasm32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmVectorMode {
    ScalarI64,
    HorizontalI32x4,
    VerticalI64x2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmExport {
    pub name: String,
    pub kind: u8,
    pub index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmSummary {
    pub type_count: usize,
    pub func_count: usize,
    pub mem_min_pages: u32,
    pub exports: Vec<WasmExport>,
    pub code_body_count: usize,
    pub total_insns: usize,
    pub simd_insns: usize,
    pub div_rem_insns: usize,
    pub has_simd128: bool,
    pub imports_memory: bool,
}

#[derive(Clone, Debug)]
pub struct CompiledWasm {
    pub kernel_sym: String,
    pub batch_sym: String,
    pub wasm_bytes: Vec<u8>,
    pub vector_mode: WasmVectorMode,
    pub summary: WasmSummary,
}

fn werr(msg: impl Into<String>) -> CompileError {
    CompileError::BackendError(msg.into())
}

fn push_uleb_u32(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let mut b = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn push_sleb_i32(out: &mut Vec<u8>, v: i32) {
    push_sleb_i64(out, i64::from(v));
}

fn push_sleb_i64(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        let done = (v == 0 && (b & 0x40) == 0) || (v == -1 && (b & 0x40) != 0);
        out.push(if done { b } else { b | 0x80 });
        if done {
            break;
        }
    }
}

fn push_name(out: &mut Vec<u8>, s: &str) {
    push_uleb_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn push_section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    push_uleb_u32(out, payload.len() as u32);
    out.extend_from_slice(payload);
}

#[derive(Default)]
struct WasmBuf {
    code: Vec<u8>,
}

impl WasmBuf {
    fn op(&mut self, b: u8) {
        self.code.push(b);
    }
    fn local_get(&mut self, idx: u32) {
        self.op(0x20);
        push_uleb_u32(&mut self.code, idx);
    }
    fn local_set(&mut self, idx: u32) {
        self.op(0x21);
        push_uleb_u32(&mut self.code, idx);
    }
    fn local_tee(&mut self, idx: u32) {
        self.op(0x22);
        push_uleb_u32(&mut self.code, idx);
    }
    fn i32_const(&mut self, v: i32) {
        self.op(0x41);
        push_sleb_i32(&mut self.code, v);
    }
    fn i64_const(&mut self, v: i64) {
        self.op(0x42);
        push_sleb_i64(&mut self.code, v);
    }
    fn memarg(&mut self, align_log2: u32, offset: u32) {
        push_uleb_u32(&mut self.code, align_log2);
        push_uleb_u32(&mut self.code, offset);
    }
    fn simd(&mut self, sub_op: u32) {
        self.op(0xFD);
        push_uleb_u32(&mut self.code, sub_op);
    }
    fn simd_mem(&mut self, sub_op: u32, align_log2: u32, offset: u32) {
        self.simd(sub_op);
        self.memarg(align_log2, offset);
    }
    fn v128_const_i64x2(&mut self, lo: i64, hi: i64) {
        self.simd(SIMD_V128_CONST);
        self.code.extend_from_slice(&lo.to_le_bytes());
        self.code.extend_from_slice(&hi.to_le_bytes());
    }
    fn v128_const_i32x4(&mut self, v: u32) {
        self.simd(SIMD_V128_CONST);
        let b = v.to_le_bytes();
        for _ in 0..4 {
            self.code.extend_from_slice(&b);
        }
    }
    fn load_elem_i64(&mut self, ty: ElemTy, offset: u32) {
        let opcode = match ty {
            ElemTy::I64 | ElemTy::U64 => 0x29,
            ElemTy::I32 => 0x34,
            ElemTy::I16 => 0x32,
            ElemTy::U32 => 0x35,
            ElemTy::U16 => 0x33,
            ElemTy::U8 => 0x31,
        };
        self.op(opcode);
        self.memarg(u32::from(ty.scale_shift()), offset);
    }
    fn store_elem_i64(&mut self, ty: ElemTy, offset: u32) {
        let opcode = match ty {
            ElemTy::I64 | ElemTy::U64 => 0x37,
            ElemTy::I32 | ElemTy::U32 => 0x3E,
            ElemTy::I16 | ElemTy::U16 => 0x3D,
            ElemTy::U8 => 0x3C,
        };
        self.op(opcode);
        self.memarg(u32::from(ty.scale_shift()), offset);
    }
}

fn is_inline_scalar(op: Op<NodeId>) -> bool {
    matches!(op, Op::MemInit | Op::Const(_) | Op::Store { .. })
}

fn alloc_ssa_locals(k: &Kernel, sched: &ScheduleReport, base_local: u32) -> Result<(Vec<Option<u32>>, u32), CompileError> {
    let n = k.nodes.len();
    let mut last_step = vec![0usize; n];
    for (step, &id) in sched.order.iter().enumerate() {
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        let (kids, cnt) = op.children();
        for &kid in kids.iter().take(cnt) {
            if let Some(ls) = last_step.get_mut(kid) {
                *ls = (*ls).max(step);
            }
        }
    }
    for &(ostep, oi) in &sched.output_order {
        if let Some(o) = k.outputs.get(oi) {
            if let Some(ls) = last_step.get_mut(o.val) {
                *ls = (*ls).max(ostep.saturating_add(1));
            }
        }
    }
    let mut node_loc: Vec<Option<u32>> = vec![None; n];
    let mut free: Vec<u32> = Vec::new();
    let mut active: Vec<(NodeId, u32)> = Vec::new();
    let mut next_slot = 0u32;
    for (step, &id) in sched.order.iter().enumerate() {
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if !is_inline_scalar(op) {
            let slot = match free.pop() {
                Some(s) => s,
                None => {
                    let s = next_slot;
                    next_slot = next_slot.saturating_add(1);
                    s
                }
            };
            if let Some(dst) = node_loc.get_mut(id) {
                *dst = Some(base_local.saturating_add(slot));
            }
            active.push((id, slot));
        }
        let mut keep = Vec::with_capacity(active.len());
        for (aid, slot) in active.drain(..) {
            if last_step.get(aid).copied().unwrap_or(0) <= step {
                free.push(slot);
            } else {
                keep.push((aid, slot));
            }
        }
        active = keep;
    }
    Ok((node_loc, next_slot))
}

fn push_scalar_val(e: &mut WasmBuf, k: &Kernel, locs: &[Option<u32>], id: NodeId) -> Result<(), CompileError> {
    match k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))? {
        Op::Const(c) => {
            e.i64_const(c);
            Ok(())
        }
        Op::MemInit => {
            e.i64_const(0);
            Ok(())
        }
        _ => {
            let loc = locs.get(id).and_then(|&v| v).ok_or(CompileError::InvalidNode(id))?;
            e.local_get(loc);
            Ok(())
        }
    }
}

fn emit_scalar_output(e: &mut WasmBuf, k: &Kernel, locs: &[Option<u32>], oi: usize) -> Result<(), CompileError> {
    let o = k.outputs.get(oi).ok_or(CompileError::InvalidNode(oi))?;
    let b = k.bufs.get(usize::from(o.buf)).ok_or(CompileError::InvalidBuffer(o.buf))?;
    e.local_get(u32::from(o.buf));
    push_scalar_val(e, k, locs, o.val)?;
    e.store_elem_i64(b.elem_ty, u32::from(o.lane) * u32::from(b.elem_ty.bytes()));
    Ok(())
}

fn emit_scalar_body(e: &mut WasmBuf, k: &Kernel, sched: &ScheduleReport, locs: &[Option<u32>]) -> Result<(), CompileError> {
    let mut out_idx = 0usize;
    for (step, &id) in sched.order.iter().enumerate() {
        while let Some(&(ostep, oi)) = sched.output_order.get(out_idx) {
            if ostep > step {
                break;
            }
            emit_scalar_output(e, k, locs, oi)?;
            out_idx += 1;
        }
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        match op {
            Op::MemInit | Op::Const(_) => {}
            Op::Store { buf, lane, val, .. } => {
                let b = k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
                e.local_get(u32::from(buf));
                push_scalar_val(e, k, locs, val)?;
                e.store_elem_i64(b.elem_ty, u32::from(lane) * u32::from(b.elem_ty.bytes()));
            }
            _ => {
                let dst = locs.get(id).and_then(|&v| v).ok_or(CompileError::InvalidNode(id))?;
                match op {
                    Op::Load { buf, lane, .. } => {
                        let b = k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
                        e.local_get(u32::from(buf));
                        e.load_elem_i64(b.elem_ty, u32::from(lane) * u32::from(b.elem_ty.bytes()));
                        e.local_set(dst);
                    }
                    Op::LoadLut { buf, idx } => {
                        let b = k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
                        e.local_get(u32::from(buf));
                        push_scalar_val(e, k, locs, idx)?;
                        e.op(0xA7);
                        let sh = b.elem_ty.scale_shift();
                        if sh > 0 {
                            e.i32_const(i32::from(sh));
                            e.op(0x74);
                        }
                        e.op(0x6A);
                        e.load_elem_i64(b.elem_ty, 0);
                        e.local_set(dst);
                    }
                    Op::BoundsCheck { idx, limit } => {
                        push_scalar_val(e, k, locs, idx)?;
                        e.local_tee(dst);
                        e.i64_const(limit);
                        e.op(0x5A);
                        e.op(0x04);
                        e.op(0x40);
                        e.op(0x00);
                        e.op(0x0B);
                    }
                    Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::Rem(a, b)
                    | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) | Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                        push_scalar_val(e, k, locs, a)?;
                        push_scalar_val(e, k, locs, b)?;
                        let opc = match op {
                            Op::Add(_, _) => 0x7C,
                            Op::Sub(_, _) => 0x7D,
                            Op::Mul(_, _) => 0x7E,
                            Op::Div(_, _) => 0x7F,
                            Op::Rem(_, _) => 0x81,
                            Op::BitAnd(_, _) => 0x83,
                            Op::BitOr(_, _) => 0x84,
                            Op::BitXor(_, _) => 0x85,
                            Op::ShlVar(_, _) => 0x86,
                            _ => 0x88,
                        };
                        e.op(opc);
                        e.local_set(dst);
                    }
                    Op::Shl(a, s) | Op::Shr(a, s) | Op::Ushr(a, s) => {
                        push_scalar_val(e, k, locs, a)?;
                        e.i64_const(i64::from(s));
                        let opc = match op {
                            Op::Shl(_, _) => 0x86,
                            Op::Shr(_, _) => 0x87,
                            _ => 0x88,
                        };
                        e.op(opc);
                        e.local_set(dst);
                    }
                    Op::Cmp(kind, a, b) => {
                        push_scalar_val(e, k, locs, a)?;
                        push_scalar_val(e, k, locs, b)?;
                        let opc = match kind {
                            CmpKind::Eq => 0x51,
                            CmpKind::Ne => 0x52,
                            CmpKind::Lt => 0x53,
                            CmpKind::Ult => 0x54,
                            CmpKind::Gt => 0x55,
                            CmpKind::Ugt => 0x56,
                            CmpKind::Le => 0x57,
                            CmpKind::Ule => 0x58,
                            CmpKind::Ge => 0x59,
                            CmpKind::Uge => 0x5A,
                        };
                        e.op(opc);
                        e.op(0xAD);
                        e.local_set(dst);
                    }
                    Op::Select { cond, t_val, f_val } => {
                        push_scalar_val(e, k, locs, f_val)?;
                        push_scalar_val(e, k, locs, t_val)?;
                        push_scalar_val(e, k, locs, cond)?;
                        e.op(0x50);
                        e.op(0x1B);
                        e.local_set(dst);
                    }
                    Op::MemInit | Op::Const(_) | Op::Store { .. } => {}
                }
            }
        }
    }
    while let Some(&(_, oi)) = sched.output_order.get(out_idx) {
        emit_scalar_output(e, k, locs, oi)?;
        out_idx += 1;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum I32x4Op {
    Load(u8, ElemTy),
    Const,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    BitAnd(usize, usize),
    Shl(usize, u8),
    Shr(usize, u8, bool),
    Cmp(CmpKind, usize, usize, bool),
    MaskTo01(usize),
    Select { cond_mask: usize, t_val: usize, f_val: usize },
}

struct I32x4Plan {
    steps: Vec<I32x4Op>,
    outputs: Vec<(u8, ElemTy, usize, bool)>,
    consts: Vec<(usize, u32)>,
}

struct I32x4Ctx<'a> {
    k: &'a Kernel,
    doms: &'a [Domain],
    needs_full: Vec<bool>,
    unsigned: bool,
    memo: Vec<([NodeId; LANES], usize)>,
    steps: Vec<I32x4Op>,
    is_mask: Vec<bool>,
    consts: Vec<(usize, u32)>,
}

impl<'a> I32x4Ctx<'a> {
    fn fits(&self, id: NodeId) -> bool {
        let Some(d) = self.doms.get(id).copied() else {
            return false;
        };
        if self.unsigned { d.fits_u32() } else { d.fits_i32() }
    }

    fn push_step(&mut self, op: I32x4Op, mask: bool) -> usize {
        let id = self.steps.len();
        self.steps.push(op);
        self.is_mask.push(mask);
        id
    }

    fn intern_const(&mut self, val: u32) -> usize {
        if let Some(&(id, _)) = self.consts.iter().find(|&&(_, c)| c == val) {
            return id;
        }
        let id = self.push_step(I32x4Op::Const, false);
        self.consts.push((id, val));
        id
    }

    fn ensure_val(&mut self, v: usize) -> usize {
        if self.is_mask.get(v).copied().unwrap_or(false) {
            self.push_step(I32x4Op::MaskTo01(v), false)
        } else {
            v
        }
    }

    fn ensure_mask(&mut self, v: usize) -> usize {
        if self.is_mask.get(v).copied().unwrap_or(false) {
            v
        } else {
            let z = self.intern_const(0);
            self.push_step(I32x4Op::Cmp(CmpKind::Ne, v, z, self.unsigned), true)
        }
    }

    fn match_lanes(&mut self, ids: [NodeId; LANES]) -> Option<usize> {
        if let Some(&(_, v)) = self.memo.iter().find(|&&(k, _)| k == ids) {
            return Some(v);
        }
        let any_full = ids.iter().any(|&id| self.needs_full.get(id).copied().unwrap_or(false));
        if any_full && !ids.iter().all(|&id| self.fits(id)) {
            return None;
        }
        let first = self.k.nodes.get(ids[0]).copied()?;
        let res = match first {
            Op::Const(c) => {
                if !ids.iter().all(|&id| self.k.nodes.get(id).copied() == Some(Op::Const(c))) {
                    return None;
                }
                self.intern_const(c as u32)
            }
            Op::Load { buf, .. } => {
                let b = self.k.bufs.get(usize::from(buf))?;
                if !matches!(b.elem_ty, ElemTy::I32 | ElemTy::U32 | ElemTy::I16 | ElemTy::U16) {
                    return None;
                }
                for (lane_idx, &id) in ids.iter().enumerate() {
                    match self.k.nodes.get(id).copied()? {
                        Op::Load { mem, buf: b2, lane }
                            if self.k.nodes.get(mem).copied() == Some(Op::MemInit)
                                && b2 == buf
                                && usize::from(lane) == lane_idx => {}
                        _ => return None,
                    }
                }
                self.push_step(I32x4Op::Load(buf, b.elem_ty), false)
            }
            Op::Add(_, _) | Op::Sub(_, _) | Op::Mul(_, _) | Op::BitAnd(_, _) => {
                let (mut la, mut lb) = ([0usize; LANES], [0usize; LANES]);
                for (i, &id) in ids.iter().enumerate() {
                    let (a, b) = match (first, self.k.nodes.get(id).copied()?) {
                        (Op::Add(_, _), Op::Add(a, b))
                        | (Op::Sub(_, _), Op::Sub(a, b))
                        | (Op::Mul(_, _), Op::Mul(a, b))
                        | (Op::BitAnd(_, _), Op::BitAnd(a, b)) => (a, b),
                        _ => return None,
                    };
                    if let (Some(sa), Some(sb)) = (la.get_mut(i), lb.get_mut(i)) {
                        *sa = a;
                        *sb = b;
                    }
                }
                let va = self.match_lanes(la)?;
                let vb = self.match_lanes(lb)?;
                let va = self.ensure_val(va);
                let vb = self.ensure_val(vb);
                let op = match first {
                    Op::Add(_, _) => I32x4Op::Add(va, vb),
                    Op::Sub(_, _) => I32x4Op::Sub(va, vb),
                    Op::Mul(_, _) => I32x4Op::Mul(va, vb),
                    _ => I32x4Op::BitAnd(va, vb),
                };
                self.push_step(op, false)
            }
            Op::Shl(_, s) | Op::Shr(_, s) => {
                if s >= 32 {
                    return None;
                }
                let is_shr = matches!(first, Op::Shr(_, _));
                let mut la = [0usize; LANES];
                for (i, &id) in ids.iter().enumerate() {
                    let a = match (is_shr, self.k.nodes.get(id).copied()?) {
                        (false, Op::Shl(a, s2)) if s2 == s => a,
                        (true, Op::Shr(a, s2)) if s2 == s => a,
                        _ => return None,
                    };
                    if is_shr && s > 0 && !self.fits(a) {
                        return None;
                    }
                    if let Some(sa) = la.get_mut(i) {
                        *sa = a;
                    }
                }
                let va = self.match_lanes(la)?;
                let va = self.ensure_val(va);
                let op = if is_shr { I32x4Op::Shr(va, s, self.unsigned) } else { I32x4Op::Shl(va, s) };
                self.push_step(op, false)
            }
            Op::Cmp(kind, _, _) => {
                if kind.is_unsigned() && !self.unsigned {
                    return None;
                }
                let (mut la, mut lb) = ([0usize; LANES], [0usize; LANES]);
                for (i, &id) in ids.iter().enumerate() {
                    let Op::Cmp(k2, a, b) = self.k.nodes.get(id).copied()? else {
                        return None;
                    };
                    if k2 != kind || !self.fits(a) || !self.fits(b) {
                        return None;
                    }
                    if let (Some(sa), Some(sb)) = (la.get_mut(i), lb.get_mut(i)) {
                        *sa = a;
                        *sb = b;
                    }
                }
                let va = self.match_lanes(la)?;
                let vb = self.match_lanes(lb)?;
                let va = self.ensure_val(va);
                let vb = self.ensure_val(vb);
                self.push_step(I32x4Op::Cmp(kind, va, vb, self.unsigned), true)
            }
            Op::Select { .. } => {
                let (mut lc, mut lt, mut lf) = ([0usize; LANES], [0usize; LANES], [0usize; LANES]);
                for (i, &id) in ids.iter().enumerate() {
                    let Op::Select { cond, t_val, f_val } = self.k.nodes.get(id).copied()? else {
                        return None;
                    };
                    if let (Some(sc), Some(st), Some(sf)) = (lc.get_mut(i), lt.get_mut(i), lf.get_mut(i)) {
                        *sc = cond;
                        *st = t_val;
                        *sf = f_val;
                    }
                }
                let vc = self.match_lanes(lc)?;
                let vt = self.match_lanes(lt)?;
                let vf = self.match_lanes(lf)?;
                let mc = self.ensure_mask(vc);
                let vt = self.ensure_val(vt);
                let vf = self.ensure_val(vf);
                self.push_step(I32x4Op::Select { cond_mask: mc, t_val: vt, f_val: vf }, false)
            }
            _ => return None,
        };
        self.memo.push((ids, res));
        Some(res)
    }
}

fn try_plan_i32x4_mode(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], unsigned: bool) -> Option<I32x4Plan> {
    if k.outputs.is_empty()
        || k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || usize::from(b.lanes) != LANES)
    {
        return None;
    }
    if sched.order.iter().any(|&id| {
        matches!(
            k.nodes.get(id),
            Some(Op::Store { .. } | Op::LoadLut { .. } | Op::BoundsCheck { .. } | Op::Div(_, _) | Op::Rem(_, _))
        ) || matches!(k.nodes.get(id), Some(&Op::Load { mem, .. }) if k.nodes.get(mem).copied() != Some(Op::MemInit))
    }) {
        return None;
    }
    let mut out_groups: Vec<(u8, ElemTy, [NodeId; LANES])> = Vec::new();
    for b in &k.bufs {
        if b.access == BufAccess::MutNoAlias {
            if !matches!(b.elem_ty, ElemTy::I32 | ElemTy::U32 | ElemTy::I64) {
                return None;
            }
            out_groups.push((b.id, b.elem_ty, [usize::MAX; LANES]));
        }
    }
    if out_groups.is_empty() {
        return None;
    }
    for o in &k.outputs {
        let g = out_groups.iter_mut().find(|g| g.0 == o.buf)?;
        let slot = g.2.get_mut(usize::from(o.lane))?;
        if *slot != usize::MAX {
            return None;
        }
        *slot = o.val;
    }
    if out_groups.iter().any(|g| g.2.contains(&usize::MAX)) {
        return None;
    }
    let mut needs_full = vec![false; k.nodes.len()];
    for g in &out_groups {
        if g.1 == ElemTy::I64 {
            for &v in &g.2 {
                if let Some(sl) = needs_full.get_mut(v) {
                    *sl = true;
                }
            }
        }
    }
    for &id in sched.order.iter().rev() {
        let Some(&op) = k.nodes.get(id) else {
            continue;
        };
        match op {
            Op::Cmp(_, a, b) => {
                if let Some(s) = needs_full.get_mut(a) { *s = true; }
                if let Some(s) = needs_full.get_mut(b) { *s = true; }
            }
            Op::Shr(a, s) if s > 0 => {
                if let Some(sl) = needs_full.get_mut(a) { *sl = true; }
            }
            Op::Select { cond, t_val, f_val } => {
                if let Some(sl) = needs_full.get_mut(cond) { *sl = true; }
                if needs_full.get(id).copied().unwrap_or(false) {
                    if let Some(sl) = needs_full.get_mut(t_val) { *sl = true; }
                    if let Some(sl) = needs_full.get_mut(f_val) { *sl = true; }
                }
            }
            _ => {}
        }
    }
    let mut ctx = I32x4Ctx {
        k,
        doms,
        needs_full,
        unsigned,
        memo: Vec::new(),
        steps: Vec::new(),
        is_mask: Vec::new(),
        consts: Vec::new(),
    };
    let mut outputs = Vec::with_capacity(out_groups.len());
    for &(buf, ty, lanes) in &out_groups {
        let v = ctx.match_lanes(lanes)?;
        let v = ctx.ensure_val(v);
        let signed_ext = if ty == ElemTy::I64 {
            let signed = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_i32));
            let unsig = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_u32));
            if !signed && !unsig {
                return None;
            }
            !unsig
        } else {
            false
        };
        outputs.push((buf, ty, v, signed_ext));
    }
    Some(I32x4Plan { steps: ctx.steps, outputs, consts: ctx.consts })
}

fn try_plan_i32x4(k: &Kernel, sched: &ScheduleReport, doms: &[Domain]) -> Option<I32x4Plan> {
    try_plan_i32x4_mode(k, sched, doms, true).or_else(|| try_plan_i32x4_mode(k, sched, doms, false))
}

fn emit_i32x4_consts(e: &mut WasmBuf, plan: &I32x4Plan, base_v: u32) {
    for &(sid, val) in &plan.consts {
        e.v128_const_i32x4(val);
        e.local_set(base_v + sid as u32);
    }
}

fn emit_i32x4_half(e: &mut WasmBuf, plan: &I32x4Plan, base_v: u32, half_idx: u32) {
    for (sid, &step) in plan.steps.iter().enumerate() {
        let dst = base_v + sid as u32;
        match step {
            I32x4Op::Const => {}
            I32x4Op::Load(buf, ty) => {
                e.local_get(u32::from(buf));
                match ty {
                    ElemTy::I32 | ElemTy::U32 => e.simd_mem(SIMD_V128_LOAD, 2, half_idx * 16),
                    ElemTy::U16 => e.simd_mem(SIMD_V128_LOAD16X4_U, 1, half_idx * 8),
                    ElemTy::I16 => e.simd_mem(SIMD_V128_LOAD16X4_S, 1, half_idx * 8),
                    _ => e.simd_mem(SIMD_V128_LOAD, 2, half_idx * 16),
                }
                e.local_set(dst);
            }
            I32x4Op::Add(a, b) | I32x4Op::Sub(a, b) | I32x4Op::Mul(a, b) | I32x4Op::BitAnd(a, b) => {
                e.local_get(base_v + a as u32);
                e.local_get(base_v + b as u32);
                let sub = match step {
                    I32x4Op::Add(_, _) => SIMD_I32X4_ADD,
                    I32x4Op::Sub(_, _) => SIMD_I32X4_SUB,
                    I32x4Op::Mul(_, _) => SIMD_I32X4_MUL,
                    _ => SIMD_V128_AND,
                };
                e.simd(sub);
                e.local_set(dst);
            }
            I32x4Op::Shl(a, s) => {
                e.local_get(base_v + a as u32);
                e.i32_const(i32::from(s));
                e.simd(SIMD_I32X4_SHL);
                e.local_set(dst);
            }
            I32x4Op::Shr(a, s, unsigned) => {
                e.local_get(base_v + a as u32);
                e.i32_const(i32::from(s));
                e.simd(if unsigned { SIMD_I32X4_SHR_U } else { SIMD_I32X4_SHR_S });
                e.local_set(dst);
            }
            I32x4Op::Cmp(kind, a, b, unsigned) => {
                e.local_get(base_v + a as u32);
                e.local_get(base_v + b as u32);
                let sub = match (kind, unsigned) {
                    (CmpKind::Eq, _) => SIMD_I32X4_EQ,
                    (CmpKind::Ne, _) => SIMD_I32X4_NE,
                    (CmpKind::Lt, false) => SIMD_I32X4_LT_S,
                    (CmpKind::Lt, true) | (CmpKind::Ult, _) => SIMD_I32X4_LT_U,
                    (CmpKind::Gt, false) => SIMD_I32X4_GT_S,
                    (CmpKind::Gt, true) | (CmpKind::Ugt, _) => SIMD_I32X4_GT_U,
                    (CmpKind::Le, false) => SIMD_I32X4_LE_S,
                    (CmpKind::Le, true) | (CmpKind::Ule, _) => SIMD_I32X4_LE_U,
                    (CmpKind::Ge, false) => SIMD_I32X4_GE_S,
                    (CmpKind::Ge, true) | (CmpKind::Uge, _) => SIMD_I32X4_GE_U,
                };
                e.simd(sub);
                e.local_set(dst);
            }
            I32x4Op::MaskTo01(m) => {
                e.local_get(base_v + m as u32);
                e.v128_const_i32x4(1);
                e.simd(SIMD_V128_AND);
                e.local_set(dst);
            }
            I32x4Op::Select { cond_mask, t_val, f_val } => {
                e.local_get(base_v + t_val as u32);
                e.local_get(base_v + f_val as u32);
                e.local_get(base_v + cond_mask as u32);
                e.simd(SIMD_V128_BITSELECT);
                e.local_set(dst);
            }
        }
    }
    for &(buf, ty, src, signed_ext) in &plan.outputs {
        if ty == ElemTy::I64 {
            let (ext_lo, ext_hi) = if signed_ext {
                (SIMD_I64X2_EXTEND_LOW_I32X4_S, SIMD_I64X2_EXTEND_HIGH_I32X4_S)
            } else {
                (SIMD_I64X2_EXTEND_LOW_I32X4_U, SIMD_I64X2_EXTEND_HIGH_I32X4_U)
            };
            e.local_get(u32::from(buf));
            e.local_get(base_v + src as u32);
            e.simd(ext_lo);
            e.simd_mem(SIMD_V128_STORE, 3, half_idx * 32);
            e.local_get(u32::from(buf));
            e.local_get(base_v + src as u32);
            e.simd(ext_hi);
            e.simd_mem(SIMD_V128_STORE, 3, half_idx * 32 + 16);
        } else {
            e.local_get(u32::from(buf));
            e.local_get(base_v + src as u32);
            e.simd_mem(SIMD_V128_STORE, 2, half_idx * 16);
        }
    }
}

fn can_plan_i64x2(k: &Kernel, sched: &ScheduleReport) -> bool {
    if k.outputs.is_empty()
        || k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || b.lanes == 0)
    {
        return false;
    }
    sched.order.iter().all(|&id| {
        match k.nodes.get(id) {
            Some(
                Op::MemInit
                | Op::Const(_)
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Add(_, _)
                | Op::Sub(_, _)
                | Op::Mul(_, _)
                | Op::Shl(_, _)
                | Op::Shr(_, _)
                | Op::Ushr(_, _)
                | Op::BitAnd(_, _)
                | Op::BitOr(_, _)
                | Op::BitXor(_, _)
                | Op::Select { .. },
            ) => true,
            Some(Op::Cmp(kind, _, _)) => !kind.is_unsigned(),
            _ => false,
        }
    })
}

fn collect_hoisted_consts_i64(k: &Kernel, sched: &ScheduleReport) -> Vec<i64> {
    let mut counts: Vec<(i64, usize)> = Vec::new();
    let mut bump = |c: i64| {
        if let Some(entry) = counts.iter_mut().find(|e| e.0 == c) {
            entry.1 += 1;
        } else {
            counts.push((c, 1));
        }
    };
    for &id in &sched.order {
        if let Some(&op) = k.nodes.get(id) {
            let (kids, cnt) = op.children();
            for &kid in kids.iter().take(cnt) {
                if let Some(Op::Const(c)) = k.nodes.get(kid).copied() {
                    bump(c);
                }
            }
        }
    }
    for o in &k.outputs {
        if let Some(Op::Const(c)) = k.nodes.get(o.val).copied() {
            bump(c);
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts.into_iter().filter(|&(_, cnt)| cnt >= 2).take(16).map(|(c, _)| c).collect()
}

fn push_i64x2_const(e: &mut WasmBuf, hoisted: &[(i64, u32)], c: i64) {
    if let Some(&(_, loc)) = hoisted.iter().find(|&&(v, _)| v == c) {
        e.local_get(loc);
    } else {
        e.v128_const_i64x2(c, c);
    }
}

fn push_i64x2_val(
    e: &mut WasmBuf,
    k: &Kernel,
    locs: &[Option<u32>],
    is_mask: &[bool],
    hoisted: &[(i64, u32)],
    id: NodeId,
) -> Result<(), CompileError> {
    match k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))? {
        Op::Const(c) => {
            push_i64x2_const(e, hoisted, c);
            Ok(())
        }
        Op::MemInit => {
            push_i64x2_const(e, hoisted, 0);
            Ok(())
        }
        _ => {
            let loc = locs.get(id).and_then(|&v| v).ok_or(CompileError::InvalidNode(id))?;
            e.local_get(loc);
            if is_mask.get(id).copied().unwrap_or(false) {
                push_i64x2_const(e, hoisted, 1);
                e.simd(SIMD_V128_AND);
            }
            Ok(())
        }
    }
}

fn emit_i64x2_store_pair(
    e: &mut WasmBuf,
    k: &Kernel,
    locs: &[Option<u32>],
    is_mask: &[bool],
    hoisted: &[(i64, u32)],
    scratch_v: u32,
    target: (u8, u8, NodeId),
) -> Result<(), CompileError> {
    let (buf, lane, val) = target;
    let b = k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
    let off0 = u32::from(lane) * u32::from(b.elem_ty.bytes());
    let off1 = (b.block_bytes() as u32).saturating_add(off0);
    push_i64x2_val(e, k, locs, is_mask, hoisted, val)?;
    e.local_set(scratch_v);
    if matches!(b.elem_ty, ElemTy::I64 | ElemTy::U64) {
        e.local_get(u32::from(buf));
        e.local_get(scratch_v);
        e.simd_mem(SIMD_V128_STORE64_LANE, 3, off0);
        e.op(0);
        e.local_get(u32::from(buf));
        e.local_get(scratch_v);
        e.simd_mem(SIMD_V128_STORE64_LANE, 3, off1);
        e.op(1);
    } else {
        e.local_get(u32::from(buf));
        e.local_get(scratch_v);
        e.simd(SIMD_I64X2_EXTRACT_LANE);
        e.op(0);
        e.store_elem_i64(b.elem_ty, off0);
        e.local_get(u32::from(buf));
        e.local_get(scratch_v);
        e.simd(SIMD_I64X2_EXTRACT_LANE);
        e.op(1);
        e.store_elem_i64(b.elem_ty, off1);
    }
    Ok(())
}

fn emit_i64x2_body(
    e: &mut WasmBuf,
    k: &Kernel,
    sched: &ScheduleReport,
    locs: &[Option<u32>],
    hoisted: &[(i64, u32)],
    scratch_v: u32,
) -> Result<(), CompileError> {
    let mut is_mask = vec![false; k.nodes.len()];
    let mut out_idx = 0usize;
    for (step, &id) in sched.order.iter().enumerate() {
        while let Some(&(ostep, oi)) = sched.output_order.get(out_idx) {
            if ostep > step {
                break;
            }
            let o = k.outputs.get(oi).ok_or(CompileError::InvalidNode(oi))?;
            emit_i64x2_store_pair(e, k, locs, &is_mask, hoisted, scratch_v, (o.buf, o.lane, o.val))?;
            out_idx += 1;
        }
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        match op {
            Op::MemInit | Op::Const(_) => {}
            Op::Store { buf, lane, val, .. } => {
                emit_i64x2_store_pair(e, k, locs, &is_mask, hoisted, scratch_v, (buf, lane, val))?;
            }
            _ => {
                let dst = locs.get(id).and_then(|&v| v).ok_or(CompileError::InvalidNode(id))?;
                match op {
                    Op::Load { buf, lane, .. } => {
                        let b = k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
                        let off0 = u32::from(lane) * u32::from(b.elem_ty.bytes());
                        let off1 = (b.block_bytes() as u32).saturating_add(off0);
                        if matches!(b.elem_ty, ElemTy::I64 | ElemTy::U64) {
                            e.local_get(u32::from(buf));
                            e.local_get(u32::from(buf));
                            e.simd_mem(SIMD_V128_LOAD64_SPLAT, 3, off0);
                            e.simd_mem(SIMD_V128_LOAD64_LANE, 3, off1);
                            e.op(1);
                        } else {
                            e.local_get(u32::from(buf));
                            e.load_elem_i64(b.elem_ty, off0);
                            e.simd(SIMD_I64X2_SPLAT);
                            e.local_get(u32::from(buf));
                            e.load_elem_i64(b.elem_ty, off1);
                            e.simd(SIMD_I64X2_REPLACE_LANE);
                            e.op(1);
                        }
                        e.local_set(dst);
                    }
                    Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => {
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, a)?;
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, b)?;
                        let sub = match op {
                            Op::Add(_, _) => SIMD_I64X2_ADD,
                            Op::Sub(_, _) => SIMD_I64X2_SUB,
                            Op::Mul(_, _) => SIMD_I64X2_MUL,
                            Op::BitAnd(_, _) => SIMD_V128_AND,
                            Op::BitOr(_, _) => SIMD_V128_OR,
                            _ => SIMD_V128_XOR,
                        };
                        e.simd(sub);
                        e.local_set(dst);
                    }
                    Op::Shl(a, s) | Op::Shr(a, s) | Op::Ushr(a, s) => {
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, a)?;
                        e.i32_const(i32::from(s));
                        let sub = match op {
                            Op::Shl(_, _) => SIMD_I64X2_SHL,
                            Op::Shr(_, _) => SIMD_I64X2_SHR_S,
                            _ => SIMD_I64X2_SHR_U,
                        };
                        e.simd(sub);
                        e.local_set(dst);
                    }
                    Op::Cmp(kind, a, b) => {
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, a)?;
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, b)?;
                        let sub = match kind {
                            CmpKind::Eq => SIMD_I64X2_EQ,
                            CmpKind::Ne => SIMD_I64X2_NE,
                            CmpKind::Lt => SIMD_I64X2_LT_S,
                            CmpKind::Gt => SIMD_I64X2_GT_S,
                            CmpKind::Le => SIMD_I64X2_LE_S,
                            CmpKind::Ge => SIMD_I64X2_GE_S,
                            _ => return Err(werr("unsigned i64x2 cmp unsupported")),
                        };
                        e.simd(sub);
                        e.local_set(dst);
                        if let Some(m) = is_mask.get_mut(id) {
                            *m = true;
                        }
                    }
                    Op::Select { cond, t_val, f_val } => {
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, t_val)?;
                        push_i64x2_val(e, k, locs, &is_mask, hoisted, f_val)?;
                        if is_mask.get(cond).copied().unwrap_or(false) {
                            let cloc = locs.get(cond).and_then(|&v| v).ok_or(CompileError::InvalidNode(cond))?;
                            e.local_get(cloc);
                        } else {
                            push_i64x2_val(e, k, locs, &is_mask, hoisted, cond)?;
                            push_i64x2_const(e, hoisted, 0);
                            e.simd(SIMD_I64X2_NE);
                        }
                        e.simd(SIMD_V128_BITSELECT);
                        e.local_set(dst);
                    }
                    _ => return Err(werr("unsupported op in vertical i64x2 SIMD128 lowering")),
                }
            }
        }
    }
    while let Some(&(_, oi)) = sched.output_order.get(out_idx) {
        let o = k.outputs.get(oi).ok_or(CompileError::InvalidNode(oi))?;
        emit_i64x2_store_pair(e, k, locs, &is_mask, hoisted, scratch_v, (o.buf, o.lane, o.val))?;
        out_idx += 1;
    }
    Ok(())
}

fn finish_func_body(local_groups: &[(u32, u8)], code: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(code.len() + 16);
    let non_empty: Vec<(u32, u8)> = local_groups.iter().copied().filter(|&(cnt, _)| cnt > 0).collect();
    push_uleb_u32(&mut body, non_empty.len() as u32);
    for (cnt, ty) in non_empty {
        push_uleb_u32(&mut body, cnt);
        body.push(ty);
    }
    body.extend_from_slice(code);
    body.push(0x0B);
    body
}

fn emit_ptr_advance(e: &mut WasmBuf, k: &Kernel, mult: usize) {
    for (idx, b) in k.bufs.iter().enumerate() {
        if !matches!(b.access, BufAccess::LookupTable(_)) {
            let step = b.block_bytes().saturating_mul(mult) as i32;
            e.local_get(idx as u32);
            e.i32_const(step);
            e.op(0x6A);
            e.local_set(idx as u32);
        }
    }
}

struct LoweredKernel {
    kernel_sym: String,
    batch_sym: String,
    nbufs: usize,
    single_body: Vec<u8>,
    batch_body: Vec<u8>,
    vector_mode: WasmVectorMode,
}

fn lower_kernel_pair(
    k: &Kernel,
    sched: &ScheduleReport,
    doms: &[Domain],
    single_func_idx: u32,
) -> Result<LoweredKernel, CompileError> {
    let nb = k.bufs.len();
    let kernel_sym = format!("zero_cc_{}", k.name);
    let batch_sym = format!("zero_cc_{}_batch", k.name);

    if let Some(plan) = try_plan_i32x4(k, sched, doms) {
        let nv = plan.steps.len() as u32;
        let mut s_buf = WasmBuf::default();
        emit_i32x4_consts(&mut s_buf, &plan, nb as u32);
        emit_i32x4_half(&mut s_buf, &plan, nb as u32, 0);
        emit_i32x4_half(&mut s_buf, &plan, nb as u32, 1);
        let single_body = finish_func_body(&[(nv, VAL_V128)], &s_buf.code);

        let n_loc = nb as u32;
        let base_v = n_loc + 1;
        let mut b_buf = WasmBuf::default();
        emit_i32x4_consts(&mut b_buf, &plan, base_v);
        b_buf.op(0x02);
        b_buf.op(0x40);
        b_buf.op(0x03);
        b_buf.op(0x40);
        b_buf.local_get(n_loc);
        b_buf.i32_const(0);
        b_buf.op(0x4C);
        b_buf.op(0x0D);
        push_uleb_u32(&mut b_buf.code, 1);
        emit_i32x4_half(&mut b_buf, &plan, base_v, 0);
        emit_i32x4_half(&mut b_buf, &plan, base_v, 1);
        emit_ptr_advance(&mut b_buf, k, 1);
        b_buf.local_get(n_loc);
        b_buf.i32_const(1);
        b_buf.op(0x6B);
        b_buf.local_set(n_loc);
        b_buf.op(0x0C);
        push_uleb_u32(&mut b_buf.code, 0);
        b_buf.op(0x0B);
        b_buf.op(0x0B);
        let batch_body = finish_func_body(&[(nv, VAL_V128)], &b_buf.code);
        return Ok(LoweredKernel {
            kernel_sym,
            batch_sym,
            nbufs: nb,
            single_body,
            batch_body,
            vector_mode: WasmVectorMode::HorizontalI32x4,
        });
    }

    let (s_locs, s_ni64) = alloc_ssa_locals(k, sched, nb as u32)?;
    let mut s_buf = WasmBuf::default();
    emit_scalar_body(&mut s_buf, k, sched, &s_locs)?;
    let single_body = finish_func_body(&[(s_ni64, VAL_I64)], &s_buf.code);

    let n_loc = nb as u32;
    if can_plan_i64x2(k, sched) {
        let base_v = n_loc + 1;
        let (v_locs, v_slots) = alloc_ssa_locals(k, sched, base_v)?;
        let hoisted_vals = collect_hoisted_consts_i64(k, sched);
        let hoist_base = base_v + v_slots;
        let hoisted: Vec<(i64, u32)> = hoisted_vals
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, hoist_base + i as u32))
            .collect();
        let scratch_v = hoist_base + hoisted.len() as u32;
        let total_v = v_slots + hoisted.len() as u32 + 1;

        let mut b_buf = WasmBuf::default();
        for &(c, loc) in &hoisted {
            b_buf.v128_const_i64x2(c, c);
            b_buf.local_set(loc);
        }
        b_buf.op(0x02);
        b_buf.op(0x40);
        b_buf.op(0x03);
        b_buf.op(0x40);
        b_buf.local_get(n_loc);
        b_buf.i32_const(2);
        b_buf.op(0x48);
        b_buf.op(0x0D);
        push_uleb_u32(&mut b_buf.code, 1);
        emit_i64x2_body(&mut b_buf, k, sched, &v_locs, &hoisted, scratch_v)?;
        emit_ptr_advance(&mut b_buf, k, 2);
        b_buf.local_get(n_loc);
        b_buf.i32_const(2);
        b_buf.op(0x6B);
        b_buf.local_set(n_loc);
        b_buf.op(0x0C);
        push_uleb_u32(&mut b_buf.code, 0);
        b_buf.op(0x0B);
        b_buf.op(0x0B);

        b_buf.local_get(n_loc);
        b_buf.i32_const(0);
        b_buf.op(0x4A);
        b_buf.op(0x04);
        b_buf.op(0x40);
        for idx in 0..nb {
            b_buf.local_get(idx as u32);
        }
        b_buf.op(0x10);
        push_uleb_u32(&mut b_buf.code, single_func_idx);
        b_buf.op(0x0B);
        let batch_body = finish_func_body(&[(total_v, VAL_V128)], &b_buf.code);
        return Ok(LoweredKernel {
            kernel_sym,
            batch_sym,
            nbufs: nb,
            single_body,
            batch_body,
            vector_mode: WasmVectorMode::VerticalI64x2,
        });
    }

    let (b_locs, b_ni64) = alloc_ssa_locals(k, sched, n_loc + 1)?;
    let mut b_buf = WasmBuf::default();
    b_buf.op(0x02);
    b_buf.op(0x40);
    b_buf.op(0x03);
    b_buf.op(0x40);
    b_buf.local_get(n_loc);
    b_buf.i32_const(0);
    b_buf.op(0x4C);
    b_buf.op(0x0D);
    push_uleb_u32(&mut b_buf.code, 1);
    emit_scalar_body(&mut b_buf, k, sched, &b_locs)?;
    emit_ptr_advance(&mut b_buf, k, 1);
    b_buf.local_get(n_loc);
    b_buf.i32_const(1);
    b_buf.op(0x6B);
    b_buf.local_set(n_loc);
    b_buf.op(0x0C);
    push_uleb_u32(&mut b_buf.code, 0);
    b_buf.op(0x0B);
    b_buf.op(0x0B);
    let batch_body = finish_func_body(&[(b_ni64, VAL_I64)], &b_buf.code);
    Ok(LoweredKernel {
        kernel_sym,
        batch_sym,
        nbufs: nb,
        single_body,
        batch_body,
        vector_mode: WasmVectorMode::ScalarI64,
    })
}

fn assemble_wasm_module(lowered: &[LoweredKernel], mem_pages: u32, import_memory: bool) -> Vec<u8> {
    let mut type_sec = Vec::new();
    let mut func_sec = Vec::new();
    let mut exp_sec = Vec::new();
    let mut code_sec = Vec::new();

    let fn_count = (lowered.len() * 2) as u32;
    push_uleb_u32(&mut type_sec, fn_count);
    push_uleb_u32(&mut func_sec, fn_count);
    push_uleb_u32(&mut exp_sec, if import_memory { fn_count } else { fn_count + 1 });
    if !import_memory {
        push_name(&mut exp_sec, "memory");
        exp_sec.push(EXPORT_MEM);
        push_uleb_u32(&mut exp_sec, 0);
    }
    push_uleb_u32(&mut code_sec, fn_count);

    for (idx, lk) in lowered.iter().enumerate() {
        let f_single = (idx * 2) as u32;
        let f_batch = f_single + 1;

        type_sec.push(0x60);
        push_uleb_u32(&mut type_sec, lk.nbufs as u32);
        type_sec.resize(type_sec.len() + lk.nbufs, VAL_I32);
        type_sec.push(0x00);

        type_sec.push(0x60);
        push_uleb_u32(&mut type_sec, (lk.nbufs + 1) as u32);
        type_sec.resize(type_sec.len() + lk.nbufs + 1, VAL_I32);
        type_sec.push(0x00);

        push_uleb_u32(&mut func_sec, f_single);
        push_uleb_u32(&mut func_sec, f_batch);

        push_name(&mut exp_sec, &lk.kernel_sym);
        exp_sec.push(EXPORT_FUNC);
        push_uleb_u32(&mut exp_sec, f_single);

        push_name(&mut exp_sec, &lk.batch_sym);
        exp_sec.push(EXPORT_FUNC);
        push_uleb_u32(&mut exp_sec, f_batch);

        push_uleb_u32(&mut code_sec, lk.single_body.len() as u32);
        code_sec.extend_from_slice(&lk.single_body);
        push_uleb_u32(&mut code_sec, lk.batch_body.len() as u32);
        code_sec.extend_from_slice(&lk.batch_body);
    }

    let mut out = Vec::with_capacity(64 + type_sec.len() + func_sec.len() + exp_sec.len() + code_sec.len());
    out.extend_from_slice(&WASM_MAGIC_VERSION);
    push_section(&mut out, SEC_TYPE, &type_sec);
    if import_memory {
        let mut imp_sec = Vec::new();
        push_uleb_u32(&mut imp_sec, 1);
        push_name(&mut imp_sec, "env");
        push_name(&mut imp_sec, "memory");
        imp_sec.push(EXPORT_MEM);
        imp_sec.push(0x00);
        push_uleb_u32(&mut imp_sec, mem_pages.max(1));
        push_section(&mut out, SEC_IMPORT, &imp_sec);
        push_section(&mut out, SEC_FUNC, &func_sec);
    } else {
        let mut mem_sec = Vec::new();
        push_uleb_u32(&mut mem_sec, 1);
        mem_sec.push(0x00);
        push_uleb_u32(&mut mem_sec, mem_pages);
        push_section(&mut out, SEC_FUNC, &func_sec);
        push_section(&mut out, SEC_MEM, &mem_sec);
    }
    push_section(&mut out, SEC_EXPORT, &exp_sec);
    push_section(&mut out, SEC_CODE, &code_sec);
    out
}

/// Compile a single `Kernel` directly to a standalone 0-import `WASM32` (+`SIMD128`) binary module.
pub fn compile_kernel_wasm_opts(kernel: &Kernel, budget: RegBudget, optimize: bool) -> Result<CompiledWasm, CompileError> {
    kernel.validate()?;
    let opt = if optimize { optimize_kernel(kernel)? } else { kernel.clone() };
    let sched = schedule_and_verify(&opt, budget)?;
    let doms = opt.infer_domains();
    let lk = lower_kernel_pair(&opt, &sched, &doms, 0)?;
    let kernel_sym = lk.kernel_sym.clone();
    let batch_sym = lk.batch_sym.clone();
    let vector_mode = lk.vector_mode;
    let wasm_bytes = assemble_wasm_module(&[lk], WASM_DEFAULT_MEM_PAGES, false);
    let summary = verify_wasm_module(&wasm_bytes)?;
    Ok(CompiledWasm { kernel_sym, batch_sym, wasm_bytes, vector_mode, summary })
}

/// Compile a single `Kernel` (with full e-graph optimization) directly to standalone `WASM32` (+`SIMD128`).
pub fn compile_kernel_wasm(kernel: &Kernel, budget: RegBudget) -> Result<CompiledWasm, CompileError> {
    compile_kernel_wasm_opts(kernel, budget, true)
}

/// Parse safe Rust kernel source and compile directly to a standalone `WASM32` (+`SIMD128`) binary module.
pub fn compile_rust_to_wasm(src: &str, budget: RegBudget) -> Result<CompiledWasm, CompileError> {
    let kernel = parse_rust_kernel(src)?;
    compile_kernel_wasm(&kernel, budget)
}

/// Compile multiple kernels into a single `WASM32` (+`SIMD128`) binary module, either owning
/// and exporting its own memory (`import_memory == false`) or importing `("env", "memory")`
/// (`import_memory == true`) for zero-copy shared-memory linking with a host WASM module.
pub fn compile_kernels_wasm_with_memory(
    kernels: &[Kernel],
    budget: RegBudget,
    import_memory: bool,
) -> Result<(Vec<u8>, WasmSummary), CompileError> {
    if kernels.is_empty() {
        return Err(werr("cannot emit wasm module with zero kernels"));
    }
    let mut lowered = Vec::with_capacity(kernels.len());
    for (idx, k) in kernels.iter().enumerate() {
        k.validate()?;
        let opt = optimize_kernel(k)?;
        let sched = schedule_and_verify(&opt, budget)?;
        let doms = opt.infer_domains();
        lowered.push(lower_kernel_pair(&opt, &sched, &doms, (idx * 2) as u32)?);
    }
    let pages = if import_memory { 1 } else { WASM_DEFAULT_MEM_PAGES };
    let wasm_bytes = assemble_wasm_module(&lowered, pages, import_memory);
    let summary = verify_wasm_module(&wasm_bytes)?;
    Ok((wasm_bytes, summary))
}

/// Compile multiple kernels into a single standalone 0-import `WASM32` (+`SIMD128`) binary module.
pub fn compile_kernels_wasm(kernels: &[Kernel], budget: RegBudget) -> Result<(Vec<u8>, WasmSummary), CompileError> {
    compile_kernels_wasm_with_memory(kernels, budget, false)
}

fn read_byte(bytes: &[u8], pos: &mut usize) -> Result<u8, CompileError> {
    let b = bytes.get(*pos).copied().ok_or_else(|| werr("wasm unexpected EOF"))?;
    *pos += 1;
    Ok(b)
}

fn read_uleb_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, CompileError> {
    let mut val = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let b = read_byte(bytes, pos)?;
        let low = u32::from(b & 0x7F);
        if shift == 28 && (b & 0xF0) != 0 {
            return Err(werr("wasm uleb32 overflow"));
        }
        val |= low << shift;
        if (b & 0x80) == 0 {
            return Ok(val);
        }
        shift += 7;
    }
    Err(werr("wasm uleb32 exceeds 5 bytes"))
}

fn skip_sleb(bytes: &[u8], pos: &mut usize, max_bytes: usize) -> Result<(), CompileError> {
    for _ in 0..max_bytes {
        let b = read_byte(bytes, pos)?;
        if (b & 0x80) == 0 {
            return Ok(());
        }
    }
    Err(werr("wasm sleb exceeds max byte width"))
}

fn is_valtype(b: u8) -> bool {
    matches!(b, 0x7B..=0x7F)
}

fn read_utf8_name(bytes: &[u8], pos: &mut usize) -> Result<String, CompileError> {
    let len = read_uleb_u32(bytes, pos)? as usize;
    let end = pos.checked_add(len).ok_or_else(|| werr("wasm name length overflow"))?;
    let slice = bytes.get(*pos..end).ok_or_else(|| werr("wasm name truncated"))?;
    *pos = end;
    let s = std::str::from_utf8(slice).map_err(|_| werr("wasm export name invalid UTF-8"))?;
    Ok(s.to_string())
}

fn skip_memarg(bytes: &[u8], pos: &mut usize, max_align_log2: u32) -> Result<(), CompileError> {
    let align = read_uleb_u32(bytes, pos)?;
    if align > max_align_log2 {
        return Err(werr("wasm memarg alignment exceeds natural width"));
    }
    let _ = read_uleb_u32(bytes, pos)?;
    Ok(())
}

fn verify_code_body(
    body: &[u8],
    func_count: usize,
    total_insns: &mut usize,
    simd_insns: &mut usize,
    div_rem_insns: &mut usize,
) -> Result<(), CompileError> {
    let mut pos = 0usize;
    let local_groups = read_uleb_u32(body, &mut pos)? as usize;
    for _ in 0..local_groups {
        let _ = read_uleb_u32(body, &mut pos)?;
        let ty = read_byte(body, &mut pos)?;
        if !is_valtype(ty) {
            return Err(werr("wasm invalid local valtype"));
        }
    }
    let mut depth = 1usize;
    while pos < body.len() {
        let op = read_byte(body, &mut pos)?;
        *total_insns += 1;
        match op {
            0x00 | 0x01 | 0x0F | 0x1A | 0x1B => {}
            0x02..=0x04 => {
                let bt = read_byte(body, &mut pos)?;
                if bt != 0x40 && !is_valtype(bt) {
                    return Err(werr("wasm invalid blocktype"));
                }
                depth += 1;
            }
            0x05 => {
                if depth < 2 {
                    return Err(werr("wasm stray else outside if block"));
                }
            }
            0x0B => {
                if depth == 0 {
                    return Err(werr("wasm unbalanced end"));
                }
                depth -= 1;
                if depth == 0 && pos != body.len() {
                    return Err(werr("wasm trailing bytes after function end"));
                }
            }
            0x0C | 0x0D => {
                let lbl = read_uleb_u32(body, &mut pos)? as usize;
                if lbl >= depth {
                    return Err(werr("wasm branch label exceeds block depth"));
                }
            }
            0x10 => {
                let fidx = read_uleb_u32(body, &mut pos)? as usize;
                if fidx >= func_count {
                    return Err(werr("wasm call target out of bounds"));
                }
            }
            0x20..=0x24 => {
                let _ = read_uleb_u32(body, &mut pos)?;
            }
            0x28 | 0x36 => skip_memarg(body, &mut pos, 2)?,
            0x29 | 0x37 => skip_memarg(body, &mut pos, 3)?,
            0x2C | 0x2D | 0x30 | 0x31 | 0x3A | 0x3C => skip_memarg(body, &mut pos, 0)?,
            0x2E | 0x2F | 0x32 | 0x33 | 0x3B | 0x3D => skip_memarg(body, &mut pos, 1)?,
            0x34 | 0x35 | 0x3E => skip_memarg(body, &mut pos, 2)?,
            0x41 => skip_sleb(body, &mut pos, 5)?,
            0x42 => skip_sleb(body, &mut pos, 10)?,
            0x45..=0x6C | 0x71..=0x7E | 0x83..=0x8A | 0xA7..=0xC4 => {}
            0x6D..=0x70 | 0x7F..=0x82 => {
                *div_rem_insns += 1;
            }
            0xFD => {
                *simd_insns += 1;
                let sub = read_uleb_u32(body, &mut pos)?;
                match sub {
                    SIMD_V128_LOAD | SIMD_V128_STORE => skip_memarg(body, &mut pos, 4)?,
                    0x01..=0x06 | SIMD_V128_LOAD64_SPLAT => skip_memarg(body, &mut pos, 3)?,
                    0x07 => skip_memarg(body, &mut pos, 0)?,
                    0x08 => skip_memarg(body, &mut pos, 1)?,
                    0x09 => skip_memarg(body, &mut pos, 2)?,
                    SIMD_V128_CONST | 0x0D => {
                        let end = pos.checked_add(16).ok_or_else(|| werr("wasm v128 imm overflow"))?;
                        if end > body.len() {
                            return Err(werr("wasm v128 immediate truncated"));
                        }
                        pos = end;
                    }
                    0x15..=0x22 => {
                        let _ = read_byte(body, &mut pos)?;
                    }
                    0x54 | 0x58 => {
                        skip_memarg(body, &mut pos, 0)?;
                        let _ = read_byte(body, &mut pos)?;
                    }
                    0x55 | 0x59 => {
                        skip_memarg(body, &mut pos, 1)?;
                        let _ = read_byte(body, &mut pos)?;
                    }
                    0x56 | 0x5A => {
                        skip_memarg(body, &mut pos, 2)?;
                        let _ = read_byte(body, &mut pos)?;
                    }
                    SIMD_V128_LOAD64_LANE | SIMD_V128_STORE64_LANE => {
                        skip_memarg(body, &mut pos, 3)?;
                        let lane = read_byte(body, &mut pos)?;
                        if lane > 1 {
                            return Err(werr("wasm v128 64-bit lane index out of range"));
                        }
                    }
                    0x0F..=0x14 | 0x23..=0x53 | 0x5C..=0xFF => {}
                    _ => return Err(werr(format!("wasm unsupported simd opcode 0x{sub:x}"))),
                }
            }
            _ => return Err(werr(format!("wasm unsupported opcode 0x{op:02x}"))),
        }
    }
    if depth != 0 {
        return Err(werr("wasm unbalanced block/loop/if without closing end"));
    }
    Ok(())
}

/// Self-verifying `0t` airlock parser for standalone `WASM32` (+`SIMD128`) binary modules emitted by `zero-cc`.
pub fn verify_wasm_module(bytes: &[u8]) -> Result<WasmSummary, CompileError> {
    if bytes.get(0..8) != Some(&WASM_MAGIC_VERSION) {
        return Err(werr("bad wasm magic or version"));
    }
    let mut pos = 8usize;
    let mut prev_sec_id = 0u8;
    let mut type_count = 0usize;
    let mut func_count = 0usize;
    let mut mem_count = 0usize;
    let mut mem_min_pages = 0u32;
    let mut imports_memory = false;
    let mut exports = Vec::new();
    let mut code_body_count = 0usize;
    let mut total_insns = 0usize;
    let mut simd_insns = 0usize;
    let mut div_rem_insns = 0usize;

    while pos < bytes.len() {
        let sec_id = read_byte(bytes, &mut pos)?;
        let sec_len = read_uleb_u32(bytes, &mut pos)? as usize;
        if sec_id != 0 {
            if sec_id <= prev_sec_id || sec_id > 12 {
                return Err(werr("wasm out-of-order or duplicate section id"));
            }
            prev_sec_id = sec_id;
        }
        let sec_end = pos.checked_add(sec_len).ok_or_else(|| werr("wasm section length overflow"))?;
        let payload = bytes.get(pos..sec_end).ok_or_else(|| werr("wasm section truncated"))?;
        pos = sec_end;

        let mut p = 0usize;
        match sec_id {
            SEC_TYPE => {
                type_count = read_uleb_u32(payload, &mut p)? as usize;
                for _ in 0..type_count {
                    if read_byte(payload, &mut p)? != 0x60 {
                        return Err(werr("wasm type section expected 0x60 functype"));
                    }
                    let nparams = read_uleb_u32(payload, &mut p)? as usize;
                    for _ in 0..nparams {
                        if !is_valtype(read_byte(payload, &mut p)?) {
                            return Err(werr("wasm invalid param valtype"));
                        }
                    }
                    let nres = read_uleb_u32(payload, &mut p)? as usize;
                    for _ in 0..nres {
                        if !is_valtype(read_byte(payload, &mut p)?) {
                            return Err(werr("wasm invalid result valtype"));
                        }
                    }
                }
            }
            SEC_IMPORT => {
                let nimp = read_uleb_u32(payload, &mut p)? as usize;
                for _ in 0..nimp {
                    let mod_name = read_utf8_name(payload, &mut p)?;
                    let field_name = read_utf8_name(payload, &mut p)?;
                    let kind = read_byte(payload, &mut p)?;
                    if kind != EXPORT_MEM || mod_name != "env" || field_name != "memory" {
                        return Err(werr("wasm import section only permits env.memory"));
                    }
                    let flags = read_byte(payload, &mut p)?;
                    let min_p = read_uleb_u32(payload, &mut p)?;
                    if mem_count == 0 {
                        mem_min_pages = min_p;
                    }
                    match flags {
                        0x00 => {}
                        0x01 => {
                            let max_p = read_uleb_u32(payload, &mut p)?;
                            if max_p < min_p {
                                return Err(werr("wasm imported memory max < min"));
                            }
                        }
                        _ => return Err(werr("wasm invalid imported memory limits flag")),
                    }
                    mem_count += 1;
                    imports_memory = true;
                }
            }
            SEC_FUNC => {
                func_count = read_uleb_u32(payload, &mut p)? as usize;
                for _ in 0..func_count {
                    let tidx = read_uleb_u32(payload, &mut p)? as usize;
                    if tidx >= type_count {
                        return Err(werr("wasm function type index out of range"));
                    }
                }
            }
            SEC_MEM => {
                if imports_memory {
                    return Err(werr("wasm module cannot both import and define memory"));
                }
                mem_count = read_uleb_u32(payload, &mut p)? as usize;
                if mem_count == 0 {
                    return Err(werr("wasm memory section empty"));
                }
                for i in 0..mem_count {
                    let flags = read_byte(payload, &mut p)?;
                    let min_p = read_uleb_u32(payload, &mut p)?;
                    if i == 0 {
                        mem_min_pages = min_p;
                    }
                    match flags {
                        0x00 => {}
                        0x01 => {
                            let max_p = read_uleb_u32(payload, &mut p)?;
                            if max_p < min_p {
                                return Err(werr("wasm memory max < min"));
                            }
                        }
                        _ => return Err(werr("wasm invalid memory limits flag")),
                    }
                }
            }
            SEC_EXPORT => {
                let nexp = read_uleb_u32(payload, &mut p)? as usize;
                for _ in 0..nexp {
                    let name = read_utf8_name(payload, &mut p)?;
                    let kind = read_byte(payload, &mut p)?;
                    let index = read_uleb_u32(payload, &mut p)?;
                    match kind {
                        EXPORT_FUNC if (index as usize) < func_count => {}
                        EXPORT_MEM if (index as usize) < mem_count => {}
                        0x01 | 0x03 => {}
                        _ => return Err(werr("wasm export index out of range")),
                    }
                    exports.push(WasmExport { name, kind, index });
                }
            }
            SEC_CODE => {
                code_body_count = read_uleb_u32(payload, &mut p)? as usize;
                if code_body_count != func_count {
                    return Err(werr("wasm code body count mismatches function section"));
                }
                for _ in 0..code_body_count {
                    let bsz = read_uleb_u32(payload, &mut p)? as usize;
                    let bend = p.checked_add(bsz).ok_or_else(|| werr("wasm code body length overflow"))?;
                    let body = payload.get(p..bend).ok_or_else(|| werr("wasm code body truncated"))?;
                    p = bend;
                    verify_code_body(body, func_count, &mut total_insns, &mut simd_insns, &mut div_rem_insns)?;
                }
            }
            0 => {
                p = payload.len();
            }
            _ => return Err(werr(format!("wasm unexpected section id {sec_id}"))),
        }
        if p != payload.len() {
            return Err(werr("wasm section payload length mismatch"));
        }
    }

    if type_count == 0 || func_count == 0 || mem_count == 0 || code_body_count != func_count {
        return Err(werr("wasm module missing mandatory section"));
    }
    if !imports_memory && !exports.iter().any(|e| e.kind == EXPORT_MEM && e.name == "memory") {
        return Err(werr("wasm module missing exported memory"));
    }
    Ok(WasmSummary {
        type_count,
        func_count,
        mem_min_pages,
        exports,
        code_body_count,
        total_insns,
        simd_insns,
        div_rem_insns,
        has_simd128: simd_insns > 0,
        imports_memory,
    })
}

/// Self-verifying `0t` airlock parser enforcing an explicit imported-memory expectation.
pub fn verify_wasm_module_with_opts(
    bytes: &[u8],
    expect_imported_memory: bool,
) -> Result<WasmSummary, CompileError> {
    let sum = verify_wasm_module(bytes)?;
    if sum.imports_memory != expect_imported_memory {
        return Err(werr(if expect_imported_memory {
            "wasm module expected imported env.memory"
        } else {
            "wasm module unexpectedly imported memory"
        }));
    }
    Ok(sum)
}
