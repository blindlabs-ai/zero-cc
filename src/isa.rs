use std::fmt::Write as _;

use crate::codegen::{schedule_and_verify, RegBudget, ScheduleReport};
use crate::elf64::{
    elf64_exec_data_vaddr, emit_elf64_executable, emit_elf64_object, verify_elf64_executable,
    verify_elf64_object, ElfReloc, ElfSymbol, TargetIsa, ELF_EXEC_TEXT_VADDR, R_AARCH64_CALL26,
    R_X86_64_PLT32,
};
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op, LANES};
use crate::optimize_kernel;
use crate::rust_fe::parse_rust_kernel;

const X86_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
const X86_CALLEE: [u8; 5] = [3, 12, 13, 14, 15];
const X86_POOL: [u8; 12] = [10, 1, 2, 6, 7, 8, 9, 3, 12, 13, 14, 15];
const ARM_POOL: [u8; 14] = [9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25];

#[derive(Clone, Debug)]
pub struct RegAlloc {
    pub node_reg: Vec<Option<u8>>, pub buf_reg: Vec<u8>,
    pub callee_saved: Vec<u8>, pub spills: usize,
}

#[derive(Clone, Debug)]
pub struct CompiledObject {
    pub isa: TargetIsa, pub kernel_sym: String, pub batch_sym: String,
    pub code_bytes: Vec<u8>, pub elf_bytes: Vec<u8>,
    pub direct_insns: usize, pub shim_glue_insns: usize, pub direct_share_pct: u8,
    pub zero_spills: bool, pub rust_ffi_shim: String,
}

#[derive(Clone, Debug)]
pub struct StandaloneExec {
    pub isa: TargetIsa, pub kernel_sym: String, pub batch_sym: String, pub entry_sym: String,
    pub entry_vaddr: u64, pub data_vaddr: u64, pub code_bytes: Vec<u8>, pub elf_bytes: Vec<u8>,
    pub direct_insns: usize, pub shim_glue_insns: usize, pub direct_share_pct: u8,
    pub zero_spills: bool, pub blocks: usize, pub output_bytes: usize,
}

fn berr(msg: impl Into<String>) -> CompileError { CompileError::BackendError(msg.into()) }
fn is_inline(op: Op<NodeId>) -> bool { matches!(op, Op::MemInit | Op::Const(_) | Op::Store { .. }) }

/// @domain reg: 0..=15, idx: 0..=15, base: 0..=15
pub const fn x86_rex_byte(w: bool, reg: u8, idx: u8, base: u8) -> u8 {
    0x40 | (if w { 8 } else { 0 }) | (((reg >> 3) & 1) << 2) | (((idx >> 3) & 1) << 1) | ((base >> 3) & 1)
}

/// @domain reg: 0..=15, rm: 0..=15
pub const fn x86_rex_w(reg: u8, rm: u8) -> u8 { 0x48 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1) }

/// @domain reg: 0..=15, idx: 0..=15, base: 0..=15
pub const fn x86_rex_sib(reg: u8, idx: u8, base: u8) -> u8 { x86_rex_byte(true, reg, idx, base) }

/// @domain md: 0..=3, reg: 0..=15, rm: 0..=15
pub const fn x86_modrm(md: u8, reg: u8, rm: u8) -> u8 { ((md & 3) << 6) | ((reg & 7) << 3) | (rm & 7) }

/// @domain scale: 0..=3, idx: 0..=15, base: 0..=15
pub const fn x86_sib(scale: u8, idx: u8, base: u8) -> u8 { ((scale & 3) << 6) | ((idx & 7) << 3) | (base & 7) }

type ModuleCode = (Vec<u8>, Vec<ElfSymbol>, Vec<ElfReloc>, usize);

/// @domain r: 0..=1, b: 0..=1, map: 1..=3, w: 0..=1, vvvv: 0..=15, l: 0..=1, pp: 0..=3
pub const fn x86_vex3_word(r: u8, b: u8, map: u8, w: u8, vvvv: u8, l: u8, pp: u8) -> u16 {
    let b1 = ((!r & 1) << 7) | 0x40 | ((!b & 1) << 5) | (map & 0x1F);
    let b2 = ((w & 1) << 7) | ((!vvvv & 15) << 3) | ((l & 1) << 2) | (pp & 3);
    ((b2 as u16) << 8) | (b1 as u16)
}

/// @domain rd: 0..=31, rn: 0..=31, rm: 0..=31
pub const fn arm64_rrr(op_hi: u32, rm: u8, mid: u32, rn: u8, rd: u8) -> u32 {
    op_hi | ((rm as u32 & 31) << 16) | mid | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}

/// @domain rd: 0..=31, rn: 0..=31, imm12: 0..=4095
pub const fn arm64_imm12(op_hi: u32, imm12: u16, rn: u8, rd: u8) -> u32 {
    op_hi | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}

/// Linear-scan physical GPR allocator over `ScheduleReport` proving `spills == 0`.
pub fn allocate_registers(kernel: &Kernel, sched: &ScheduleReport, isa: TargetIsa, budget: RegBudget) -> Result<RegAlloc, CompileError> {
    let nb = kernel.bufs.len();
    if (isa == TargetIsa::X86_64 && nb > 5) || (isa == TargetIsa::Aarch64 && nb > 7) {
        return Err(berr("kernel buffer count exceeds register-passed ABI args"));
    }
    let has_div = sched.order.iter().any(|&id| matches!(kernel.nodes.get(id), Some(Op::Div(_, _) | Op::Rem(_, _))));
    let mut buf_reg: Vec<u8> = match isa {
        TargetIsa::X86_64 => X86_ARGS.iter().copied().take(nb).collect(),
        TargetIsa::Aarch64 => (0..nb as u8).collect(),
    };
    let mut rdx_reloc = None;
    if isa == TargetIsa::X86_64 && has_div && nb > 2 {
        let alt: u8 = if nb < 5 { 9 } else { 15 };
        if let Some(slot) = buf_reg.get_mut(2) { *slot = alt; }
        rdx_reloc = Some(alt);
    }

    let cnt_x86 = X86_ARGS.get(nb).copied().unwrap_or(255);
    let pool: Vec<u8> = match isa {
        TargetIsa::X86_64 => X86_POOL.iter().copied().filter(|&r| !buf_reg.contains(&r) && r != cnt_x86 && (!has_div || r != 2) && rdx_reloc != Some(r)).collect(),
        TargetIsa::Aarch64 => ARM_POOL.to_vec(),
    };
    let n = kernel.nodes.len();
    let mut last_step = vec![0usize; n];
    for (step, &id) in sched.order.iter().enumerate() {
        let (kids, cnt) = kernel.nodes[id].children();
        for &k in &kids[..cnt] { if let Some(ls) = last_step.get_mut(k) { *ls = (*ls).max(step); } }
    }
    for &(ostep, oi) in &sched.output_order {
        if let Some(o) = kernel.outputs.get(oi) {
            if let Some(ls) = last_step.get_mut(o.val) { *ls = (*ls).max(ostep.saturating_add(1)); }
        }
    }

    let mut free: Vec<u8> = pool.into_iter().rev().collect();
    let (mut node_reg, mut active, mut used_callee) = (vec![None; n], Vec::<(NodeId, u8)>::new(), Vec::new());
    for (step, &id) in sched.order.iter().enumerate() {
        let op = kernel.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if !is_inline(op) {
            let Some(reg) = free.pop() else {
                return Err(CompileError::RegisterSpill { peak_gprs: nb + active.len() + 1, max_gprs: budget.max_gprs, peak_vecs: sched.peak_vecs, max_vecs: budget.max_vecs });
            };
            let is_cs = if isa == TargetIsa::X86_64 { X86_CALLEE.contains(&reg) } else { reg >= 19 };
            if is_cs && !used_callee.contains(&reg) { used_callee.push(reg); }
            if let Some(slot) = node_reg.get_mut(id) { *slot = Some(reg); }
            if last_step.get(id).copied().unwrap_or(0) > step { active.push((id, reg)); } else { free.push(reg); }
        }
        let mut i = 0usize;
        while i < active.len() {
            let (aid, areg) = active[i];
            if last_step.get(aid).copied().unwrap_or(0) <= step { active.swap_remove(i); free.push(areg); } else { i += 1; }
        }
        for &(ostep, _) in sched.output_order.iter().filter(|&&(os, _)| os == step + 1) {
            let mut j = 0usize;
            while j < active.len() {
                let (aid, areg) = active[j];
                if last_step.get(aid).copied().unwrap_or(0) <= ostep { active.swap_remove(j); free.push(areg); } else { j += 1; }
            }
        }
    }
    used_callee.sort_unstable();
    Ok(RegAlloc { node_reg, buf_reg, callee_saved: used_callee, spills: 0 })
}

struct X86Emit { code: Vec<u8>, insns: usize }
impl X86Emit {
    fn new() -> Self { Self { code: Vec::with_capacity(512), insns: 0 } }
    fn push_rr(&mut self, op: u8, reg: u8, rm: u8) { self.code.extend_from_slice(&[x86_rex_w(reg, rm), op, x86_modrm(3, reg, rm)]); self.insns += 1; }
    fn mov_rr(&mut self, dst: u8, src: u8) { if dst != src { self.push_rr(0x89, src, dst); } }
    fn movabs_imm64(&mut self, dst: u8, val: u64) {
        self.code.extend_from_slice(&[x86_rex_w(0, dst), 0xB8 + (dst & 7)]);
        self.code.extend_from_slice(&val.to_le_bytes());
        self.insns += 1;
    }
    fn mov_imm(&mut self, dst: u8, v: i64) {
        if v == 0 {
            let hi = (dst >> 3) & 1;
            if hi != 0 { self.code.push(0x40 | (hi << 2) | hi); }
            self.code.extend_from_slice(&[0x31, x86_modrm(3, dst, dst)]);
            self.insns += 1;
        } else if let Ok(v32) = i32::try_from(v) {
            self.code.extend_from_slice(&[x86_rex_w(0, dst), 0xC7, x86_modrm(3, 0, dst)]);
            self.code.extend_from_slice(&v32.to_le_bytes());
            self.insns += 1;
        } else {
            self.movabs_imm64(dst, v as u64);
        }
    }
    fn push_mem_disp(&mut self, reg: u8, base: u8, disp: i32) {
        let b7 = base & 7;
        if disp == 0 && b7 != 5 {
            self.code.push(x86_modrm(0, reg, base));
            if b7 == 4 { self.code.push(0x24); }
        } else if let Ok(d8) = i8::try_from(disp) {
            self.code.push(x86_modrm(1, reg, base));
            if b7 == 4 { self.code.push(0x24); }
            self.code.push(d8 as u8);
        } else {
            self.code.push(x86_modrm(2, reg, base));
            if b7 == 4 { self.code.push(0x24); }
            self.code.extend_from_slice(&disp.to_le_bytes());
        }
        self.insns += 1;
    }
    fn push_load_prefix_opc(&mut self, ty: ElemTy, dst: u8, idx: u8, base: u8) {
        let need_w = matches!(ty, ElemTy::I64 | ElemTy::I32 | ElemTy::I16);
        if need_w || dst >= 8 || idx >= 8 || base >= 8 {
            self.code.push(x86_rex_byte(need_w, dst, idx, base));
        }
        match ty {
            ElemTy::I64 | ElemTy::U32 => self.code.push(0x8B),
            ElemTy::I32 => self.code.push(0x63),
            ElemTy::I16 => self.code.extend_from_slice(&[0x0F, 0xBF]),
            ElemTy::U16 => self.code.extend_from_slice(&[0x0F, 0xB7]),
            ElemTy::U8 => self.code.extend_from_slice(&[0x0F, 0xB6]),
        }
    }
    fn load_lane(&mut self, ty: ElemTy, dst: u8, base: u8, lane: u8) {
        self.push_load_prefix_opc(ty, dst, 0, base);
        self.push_mem_disp(dst, base, i32::from(lane) * i32::from(ty.bytes()));
    }
    fn store_lane(&mut self, ty: ElemTy, src: u8, base: u8, lane: u8) {
        match ty {
            ElemTy::I64 => self.code.extend_from_slice(&[x86_rex_w(src, base), 0x89]),
            ElemTy::I32 | ElemTy::U32 => {
                if src >= 8 || base >= 8 { self.code.push(x86_rex_byte(false, src, 0, base)); }
                self.code.push(0x89);
            }
            ElemTy::I16 | ElemTy::U16 => {
                self.code.push(0x66);
                if src >= 8 || base >= 8 { self.code.push(x86_rex_byte(false, src, 0, base)); }
                self.code.push(0x89);
            }
            ElemTy::U8 => {
                if src >= 4 || base >= 8 { self.code.push(x86_rex_byte(false, src, 0, base)); }
                self.code.push(0x88);
            }
        }
        self.push_mem_disp(src, base, i32::from(lane) * i32::from(ty.bytes()));
    }
    fn load_lut(&mut self, ty: ElemTy, dst: u8, base: u8, idx: u8) {
        self.push_load_prefix_opc(ty, dst, idx, base);
        let scale = ty.scale_shift();
        if (base & 7) == 5 {
            self.code.extend_from_slice(&[x86_modrm(1, dst, 4), x86_sib(scale, idx, base), 0]);
        } else {
            self.code.extend_from_slice(&[x86_modrm(0, dst, 4), x86_sib(scale, idx, base)]);
        }
        self.insns += 1;
    }
    fn op_imm8(&mut self, opc: u8, subop: u8, dst: u8, imm: u8, skip_zero: bool) {
        if !(skip_zero && imm == 0) {
            self.code.extend_from_slice(&[x86_rex_w(0, dst), opc, x86_modrm(3, subop, dst), imm]);
            self.insns += 1;
        }
    }
    fn stack_op(&mut self, r: u8, is_pop: bool) {
        if r >= 8 { self.code.push(0x41); }
        self.code.push((if is_pop { 0x58 } else { 0x50 }) + (r & 7));
        self.insns += 1;
    }
    fn vex_ymm_op(&mut self, map: u8, opc: u8, dst: u8, vvvv: u8, rm: u8) {
        let vw = x86_vex3_word((dst >> 3) & 1, (rm >> 3) & 1, map, 0, vvvv, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], opc, x86_modrm(3, dst, rm)]);
        self.insns += 1;
    }
    fn vex_mem(&mut self, l: u8, pp: u8, opc: u8, reg: u8, base: u8, disp: i8) {
        let (vw, b7) = (x86_vex3_word((reg >> 3) & 1, (base >> 3) & 1, 1, 0, 0, l, pp).to_le_bytes(), base & 7);
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], opc]);
        if disp == 0 && b7 != 5 { self.code.push(x86_modrm(0, reg, base)); if b7 == 4 { self.code.push(0x24); } }
        else { self.code.push(x86_modrm(1, reg, base)); if b7 == 4 { self.code.push(0x24); } self.code.push(disp as u8); }
        self.insns += 1;
    }
    fn vzeroall(&mut self) { self.code.extend_from_slice(&[0xC5, 0xFC, 0x77]); self.insns += 1; }
    fn ret(&mut self) { self.code.push(0xC3); self.insns += 1; }
    fn ud2(&mut self) { self.code.extend_from_slice(&[0x0F, 0x0B]); self.insns += 1; }
}

fn resolve_node_reg(kernel: &Kernel, ra: &RegAlloc, id: NodeId, scratch: u8, mut set_imm: impl FnMut(u8, i64)) -> Result<u8, CompileError> {
    match kernel.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))? {
        Op::Const(c) => { set_imm(scratch, c); Ok(scratch) }
        Op::MemInit => { set_imm(scratch, 0); Ok(scratch) }
        _ => ra.node_reg.get(id).and_then(|&r| r).ok_or(CompileError::InvalidNode(id)),
    }
}

fn buf_elem_ty(kernel: &Kernel, buf: u8) -> Result<ElemTy, CompileError> {
    kernel.bufs.get(buf as usize).map(|b| b.elem_ty).ok_or(CompileError::InvalidBuffer(buf))
}

fn drain_scheduled_outputs(
    kernel: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, oc: &mut usize, step: Option<usize>,
    mut emit_store: impl FnMut(ElemTy, u8, u8, NodeId) -> Result<(), CompileError>,
) -> Result<(), CompileError> {
    while let Some(&(ostep, oi)) = sched.output_order.get(*oc) {
        if step.is_some_and(|st| ostep != st) { break; }
        let o = kernel.outputs.get(oi).copied().ok_or(CompileError::InvalidNode(oi))?;
        let breg = *ra.buf_reg.get(o.buf as usize).ok_or(CompileError::InvalidBuffer(o.buf))?;
        emit_store(buf_elem_ty(kernel, o.buf)?, breg, o.lane, o.val)?;
        *oc += 1;
    }
    Ok(())
}

fn x86_val_into(e: &mut X86Emit, kernel: &Kernel, ra: &RegAlloc, id: NodeId, scratch: u8) -> Result<u8, CompileError> {
    resolve_node_reg(kernel, ra, id, scratch, |r, c| e.mov_imm(r, c))
}

#[derive(Clone, Debug)]
enum AvxStep {
    Const(u8, i64), Load(u8, u8, i8), Store(u8, u8, i8), Add(u8, u8, u8), Sub(u8, u8, u8),
    BitAnd(u8, u8, u8), MulU32(u8, u8, u8), Shl(u8, u8, u8), ShrU64(u8, u8, u8),
}

fn match_vec4(
    k: &Kernel, doms: &[Domain], ids: [NodeId; 4], memo: &mut Vec<([NodeId; 4], u8)>,
    steps: &mut Vec<AvxStep>, next_ymm: &mut u8, max_vecs: usize,
) -> Option<u8> {
    if let Some(&(_, r)) = memo.iter().find(|&&(m, _)| m == ids) { return Some(r); }
    let o0 = *k.nodes.get(ids[0])?;
    let alloc = |ny: &mut u8| -> Option<u8> {
        if usize::from(*ny) >= max_vecs.min(16) { None } else { let r = *ny; *ny += 1; Some(r) }
    };
    let r = match o0 {
        Op::Const(c) => {
            for &id in &ids[1..] { if k.nodes.get(id).copied() != Some(Op::Const(c)) { return None; } }
            if let Some(&(_, prev)) = memo.iter().find(|&&(m, _)| k.nodes.get(m[0]).copied() == Some(Op::Const(c)) && m[1..].iter().all(|&x| k.nodes.get(x).copied() == Some(Op::Const(c)))) {
                prev
            } else {
                let d = alloc(next_ymm)?; steps.push(AvxStep::Const(d, c)); d
            }
        }
        Op::Load { mem, buf, lane } if (lane == 0 || lane == 4) && k.nodes.get(mem).copied() == Some(Op::MemInit) => {
            for (i, &id) in ids.iter().enumerate().skip(1) {
                if k.nodes.get(id).copied() != Some(Op::Load { mem, buf, lane: lane + i as u8 }) { return None; }
            }
            let d = alloc(next_ymm)?; steps.push(AvxStep::Load(d, buf, if lane == 4 { 32 } else { 0 })); d
        }
        Op::Add(a0, b0) | Op::Sub(a0, b0) | Op::BitAnd(a0, b0) | Op::Mul(a0, b0) => {
            let (mut la, mut lb) = ([a0; 4], [b0; 4]);
            for i in 1..4 {
                match (o0, *k.nodes.get(ids[i])?) {
                    (Op::Add(_, _), Op::Add(ai, bi)) | (Op::Sub(_, _), Op::Sub(ai, bi))
                    | (Op::BitAnd(_, _), Op::BitAnd(ai, bi)) | (Op::Mul(_, _), Op::Mul(ai, bi)) => { la[i] = ai; lb[i] = bi; }
                    _ => return None,
                }
            }
            if matches!(o0, Op::Mul(_, _)) && !ids.iter().zip(la.iter().zip(lb.iter())).all(|(_, (&a, &b))| doms.get(a).copied().is_some_and(Domain::fits_u32) && doms.get(b).copied().is_some_and(Domain::fits_u32)) {
                return None;
            }
            let va = match_vec4(k, doms, la, memo, steps, next_ymm, max_vecs)?;
            let vb = match_vec4(k, doms, lb, memo, steps, next_ymm, max_vecs)?;
            let d = alloc(next_ymm)?;
            steps.push(match o0 {
                Op::Add(_, _) => AvxStep::Add(d, va, vb), Op::Sub(_, _) => AvxStep::Sub(d, va, vb),
                Op::BitAnd(_, _) => AvxStep::BitAnd(d, va, vb), _ => AvxStep::MulU32(d, va, vb),
            });
            d
        }
        Op::Shl(a0, s) | Op::Shr(a0, s) => {
            let mut la = [a0; 4];
            for i in 1..4 {
                match (o0, *k.nodes.get(ids[i])?) {
                    (Op::Shl(_, _), Op::Shl(ai, si)) | (Op::Shr(_, _), Op::Shr(ai, si)) if si == s => la[i] = ai,
                    _ => return None,
                }
            }
            if matches!(o0, Op::Shr(_, _)) && !la.iter().all(|&a| doms.get(a).copied().is_some_and(Domain::is_nonneg)) { return None; }
            let va = match_vec4(k, doms, la, memo, steps, next_ymm, max_vecs)?;
            let d = alloc(next_ymm)?;
            steps.push(if matches!(o0, Op::Shl(_, _)) { AvxStep::Shl(d, va, s) } else { AvxStep::ShrU64(d, va, s) });
            d
        }
        _ => return None,
    };
    memo.push((ids, r));
    Some(r)
}

fn try_plan_avx2(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], max_vecs: usize) -> Option<Vec<AvxStep>> {
    if k.outputs.len() != LANES { return None; }
    if k.bufs.iter().any(|b| b.elem_ty != ElemTy::I64 || (!matches!(b.access, BufAccess::LookupTable(_)) && usize::from(b.lanes) != LANES)) {
        return None;
    }
    if sched.order.iter().any(|&id| matches!(k.nodes.get(id), Some(Op::Store { .. })) || matches!(k.nodes.get(id), Some(&Op::Load { mem, .. }) if k.nodes.get(mem).copied() != Some(Op::MemInit))) {
        return None;
    }
    let out_buf = k.outputs.first()?.buf;
    if k.bufs.iter().any(|b| b.id != out_buf && b.access == BufAccess::MutNoAlias) { return None; }
    let mut lane_vals = [usize::MAX; LANES];
    for o in &k.outputs {
        if o.buf != out_buf || usize::from(o.lane) >= LANES || lane_vals[usize::from(o.lane)] != usize::MAX { return None; }
        lane_vals[usize::from(o.lane)] = o.val;
    }
    let mut steps = Vec::new();
    let (mut m0, mut n0) = (Vec::new(), 0u8);
    let r_lo = match_vec4(k, doms, [lane_vals[0], lane_vals[1], lane_vals[2], lane_vals[3]], &mut m0, &mut steps, &mut n0, max_vecs)?;
    let (mut m1, mut s_hi, mut n1) = (Vec::new(), Vec::new(), 0u8);
    let mut s_both = steps.clone();
    let mut m_shared = m0;
    if let Some(r_hi) = match_vec4(k, doms, [lane_vals[4], lane_vals[5], lane_vals[6], lane_vals[7]], &mut m_shared, &mut s_both, &mut n0, max_vecs) {
        s_both.push(AvxStep::Store(r_lo, out_buf, 0));
        s_both.push(AvxStep::Store(r_hi, out_buf, 32));
        return Some(s_both);
    }
    let out_is_read = k.nodes.iter().any(|&op| matches!(op, Op::Load { buf, .. } if buf == out_buf));
    if out_is_read { return None; }
    steps.push(AvxStep::Store(r_lo, out_buf, 0));
    let r_hi = match_vec4(k, doms, [lane_vals[4], lane_vals[5], lane_vals[6], lane_vals[7]], &mut m1, &mut s_hi, &mut n1, max_vecs)?;
    steps.extend(s_hi);
    steps.push(AvxStep::Store(r_hi, out_buf, 32));
    Some(steps)
}

fn emit_x86_avx2_steps(e: &mut X86Emit, steps: &[AvxStep], ra: &RegAlloc) -> Result<(), CompileError> {
    for st in steps {
        match *st {
            AvxStep::Const(d, c) => {
                e.mov_imm(0, c);
                let vw = x86_vex3_word((d >> 3) & 1, 0, 1, 1, 0, 0, 1).to_le_bytes();
                e.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x6E, x86_modrm(3, d, 0)]); e.insns += 1;
                e.vex_ymm_op(2, 0x59, d, 0, d);
            }
            AvxStep::Load(d, buf, disp) => e.vex_mem(1, 2, 0x6F, d, *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, disp),
            AvxStep::Store(s, buf, disp) => e.vex_mem(1, 2, 0x7F, s, *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, disp),
            AvxStep::Add(d, a, b) => e.vex_ymm_op(1, 0xD4, d, a, b),
            AvxStep::Sub(d, a, b) => e.vex_ymm_op(1, 0xFB, d, a, b),
            AvxStep::BitAnd(d, a, b) => e.vex_ymm_op(1, 0xDB, d, a, b),
            AvxStep::MulU32(d, a, b) => e.vex_ymm_op(1, 0xF4, d, a, b),
            AvxStep::Shl(d, a, s) | AvxStep::ShrU64(d, a, s) => {
                let sub = if matches!(*st, AvxStep::Shl(_, _, _)) { 6 } else { 2 };
                let vw = x86_vex3_word(0, (a >> 3) & 1, 1, 0, d, 1, 1).to_le_bytes();
                e.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x73, x86_modrm(3, sub, a), s]);
                e.insns += 1;
            }
        }
    }
    Ok(())
}

fn emit_x86_cmp(e: &mut X86Emit, k: CmpKind, dst: u8, ra: u8, rb: u8) {
    e.push_rr(0x39, rb, ra);
    let cc = match k { CmpKind::Eq => 0x94, CmpKind::Ne => 0x95, CmpKind::Lt => 0x9C, CmpKind::Ge => 0x9D, CmpKind::Le => 0x9E, CmpKind::Gt => 0x9F };
    e.code.extend_from_slice(&[0x0F, cc, 0xC0]); e.insns += 1;
    e.code.extend_from_slice(&[x86_rex_w(dst, 0), 0x0F, 0xB6, x86_modrm(3, dst, 0)]); e.insns += 1;
}

fn emit_x86_div_rem(e: &mut X86Emit, dst: u8, ra: u8, rb: u8, is_rem: bool) {
    e.mov_rr(11, rb);
    e.push_rr(0x85, 11, 11);
    e.code.extend_from_slice(&[0x75, 0x02]); e.insns += 1;
    e.ud2();
    e.mov_rr(0, ra);
    e.code.extend_from_slice(&[0x48, 0xBA]);
    e.code.extend_from_slice(&i64::MIN.to_le_bytes()); e.insns += 1;
    e.push_rr(0x39, 2, 0);
    e.code.extend_from_slice(&[0x75, 0x08]); e.insns += 1;
    e.op_imm8(0x83, 7, 11, 0xFF, false);
    e.code.extend_from_slice(&[0x75, 0x02]); e.insns += 1;
    e.ud2();
    e.code.extend_from_slice(&[0x48, 0x99]); e.insns += 1;
    e.code.extend_from_slice(&[x86_rex_w(0, 11), 0xF7, x86_modrm(3, 7, 11)]); e.insns += 1;
    e.mov_rr(dst, if is_rem { 2 } else { 0 });
}

fn emit_x86_body(e: &mut X86Emit, kernel: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain]) -> Result<(), CompileError> {
    let mut oc = 0usize;
    for (step, &id) in sched.order.iter().enumerate() {
        drain_scheduled_outputs(kernel, sched, ra, &mut oc, Some(step), |ty, breg, lane, val| {
            let vreg = x86_val_into(e, kernel, ra, val, 0)?;
            e.store_lane(ty, vreg, breg, lane);
            Ok(())
        })?;
        let op = kernel.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if let Op::Store { buf, lane, val, .. } = op {
            let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
            let vreg = x86_val_into(e, kernel, ra, val, 0)?;
            e.store_lane(buf_elem_ty(kernel, buf)?, vreg, breg, lane);
            continue;
        }
        if matches!(op, Op::MemInit | Op::Const(_)) { continue; }
        let dst = ra.node_reg.get(id).and_then(|&r| r).ok_or(CompileError::InvalidNode(id))?;
        match op {
            Op::Load { buf, lane, .. } => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                e.load_lane(buf_elem_ty(kernel, buf)?, dst, breg, lane);
            }
            Op::LoadLut { buf, idx } => {
                let (breg, ireg) = (*ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, x86_val_into(e, kernel, ra, idx, 0)?);
                let b = kernel.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let BufAccess::LookupTable(len) = b.access else { return Err(CompileError::InvalidBuffer(buf)); };
                let d = doms.get(idx).copied().unwrap_or_else(Domain::top);
                if d.is_empty() || d.lo < 0 || d.hi >= len as i64 {
                    e.mov_imm(11, len as i64); e.push_rr(0x39, 11, ireg);
                    e.code.extend_from_slice(&[0x72, 0x02]); e.insns += 1; e.ud2();
                }
                e.load_lut(b.elem_ty, dst, breg, ireg);
            }
            Op::BoundsCheck { idx, limit } => {
                let ireg = x86_val_into(e, kernel, ra, idx, 0)?;
                e.mov_imm(11, limit); e.push_rr(0x39, 11, ireg);
                e.code.extend_from_slice(&[0x72, 0x02]); e.insns += 1; e.ud2();
                e.mov_rr(dst, ireg);
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) => {
                let (is_add, is_sub) = (matches!(op, Op::Add(_, _)), matches!(op, Op::Sub(_, _)));
                let imm_rhs = kernel.nodes.get(b).and_then(|&o| if let Op::Const(c) = o { i32::try_from(c).ok().map(|v| (a, v)) } else { None })
                    .or_else(|| if !is_sub { kernel.nodes.get(a).and_then(|&o| if let Op::Const(c) = o { i32::try_from(c).ok().map(|v| (b, v)) } else { None }) } else { None });
                if let Some((x, c32)) = imm_rhs {
                    let vx = x86_val_into(e, kernel, ra, x, 0)?;
                    e.mov_rr(dst, vx);
                    let subop = if is_add { 0 } else if is_sub { 5 } else { 4 };
                    if let Ok(c8) = i8::try_from(c32) { e.op_imm8(0x83, subop, dst, c8 as u8, false); }
                    else { e.code.extend_from_slice(&[x86_rex_w(0, dst), 0x81, x86_modrm(3, subop, dst)]); e.code.extend_from_slice(&c32.to_le_bytes()); e.insns += 1; }
                } else {
                    let (opc, va) = (if is_add { 0x01 } else if is_sub { 0x29 } else { 0x21 }, x86_val_into(e, kernel, ra, a, 0)?);
                    let vb = x86_val_into(e, kernel, ra, b, 11)?;
                    if dst == vb && dst != va { e.mov_rr(11, vb); e.mov_rr(dst, va); e.push_rr(opc, 11, dst); }
                    else { e.mov_rr(dst, va); e.push_rr(opc, vb, dst); }
                }
            }
            Op::Mul(a, b) => {
                if let Some((x, c32)) = kernel.nodes.get(b).and_then(|&op| if let Op::Const(c) = op { i32::try_from(c).ok().map(|v| (a, v)) } else { None })
                    .or_else(|| kernel.nodes.get(a).and_then(|&op| if let Op::Const(c) = op { i32::try_from(c).ok().map(|v| (b, v)) } else { None }))
                {
                    let vx = x86_val_into(e, kernel, ra, x, 0)?;
                    e.code.extend_from_slice(&[x86_rex_w(dst, vx), 0x69, x86_modrm(3, dst, vx)]);
                    e.code.extend_from_slice(&c32.to_le_bytes()); e.insns += 1;
                } else {
                    let (va, vb) = (x86_val_into(e, kernel, ra, a, 0)?, x86_val_into(e, kernel, ra, b, 11)?);
                    let (r1, r2) = if dst == vb && dst != va { (vb, va) } else { (va, vb) };
                    e.mov_rr(dst, r1);
                    e.code.extend_from_slice(&[x86_rex_w(dst, r2), 0x0F, 0xAF, x86_modrm(3, dst, r2)]); e.insns += 1;
                }
            }
            Op::Div(a, b) | Op::Rem(a, b) => {
                let (va, vb) = (x86_val_into(e, kernel, ra, a, 0)?, x86_val_into(e, kernel, ra, b, 11)?);
                emit_x86_div_rem(e, dst, va, vb, matches!(op, Op::Rem(_, _)));
            }
            Op::Shl(a, s) => { let va = x86_val_into(e, kernel, ra, a, 0)?; e.mov_rr(dst, va); e.op_imm8(0xC1, 4, dst, s, true); }
            Op::Shr(a, s) => {
                let va = x86_val_into(e, kernel, ra, a, 0)?; e.mov_rr(dst, va);
                e.op_imm8(0xC1, if doms.get(a).copied().is_some_and(Domain::is_nonneg) { 5 } else { 7 }, dst, s, true);
            }
            Op::Cmp(k, a, b) => { let (va, vb) = (x86_val_into(e, kernel, ra, a, 0)?, x86_val_into(e, kernel, ra, b, 11)?); emit_x86_cmp(e, k, dst, va, vb); }
            Op::Select { cond, t_val, f_val } => {
                let vc = x86_val_into(e, kernel, ra, cond, 0)?;
                e.mov_rr(0, vc);
                let vt = x86_val_into(e, kernel, ra, t_val, 11)?;
                if vt == 11 { e.mov_rr(11, vt); }
                let mut t_hold = vt;
                if dst == vt && t_val != f_val { e.mov_rr(11, vt); t_hold = 11; }
                let vf = if f_val == cond { 0 } else { x86_val_into(e, kernel, ra, f_val, dst)? };
                e.mov_rr(dst, vf);
                e.push_rr(0x85, 0, 0);
                e.code.extend_from_slice(&[x86_rex_w(dst, t_hold), 0x0F, 0x45, x86_modrm(3, dst, t_hold)]); e.insns += 1;
            }
            Op::MemInit | Op::Const(_) | Op::Store { .. } => {}
        }
    }
    drain_scheduled_outputs(kernel, sched, ra, &mut oc, None, |ty, breg, lane, val| {
        let vreg = x86_val_into(e, kernel, ra, val, 0)?;
        e.store_lane(ty, vreg, breg, lane);
        Ok(())
    })
}

fn emit_x86_module(kernel: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain], max_vecs: usize, call_batch: bool) -> Result<ModuleCode, CompileError> {
    let mut e = X86Emit::new();
    let avx_plan = try_plan_avx2(kernel, sched, doms, max_vecs);
    let mut saved: Vec<u8> = if avx_plan.is_some() { Vec::new() } else { ra.callee_saved.clone() };
    if ra.buf_reg.get(2).copied() == Some(15) && !saved.contains(&15) { saved.push(15); }
    for &r in &saved { e.stack_op(r, false); }
    if let Some(&b2) = ra.buf_reg.get(2).filter(|&&r| r != 2) { e.mov_rr(b2, 2); }
    if let Some(ref steps) = avx_plan {
        emit_x86_avx2_steps(&mut e, steps, ra)?;
        e.vzeroall();
    } else {
        emit_x86_body(&mut e, kernel, sched, ra, doms)?;
    }
    for &r in saved.iter().rev() { e.stack_op(r, true); }
    e.ret();
    let k_size = e.code.len() as u64;
    while e.code.len() % 16 != 0 { e.code.push(0x90); }
    let b_off = e.code.len() as u64;

    let nb = kernel.bufs.len();
    let cnt_reg = *X86_ARGS.get(nb).ok_or_else(|| berr("x86_64 batch requires <= 5 kernel buffers"))?;
    let mut relocs = Vec::new();
    if call_batch {
        const BSAVED_ALL: [u8; 6] = [12, 13, 14, 15, 3, 5];
        let bsaved = BSAVED_ALL.get(..nb + 1).ok_or_else(|| berr("bsaved overflow"))?;
        let pad_rsp = (bsaved.len() & 1) == 0;
        for &r in bsaved { e.stack_op(r, false); }
        if pad_rsp { e.op_imm8(0x83, 5, 4, 8, false); }
        let c_hold = *bsaved.get(nb).ok_or_else(|| berr("missing c_hold"))?;
        e.mov_rr(c_hold, cnt_reg);
        for (&bs, &arg) in bsaved.iter().zip(X86_ARGS.iter()).take(nb) { e.mov_rr(bs, arg); }
        e.push_rr(0x85, c_hold, c_hold);
        let jz_pos = e.code.len();
        e.code.extend_from_slice(&[0x0F, 0x84, 0, 0, 0, 0]); e.insns += 1;
        let loop_top = e.code.len();
        for (&bs, &arg) in bsaved.iter().zip(X86_ARGS.iter()).take(nb) { e.mov_rr(arg, bs); }
        e.code.push(0xE8);
        let rel_off = e.code.len() as u64;
        e.code.extend_from_slice(&0i32.to_le_bytes()); e.insns += 1;
        relocs.push(ElfReloc { offset: rel_off, sym_idx: 0, r_type: R_X86_64_PLT32, addend: -4 });
        for (&bs, b) in bsaved.iter().zip(kernel.bufs.iter()) {
            if !matches!(b.access, BufAccess::LookupTable(_)) { e.op_imm8(0x83, 0, bs, b.block_bytes() as u8, false); }
        }
        e.op_imm8(0x83, 0, c_hold, 0xFF, false);
        let back = (loop_top as isize) - ((e.code.len() + 6) as isize);
        e.code.extend_from_slice(&[0x0F, 0x85]); e.code.extend_from_slice(&(back as i32).to_le_bytes()); e.insns += 1;
        let done_pos = e.code.len();
        let fwd = (done_pos as isize) - ((jz_pos + 6) as isize);
        if let Some(s) = e.code.get_mut(jz_pos + 2..jz_pos + 6) { s.copy_from_slice(&(fwd as i32).to_le_bytes()); }
        if pad_rsp { e.op_imm8(0x83, 0, 4, 8, false); }
        for &r in bsaved.iter().rev() { e.stack_op(r, true); }
        e.ret();
    } else {
        for &r in &saved { e.stack_op(r, false); }
        if let Some(&b2) = ra.buf_reg.get(2).filter(|&&r| r != 2) { e.mov_rr(b2, 2); }
        e.push_rr(0x85, cnt_reg, cnt_reg);
        let jz_pos = e.code.len();
        e.code.extend_from_slice(&[0x0F, 0x84, 0, 0, 0, 0]); e.insns += 1;
        let loop_top = e.code.len();
        if let Some(ref steps) = avx_plan {
            emit_x86_avx2_steps(&mut e, steps, ra)?;
        } else {
            emit_x86_body(&mut e, kernel, sched, ra, doms)?;
        }
        for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
            if !matches!(b.access, BufAccess::LookupTable(_)) { e.op_imm8(0x83, 0, br, b.block_bytes() as u8, false); }
        }
        e.op_imm8(0x83, 0, cnt_reg, 0xFF, false);
        let back = (loop_top as isize) - ((e.code.len() + 6) as isize);
        e.code.extend_from_slice(&[0x0F, 0x85]); e.code.extend_from_slice(&(back as i32).to_le_bytes()); e.insns += 1;
        let done_pos = e.code.len();
        let fwd = (done_pos as isize) - ((jz_pos + 6) as isize);
        if let Some(s) = e.code.get_mut(jz_pos + 2..jz_pos + 6) { s.copy_from_slice(&(fwd as i32).to_le_bytes()); }
        if avx_plan.is_some() { e.vzeroall(); }
        for &r in saved.iter().rev() { e.stack_op(r, true); }
        e.ret();
    }
    let b_size = (e.code.len() as u64) - b_off;
    let syms = vec![
        ElfSymbol { name: format!("zero_cc_{}", kernel.name), offset: 0, size: k_size },
        ElfSymbol { name: format!("zero_cc_{}_batch", kernel.name), offset: b_off, size: b_size },
    ];
    Ok((e.code, syms, relocs, e.insns))
}

struct ArmEmit { code: Vec<u8>, insns: usize }
impl ArmEmit {
    fn new() -> Self { Self { code: Vec::with_capacity(512), insns: 0 } }
    fn put(&mut self, w: u32) { self.code.extend_from_slice(&w.to_le_bytes()); self.insns += 1; }
    fn mov_rr(&mut self, dst: u8, src: u8) { if dst != src { self.put(arm64_rrr(0xAA00_0000, src, 0, 31, dst)); } }
    fn mov_imm_u64(&mut self, dst: u8, u: u64, fixed: bool) {
        self.put(0xD280_0000 | (((u & 0xFFFF) as u32) << 5) | (dst as u32 & 31));
        for hw in 1..4u32 {
            let part = (u >> (hw * 16)) & 0xFFFF;
            if fixed || part != 0 { self.put(0xF280_0000 | (hw << 21) | ((part as u32) << 5) | (dst as u32 & 31)); }
        }
    }
    fn mov_imm(&mut self, dst: u8, v: i64) { self.mov_imm_u64(dst, v as u64, false); }
    fn mov_imm64_fixed(&mut self, dst: u8, u: u64) { self.mov_imm_u64(dst, u, true); }
    fn ldst_lane(&mut self, ty: ElemTy, reg: u8, base: u8, lane: u8, store: bool) {
        let op_hi = match (store, ty) {
            (false, ElemTy::I64) => 0xF940_0000,
            (false, ElemTy::I32) => 0xB980_0000,
            (false, ElemTy::I16) => 0x7980_0000,
            (false, ElemTy::U32) => 0xB940_0000,
            (false, ElemTy::U16) => 0x7940_0000,
            (false, ElemTy::U8) => 0x3940_0000,
            (true, ElemTy::I64) => 0xF900_0000,
            (true, ElemTy::I32 | ElemTy::U32) => 0xB900_0000,
            (true, ElemTy::I16 | ElemTy::U16) => 0x7900_0000,
            (true, ElemTy::U8) => 0x3900_0000,
        };
        self.put(arm64_imm12(op_hi, u16::from(lane), base, reg));
    }
    fn ldr_lane(&mut self, ty: ElemTy, dst: u8, base: u8, lane: u8) { self.ldst_lane(ty, dst, base, lane, false); }
    fn str_lane(&mut self, ty: ElemTy, src: u8, base: u8, lane: u8) { self.ldst_lane(ty, src, base, lane, true); }
    fn ldr_lut(&mut self, ty: ElemTy, dst: u8, base: u8, idx: u8) {
        let (op_hi, mid) = match ty {
            ElemTy::I64 => (0xF860_0000, 0x7800), ElemTy::I32 => (0xB8A0_0000, 0x7800),
            ElemTy::I16 => (0x78A0_0000, 0x7800), ElemTy::U32 => (0xB860_0000, 0x7800),
            ElemTy::U16 => (0x7860_0000, 0x7800), ElemTy::U8 => (0x3860_0000, 0x6800),
        };
        self.put(arm64_rrr(op_hi, idx, mid, base, dst));
    }
}

fn arm_val(e: &mut ArmEmit, kernel: &Kernel, ra: &RegAlloc, id: NodeId, scratch: u8) -> Result<u8, CompileError> {
    resolve_node_reg(kernel, ra, id, scratch, |r, c| e.mov_imm(r, c))
}

fn emit_arm_body(e: &mut ArmEmit, kernel: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain]) -> Result<(), CompileError> {
    let mut oc = 0usize;
    for (step, &id) in sched.order.iter().enumerate() {
        drain_scheduled_outputs(kernel, sched, ra, &mut oc, Some(step), |ty, breg, lane, val| {
            let vreg = arm_val(e, kernel, ra, val, 16)?;
            e.str_lane(ty, vreg, breg, lane);
            Ok(())
        })?;
        let op = kernel.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if let Op::Store { buf, lane, val, .. } = op {
            let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
            let vreg = arm_val(e, kernel, ra, val, 16)?;
            e.str_lane(buf_elem_ty(kernel, buf)?, vreg, breg, lane);
            continue;
        }
        if matches!(op, Op::MemInit | Op::Const(_)) { continue; }
        let dst = ra.node_reg.get(id).and_then(|&r| r).ok_or(CompileError::InvalidNode(id))?;
        match op {
            Op::Load { buf, lane, .. } => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                e.ldr_lane(buf_elem_ty(kernel, buf)?, dst, breg, lane);
            }
            Op::LoadLut { buf, idx } => {
                let (breg, ireg) = (*ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, arm_val(e, kernel, ra, idx, 16)?);
                let b = kernel.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let BufAccess::LookupTable(len) = b.access else { return Err(CompileError::InvalidBuffer(buf)); };
                let d = doms.get(idx).copied().unwrap_or_else(Domain::top);
                if d.is_empty() || d.lo < 0 || d.hi >= len as i64 {
                    e.mov_imm(17, len as i64); e.put(arm64_rrr(0xEB00_0000, 17, 0, ireg, 31));
                    e.put(0x5400_0043); e.put(0xD420_0000);
                }
                e.ldr_lut(b.elem_ty, dst, breg, ireg);
            }
            Op::BoundsCheck { idx, limit } => {
                let ireg = arm_val(e, kernel, ra, idx, 16)?;
                e.mov_imm(17, limit); e.put(arm64_rrr(0xEB00_0000, 17, 0, ireg, 31));
                e.put(0x5400_0043); e.put(0xD420_0000); e.mov_rr(dst, ireg);
            }
            Op::Add(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x8B00_0000, vb, 0, va, dst)); }
            Op::Sub(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0xCB00_0000, vb, 0, va, dst)); }
            Op::BitAnd(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x8A00_0000, vb, 0, va, dst)); }
            Op::Mul(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x9B00_0000, vb, 0x7C00, va, dst)); }
            Op::Div(a, b) | Op::Rem(a, b) => {
                let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?);
                let (da, db) = (doms.get(a).copied().unwrap_or_else(Domain::top), doms.get(b).copied().unwrap_or_else(Domain::top));
                if db.is_empty() || db.contains(0) {
                    e.put(0xB400_0040 | (vb as u32 & 31)); e.put(0xD420_0000);
                }
                if (da.is_empty() || da.contains(i64::MIN)) && (db.is_empty() || db.contains(-1)) {
                    e.put(arm64_imm12(0xB100_0000, 1, vb, 31));
                    let skip = if vb == 17 { 0x5400_00C1 } else { 0x5400_00A1 };
                    e.put(skip);
                    e.put(0xD2E0_0000 | (0x8000 << 5) | 17);
                    e.put(arm64_rrr(0xEB00_0000, 17, 0, va, 31));
                    e.put(0x5400_0041);
                    e.put(0xD420_0000);
                    if vb == 17 { e.put(0x9280_0011); }
                }
                if matches!(op, Op::Div(_, _)) {
                    e.put(arm64_rrr(0x9AC0_0000, vb, 0x0C00, va, dst));
                } else {
                    let q_tmp = if va == 16 && vb != 17 { 17 } else if va != 16 { 16 } else { dst };
                    if q_tmp == dst && (dst == va || dst == vb) { return Err(berr("arm64 rem scratch conflict")); }
                    e.put(arm64_rrr(0x9AC0_0000, vb, 0x0C00, va, q_tmp));
                    e.put(arm64_rrr(0x9B00_0000, vb, 0x8000 | ((va as u32 & 31) << 10), q_tmp, dst));
                }
            }
            Op::Shl(a, s) => {
                let va = arm_val(e, kernel, ra, a, 16)?;
                if s == 0 { e.mov_rr(dst, va); } else {
                    let (immr, imms) = (((64 - u32::from(s)) & 63) << 16, (63 - u32::from(s)) << 10);
                    e.put(0xD340_0000 | immr | imms | ((va as u32 & 31) << 5) | (dst as u32 & 31));
                }
            }
            Op::Shr(a, s) => {
                let va = arm_val(e, kernel, ra, a, 16)?;
                if s == 0 { e.mov_rr(dst, va); } else {
                    let op_hi = if doms.get(a).copied().is_some_and(Domain::is_nonneg) { 0xD340_FC00 } else { 0x9340_FC00 };
                    e.put(op_hi | ((u32::from(s) & 63) << 16) | ((va as u32 & 31) << 5) | (dst as u32 & 31));
                }
            }
            Op::Cmp(k, a, b) => {
                let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?);
                e.put(arm64_rrr(0xEB00_0000, vb, 0, va, 31));
                let inv_cc = match k { CmpKind::Eq => 1, CmpKind::Ne => 0, CmpKind::Lt => 10, CmpKind::Ge => 11, CmpKind::Le => 12, CmpKind::Gt => 13 };
                e.put(0x9A9F_07E0 | (inv_cc << 12) | (dst as u32 & 31));
            }
            Op::Select { cond, t_val, f_val } => {
                let vc = arm_val(e, kernel, ra, cond, 16)?;
                e.put(arm64_rrr(0xEB00_0000, 31, 0, vc, 31));
                let (vt, vf) = (arm_val(e, kernel, ra, t_val, 16)?, arm_val(e, kernel, ra, f_val, 17)?);
                e.put(arm64_rrr(0x9A80_0000, vf, 0x1000, vt, dst));
            }
            Op::MemInit | Op::Const(_) | Op::Store { .. } => {}
        }
    }
    drain_scheduled_outputs(kernel, sched, ra, &mut oc, None, |ty, breg, lane, val| {
        let vreg = arm_val(e, kernel, ra, val, 16)?;
        e.str_lane(ty, vreg, breg, lane);
        Ok(())
    })
}

fn emit_arm_callee_save(e: &mut ArmEmit, saved: &[u8], load: bool) {
    if saved.is_empty() { return; }
    let pairs = saved.len().div_ceil(2);
    let frame = (pairs as u32) * 16;
    if !load { e.put(arm64_imm12(0xD100_0000, frame as u16, 31, 31)); }
    for p in 0..pairs {
        let r1 = saved[p * 2];
        let r2 = saved.get(p * 2 + 1).copied().unwrap_or(31);
        let imm7 = ((p as u32) * 2) & 0x7F;
        let opc = if load { 0xA940_0000 } else { 0xA900_0000 };
        e.put(opc | (imm7 << 15) | ((r2 as u32 & 31) << 10) | (31 << 5) | (r1 as u32 & 31));
    }
    if load { e.put(arm64_imm12(0x9100_0000, frame as u16, 31, 31)); }
}

fn emit_arm_module(kernel: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain], call_batch: bool) -> Result<ModuleCode, CompileError> {
    let mut e = ArmEmit::new();
    emit_arm_callee_save(&mut e, &ra.callee_saved, false);
    emit_arm_body(&mut e, kernel, sched, ra, doms)?;
    emit_arm_callee_save(&mut e, &ra.callee_saved, true);
    e.put(0xD65F_03C0);
    let k_size = e.code.len() as u64;
    while e.code.len() % 16 != 0 { e.put(0xD503_201F); }
    let b_off = e.code.len() as u64;

    let cnt_reg = kernel.bufs.len() as u8;
    let mut relocs = Vec::new();
    if call_batch {
        let mut bsaved = vec![30u8];
        for i in 0..=kernel.bufs.len() { bsaved.push(19 + i as u8); }
        emit_arm_callee_save(&mut e, &bsaved, false);
        let c_hold = 19 + kernel.bufs.len() as u8;
        e.mov_rr(c_hold, cnt_reg);
        for i in 0..kernel.bufs.len() { e.mov_rr(19 + i as u8, i as u8); }
        let cbz_pos = e.code.len();
        e.put(0xB400_0000 | (c_hold as u32 & 31));
        let loop_top = e.code.len();
        for i in 0..kernel.bufs.len() { e.mov_rr(i as u8, 19 + i as u8); }
        let bl_off = e.code.len() as u64;
        e.put(0x9400_0000);
        relocs.push(ElfReloc { offset: bl_off, sym_idx: 0, r_type: R_AARCH64_CALL26, addend: 0 });
        for (i, b) in kernel.bufs.iter().enumerate() {
            if !matches!(b.access, BufAccess::LookupTable(_)) { e.put(arm64_imm12(0x9100_0000, b.block_bytes() as u16, 19 + i as u8, 19 + i as u8)); }
        }
        e.put(arm64_imm12(0xF100_0000, 1, c_hold, c_hold));
        let back_words = ((loop_top as isize) - (e.code.len() as isize)) / 4;
        e.put(0x5400_0001 | (((back_words as u32) & 0x7_FFFF) << 5));
        let done_words = ((e.code.len() as isize) - (cbz_pos as isize)) / 4;
        let cbz_w = 0xB400_0000 | (((done_words as u32) & 0x7_FFFF) << 5) | (c_hold as u32 & 31);
        if let Some(s) = e.code.get_mut(cbz_pos..cbz_pos + 4) { s.copy_from_slice(&cbz_w.to_le_bytes()); }
        emit_arm_callee_save(&mut e, &bsaved, true);
        e.put(0xD65F_03C0);
    } else {
        emit_arm_callee_save(&mut e, &ra.callee_saved, false);
        let cbz_pos = e.code.len();
        e.put(0xB400_0000 | (cnt_reg as u32 & 31));
        let loop_top = e.code.len();
        emit_arm_body(&mut e, kernel, sched, ra, doms)?;
        for (i, b) in kernel.bufs.iter().enumerate() {
            if !matches!(b.access, BufAccess::LookupTable(_)) { e.put(arm64_imm12(0x9100_0000, b.block_bytes() as u16, i as u8, i as u8)); }
        }
        e.put(arm64_imm12(0xF100_0000, 1, cnt_reg, cnt_reg));
        let back_words = ((loop_top as isize) - (e.code.len() as isize)) / 4;
        e.put(0x5400_0001 | (((back_words as u32) & 0x7_FFFF) << 5));
        let done_words = ((e.code.len() as isize) - (cbz_pos as isize)) / 4;
        let cbz_w = 0xB400_0000 | (((done_words as u32) & 0x7_FFFF) << 5) | (cnt_reg as u32 & 31);
        if let Some(s) = e.code.get_mut(cbz_pos..cbz_pos + 4) { s.copy_from_slice(&cbz_w.to_le_bytes()); }
        emit_arm_callee_save(&mut e, &ra.callee_saved, true);
        e.put(0xD65F_03C0);
    }
    let b_size = (e.code.len() as u64) - b_off;
    let syms = vec![
        ElfSymbol { name: format!("zero_cc_{}", kernel.name), offset: 0, size: k_size },
        ElfSymbol { name: format!("zero_cc_{}_batch", kernel.name), offset: b_off, size: b_size },
    ];
    Ok((e.code, syms, relocs, e.insns))
}

fn emit_ffi_shim(kernel: &Kernel, ksym: &str, bsym: &str) -> (String, usize) {
    let (mut c_args, mut k_params, mut k_call, mut b_params, mut b_call, mut lbufs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in &kernel.bufs {
        let (ty, n) = (b.elem_ty.rust_name(), b.lanes);
        match b.access {
            BufAccess::ReadOnly => {
                c_args.push(format!("{}: *const [{ty}; {n}]", b.name));
                k_params.push(format!("{}: &[{ty}; {n}]", b.name)); k_call.push(format!("{}.as_ptr().cast()", b.name));
                b_params.push(format!("{}: &[[{ty}; {n}]]", b.name)); b_call.push(format!("{}.as_ptr()", b.name)); lbufs.push(b.name.as_str());
            }
            BufAccess::MutNoAlias => {
                c_args.push(format!("{}: *mut [{ty}; {n}]", b.name));
                k_params.push(format!("{}: &mut [{ty}; {n}]", b.name)); k_call.push(format!("{}.as_mut_ptr().cast()", b.name));
                b_params.push(format!("{}: &mut [[{ty}; {n}]]", b.name)); b_call.push(format!("{}.as_mut_ptr()", b.name)); lbufs.push(b.name.as_str());
            }
            BufAccess::LookupTable(len) => {
                c_args.push(format!("{}: *const {ty}", b.name));
                k_params.push(format!("{}: &[{ty}; {len}]", b.name)); k_call.push(format!("{}.as_ptr()", b.name));
                b_params.push(format!("{}: &[{ty}; {len}]", b.name)); b_call.push(format!("{}.as_ptr()", b.name));
            }
        }
    }
    let mut b_cargs = c_args.clone(); b_cargs.push(String::from("blocks: usize"));
    let mut s = String::with_capacity(512);
    let _ = writeln!(s, "extern \"C\" {{\n    fn {ksym}({});\n    fn {bsym}({});\n}}", c_args.join(", "), b_cargs.join(", "));
    let _ = writeln!(s, "#[inline(never)]\npub fn {}({}) {{\n    unsafe {{ {ksym}({}); }}\n}}", kernel.name, k_params.join(", "), k_call.join(", "));
    let len_expr = lbufs.first().map_or_else(|| String::from("0"), |f| lbufs[1..].iter().fold(format!("{f}.len()"), |acc, &lb| format!("{acc}.min({lb}.len())")));
    b_call.push(String::from("blocks"));
    let _ = writeln!(s, "#[inline(never)]\npub fn {}_batch({}) {{\n    let blocks = {len_expr};\n    unsafe {{ {bsym}({}); }}\n}}", kernel.name, b_params.join(", "), b_call.join(", "));
    let glue_insns = 4 + lbufs.len().saturating_sub(1) * 2;
    (s, glue_insns)
}

type PackedExecData = (Vec<u8>, Vec<u64>, usize, Option<(u64, usize)>);

fn pack_exec_data(kernel: &Kernel, inputs: &[&[i64]]) -> Result<PackedExecData, CompileError> {
    let blocks = if inputs.is_empty() {
        1usize
    } else {
        if inputs.len() != kernel.bufs.len() { return Err(CompileError::InvalidBuffer(inputs.len() as u8)); }
        let mut blk_opt: Option<usize> = None;
        for (b, slice) in kernel.bufs.iter().zip(inputs.iter()) {
            match b.access {
                BufAccess::LookupTable(len) => {
                    if slice.len() < len { return Err(CompileError::OutOfBounds { idx: slice.len() as i64, limit: len as i64 }); }
                }
                _ => {
                    let lanes = usize::from(b.lanes);
                    if lanes == 0 || slice.len() % lanes != 0 { return Err(berr("standalone input length not multiple of lanes")); }
                    let nb = slice.len() / lanes;
                    if let Some(prev) = blk_opt {
                        if prev != nb { return Err(berr("standalone inputs have mismatched block counts")); }
                    } else {
                        blk_opt = Some(nb);
                    }
                }
            }
        }
        blk_opt.unwrap_or(1)
    };
    let (mut data, mut buf_offs, mut primary_out) = (Vec::new(), Vec::with_capacity(kernel.bufs.len()), None);
    for (idx, b) in kernel.bufs.iter().enumerate() {
        while data.len() % 64 != 0 { data.push(0); }
        let off = data.len() as u64;
        buf_offs.push(off);
        let elem_count = match b.access {
            BufAccess::LookupTable(len) => len,
            _ => blocks.saturating_mul(usize::from(b.lanes)),
        };
        if inputs.is_empty() {
            for _ in 0..elem_count { b.elem_ty.push_le_elem(&mut data, 0); }
        } else if let Some(slice) = inputs.get(idx) {
            for &v in slice.iter().take(elem_count) { b.elem_ty.push_le_elem(&mut data, v); }
        }
        if b.access == BufAccess::MutNoAlias && primary_out.is_none() {
            primary_out = Some((off, blocks.saturating_mul(b.block_bytes())));
        }
    }
    Ok((data, buf_offs, blocks, primary_out))
}

fn emit_x86_start(start_off: usize, b_off: u64, data_vaddr: u64, buf_offs: &[u64], blocks: usize, primary_out: Option<(u64, usize)>) -> Result<(Vec<u8>, usize), CompileError> {
    let mut e = X86Emit::new();
    e.code.extend_from_slice(&[0x48, 0x83, 0xE4, 0xF0]); e.insns += 1;
    for (i, &boff) in buf_offs.iter().enumerate() {
        let arg = *X86_ARGS.get(i).ok_or_else(|| berr("x86_64 _start arg overflow"))?;
        e.movabs_imm64(arg, data_vaddr.saturating_add(boff));
    }
    let cnt_reg = *X86_ARGS.get(buf_offs.len()).ok_or_else(|| berr("x86_64 _start count reg overflow"))?;
    e.movabs_imm64(cnt_reg, blocks as u64);
    let call_end = start_off + e.code.len() + 5;
    let rel = (b_off as isize) - (call_end as isize);
    e.code.push(0xE8); e.code.extend_from_slice(&(rel as i32).to_le_bytes()); e.insns += 1;

    if let Some((out_off, out_bytes)) = primary_out.filter(|&(_, n)| n > 0) {
        e.movabs_imm64(6, data_vaddr.saturating_add(out_off));
        e.movabs_imm64(2, out_bytes as u64);
        e.code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); e.insns += 1;
        e.code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]); e.insns += 1;
        e.code.extend_from_slice(&[0x0F, 0x05]); e.insns += 1;
        e.code.extend_from_slice(&[0x31, 0xFF]); e.insns += 1;
        e.code.extend_from_slice(&[0x31, 0xC9]); e.insns += 1;
        let loop_top = e.code.len();
        e.code.extend_from_slice(&[0x0F, 0xB6, 0x04, 0x0E]); e.insns += 1;
        e.code.extend_from_slice(&[0x01, 0xC7]); e.insns += 1;
        e.code.extend_from_slice(&[0x48, 0xFF, 0xC1]); e.insns += 1;
        e.code.extend_from_slice(&[0x48, 0x39, 0xD1]); e.insns += 1;
        let rel8 = (loop_top as isize) - ((e.code.len() + 2) as isize);
        e.code.extend_from_slice(&[0x72, rel8 as u8]); e.insns += 1;
        e.code.extend_from_slice(&[0x83, 0xE7, 0x7F]); e.insns += 1;
    } else {
        e.code.extend_from_slice(&[0x31, 0xFF]); e.insns += 1;
    }
    e.code.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]); e.insns += 1;
    e.code.extend_from_slice(&[0x0F, 0x05]); e.insns += 1;
    e.ud2();
    Ok((e.code, e.insns))
}

fn emit_arm_start(start_off: usize, b_off: u64, data_vaddr: u64, buf_offs: &[u64], blocks: usize, primary_out: Option<(u64, usize)>) -> Result<(Vec<u8>, usize), CompileError> {
    if buf_offs.len() > 7 { return Err(berr("aarch64 _start arg overflow")); }
    let mut e = ArmEmit::new();
    for (i, &boff) in buf_offs.iter().enumerate() {
        e.mov_imm64_fixed(i as u8, data_vaddr.saturating_add(boff));
    }
    e.mov_imm64_fixed(buf_offs.len() as u8, blocks as u64);
    let bl_words = ((b_off as isize) - ((start_off + e.code.len()) as isize)) / 4;
    e.put(0x9400_0000 | ((bl_words as u32) & 0x03FF_FFFF));

    if let Some((out_off, out_bytes)) = primary_out.filter(|&(_, n)| n > 0) {
        e.mov_imm64_fixed(19, data_vaddr.saturating_add(out_off));
        e.mov_imm64_fixed(20, out_bytes as u64);
        e.put(0xD280_0020);
        e.mov_rr(1, 19);
        e.mov_rr(2, 20);
        e.put(0xD280_0808);
        e.put(0xD400_0001);
        e.put(0xD280_0000);
        e.put(0xD280_0001);
        let loop_top = e.code.len();
        e.put(arm64_rrr(0x3860_0000, 1, 0x6800, 19, 2));
        e.put(arm64_rrr(0x8B00_0000, 2, 0, 0, 0));
        e.put(arm64_imm12(0x9100_0000, 1, 1, 1));
        e.put(arm64_rrr(0xEB00_0000, 20, 0, 1, 31));
        let back_words = ((loop_top as isize) - (e.code.len() as isize)) / 4;
        e.put(0x5400_0003 | (((back_words as u32) & 0x7_FFFF) << 5));
        e.put(0x9240_1800);
    } else {
        e.put(0xD280_0000);
    }
    e.put(0xD280_0BA8);
    e.put(0xD400_0001);
    e.put(0xD420_0000);
    Ok((e.code, e.insns))
}

fn lower_kernel_code(kernel: &Kernel, budget: RegBudget, isa: TargetIsa, optimize: bool, call_batch: bool) -> Result<(Kernel, RegAlloc, ModuleCode), CompileError> {
    kernel.validate()?;
    let lowered = if optimize { optimize_kernel(kernel)? } else { kernel.clone() };
    let sched = schedule_and_verify(&lowered, budget)?;
    let ra = allocate_registers(&lowered, &sched, isa, budget)?;
    let doms = lowered.infer_domains();
    let module = match isa {
        TargetIsa::X86_64 => emit_x86_module(&lowered, &sched, &ra, &doms, budget.max_vecs, call_batch)?,
        TargetIsa::Aarch64 => emit_arm_module(&lowered, &sched, &ra, &doms, call_batch)?,
    };
    Ok((lowered, ra, module))
}

/// Lower an already-validated `Kernel` (with optional E-Graph pass) directly into a standalone `ELF64` `.o`.
pub fn emit_kernel_elf(kernel: &Kernel, budget: RegBudget, isa: TargetIsa, optimize: bool, call_batch: bool) -> Result<CompiledObject, CompileError> {
    let (lowered, ra, (code_bytes, syms, relocs, direct_insns)) = lower_kernel_code(kernel, budget, isa, optimize, call_batch)?;
    let elf_bytes = emit_elf64_object(isa, &code_bytes, &syms, &relocs)?;
    let summary = verify_elf64_object(&elf_bytes)?;
    if summary.machine != isa.e_machine() || !summary.has_nonexec_gnu_stack || summary.symbols.len() != 2 {
        return Err(berr("elf64 airlock verification mismatch"));
    }
    let (kernel_sym, batch_sym) = (syms[0].name.clone(), syms[1].name.clone());
    let (rust_ffi_shim, shim_glue_insns) = emit_ffi_shim(&lowered, &kernel_sym, &batch_sym);
    let total = direct_insns + shim_glue_insns;
    let direct_share_pct = if total == 0 { 100 } else { ((direct_insns * 100) / total) as u8 };
    Ok(CompiledObject { isa, kernel_sym, batch_sym, code_bytes, elf_bytes, direct_insns, shim_glue_insns, direct_share_pct, zero_spills: ra.spills == 0, rust_ffi_shim })
}

/// Full sovereign compilation pipeline: validate -> E-Graph -> Sethi-Ullman -> Linear-Scan RegAlloc -> Direct `ELF64` `.o`.
pub fn compile_kernel_direct(kernel: &Kernel, budget: RegBudget, isa: TargetIsa) -> Result<CompiledObject, CompileError> {
    emit_kernel_elf(kernel, budget, isa, true, false)
}

/// Compile raw safe-Rust `.rs` kernel source directly to a standalone `ELF64` `.o` object.
pub fn compile_rust_to_elf(src: &str, budget: RegBudget, isa: TargetIsa) -> Result<CompiledObject, CompileError> {
    compile_kernel_direct(&parse_rust_kernel(src)?, budget, isa)
}

/// Compile a `Kernel` directly to a `100.0%` LLVM-free, static Linux `ELF64` `ET_EXEC` binary
/// (`_start` -> `zero_cc_<name>_batch` -> `sys_write` primary output -> `sys_exit(byte_sum & 0x7F)`).
pub fn compile_standalone_elf_exec(kernel: &Kernel, budget: RegBudget, isa: TargetIsa, inputs: &[&[i64]]) -> Result<StandaloneExec, CompileError> {
    let (lowered, ra, (mut code_bytes, mut syms, _, mut direct_insns)) = lower_kernel_code(kernel, budget, isa, true, false)?;
    let (data, buf_offs, blocks, primary_out) = pack_exec_data(&lowered, inputs)?;
    match isa {
        TargetIsa::X86_64 => while code_bytes.len() % 16 != 0 { code_bytes.push(0x90); },
        TargetIsa::Aarch64 => while code_bytes.len() % 16 != 0 { code_bytes.extend_from_slice(&0xD503_201Fu32.to_le_bytes()); },
    }
    let b_off = syms.get(1).ok_or_else(|| berr("missing batch symbol"))?.offset;
    let start_off = code_bytes.len();
    let emit_start = |vaddr: u64| match isa {
        TargetIsa::X86_64 => emit_x86_start(start_off, b_off, vaddr, &buf_offs, blocks, primary_out),
        TargetIsa::Aarch64 => emit_arm_start(start_off, b_off, vaddr, &buf_offs, blocks, primary_out),
    };
    let (pass1, _) = emit_start(0)?;
    let data_vaddr = elf64_exec_data_vaddr(start_off + pass1.len());
    let (start_bytes, start_insns) = emit_start(data_vaddr)?;
    if start_bytes.len() != pass1.len() { return Err(berr("standalone _start length invariance violated")); }
    code_bytes.extend_from_slice(&start_bytes);
    direct_insns += start_insns;
    syms.push(ElfSymbol { name: String::from("_start"), offset: start_off as u64, size: start_bytes.len() as u64 });

    let elf_bytes = emit_elf64_executable(isa, &code_bytes, start_off as u64, &data, 0, &syms)?;
    let summary = verify_elf64_executable(&elf_bytes)?;
    let entry_vaddr = ELF_EXEC_TEXT_VADDR + start_off as u64;
    if summary.machine != isa.e_machine() || !summary.has_nonexec_gnu_stack || summary.symbols.len() != 3
        || summary.entry_vaddr != entry_vaddr || summary.data_vaddr != data_vaddr
    {
        return Err(berr("elf64 exec airlock verification mismatch"));
    }
    let output_bytes = primary_out.map_or(0, |(_, n)| n);
    Ok(StandaloneExec {
        isa, kernel_sym: syms[0].name.clone(), batch_sym: syms[1].name.clone(), entry_sym: String::from("_start"),
        entry_vaddr, data_vaddr, code_bytes, elf_bytes, direct_insns, shim_glue_insns: 0, direct_share_pct: 100,
        zero_spills: ra.spills == 0, blocks, output_bytes,
    })
}

/// Compile raw safe-Rust `.rs` kernel source directly into a `100.0%` LLVM-free static Linux `ELF64` `ET_EXEC` binary.
pub fn compile_rust_to_elf_exec(src: &str, budget: RegBudget, isa: TargetIsa, inputs: &[&[i64]]) -> Result<StandaloneExec, CompileError> {
    compile_standalone_elf_exec(&parse_rust_kernel(src)?, budget, isa, inputs)
}
