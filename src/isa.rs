use std::fmt::Write as _;

use crate::codegen::{remat_readonly_loads, schedule_and_verify, RegBudget, ScheduleReport};
use crate::elf64::{
    elf64_exec_data_vaddr, emit_elf64_executable, emit_elf64_object, verify_elf64_executable,
    verify_elf64_object, ElfReloc, ElfSymbol, TargetIsa, ELF_EXEC_TEXT_VADDR, R_AARCH64_CALL26,
    R_X86_64_PLT32,
};
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op, LANES};
use crate::optimize_kernel;
use crate::rust_fe::parse_rust_kernel;

const X86_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
const X86_CALLEE: [u8; 6] = [3, 5, 12, 13, 14, 15];
const X86_POOL: [u8; 13] = [10, 1, 2, 6, 7, 8, 9, 3, 5, 12, 13, 14, 15];
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
            let need = if isa == TargetIsa::X86_64 { ostep } else { ostep.saturating_add(1) };
            if let Some(ls) = last_step.get_mut(o.val) { *ls = (*ls).max(need); }
        }
    }

    let mut free: Vec<u8> = pool.into_iter().rev().collect();
    let (mut node_reg, mut active, mut used_callee) = (vec![None; n], Vec::<(NodeId, u8)>::new(), Vec::new());
    for (step, &id) in sched.order.iter().enumerate() {
        let op = kernel.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        let reuse_dying = isa == TargetIsa::X86_64;
        if reuse_dying {
            let mut i = 0usize;
            while i < active.len() {
                let (aid, areg) = active[i];
                if last_step.get(aid).copied().unwrap_or(0) <= step { active.swap_remove(i); free.push(areg); } else { i += 1; }
            }
        }
        if !is_inline(op) {
            let Some(reg) = free.pop() else {
                return Err(CompileError::RegisterSpill { peak_gprs: nb + active.len() + 1, max_gprs: budget.max_gprs, peak_vecs: sched.peak_vecs, max_vecs: budget.max_vecs });
            };
            let is_cs = if isa == TargetIsa::X86_64 { X86_CALLEE.contains(&reg) } else { reg >= 19 };
            if is_cs && !used_callee.contains(&reg) { used_callee.push(reg); }
            if let Some(slot) = node_reg.get_mut(id) { *slot = Some(reg); }
            if last_step.get(id).copied().unwrap_or(0) > step { active.push((id, reg)); } else { free.push(reg); }
        }
        if !reuse_dying {
            let mut i = 0usize;
            while i < active.len() {
                let (aid, areg) = active[i];
                if last_step.get(aid).copied().unwrap_or(0) <= step { active.swap_remove(i); free.push(areg); } else { i += 1; }
            }
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

pub(crate) struct X86Emit { pub(crate) code: Vec<u8>, pub(crate) insns: usize, const_pool: Vec<i64>, const_patches: Vec<(usize, usize)> }
impl X86Emit {
    fn new() -> Self { Self { code: Vec::with_capacity(512), insns: 0, const_pool: Vec::new(), const_patches: Vec::new() } }
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
        let need_w = matches!(ty, ElemTy::I64 | ElemTy::U64 | ElemTy::I32 | ElemTy::I16);
        if need_w || dst >= 8 || idx >= 8 || base >= 8 {
            self.code.push(x86_rex_byte(need_w, dst, idx, base));
        }
        match ty {
            ElemTy::I64 | ElemTy::U64 | ElemTy::U32 => self.code.push(0x8B),
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
            ElemTy::I64 | ElemTy::U64 => self.code.extend_from_slice(&[x86_rex_w(src, base), 0x89]),
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
    fn add_imm(&mut self, dst: u8, imm: i32) {
        if imm == 0 { return; }
        if let Ok(c8) = i8::try_from(imm) { self.op_imm8(0x83, 0, dst, c8 as u8, false); }
        else { self.code.extend_from_slice(&[x86_rex_w(0, dst), 0x81, x86_modrm(3, 0, dst)]); self.code.extend_from_slice(&imm.to_le_bytes()); self.insns += 1; }
    }
    fn stack_op(&mut self, r: u8, is_pop: bool) {
        if r >= 8 { self.code.push(0x41); }
        self.code.push((if is_pop { 0x58 } else { 0x50 }) + (r & 7));
        self.insns += 1;
    }
    pub(crate) fn vex_ymm_op_w(&mut self, map: u8, w: u8, opc: u8, dst: u8, vvvv: u8, rm: u8) {
        let vw = x86_vex3_word((dst >> 3) & 1, (rm >> 3) & 1, map, w, vvvv, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], opc, x86_modrm(3, dst, rm)]);
        self.insns += 1;
    }
    pub(crate) fn vex_ymm_op(&mut self, map: u8, opc: u8, dst: u8, vvvv: u8, rm: u8) {
        self.vex_ymm_op_w(map, 0, opc, dst, vvvv, rm);
    }
    fn prefetcht0(&mut self, base: u8, disp: i32) {
        if base >= 8 { self.code.push(x86_rex_byte(false, 0, 0, base)); }
        self.code.extend_from_slice(&[0x0F, 0x18]);
        self.push_mem_disp(1, base, disp);
    }
    fn bmi2_shift(&mut self, pp: u8, dst: u8, src: u8, amt: u8) {
        let vw = x86_vex3_word((dst >> 3) & 1, (src >> 3) & 1, 2, 1, amt, 0, pp).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0xF7, x86_modrm(3, dst, src)]);
        self.insns += 1;
    }
    pub(crate) fn vex_ymm_imm(&mut self, map: u8, opc: u8, dst: u8, vvvv: u8, rm: u8, imm: u8) {
        self.vex_ymm_op(map, opc, dst, vvvv, rm);
        self.code.push(imm);
    }
    pub(crate) fn vex_vshufps(&mut self, dst: u8, vvvv: u8, rm: u8, imm: u8) {
        let vw = x86_vex3_word((dst >> 3) & 1, (rm >> 3) & 1, 1, 0, vvvv, 1, 0).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0xC6, x86_modrm(3, dst, rm), imm]);
        self.insns += 1;
    }
    pub(crate) fn vex_vpermq(&mut self, dst: u8, src: u8, imm: u8) {
        let vw = x86_vex3_word((dst >> 3) & 1, (src >> 3) & 1, 3, 1, 0, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x00, x86_modrm(3, dst, src), imm]);
        self.insns += 1;
    }
    pub(crate) fn vex_vptest(&mut self, a: u8, b: u8) {
        let vw = x86_vex3_word((a >> 3) & 1, (b >> 3) & 1, 2, 0, 0, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x17, x86_modrm(3, a, b)]);
        self.insns += 1;
    }
    pub(crate) fn jcc_placeholder(&mut self, cc: u8) -> usize {
        let pos = self.code.len();
        self.code.extend_from_slice(&[0x0F, cc, 0, 0, 0, 0]);
        self.insns += 1;
        pos
    }
    pub(crate) fn jmp_placeholder(&mut self) -> usize {
        let pos = self.code.len();
        self.code.extend_from_slice(&[0xE9, 0, 0, 0, 0]);
        self.insns += 1;
        pos
    }
    pub(crate) fn patch_rel32(&mut self, pos: usize, op_len: usize, target: usize) {
        let rel = (target as isize) - ((pos + op_len + 4) as isize);
        if let Some(s) = self.code.get_mut(pos + op_len..pos + op_len + 4) { s.copy_from_slice(&(rel as i32).to_le_bytes()); }
    }
    fn vex_extracti128(&mut self, dst_xmm: u8, src_ymm: u8, imm: u8) {
        self.vex_ymm_imm(3, 0x39, src_ymm, 0, dst_xmm, imm);
    }
    pub(crate) fn vex_shift_imm(&mut self, opc: u8, sub: u8, dst: u8, src: u8, imm: u8) {
        let vw = x86_vex3_word(0, (src >> 3) & 1, 1, 0, dst, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], opc, x86_modrm(3, sub, src), imm]);
        self.insns += 1;
    }
    pub(crate) fn vex_mem_op(&mut self, enc: [u8; 6], reg: u8, base: u8, disp: i32) {
        let [map, l, pp, w, vvvv, opc] = enc;
        let vw = x86_vex3_word((reg >> 3) & 1, (base >> 3) & 1, map, w, vvvv, l, pp).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], opc]);
        self.push_mem_disp(reg, base, disp);
    }
    fn vex_mem_imm(&mut self, enc: [u8; 6], reg: u8, base: u8, disp: i32, imm: u8) {
        self.vex_mem_op(enc, reg, base, disp);
        self.code.push(imm);
    }
    fn vex_broadcast_imm(&mut self, dst: u8, val: i64, is_64: bool) {
        if val == 0 { self.vex_ymm_op(1, 0xEF, dst, dst, dst); return; }
        self.mov_imm(0, if is_64 { val } else { i64::from(val as i32) });
        let vw = x86_vex3_word((dst >> 3) & 1, 0, 1, u8::from(is_64), 0, 0, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x6E, x86_modrm(3, dst, 0)]);
        self.insns += 1;
        self.vex_ymm_op(2, if is_64 { 0x59 } else { 0x58 }, dst, 0, dst);
    }
    pub(crate) fn vex_broadcast_rip(&mut self, dst: u8, val: i64) {
        if val == 0 { self.vex_ymm_op(1, 0xEF, dst, dst, dst); return; }
        let idx = match self.const_pool.iter().position(|&c| c == val) {
            Some(i) => i,
            None => { let i = self.const_pool.len(); self.const_pool.push(val); i }
        };
        let vw = x86_vex3_word((dst >> 3) & 1, 0, 2, 0, 0, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x59, x86_modrm(0, dst, 5)]);
        let disp_off = self.code.len();
        self.code.extend_from_slice(&0i32.to_le_bytes());
        self.const_patches.push((disp_off, idx));
        self.insns += 1;
    }
    pub(crate) fn vex_broadcast32_rip(&mut self, dst: u8, val: u32) {
        if val == 0 { self.vex_ymm_op(1, 0xEF, dst, dst, dst); return; }
        if val == u32::MAX { self.vex_ymm_op(1, 0x76, dst, dst, dst); return; }
        let c64 = i64::from(val);
        let idx = match self.const_pool.iter().position(|&c| c == c64) {
            Some(i) => i,
            None => { let i = self.const_pool.len(); self.const_pool.push(c64); i }
        };
        let vw = x86_vex3_word((dst >> 3) & 1, 0, 2, 0, 0, 1, 1).to_le_bytes();
        self.code.extend_from_slice(&[0xC4, vw[0], vw[1], 0x58, x86_modrm(0, dst, 5)]);
        let disp_off = self.code.len();
        self.code.extend_from_slice(&0i32.to_le_bytes());
        self.const_patches.push((disp_off, idx));
        self.insns += 1;
    }
    fn finalize_const_pool(&mut self) {
        if self.const_pool.is_empty() { return; }
        while self.code.len() % 8 != 0 { self.code.push(0x90); }
        let pool_base = self.code.len();
        for &c in &self.const_pool { self.code.extend_from_slice(&c.to_le_bytes()); }
        for &(disp_off, idx) in &self.const_patches {
            let target = pool_base.saturating_add(idx.saturating_mul(8));
            let disp = (target as isize) - ((disp_off + 4) as isize);
            if let Some(s) = self.code.get_mut(disp_off..disp_off + 4) { s.copy_from_slice(&(disp as i32).to_le_bytes()); }
        }
    }
    fn vzeroupper(&mut self) { self.code.extend_from_slice(&[0xC5, 0xF8, 0x77]); self.insns += 1; }
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

impl AvxStep {
    fn dst(&self) -> Option<u8> {
        match *self {
            Self::Const(d, _) | Self::Load(d, _, _) | Self::Add(d, _, _) | Self::Sub(d, _, _)
            | Self::BitAnd(d, _, _) | Self::MulU32(d, _, _) | Self::Shl(d, _, _) | Self::ShrU64(d, _, _) => Some(d),
            Self::Store(_, _, _) => None,
        }
    }
}

fn emit_x86_avx2_steps(e: &mut X86Emit, steps: &[AvxStep], ra: &RegAlloc, only_hoist: Option<bool>) -> Result<(), CompileError> {
    let mut dst_cnt = [0u16; 16];
    for st in steps { if let Some(d) = st.dst() { dst_cnt[usize::from(d & 15)] += 1; } }
    for st in steps {
        let is_hoisted = matches!(*st, AvxStep::Const(d, _) if dst_cnt[usize::from(d & 15)] == 1);
        if only_hoist.is_some_and(|want| want != is_hoisted) { continue; }
        match *st {
            AvxStep::Const(d, c) => e.vex_broadcast_imm(d, c, true),
            AvxStep::Load(d, buf, disp) => e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], d, *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, i32::from(disp)),
            AvxStep::Store(s, buf, disp) => e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], s, *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?, i32::from(disp)),
            AvxStep::Add(d, a, b) => e.vex_ymm_op(1, 0xD4, d, a, b),
            AvxStep::Sub(d, a, b) => e.vex_ymm_op(1, 0xFB, d, a, b),
            AvxStep::BitAnd(d, a, b) => e.vex_ymm_op(1, 0xDB, d, a, b),
            AvxStep::MulU32(d, a, b) => e.vex_ymm_op(1, 0xF4, d, a, b),
            AvxStep::Shl(d, a, s) | AvxStep::ShrU64(d, a, s) => e.vex_shift_imm(0x73, if matches!(*st, AvxStep::Shl(_, _, _)) { 6 } else { 2 }, d, a, s),
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
enum W32Step {
    Load(usize, u8, ElemTy),
    Store32(usize, u8),
    StoreI64 { src: usize, buf: u8, signed: bool },
    Add(usize, usize, usize),
    Sub(usize, usize, usize),
    BitAnd(usize, usize, usize),
    Mul(usize, usize, usize),
    Shl(usize, usize, u8),
    ShrU(usize, usize, u8),
    ShrS(usize, usize, u8),
    Cmp { dst: usize, kind: CmpKind, a: usize, b: usize, bias: Option<usize>, one: Option<usize> },
    Select { dst: usize, cond: usize, t_val: usize, f_val: usize, zero: usize },
}

impl W32Step {
    fn dst(&self) -> Option<usize> {
        match *self {
            Self::Load(d, _, _) | Self::Add(d, _, _) | Self::Sub(d, _, _) | Self::BitAnd(d, _, _)
            | Self::Mul(d, _, _) | Self::Shl(d, _, _) | Self::ShrU(d, _, _) | Self::ShrS(d, _, _)
            | Self::Cmp { dst: d, .. } | Self::Select { dst: d, .. } => Some(d),
            Self::Store32(_, _) | Self::StoreI64 { .. } => None,
        }
    }
    fn for_each_src(&self, mut f: impl FnMut(usize)) {
        match *self {
            Self::Load(_, _, _) => {}
            Self::Store32(s, _) | Self::StoreI64 { src: s, .. } | Self::Shl(_, s, _) | Self::ShrU(_, s, _) | Self::ShrS(_, s, _) => f(s),
            Self::Add(_, a, b) | Self::Sub(_, a, b) | Self::BitAnd(_, a, b) | Self::Mul(_, a, b) => { f(a); f(b); }
            Self::Cmp { a, b, bias, one, .. } => {
                f(a);
                f(b);
                if let Some(x) = bias { f(x); }
                if let Some(x) = one { f(x); }
            }
            Self::Select { cond, t_val, f_val, zero, .. } => { f(cond); f(t_val); f(f_val); f(zero); }
        }
    }
    fn remap(&self, m: &[u8]) -> Self {
        let g = |v: usize| usize::from(m[v]);
        match *self {
            Self::Load(d, buf, ty) => Self::Load(g(d), buf, ty),
            Self::Store32(s, buf) => Self::Store32(g(s), buf),
            Self::StoreI64 { src, buf, signed } => Self::StoreI64 { src: g(src), buf, signed },
            Self::Add(d, a, b) => Self::Add(g(d), g(a), g(b)),
            Self::Sub(d, a, b) => Self::Sub(g(d), g(a), g(b)),
            Self::BitAnd(d, a, b) => Self::BitAnd(g(d), g(a), g(b)),
            Self::Mul(d, a, b) => Self::Mul(g(d), g(a), g(b)),
            Self::Shl(d, a, s) => Self::Shl(g(d), g(a), s),
            Self::ShrU(d, a, s) => Self::ShrU(g(d), g(a), s),
            Self::ShrS(d, a, s) => Self::ShrS(g(d), g(a), s),
            Self::Cmp { dst, kind, a, b, bias, one } => Self::Cmp { dst: g(dst), kind, a: g(a), b: g(b), bias: bias.map(g), one: one.map(g) },
            Self::Select { dst, cond, t_val, f_val, zero } => Self::Select { dst: g(dst), cond: g(cond), t_val: g(t_val), f_val: g(f_val), zero: g(zero) },
        }
    }
}

struct W32Plan { consts: Vec<(u8, u32)>, steps: Vec<W32Step> }

struct W32Ctx<'a> {
    k: &'a Kernel, doms: &'a [Domain], needs_full: Vec<bool>,
    memo: Vec<([NodeId; LANES], usize)>, consts: Vec<(usize, u32)>,
    steps: Vec<W32Step>, next_v: usize,
}
impl<'a> W32Ctx<'a> {
    fn alloc_v(&mut self) -> usize { let v = self.next_v; self.next_v += 1; v }
    fn get_const(&mut self, val: u32) -> usize {
        if let Some(&(v, _)) = self.consts.iter().find(|&&(_, c)| c == val) { return v; }
        let v = self.alloc_v();
        self.consts.push((v, val));
        v
    }
    fn match_vec8(&mut self, ids: [NodeId; LANES]) -> Option<usize> {
        if let Some(&(_, v)) = self.memo.iter().find(|&&(m, _)| m == ids) { return Some(v); }
        for &id in &ids {
            if self.needs_full[id] {
                let d = self.doms.get(id).copied()?;
                if !(d.fits_i32() || d.fits_u32()) { return None; }
            }
        }
        let o0 = *self.k.nodes.get(ids[0])?;
        let r = match o0 {
            Op::Const(c) => {
                if !ids[1..].iter().all(|&id| self.k.nodes.get(id).copied() == Some(Op::Const(c))) { return None; }
                self.get_const(c as u32)
            }
            Op::Load { mem, buf, lane: 0 } if self.k.nodes.get(mem).copied() == Some(Op::MemInit) => {
                for (i, &id) in ids.iter().enumerate().skip(1) {
                    if self.k.nodes.get(id).copied() != Some(Op::Load { mem, buf, lane: i as u8 }) { return None; }
                }
                let ty = self.k.bufs.get(buf as usize)?.elem_ty;
                if matches!(ty, ElemTy::I64 | ElemTy::U64) { return None; }
                let d = self.alloc_v();
                self.steps.push(W32Step::Load(d, buf, ty));
                d
            }
            Op::Add(a0, b0) | Op::Sub(a0, b0) | Op::BitAnd(a0, b0) | Op::Mul(a0, b0) => {
                let (mut la, mut lb) = ([a0; LANES], [b0; LANES]);
                for i in 1..LANES {
                    match (o0, *self.k.nodes.get(ids[i])?) {
                        (Op::Add(_, _), Op::Add(ai, bi)) | (Op::Sub(_, _), Op::Sub(ai, bi))
                        | (Op::BitAnd(_, _), Op::BitAnd(ai, bi)) | (Op::Mul(_, _), Op::Mul(ai, bi)) => { la[i] = ai; lb[i] = bi; }
                        _ => return None,
                    }
                }
                let (va, vb) = (self.match_vec8(la)?, self.match_vec8(lb)?);
                let d = self.alloc_v();
                self.steps.push(match o0 {
                    Op::Add(_, _) => W32Step::Add(d, va, vb), Op::Sub(_, _) => W32Step::Sub(d, va, vb),
                    Op::BitAnd(_, _) => W32Step::BitAnd(d, va, vb), _ => W32Step::Mul(d, va, vb),
                });
                d
            }
            Op::Shl(a0, s) | Op::Shr(a0, s) if s < 32 => {
                let mut la = [a0; LANES];
                for i in 1..LANES {
                    match (o0, *self.k.nodes.get(ids[i])?) {
                        (Op::Shl(_, _), Op::Shl(ai, si)) | (Op::Shr(_, _), Op::Shr(ai, si)) if si == s => la[i] = ai,
                        _ => return None,
                    }
                }
                let is_shr = matches!(o0, Op::Shr(_, _));
                let is_u = is_shr && la.iter().all(|&a| self.doms.get(a).copied().is_some_and(|dm| dm.is_nonneg() && dm.fits_u32()));
                if is_shr && !is_u && !la.iter().all(|&a| self.doms.get(a).copied().is_some_and(Domain::fits_i32)) { return None; }
                let va = self.match_vec8(la)?;
                let d = self.alloc_v();
                self.steps.push(if !is_shr { W32Step::Shl(d, va, s) } else if is_u { W32Step::ShrU(d, va, s) } else { W32Step::ShrS(d, va, s) });
                d
            }
            Op::Cmp(kind, a0, b0) => {
                let (mut la, mut lb) = ([a0; LANES], [b0; LANES]);
                for i in 1..LANES {
                    match *self.k.nodes.get(ids[i])? {
                        Op::Cmp(ki, ai, bi) if ki == kind => { la[i] = ai; lb[i] = bi; }
                        _ => return None,
                    }
                }
                let all_i32 = la.iter().chain(lb.iter()).all(|&x| self.doms.get(x).copied().is_some_and(Domain::fits_i32));
                let all_u32 = la.iter().chain(lb.iter()).all(|&x| self.doms.get(x).copied().is_some_and(Domain::fits_u32));
                if (!all_i32 || kind.is_unsigned()) && !all_u32 { return None; }
                let need_bias = (kind.is_unsigned() || !all_i32) && !matches!(kind, CmpKind::Eq | CmpKind::Ne);
                let need_one = matches!(kind, CmpKind::Ne | CmpKind::Ge | CmpKind::Le | CmpKind::Uge | CmpKind::Ule);
                let (va, vb) = (self.match_vec8(la)?, self.match_vec8(lb)?);
                let bias = if need_bias { Some(self.get_const(0x8000_0000)) } else { None };
                let one = if need_one { Some(self.get_const(1)) } else { None };
                let d = self.alloc_v();
                self.steps.push(W32Step::Cmp { dst: d, kind, a: va, b: vb, bias, one });
                d
            }
            Op::Select { cond: c0, t_val: t0, f_val: f0 } => {
                let (mut lc, mut lt, mut lf) = ([c0; LANES], [t0; LANES], [f0; LANES]);
                for i in 1..LANES {
                    match *self.k.nodes.get(ids[i])? {
                        Op::Select { cond, t_val, f_val } => { lc[i] = cond; lt[i] = t_val; lf[i] = f_val; }
                        _ => return None,
                    }
                }
                let (vc, vt, vf) = (self.match_vec8(lc)?, self.match_vec8(lt)?, self.match_vec8(lf)?);
                let zero = self.get_const(0);
                let d = self.alloc_v();
                self.steps.push(W32Step::Select { dst: d, cond: vc, t_val: vt, f_val: vf, zero });
                d
            }
            _ => return None,
        };
        self.memo.push((ids, r));
        Some(r)
    }
}

fn try_plan_avx2_w32(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], max_vecs: usize) -> Option<W32Plan> {
    if k.outputs.is_empty() || k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || usize::from(b.lanes) != LANES) {
        return None;
    }
    if sched.order.iter().any(|&id| matches!(k.nodes.get(id), Some(Op::Store { .. } | Op::LoadLut { .. } | Op::BoundsCheck { .. } | Op::Div(_, _) | Op::Rem(_, _)))
        || matches!(k.nodes.get(id), Some(&Op::Load { mem, .. }) if k.nodes.get(mem).copied() != Some(Op::MemInit)))
    {
        return None;
    }
    let mut out_groups: Vec<(u8, ElemTy, [NodeId; LANES])> = Vec::new();
    for b in &k.bufs {
        if b.access == BufAccess::MutNoAlias {
            if !matches!(b.elem_ty, ElemTy::I32 | ElemTy::U32 | ElemTy::I64) { return None; }
            out_groups.push((b.id, b.elem_ty, [usize::MAX; LANES]));
        }
    }
    if out_groups.is_empty() { return None; }
    for o in &k.outputs {
        let g = out_groups.iter_mut().find(|g| g.0 == o.buf)?;
        if usize::from(o.lane) >= LANES || g.2[usize::from(o.lane)] != usize::MAX { return None; }
        g.2[usize::from(o.lane)] = o.val;
    }
    if out_groups.iter().any(|g| g.2.contains(&usize::MAX)) { return None; }

    let mut needs_full = vec![false; k.nodes.len()];
    for g in &out_groups {
        if g.1 == ElemTy::I64 { for &v in &g.2 { needs_full[v] = true; } }
    }
    for &id in sched.order.iter().rev() {
        match k.nodes[id] {
            Op::Cmp(_, a, b) => { needs_full[a] = true; needs_full[b] = true; }
            Op::Shr(a, s) if s > 0 => { needs_full[a] = true; }
            Op::Select { cond, t_val, f_val } => {
                needs_full[cond] = true;
                if needs_full[id] { needs_full[t_val] = true; needs_full[f_val] = true; }
            }
            _ => {}
        }
    }

    let mut ctx = W32Ctx { k, doms, needs_full, memo: Vec::new(), consts: Vec::new(), steps: Vec::new(), next_v: 0 };
    let mut out_vregs = Vec::with_capacity(out_groups.len());
    for &(buf, ty, lanes) in &out_groups {
        let vr = ctx.match_vec8(lanes)?;
        out_vregs.push((buf, ty, lanes, vr));
    }
    for (buf, ty, lanes, vr) in out_vregs {
        if ty == ElemTy::I64 {
            let signed = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_i32));
            let unsigned = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_u32));
            if !signed && !unsigned { return None; }
            ctx.steps.push(W32Step::StoreI64 { src: vr, buf, signed });
        } else {
            ctx.steps.push(W32Step::Store32(vr, buf));
        }
    }

    let limit = max_vecs.min(14);
    if ctx.consts.len() >= limit { return None; }
    let mut phys = vec![255u8; ctx.next_v];
    let mut is_const = vec![false; ctx.next_v];
    let mut hoisted = Vec::with_capacity(ctx.consts.len());
    for (idx, &(v, c)) in ctx.consts.iter().enumerate() {
        phys[v] = idx as u8;
        is_const[v] = true;
        hoisted.push((idx as u8, c));
    }
    let mut last_use = vec![0usize; ctx.next_v];
    for (i, st) in ctx.steps.iter().enumerate() { st.for_each_src(|s| last_use[s] = last_use[s].max(i)); }
    let mut free: Vec<u8> = (ctx.consts.len() as u8..limit as u8).rev().collect();
    let mut mapped = Vec::with_capacity(ctx.steps.len());
    for (i, st) in ctx.steps.iter().enumerate() {
        let mut to_free = Vec::new();
        st.for_each_src(|s| {
            if !is_const[s] && last_use[s] == i && !to_free.contains(&s) { to_free.push(s); }
        });
        for s in to_free { free.push(phys[s]); }
        if let Some(d) = st.dst() {
            let pr = free.pop()?;
            phys[d] = pr;
            if last_use[d] <= i { free.push(pr); }
        }
        mapped.push(st.remap(&phys));
    }
    Some(W32Plan { consts: hoisted, steps: mapped })
}

fn emit_x86_w32_consts(e: &mut X86Emit, plan: &W32Plan) {
    for &(reg, val) in &plan.consts { e.vex_broadcast_imm(reg, i64::from(val as i32), false); }
}

fn emit_x86_w32_body(e: &mut X86Emit, k: &Kernel, plan: &W32Plan, ra: &RegAlloc, base_blk: i32) -> Result<(), CompileError> {
    for st in &plan.steps {
        match *st {
            W32Step::Load(d, buf, ty) => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * (k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32);
                match ty {
                    ElemTy::I32 | ElemTy::U32 => e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], d as u8, breg, off),
                    ElemTy::U16 => e.vex_mem_op([2, 1, 1, 0, 0, 0x33], d as u8, breg, off),
                    ElemTy::I16 => e.vex_mem_op([2, 1, 1, 0, 0, 0x23], d as u8, breg, off),
                    ElemTy::U8 => e.vex_mem_op([2, 1, 1, 0, 0, 0x31], d as u8, breg, off),
                    ElemTy::I64 | ElemTy::U64 => return Err(berr("unexpected 64-bit load in w32")),
                }
            }
            W32Step::Store32(s, buf) => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * (k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], s as u8, breg, off);
            }
            W32Step::StoreI64 { src, buf, signed } => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * (k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32);
                let ext_opc = if signed { 0x25 } else { 0x35 };
                e.vex_ymm_op(2, ext_opc, 15, 0, src as u8);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, off);
                e.vex_extracti128(14, src as u8, 1);
                e.vex_ymm_op(2, ext_opc, 15, 0, 14);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, off + 32);
            }
            W32Step::Add(d, a, b) => e.vex_ymm_op(1, 0xFE, d as u8, a as u8, b as u8),
            W32Step::Sub(d, a, b) => e.vex_ymm_op(1, 0xFA, d as u8, a as u8, b as u8),
            W32Step::BitAnd(d, a, b) => e.vex_ymm_op(1, 0xDB, d as u8, a as u8, b as u8),
            W32Step::Mul(d, a, b) => e.vex_ymm_op(2, 0x40, d as u8, a as u8, b as u8),
            W32Step::Shl(d, a, s) => e.vex_shift_imm(0x72, 6, d as u8, a as u8, s),
            W32Step::ShrU(d, a, s) => e.vex_shift_imm(0x72, 2, d as u8, a as u8, s),
            W32Step::ShrS(d, a, s) => e.vex_shift_imm(0x72, 4, d as u8, a as u8, s),
            W32Step::Cmp { dst, kind, a, b, bias, one } => {
                let (d, mut va, mut vb) = (dst as u8, a as u8, b as u8);
                if let Some(br) = bias {
                    e.vex_ymm_op(1, 0xEF, 14, va, br as u8);
                    e.vex_ymm_op(1, 0xEF, 15, vb, br as u8);
                    va = 14; vb = 15;
                }
                match kind {
                    CmpKind::Eq | CmpKind::Ne => e.vex_ymm_op(1, 0x76, d, va, vb),
                    CmpKind::Gt | CmpKind::Le | CmpKind::Ugt | CmpKind::Ule => e.vex_ymm_op(1, 0x66, d, va, vb),
                    CmpKind::Lt | CmpKind::Ge | CmpKind::Ult | CmpKind::Uge => e.vex_ymm_op(1, 0x66, d, vb, va),
                }
                e.vex_shift_imm(0x72, 2, d, d, 31);
                if let Some(or) = one { e.vex_ymm_op(1, 0xEF, d, d, or as u8); }
            }
            W32Step::Select { dst, cond, t_val, f_val, zero } => {
                e.vex_ymm_op(1, 0x76, 15, cond as u8, zero as u8);
                e.vex_ymm_imm(3, 0x4C, dst as u8, t_val as u8, f_val as u8, 15 << 4);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum TransposeOut {
    Lanes2(u8, [NodeId; 2]),
    Lanes4(u8, [NodeId; 4]),
}

struct Batch4Plan {
    node_ymm: Vec<Option<u8>>,
    hoisted: Vec<(u8, i64)>,
    transpose_out: Option<TransposeOut>,
}

fn alloc_batch4_ymms(k: &Kernel, sched: &ScheduleReport, pool_limit: usize, hold_outs: &[NodeId]) -> Option<(Vec<Option<u8>>, u16)> {
    let n = k.nodes.len();
    let mut last_step = vec![0usize; n];
    for (step, &id) in sched.order.iter().enumerate() {
        let (kids, cnt) = k.nodes[id].children();
        for &kid in &kids[..cnt] { if let Some(ls) = last_step.get_mut(kid) { *ls = (*ls).max(step); } }
    }
    if !hold_outs.is_empty() {
        for &o in hold_outs { if let Some(ls) = last_step.get_mut(o) { *ls = sched.order.len().saturating_add(1); } }
    } else {
        for &(ostep, oi) in &sched.output_order {
            if let Some(o) = k.outputs.get(oi) {
                if let Some(ls) = last_step.get_mut(o.val) { *ls = (*ls).max(ostep.saturating_add(1)); }
            }
        }
    }
    let mut free: Vec<u8> = (0..pool_limit as u8).rev().collect();
    let (mut node_ymm, mut active, mut used_mask) = (vec![None; n], Vec::<(NodeId, u8)>::new(), 0u16);
    for (step, &id) in sched.order.iter().enumerate() {
        let op = *k.nodes.get(id)?;
        if !is_inline(op) {
            let reg = free.pop()?;
            used_mask |= 1u16 << reg;
            node_ymm[id] = Some(reg);
            if last_step[id] > step { active.push((id, reg)); } else { free.push(reg); }
        }
        let mut i = 0usize;
        while i < active.len() {
            let (aid, areg) = active[i];
            if last_step[aid] <= step { active.swap_remove(i); free.push(areg); } else { i += 1; }
        }
        if hold_outs.is_empty() {
            for &(ostep, _) in sched.output_order.iter().filter(|&&(os, _)| os == step + 1) {
                let mut j = 0usize;
                while j < active.len() {
                    let (aid, areg) = active[j];
                    if last_step[aid] <= ostep { active.swap_remove(j); free.push(areg); } else { j += 1; }
                }
            }
        }
    }
    Some((node_ymm, used_mask))
}

fn b4_const_of(k: &Kernel, id: NodeId) -> Option<i64> {
    match k.nodes.get(id).copied() {
        Some(Op::Const(c)) => Some(c),
        Some(Op::MemInit) => Some(0),
        _ => None,
    }
}

fn try_plan_avx2_batch4(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], max_vecs: usize) -> Option<Batch4Plan> {
    if k.outputs.is_empty() || k.bufs.iter().any(|b| {
        matches!(b.access, BufAccess::LookupTable(_))
            || b.lanes == 0
            || if b.access == BufAccess::ReadOnly && b.lanes == 1 {
                false
            } else {
                !matches!(b.elem_ty, ElemTy::I64 | ElemTy::U64 | ElemTy::I32 | ElemTy::U32)
            }
    }) {
        return None;
    }
    for &id in &sched.order {
        match *k.nodes.get(id)? {
            Op::MemInit | Op::Const(_) | Op::Add(_, _) | Op::Sub(_, _) | Op::BitAnd(_, _)
            | Op::BitOr(_, _) | Op::BitXor(_, _) | Op::ShlVar(_, _) | Op::ShrVarU(_, _)
            | Op::Cmp(_, _, _) | Op::Select { .. } => {}
            Op::Load { mem, .. } if k.nodes.get(mem).copied() == Some(Op::MemInit) => {}
            Op::Mul(a, b) => {
                let (da, db) = (doms.get(a).copied()?, doms.get(b).copied()?);
                if !((da.fits_i32() && db.fits_i32()) || da.fits_u32() || db.fits_u32()) { return None; }
            }
            Op::Shl(_, s) | Op::Shr(_, s) | Op::Ushr(_, s) if s < 64 => {}
            _ => return None,
        }
    }
    let mut cand_transpose: Option<TransposeOut> = None;
    if k.outputs.len() == 2 || k.outputs.len() == 4 {
        let n_lanes = k.outputs.len();
        let ob = k.outputs[0].buf;
        let bdecl = k.bufs.get(ob as usize)?;
        let mut lanes = [usize::MAX; 4];
        let mut ok = matches!(bdecl.elem_ty, ElemTy::I64 | ElemTy::U64)
            && usize::from(bdecl.lanes) == n_lanes
            && k.bufs.iter().filter(|b| b.access == BufAccess::MutNoAlias).count() == 1;
        for o in &k.outputs {
            if o.buf != ob || usize::from(o.lane) >= n_lanes || lanes[usize::from(o.lane)] != usize::MAX
                || matches!(k.nodes.get(o.val), Some(Op::Const(_) | Op::MemInit))
            {
                ok = false; break;
            }
            lanes[usize::from(o.lane)] = o.val;
        }
        if ok && !lanes[..n_lanes].contains(&usize::MAX) {
            cand_transpose = Some(if n_lanes == 2 {
                TransposeOut::Lanes2(ob, [lanes[0], lanes[1]])
            } else {
                TransposeOut::Lanes4(ob, lanes)
            });
        }
    }
    let (node_ymm, used_mask, pool_limit, transpose_out) = if let Some(tr) = cand_transpose {
        let tr_lim = max_vecs.min(if matches!(tr, TransposeOut::Lanes2(_, _)) { 14 } else { 13 });
        let hold: &[NodeId] = match &tr { TransposeOut::Lanes2(_, l) => l, TransposeOut::Lanes4(_, l) => l };
        if let Some((ny, um)) = alloc_batch4_ymms(k, sched, tr_lim, hold) {
            (ny, um, tr_lim, Some(tr))
        } else {
            let lim14 = max_vecs.min(14);
            let (ny, um) = alloc_batch4_ymms(k, sched, lim14, &[])?;
            (ny, um, lim14, None)
        }
    } else {
        let lim14 = max_vecs.min(14);
        let (ny, um) = alloc_batch4_ymms(k, sched, lim14, &[])?;
        (ny, um, lim14, None)
    };

    let mut const_counts: Vec<(i64, usize)> = Vec::new();
    let mut bump = |c: i64, w: usize| {
        if let Some(e) = const_counts.iter_mut().find(|e| e.0 == c) { e.1 += w; } else { const_counts.push((c, w)); }
    };
    for &id in &sched.order {
        let Some(&op) = k.nodes.get(id) else { continue; };
        match op {
            Op::Cmp(kind, a, b) => {
                let need_bias = kind.is_unsigned() && !(doms[a].is_nonneg() && doms[b].is_nonneg());
                if need_bias {
                    if let Some(ca) = b4_const_of(k, a) { bump(ca ^ i64::MIN, 3); } else { bump(i64::MIN, 6); }
                    if let Some(cb) = b4_const_of(k, b) { bump(cb ^ i64::MIN, 3); } else { bump(i64::MIN, 6); }
                } else {
                    if let Some(ca) = b4_const_of(k, a) { bump(ca, 2); }
                    if let Some(cb) = b4_const_of(k, b) { bump(cb, 2); }
                }
                if matches!(kind, CmpKind::Ne | CmpKind::Le | CmpKind::Ge | CmpKind::Ule | CmpKind::Uge) { bump(1, 6); }
            }
            Op::Select { cond, t_val, f_val } => {
                bump(0, 6);
                for kid in [cond, t_val, f_val] {
                    if let Some(Op::Const(c)) = k.nodes.get(kid).copied() { bump(c, 2); }
                }
            }
            Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                if let Some(Op::Const(c)) = k.nodes.get(a).copied() { bump(c, 2); }
                if let Some(Op::Const(c)) = k.nodes.get(b).copied() { bump(c & 63, 2); }
                else if !(doms[b].is_nonneg() && doms[b].hi < 64) { bump(63, 4); }
            }
            _ => {
                let (kids, cnt) = op.children();
                for &kid in &kids[..cnt] {
                    if let Some(Op::Const(c)) = k.nodes.get(kid).copied() { bump(c, 2); }
                }
                if let Op::Shr(a, s) = op {
                    if s > 0 && doms.get(a).is_some_and(|d| !d.is_nonneg()) {
                        if s == 63 { bump(0, 3); } else { bump(i64::MIN, 3); bump(((1u64 << 63) >> s) as i64, 2); }
                    }
                }
            }
        }
    }
    for o in &k.outputs { if let Op::Const(c) = k.nodes[o.val] { bump(c, 2); } }
    const_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut hoisted = Vec::new();
    for r in 0..pool_limit as u8 {
        if (used_mask & (1u16 << r)) == 0 {
            if let Some(&(c, _)) = const_counts.get(hoisted.len()) { hoisted.push((r, c)); }
        }
    }
    Some(Batch4Plan { node_ymm, hoisted, transpose_out })
}

fn b4_const_reg(e: &mut X86Emit, plan: &Batch4Plan, c: i64, scratch: u8) -> u8 {
    if let Some(&(r, _)) = plan.hoisted.iter().find(|&&(_, v)| v == c) { return r; }
    e.vex_broadcast_rip(scratch, c);
    scratch
}

fn b4_val_reg(e: &mut X86Emit, k: &Kernel, plan: &Batch4Plan, id: NodeId, scratch: u8) -> Result<u8, CompileError> {
    match k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))? {
        Op::Const(c) => Ok(b4_const_reg(e, plan, c, scratch)),
        Op::MemInit => Ok(b4_const_reg(e, plan, 0, scratch)),
        _ => plan.node_ymm.get(id).and_then(|&r| r).ok_or(CompileError::InvalidNode(id)),
    }
}

fn emit_b4_store_scatter(e: &mut X86Emit, bdecl: &crate::ir::BufDecl, vreg: u8, breg: u8, base_blk: i32, lane: u8) {
    let ty = bdecl.elem_ty;
    let stride = bdecl.block_bytes() as i32;
    let base_off = base_blk * stride;
    if bdecl.lanes == 1 && matches!(ty, ElemTy::I64 | ElemTy::U64) {
        e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], vreg, breg, base_off);
        return;
    }
    let off = base_off + i32::from(lane) * i32::from(ty.bytes());
    let (mov_opc, w, idx) = if matches!(ty, ElemTy::I64 | ElemTy::U64) { (0xD6u8, 1u8, 1u8) } else { (0x7Eu8, 0u8, 2u8) };
    e.vex_mem_op([1, 0, 1, 0, 0, mov_opc], vreg, breg, off);
    e.vex_mem_imm([3, 0, 1, w, 0, 0x16], vreg, breg, stride + off, idx);
    e.vex_extracti128(15, vreg, 1);
    e.vex_mem_op([1, 0, 1, 0, 0, mov_opc], 15, breg, 2 * stride + off);
    e.vex_mem_imm([3, 0, 1, w, 0, 0x16], 15, breg, 3 * stride + off, idx);
}

fn emit_x86_batch4_body(e: &mut X86Emit, k: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain], plan: &Batch4Plan, base_blk: i32) -> Result<(), CompileError> {
    let mut oc = 0usize;
    let scatter_outputs = plan.transpose_out.is_none();
    for (step, &id) in sched.order.iter().enumerate() {
        if scatter_outputs {
            drain_scheduled_outputs(k, sched, ra, &mut oc, Some(step), |_ty, breg, lane, val| {
                let bdecl = k.bufs.iter().find(|b| ra.buf_reg.get(usize::from(b.id)).copied() == Some(breg)).ok_or(CompileError::InvalidBuffer(0))?;
                let vreg = b4_val_reg(e, k, plan, val, 14)?;
                emit_b4_store_scatter(e, bdecl, vreg, breg, base_blk, lane);
                Ok(())
            })?;
        }
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if is_inline(op) { continue; }
        let dst = plan.node_ymm.get(id).and_then(|&r| r).ok_or(CompileError::InvalidNode(id))?;
        match op {
            Op::Load { buf, lane, .. } => {
                let bdecl = k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let stride = bdecl.block_bytes() as i32;
                let base_off = base_blk * stride;
                if bdecl.lanes == 1 {
                    let enc = match bdecl.elem_ty {
                        ElemTy::I64 | ElemTy::U64 => [1, 1, 2, 0, 0, 0x6F],
                        ElemTy::I32 => [2, 1, 1, 0, 0, 0x25],
                        ElemTy::U32 => [2, 1, 1, 0, 0, 0x35],
                        ElemTy::I16 => [2, 1, 1, 0, 0, 0x24],
                        ElemTy::U16 => [2, 1, 1, 0, 0, 0x34],
                        ElemTy::U8 => [2, 1, 1, 0, 0, 0x32],
                    };
                    e.vex_mem_op(enc, dst, breg, base_off);
                } else {
                    let off = base_off + i32::from(lane) * i32::from(bdecl.elem_ty.bytes());
                    if matches!(bdecl.elem_ty, ElemTy::I64 | ElemTy::U64) {
                        e.vex_mem_op([1, 0, 2, 0, 0, 0x7E], dst, breg, off);
                        e.vex_mem_imm([3, 0, 1, 1, dst, 0x22], dst, breg, stride + off, 1);
                        e.vex_mem_op([1, 0, 2, 0, 0, 0x7E], 15, breg, 2 * stride + off);
                        e.vex_mem_imm([3, 0, 1, 1, 15, 0x22], 15, breg, 3 * stride + off, 1);
                        e.vex_ymm_imm(3, 0x38, dst, dst, 15, 1);
                    } else {
                        e.vex_mem_op([1, 0, 1, 0, 0, 0x6E], dst, breg, off);
                        e.vex_mem_imm([3, 0, 1, 0, dst, 0x22], dst, breg, stride + off, 1);
                        e.vex_mem_imm([3, 0, 1, 0, dst, 0x22], dst, breg, 2 * stride + off, 2);
                        e.vex_mem_imm([3, 0, 1, 0, dst, 0x22], dst, breg, 3 * stride + off, 3);
                        e.vex_ymm_op(2, if bdecl.elem_ty == ElemTy::I32 { 0x25 } else { 0x35 }, dst, 0, dst);
                    }
                }
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => {
                let (va, vb) = (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?);
                let opc = match op {
                    Op::Add(_, _) => 0xD4,
                    Op::Sub(_, _) => 0xFB,
                    Op::BitAnd(_, _) => 0xDB,
                    Op::BitOr(_, _) => 0xEB,
                    _ => 0xEF,
                };
                e.vex_ymm_op(1, opc, dst, va, vb);
            }
            Op::Mul(a, b) => {
                if doms[a].fits_i32() && doms[b].fits_i32() {
                    let (va, vb) = (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?);
                    e.vex_ymm_op(2, 0x28, dst, va, vb);
                } else if doms[a].fits_u32() && doms[b].fits_u32() {
                    let (va, vb) = (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?);
                    e.vex_ymm_op(1, 0xF4, dst, va, vb);
                } else {
                    let (x, y) = if doms[b].fits_u32() { (a, b) } else { (b, a) };
                    let (vx, vy) = (b4_val_reg(e, k, plan, x, 14)?, b4_val_reg(e, k, plan, y, 15)?);
                    let tmp = if dst == vx { 14 } else if dst == vy { 15 } else { dst };
                    e.vex_shift_imm(0x73, 2, tmp, vx, 32);
                    e.vex_ymm_op(1, 0xF4, tmp, tmp, vy);
                    e.vex_shift_imm(0x73, 6, tmp, tmp, 32);
                    if tmp == dst {
                        e.vex_ymm_op(1, 0xF4, 14, vx, vy);
                        e.vex_ymm_op(1, 0xD4, dst, dst, 14);
                    } else {
                        e.vex_ymm_op(1, 0xF4, dst, vx, vy);
                        e.vex_ymm_op(1, 0xD4, dst, dst, tmp);
                    }
                }
            }
            Op::Shl(a, s) => {
                let va = b4_val_reg(e, k, plan, a, dst)?;
                if s > 0 { e.vex_shift_imm(0x73, 6, dst, va, s); } else if dst != va { e.vex_ymm_op(1, 0xEB, dst, va, va); }
            }
            Op::Ushr(a, s) => {
                let va = b4_val_reg(e, k, plan, a, dst)?;
                if s > 0 { e.vex_shift_imm(0x73, 2, dst, va, s); } else if dst != va { e.vex_ymm_op(1, 0xEB, dst, va, va); }
            }
            Op::Shr(a, s) => {
                let va = b4_val_reg(e, k, plan, a, dst)?;
                if s == 0 {
                    if dst != va { e.vex_ymm_op(1, 0xEB, dst, va, va); }
                } else if doms[a].is_nonneg() {
                    e.vex_shift_imm(0x73, 2, dst, va, s);
                } else if s == 63 {
                    let zr = b4_const_reg(e, plan, 0, 15);
                    e.vex_ymm_op(2, 0x37, dst, zr, va);
                } else {
                    let sb = b4_const_reg(e, plan, i64::MIN, 14);
                    let sbs = b4_const_reg(e, plan, ((1u64 << 63) >> s) as i64, 15);
                    e.vex_ymm_op(1, 0xEF, dst, va, sb);
                    e.vex_shift_imm(0x73, 2, dst, dst, s);
                    e.vex_ymm_op(1, 0xFB, dst, dst, sbs);
                }
            }
            Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                let va = b4_val_reg(e, k, plan, a, 14)?;
                let vb = match b4_const_of(k, b) {
                    Some(c) => b4_const_reg(e, plan, c & 63, 15),
                    None => {
                        let r = b4_val_reg(e, k, plan, b, 15)?;
                        if doms[b].is_nonneg() && doms[b].hi < 64 {
                            r
                        } else {
                            let m63 = b4_const_reg(e, plan, 63, 15);
                            e.vex_ymm_op(1, 0xDB, 15, r, m63);
                            15
                        }
                    }
                };
                e.vex_ymm_op_w(2, 1, if matches!(op, Op::ShlVar(_, _)) { 0x47 } else { 0x45 }, dst, va, vb);
            }
            Op::Cmp(kind, a, b) => {
                if matches!(kind, CmpKind::Eq | CmpKind::Ne) {
                    let (va, vb) = (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?);
                    e.vex_ymm_op(2, 0x29, dst, va, vb);
                } else {
                    let need_bias = kind.is_unsigned() && !(doms[a].is_nonneg() && doms[b].is_nonneg());
                    let (va, vb) = if !need_bias {
                        (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?)
                    } else {
                        match (b4_const_of(k, a), b4_const_of(k, b)) {
                            (Some(ca), Some(cb)) => (b4_const_reg(e, plan, ca ^ i64::MIN, 14), b4_const_reg(e, plan, cb ^ i64::MIN, 15)),
                            (Some(ca), None) => {
                                let rb = b4_val_reg(e, k, plan, b, 15)?;
                                let br = b4_const_reg(e, plan, i64::MIN, 15);
                                e.vex_ymm_op(1, 0xEF, 15, rb, br);
                                (b4_const_reg(e, plan, ca ^ i64::MIN, 14), 15)
                            }
                            (None, Some(cb)) => {
                                let ra = b4_val_reg(e, k, plan, a, 14)?;
                                let br = b4_const_reg(e, plan, i64::MIN, 14);
                                e.vex_ymm_op(1, 0xEF, 14, ra, br);
                                (14, b4_const_reg(e, plan, cb ^ i64::MIN, 15))
                            }
                            (None, None) => {
                                let (ra, rb) = (b4_val_reg(e, k, plan, a, 14)?, b4_val_reg(e, k, plan, b, 15)?);
                                let br = b4_const_reg(e, plan, i64::MIN, 14);
                                e.vex_ymm_op(1, 0xEF, 15, rb, br);
                                e.vex_ymm_op(1, 0xEF, 14, ra, br);
                                (14, 15)
                            }
                        }
                    };
                    if matches!(kind, CmpKind::Gt | CmpKind::Le | CmpKind::Ugt | CmpKind::Ule) {
                        e.vex_ymm_op(2, 0x37, dst, va, vb);
                    } else {
                        e.vex_ymm_op(2, 0x37, dst, vb, va);
                    }
                }
                e.vex_shift_imm(0x73, 2, dst, dst, 63);
                if matches!(kind, CmpKind::Ne | CmpKind::Le | CmpKind::Ge | CmpKind::Ule | CmpKind::Uge) {
                    let one = b4_const_reg(e, plan, 1, 15);
                    e.vex_ymm_op(1, 0xEF, dst, dst, one);
                }
            }
            Op::Select { cond, t_val, f_val } => {
                let vc = b4_val_reg(e, k, plan, cond, 14)?;
                let zr = b4_const_reg(e, plan, 0, 15);
                e.vex_ymm_op(2, 0x29, 15, vc, zr);
                if b4_const_of(k, f_val) == Some(0) {
                    let vt = b4_val_reg(e, k, plan, t_val, 14)?;
                    e.vex_ymm_op(1, 0xDF, dst, 15, vt);
                } else if b4_const_of(k, t_val) == Some(0) {
                    let vf = b4_val_reg(e, k, plan, f_val, 14)?;
                    e.vex_ymm_op(1, 0xDB, dst, 15, vf);
                } else {
                    let mut vt = b4_val_reg(e, k, plan, t_val, 14)?;
                    if vt == dst {
                        e.vex_ymm_op(1, 0xEB, 14, vt, vt);
                        vt = 14;
                    }
                    let vf = b4_val_reg(e, k, plan, f_val, dst)?;
                    e.vex_ymm_imm(3, 0x4C, dst, vt, vf, 15 << 4);
                }
            }
            _ => return Err(berr("unsupported op in batch4")),
        }
    }
    match plan.transpose_out {
        Some(TransposeOut::Lanes2(ob, lanes)) => {
            let breg = *ra.buf_reg.get(ob as usize).ok_or(CompileError::InvalidBuffer(ob))?;
            let base_off = base_blk * 16;
            let r0 = b4_val_reg(e, k, plan, lanes[0], 14)?;
            let r1 = b4_val_reg(e, k, plan, lanes[1], 15)?;
            e.vex_ymm_op(1, 0x6C, 14, r0, r1);
            e.vex_ymm_op(1, 0x6D, 15, r0, r1);
            e.vex_ymm_imm(3, 0x46, r0, 14, 15, 0x20);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], r0, breg, base_off);
            e.vex_ymm_imm(3, 0x46, 14, 14, 15, 0x31);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, base_off + 32);
        }
        Some(TransposeOut::Lanes4(ob, lanes)) => {
            let breg = *ra.buf_reg.get(ob as usize).ok_or(CompileError::InvalidBuffer(ob))?;
            let base_off = base_blk * 32;
            let [r0, r1, r2, r3] = [
                b4_val_reg(e, k, plan, lanes[0], 13)?, b4_val_reg(e, k, plan, lanes[1], 13)?,
                b4_val_reg(e, k, plan, lanes[2], 13)?, b4_val_reg(e, k, plan, lanes[3], 13)?,
            ];
            e.vex_ymm_op(1, 0x6C, 13, r0, r1);
            e.vex_ymm_op(1, 0x6C, 14, r2, r3);
            e.vex_ymm_imm(3, 0x46, 15, 13, 14, 0x20);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, base_off);
            e.vex_ymm_imm(3, 0x46, 15, 13, 14, 0x31);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, base_off + 64);
            e.vex_ymm_op(1, 0x6D, 13, r0, r1);
            e.vex_ymm_op(1, 0x6D, 14, r2, r3);
            e.vex_ymm_imm(3, 0x46, 15, 13, 14, 0x20);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, base_off + 32);
            e.vex_ymm_imm(3, 0x46, 15, 13, 14, 0x31);
            e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 15, breg, base_off + 96);
        }
        None => {
            drain_scheduled_outputs(k, sched, ra, &mut oc, None, |_ty, breg, lane, val| {
                let bdecl = k.bufs.iter().find(|b| ra.buf_reg.get(usize::from(b.id)).copied() == Some(breg)).ok_or(CompileError::InvalidBuffer(0))?;
                let vreg = b4_val_reg(e, k, plan, val, 14)?;
                emit_b4_store_scatter(e, bdecl, vreg, breg, base_blk, lane);
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn emit_x86_cmp(e: &mut X86Emit, k: CmpKind, dst: u8, ra: u8, rb: u8) {
    e.push_rr(0x39, rb, ra);
    let cc = match k {
        CmpKind::Eq => 0x94, CmpKind::Ne => 0x95, CmpKind::Lt => 0x9C, CmpKind::Ge => 0x9D, CmpKind::Le => 0x9E, CmpKind::Gt => 0x9F,
        CmpKind::Ult => 0x92, CmpKind::Uge => 0x93, CmpKind::Ule => 0x96, CmpKind::Ugt => 0x97,
    };
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
            Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => {
                let is_sub = matches!(op, Op::Sub(_, _));
                let imm_rhs = kernel.nodes.get(b).and_then(|&o| if let Op::Const(c) = o { i32::try_from(c).ok().map(|v| (a, v)) } else { None })
                    .or_else(|| if !is_sub { kernel.nodes.get(a).and_then(|&o| if let Op::Const(c) = o { i32::try_from(c).ok().map(|v| (b, v)) } else { None }) } else { None });
                if let Some((x, c32)) = imm_rhs {
                    let vx = x86_val_into(e, kernel, ra, x, 0)?;
                    e.mov_rr(dst, vx);
                    let subop = match op { Op::Add(_, _) => 0, Op::BitOr(_, _) => 1, Op::BitAnd(_, _) => 4, Op::Sub(_, _) => 5, _ => 6 };
                    if let Ok(c8) = i8::try_from(c32) { e.op_imm8(0x83, subop, dst, c8 as u8, false); }
                    else { e.code.extend_from_slice(&[x86_rex_w(0, dst), 0x81, x86_modrm(3, subop, dst)]); e.code.extend_from_slice(&c32.to_le_bytes()); e.insns += 1; }
                } else {
                    let opc = match op { Op::Add(_, _) => 0x01, Op::BitOr(_, _) => 0x09, Op::BitAnd(_, _) => 0x21, Op::Sub(_, _) => 0x29, _ => 0x31 };
                    let (va, vb) = (x86_val_into(e, kernel, ra, a, 0)?, x86_val_into(e, kernel, ra, b, 11)?);
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
            Op::Ushr(a, s) => { let va = x86_val_into(e, kernel, ra, a, 0)?; e.mov_rr(dst, va); e.op_imm8(0xC1, 5, dst, s, true); }
            Op::Shr(a, s) => {
                let va = x86_val_into(e, kernel, ra, a, 0)?; e.mov_rr(dst, va);
                e.op_imm8(0xC1, if doms.get(a).copied().is_some_and(Domain::is_nonneg) { 5 } else { 7 }, dst, s, true);
            }
            Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                let (va, vb) = (x86_val_into(e, kernel, ra, a, 0)?, x86_val_into(e, kernel, ra, b, 11)?);
                e.bmi2_shift(if matches!(op, Op::ShlVar(_, _)) { 1 } else { 3 }, dst, va, vb);
            }
            Op::Popcnt(a) => {
                let va = x86_val_into(e, kernel, ra, a, 0)?;
                e.code.extend_from_slice(&[0xF3, x86_rex_w(dst, va), 0x0F, 0xB8, x86_modrm(3, dst, va)]);
                e.insns += 1;
            }
            Op::Tzcnt(a) => {
                let va = x86_val_into(e, kernel, ra, a, 0)?;
                e.code.extend_from_slice(&[0xF3, x86_rex_w(dst, va), 0x0F, 0xBC, x86_modrm(3, dst, va)]);
                e.insns += 1;
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

fn emit_x86_module(kernel: &Kernel, base: &Kernel, sched: &ScheduleReport, ra: &RegAlloc, doms: &[Domain], max_vecs: usize, call_batch: bool) -> Result<ModuleCode, CompileError> {
    let mut e = X86Emit::new();
    let avx_plan = try_plan_avx2(kernel, sched, doms, max_vecs);
    let w32_plan = if avx_plan.is_none() { try_plan_avx2_w32(kernel, sched, doms, max_vecs) } else { None };
    let b8_plan = if avx_plan.is_none() && w32_plan.is_none() {
        let mut base_b8 = None;
        if base.nodes.len() < kernel.nodes.len() {
            let b_budget = RegBudget { max_gprs: 64, max_vecs };
            for remat_n in [0u8, 1u8, 3u8, 6u8, 12u8] {
                let cand = if remat_n == 0 { base.clone() } else { remat_readonly_loads(base, remat_n) };
                if cand.nodes.len() >= kernel.nodes.len() { break; }
                if let Ok(bs) = schedule_and_verify(&cand, b_budget) {
                    let b_doms = cand.infer_domains();
                    if let Some(plan) = crate::batch8::try_plan_avx2_batch8(&cand, &bs, &b_doms, max_vecs) {
                        base_b8 = Some(plan);
                        break;
                    }
                }
            }
        }
        base_b8.or_else(|| crate::batch8::try_plan_avx2_batch8(kernel, sched, doms, max_vecs))
    } else {
        None
    };
    let b4_plan = if avx_plan.is_none() && w32_plan.is_none() && b8_plan.is_none() { try_plan_avx2_batch4(kernel, sched, doms, max_vecs) } else { None };
    let single_vec = avx_plan.is_some() || w32_plan.is_some();
    let mut saved: Vec<u8> = if single_vec { Vec::new() } else { ra.callee_saved.clone() };
    if ra.buf_reg.get(2).copied() == Some(15) && !saved.contains(&15) { saved.push(15); }
    for &r in &saved { e.stack_op(r, false); }
    if let Some(&b2) = ra.buf_reg.get(2).filter(|&&r| r != 2) { e.mov_rr(b2, 2); }
    if let Some(ref steps) = avx_plan {
        emit_x86_avx2_steps(&mut e, steps, ra, None)?;
        e.vzeroupper();
    } else if let Some(ref w32) = w32_plan {
        emit_x86_w32_consts(&mut e, w32);
        emit_x86_w32_body(&mut e, kernel, w32, ra, 0)?;
        e.vzeroupper();
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
            if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(bs, b.block_bytes() as i32); }
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
        if let Some(ref b8) = b8_plan {
            e.op_imm8(0x83, 7, cnt_reg, 8, false);
            let jb_skip_all = e.jcc_placeholder(0x82);
            let mut jmp_fp_done: Option<usize> = None;
            let mut jb_miss_done: Option<usize> = None;
            if let Some(ref fp) = b8.fastpath {
                if !ra.buf_reg.contains(&11) && cnt_reg != 11 {
                    e.mov_imm(11, 0);
                    let fp_hoist = e.code.len();
                    crate::batch8::emit_x86_batch8_fp_consts(&mut e, fp);
                    let fp_loop = e.code.len();
                    let jnz_miss = crate::batch8::emit_x86_batch8_fp_body(&mut e, ra, fp)?;
                    for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                        if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 8); }
                    }
                    e.op_imm8(0x83, 5, cnt_reg, 8, false);
                    e.op_imm8(0x83, 7, cnt_reg, 8, false);
                    let jae_fp = e.jcc_placeholder(0x83);
                    e.patch_rel32(jae_fp, 2, fp_loop);
                    jmp_fp_done = Some(e.jmp_placeholder());
                    let fp_miss = e.code.len();
                    e.patch_rel32(jnz_miss, 2, fp_miss);
                    for &(r, c) in &b8.hoisted { e.vex_broadcast32_rip(r, c); }
                    crate::batch8::emit_x86_batch8_body(&mut e, kernel, ra, b8, 0)?;
                    for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                        if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 8); }
                    }
                    e.op_imm8(0x83, 5, cnt_reg, 8, false);
                    e.op_imm8(0x83, 7, cnt_reg, 8, false);
                    jb_miss_done = Some(e.jcc_placeholder(0x82));
                    e.op_imm8(0x83, 0, 11, 1, false);
                    e.op_imm8(0x83, 7, 11, 2, false);
                    let jb_reprobe = e.jcc_placeholder(0x82);
                    e.patch_rel32(jb_reprobe, 2, fp_hoist);
                } else {
                    for &(r, c) in &b8.hoisted { e.vex_broadcast32_rip(r, c); }
                }
            } else {
                for &(r, c) in &b8.hoisted { e.vex_broadcast32_rip(r, c); }
            }
            e.op_imm8(0x83, 7, cnt_reg, 16, false);
            let jb_skip16 = e.jcc_placeholder(0x82);
            let b16_top = e.code.len();
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if b.access == BufAccess::ReadOnly { e.prefetcht0(br, 512); }
            }
            crate::batch8::emit_x86_batch8_body(&mut e, kernel, ra, b8, 0)?;
            crate::batch8::emit_x86_batch8_body(&mut e, kernel, ra, b8, 8)?;
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 16); }
            }
            e.op_imm8(0x83, 5, cnt_reg, 16, false);
            e.op_imm8(0x83, 7, cnt_reg, 16, false);
            let jae_b16 = e.jcc_placeholder(0x83);
            e.patch_rel32(jae_b16, 2, b16_top);
            let chk8_pos = e.code.len();
            e.patch_rel32(jb_skip16, 2, chk8_pos);
            e.op_imm8(0x83, 7, cnt_reg, 8, false);
            let jb_skip8 = e.jcc_placeholder(0x82);
            crate::batch8::emit_x86_batch8_body(&mut e, kernel, ra, b8, 0)?;
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 8); }
            }
            e.op_imm8(0x83, 5, cnt_reg, 8, false);
            let vzu_pos = e.code.len();
            e.patch_rel32(jb_skip8, 2, vzu_pos);
            if let Some(p) = jmp_fp_done { e.patch_rel32(p, 1, vzu_pos); }
            if let Some(p) = jb_miss_done { e.patch_rel32(p, 2, vzu_pos); }
            e.vzeroupper();
            let tail_pos = e.code.len();
            e.patch_rel32(jb_skip_all, 2, tail_pos);
        } else if let Some(ref b4) = b4_plan {
            e.op_imm8(0x83, 7, cnt_reg, 4, false);
            let jb_skip_all = e.code.len();
            e.code.extend_from_slice(&[0x0F, 0x82, 0, 0, 0, 0]); e.insns += 1;
            for &(r, c) in &b4.hoisted { e.vex_broadcast_rip(r, c); }
            e.op_imm8(0x83, 7, cnt_reg, 8, false);
            let jb_skip8 = e.code.len();
            e.code.extend_from_slice(&[0x0F, 0x82, 0, 0, 0, 0]); e.insns += 1;
            let b8_top = e.code.len();
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if b.access == BufAccess::ReadOnly { e.prefetcht0(br, 256); }
            }
            emit_x86_batch4_body(&mut e, kernel, sched, ra, doms, b4, 0)?;
            emit_x86_batch4_body(&mut e, kernel, sched, ra, doms, b4, 4)?;
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 8); }
            }
            e.op_imm8(0x83, 5, cnt_reg, 8, false);
            e.op_imm8(0x83, 7, cnt_reg, 8, false);
            let back8 = (b8_top as isize) - ((e.code.len() + 6) as isize);
            e.code.extend_from_slice(&[0x0F, 0x83]); e.code.extend_from_slice(&(back8 as i32).to_le_bytes()); e.insns += 1;
            let chk4_pos = e.code.len();
            let fwd8 = (chk4_pos as isize) - ((jb_skip8 + 6) as isize);
            if let Some(s) = e.code.get_mut(jb_skip8 + 2..jb_skip8 + 6) { s.copy_from_slice(&(fwd8 as i32).to_le_bytes()); }
            e.op_imm8(0x83, 7, cnt_reg, 4, false);
            let jb_skip4 = e.code.len();
            e.code.extend_from_slice(&[0x0F, 0x82, 0, 0, 0, 0]); e.insns += 1;
            emit_x86_batch4_body(&mut e, kernel, sched, ra, doms, b4, 0)?;
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 4); }
            }
            e.op_imm8(0x83, 5, cnt_reg, 4, false);
            let vzu_pos = e.code.len();
            let fwd4 = (vzu_pos as isize) - ((jb_skip4 + 6) as isize);
            if let Some(s) = e.code.get_mut(jb_skip4 + 2..jb_skip4 + 6) { s.copy_from_slice(&(fwd4 as i32).to_le_bytes()); }
            e.vzeroupper();
            let tail_pos = e.code.len();
            let fwd_all = (tail_pos as isize) - ((jb_skip_all + 6) as isize);
            if let Some(s) = e.code.get_mut(jb_skip_all + 2..jb_skip_all + 6) { s.copy_from_slice(&(fwd_all as i32).to_le_bytes()); }
        }
        e.push_rr(0x85, cnt_reg, cnt_reg);
        let jz_pos = e.code.len();
        e.code.extend_from_slice(&[0x0F, 0x84, 0, 0, 0, 0]); e.insns += 1;
        if let Some(ref steps) = avx_plan {
            emit_x86_avx2_steps(&mut e, steps, ra, Some(true))?;
        } else if let Some(ref w32) = w32_plan {
            emit_x86_w32_consts(&mut e, w32);
            e.op_imm8(0x83, 7, cnt_reg, 2, false);
            let jb_w32_2 = e.code.len();
            e.code.extend_from_slice(&[0x0F, 0x82, 0, 0, 0, 0]); e.insns += 1;
            let w32_2_top = e.code.len();
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if b.access == BufAccess::ReadOnly { e.prefetcht0(br, 256); }
            }
            emit_x86_w32_body(&mut e, kernel, w32, ra, 0)?;
            emit_x86_w32_body(&mut e, kernel, w32, ra, 1)?;
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, (b.block_bytes() as i32) * 2); }
            }
            e.op_imm8(0x83, 5, cnt_reg, 2, false);
            e.op_imm8(0x83, 7, cnt_reg, 2, false);
            let back_w32 = (w32_2_top as isize) - ((e.code.len() + 6) as isize);
            e.code.extend_from_slice(&[0x0F, 0x83]); e.code.extend_from_slice(&(back_w32 as i32).to_le_bytes()); e.insns += 1;
            let w32_tail_chk = e.code.len();
            let fwd_w32 = (w32_tail_chk as isize) - ((jb_w32_2 + 6) as isize);
            if let Some(s) = e.code.get_mut(jb_w32_2 + 2..jb_w32_2 + 6) { s.copy_from_slice(&(fwd_w32 as i32).to_le_bytes()); }
            e.push_rr(0x85, cnt_reg, cnt_reg);
            let jz_w32_done = e.code.len();
            e.code.extend_from_slice(&[0x0F, 0x84, 0, 0, 0, 0]); e.insns += 1;
            emit_x86_w32_body(&mut e, kernel, w32, ra, 0)?;
            let w32_vzu = e.code.len();
            let fwd_w32_done = (w32_vzu as isize) - ((jz_w32_done + 6) as isize);
            if let Some(s) = e.code.get_mut(jz_w32_done + 2..jz_w32_done + 6) { s.copy_from_slice(&(fwd_w32_done as i32).to_le_bytes()); }
        }
        if w32_plan.is_none() {
            let loop_top = e.code.len();
            if let Some(ref steps) = avx_plan {
                emit_x86_avx2_steps(&mut e, steps, ra, Some(false))?;
            } else {
                emit_x86_body(&mut e, kernel, sched, ra, doms)?;
            }
            for (&br, b) in ra.buf_reg.iter().zip(kernel.bufs.iter()) {
                if !matches!(b.access, BufAccess::LookupTable(_)) { e.add_imm(br, b.block_bytes() as i32); }
            }
            e.op_imm8(0x83, 0, cnt_reg, 0xFF, false);
            let back = (loop_top as isize) - ((e.code.len() + 6) as isize);
            e.code.extend_from_slice(&[0x0F, 0x85]); e.code.extend_from_slice(&(back as i32).to_le_bytes()); e.insns += 1;
        }
        if single_vec { e.vzeroupper(); }
        let done_pos = e.code.len();
        let fwd = (done_pos as isize) - ((jz_pos + 6) as isize);
        if let Some(s) = e.code.get_mut(jz_pos + 2..jz_pos + 6) { s.copy_from_slice(&(fwd as i32).to_le_bytes()); }
        for &r in saved.iter().rev() { e.stack_op(r, true); }
        e.ret();
    }
    let b_size = (e.code.len() as u64) - b_off;
    e.finalize_const_pool();
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
            (false, ElemTy::I64 | ElemTy::U64) => 0xF940_0000,
            (false, ElemTy::I32) => 0xB980_0000,
            (false, ElemTy::I16) => 0x7980_0000,
            (false, ElemTy::U32) => 0xB940_0000,
            (false, ElemTy::U16) => 0x7940_0000,
            (false, ElemTy::U8) => 0x3940_0000,
            (true, ElemTy::I64 | ElemTy::U64) => 0xF900_0000,
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
            ElemTy::I64 | ElemTy::U64 => (0xF860_0000, 0x7800), ElemTy::I32 => (0xB8A0_0000, 0x7800),
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
            Op::BitOr(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0xAA00_0000, vb, 0, va, dst)); }
            Op::BitXor(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0xCA00_0000, vb, 0, va, dst)); }
            Op::ShlVar(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x9AC0_0000, vb, 0x2000, va, dst)); }
            Op::ShrVarU(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x9AC0_0000, vb, 0x2400, va, dst)); }
            Op::Mul(a, b) => { let (va, vb) = (arm_val(e, kernel, ra, a, 16)?, arm_val(e, kernel, ra, b, 17)?); e.put(arm64_rrr(0x9B00_0000, vb, 0x7C00, va, dst)); }
            Op::Popcnt(a) => {
                let va = arm_val(e, kernel, ra, a, 16)?;
                e.put(0x9E67_0000 | ((va as u32 & 31) << 5));
                e.put(0x0E20_5800);
                e.put(0x0E31_B800);
                e.put(0x0E01_3C00 | (dst as u32 & 31));
            }
            Op::Tzcnt(a) => {
                let va = arm_val(e, kernel, ra, a, 16)?;
                e.put(0xDAC0_0000 | ((va as u32 & 31) << 5) | (dst as u32 & 31));
                e.put(0xDAC0_1000 | ((dst as u32 & 31) << 5) | (dst as u32 & 31));
            }
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
            Op::Ushr(a, s) => {
                let va = arm_val(e, kernel, ra, a, 16)?;
                if s == 0 { e.mov_rr(dst, va); } else {
                    e.put(0xD340_FC00 | ((u32::from(s) & 63) << 16) | ((va as u32 & 31) << 5) | (dst as u32 & 31));
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
                let inv_cc = match k {
                    CmpKind::Eq => 1, CmpKind::Ne => 0, CmpKind::Lt => 10, CmpKind::Ge => 11, CmpKind::Le => 12, CmpKind::Gt => 13,
                    CmpKind::Ult => 2, CmpKind::Uge => 3, CmpKind::Ule => 8, CmpKind::Ugt => 9,
                };
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
    let base = if optimize { optimize_kernel(kernel)? } else { kernel.clone() };
    let mut lowered = base.clone();
    let (sched, ra) = match schedule_and_verify(&lowered, budget).and_then(|s| allocate_registers(&lowered, &s, isa, budget).map(|r| (s, r))) {
        Ok(pair) => pair,
        Err(CompileError::RegisterSpill { .. }) => {
            let mut found = None;
            for max_nodes in [1u8, 3u8, 6u8, 12u8, 24u8, 48u8] {
                let cand = remat_readonly_loads(&base, max_nodes);
                if let Ok(s) = schedule_and_verify(&cand, budget) {
                    if let Ok(r) = allocate_registers(&cand, &s, isa, budget) {
                        lowered = cand;
                        found = Some((s, r));
                        break;
                    }
                }
            }
            if let Some(pair) = found {
                pair
            } else {
                let s = schedule_and_verify(&base, budget)?;
                let r = allocate_registers(&base, &s, isa, budget)?;
                (s, r)
            }
        }
        Err(e) => return Err(e),
    };
    let doms = lowered.infer_domains();
    let module = match isa {
        TargetIsa::X86_64 => emit_x86_module(&lowered, &base, &sched, &ra, &doms, budget.max_vecs, call_batch)?,
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
