use crate::codegen::{RegBudget, ScheduleReport};
use crate::elf64::{emit_elf64_object, verify_elf64_object, ElfReloc, ElfSymbol, TargetIsa};
use crate::ir::{BufAccess, CmpKind, CompileError, ElemTy, Kernel, NodeId, Op};
use crate::isa::{x86_modrm, x86_rex_byte, x86_rex_w, x86_sib, RegAlloc};
use crate::tensor::{TensorKernel, TilePlan};

type Res<T> = Result<T, CompileError>;
type ModuleCode = (Vec<u8>, Vec<ElfSymbol>, Vec<ElfReloc>, usize);

const X86_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;
const R12: u8 = 12;
const R13: u8 = 13;
const R14: u8 = 14;
const R15: u8 = 15;

#[derive(Clone, Debug)]
pub struct CompiledTensorObject {
    pub name: String,
    pub symbols: Vec<String>,
    pub code_bytes: Vec<u8>,
    pub elf_bytes: Vec<u8>,
    pub direct_insns: usize,
    pub zero_spills: bool,
    pub tile: TilePlan,
}

fn berr(m: &str) -> CompileError { CompileError::BackendError(m.into()) }

/// @domain map: 1..=3, w: 0..=1, pp: 0..=3, ll: 0..=2, aaa: 0..=7
pub const fn x86_evex4(regs: [u8; 4], is_reg_rm: bool, enc: [u8; 4], mask: [u8; 3]) -> [u8; 4] {
    let [reg, idx, base, vvvv] = regs;
    let [map, w, pp, ll] = enc;
    let [z, bcast, aaa] = mask;
    let r = (reg >> 3) & 1;
    let r2 = (reg >> 4) & 1;
    let b = (base >> 3) & 1;
    let x = if is_reg_rm { (base >> 4) & 1 } else { (idx >> 3) & 1 };
    let v2 = (vvvv >> 4) & 1;
    let p0 = ((!r & 1) << 7) | ((!x & 1) << 6) | ((!b & 1) << 5) | ((!r2 & 1) << 4) | (map & 3);
    let p1 = ((w & 1) << 7) | ((!vvvv & 0xF) << 3) | 0x04 | (pp & 3);
    let p2 = ((z & 1) << 7) | ((ll & 3) << 5) | ((bcast & 1) << 4) | ((!v2 & 1) << 3) | (aaa & 7);
    [0x62, p0, p1, p2]
}

#[derive(Clone, Copy)]
struct EvSpec { map: u8, w: u8, pp: u8, ll: u8, op: u8 }

#[derive(Clone, Copy)]
struct MemOp { base: u8, idx: Option<(u8, u8)>, disp: i32 }

impl MemOp {
    const fn bd(base: u8, disp: i32) -> Self { Self { base, idx: None, disp } }
    const fn sib(base: u8, idx: u8, scale: u8, disp: i32) -> Self { Self { base, idx: Some((idx, scale)), disp } }
}

#[derive(Default)]
struct Avx512Emit {
    code: Vec<u8>,
    insns: usize,
    f32_pool: Vec<u32>,
    f32_patches: Vec<(usize, usize)>,
}

impl Avx512Emit {
    fn raw(&mut self, b: &[u8]) -> &mut Self {
        self.code.extend_from_slice(b);
        self
    }
    fn emit(&mut self, b: &[u8]) -> &mut Self {
        self.insns += 1;
        self.raw(b)
    }
    fn align(&mut self, a: usize) {
        while self.code.len() % a != 0 { self.code.push(0x90); }
    }
    fn push_r(&mut self, r: u8) -> &mut Self {
        if r >= 8 { self.code.push(0x41); }
        self.emit(&[0x50 + (r & 7)])
    }
    fn pop_r(&mut self, r: u8) -> &mut Self {
        if r >= 8 { self.code.push(0x41); }
        self.emit(&[0x58 + (r & 7)])
    }
    fn ret(&mut self) -> &mut Self { self.emit(&[0xC3]) }
    fn vzeroupper(&mut self) -> &mut Self { self.emit(&[0xC5, 0xF8, 0x77]) }
    fn mov_rr(&mut self, d: u8, s: u8) -> &mut Self {
        if d != s { self.emit(&[x86_rex_w(s, d), 0x89, x86_modrm(3, s, d)]) } else { self }
    }
    fn movsxd(&mut self, d: u8, s: u8) -> &mut Self {
        self.emit(&[x86_rex_w(d, s), 0x63, x86_modrm(3, d, s)])
    }
    fn mov_ri32(&mut self, d: u8, v: i32) -> &mut Self {
        if v == 0 {
            if d >= 8 { self.code.push(0x45); }
            self.emit(&[0x31, x86_modrm(3, d, d)])
        } else {
            self.raw(&[x86_rex_w(0, d), 0xC7, x86_modrm(3, 0, d)]).emit(&v.to_le_bytes())
        }
    }
    fn add_rr(&mut self, d: u8, s: u8) -> &mut Self { self.emit(&[x86_rex_w(s, d), 0x01, x86_modrm(3, s, d)]) }
    fn sub_rr(&mut self, d: u8, s: u8) -> &mut Self { self.emit(&[x86_rex_w(s, d), 0x29, x86_modrm(3, s, d)]) }
    fn cmp_rr(&mut self, a: u8, b: u8) -> &mut Self { self.emit(&[x86_rex_w(b, a), 0x39, x86_modrm(3, b, a)]) }
    fn test_rr(&mut self, a: u8, b: u8) -> &mut Self { self.emit(&[x86_rex_w(b, a), 0x85, x86_modrm(3, b, a)]) }
    fn alu_ri(&mut self, subop: u8, d: u8, imm: i32) -> &mut Self {
        if let Ok(c) = i8::try_from(imm) {
            self.emit(&[x86_rex_w(0, d), 0x83, x86_modrm(3, subop, d), c as u8])
        } else {
            self.raw(&[x86_rex_w(0, d), 0x81, x86_modrm(3, subop, d)]).emit(&imm.to_le_bytes())
        }
    }
    fn add_ri(&mut self, d: u8, imm: i32) -> &mut Self { self.alu_ri(0, d, imm) }
    fn sub_ri(&mut self, d: u8, imm: i32) -> &mut Self { self.alu_ri(5, d, imm) }
    fn cmp_ri(&mut self, d: u8, imm: i32) -> &mut Self { self.alu_ri(7, d, imm) }
    fn shl_ri(&mut self, d: u8, sh: u8) -> &mut Self { self.emit(&[x86_rex_w(0, d), 0xC1, x86_modrm(3, 4, d), sh]) }
    fn shl_cl_r32(&mut self, d: u8) -> &mut Self {
        if d >= 8 { self.code.push(0x41); }
        self.emit(&[0xD3, x86_modrm(3, 4, d)])
    }
    fn and_ri(&mut self, d: u8, imm: i32) -> &mut Self { self.alu_ri(4, d, imm) }
    fn imul_rr(&mut self, d: u8, s: u8) -> &mut Self {
        self.emit(&[x86_rex_w(d, s), 0x0F, 0xAF, x86_modrm(3, d, s)])
    }
    fn gpr_mem(&mut self, w: bool, op: u8, r: u8, m: MemOp) -> &mut Self {
        let idx = m.idx.map_or(0, |(i, _)| i);
        self.code.push(x86_rex_byte(w, r, idx, m.base));
        self.code.push(op);
        self.emit_modrm_mem(r, m, 1)
    }
    fn mov_rm64(&mut self, d: u8, m: MemOp) -> &mut Self { self.gpr_mem(true, 0x8B, d, m) }
    fn mov_mr64(&mut self, m: MemOp, s: u8) -> &mut Self { self.gpr_mem(true, 0x89, s, m) }
    fn movsxd_rm(&mut self, d: u8, m: MemOp) -> &mut Self { self.gpr_mem(true, 0x63, d, m) }
    fn lea(&mut self, d: u8, m: MemOp) -> &mut Self { self.gpr_mem(true, 0x8D, d, m) }
    fn prefetcht0(&mut self, m: MemOp) -> &mut Self {
        let idx = m.idx.map_or(0, |(i, _)| i);
        if idx >= 8 || m.base >= 8 { self.code.push(x86_rex_byte(false, 0, idx, m.base)); }
        self.raw(&[0x0F, 0x18]);
        self.emit_modrm_mem(1, m, 1)
    }
    fn emit_modrm_mem(&mut self, reg: u8, m: MemOp, disp8_scale: i32) -> &mut Self {
        let b7 = m.base & 7;
        let (md, disp_bytes): (u8, Vec<u8>) = if m.disp == 0 && b7 != 5 {
            (0, Vec::new())
        } else if m.disp % disp8_scale == 0 {
            let scaled = m.disp / disp8_scale;
            if let Ok(d8) = i8::try_from(scaled) { (1, vec![d8 as u8]) }
            else { (2, m.disp.to_le_bytes().to_vec()) }
        } else {
            (2, m.disp.to_le_bytes().to_vec())
        };
        if let Some((idx, scale)) = m.idx {
            self.raw(&[x86_modrm(md, reg, 4), x86_sib(scale, idx, m.base)]).emit(&disp_bytes)
        } else if b7 == 4 {
            self.raw(&[x86_modrm(md, reg, 4), 0x24]).emit(&disp_bytes)
        } else {
            self.raw(&[x86_modrm(md, reg, m.base)]).emit(&disp_bytes)
        }
    }
    fn jcc_ph(&mut self, cc: u8) -> usize {
        let p = self.code.len();
        self.raw(&[0x0F, cc]).emit(&[0, 0, 0, 0]);
        p
    }
    fn jmp_ph(&mut self) -> usize {
        let p = self.code.len();
        self.raw(&[0xE9]).emit(&[0, 0, 0, 0]);
        p
    }
    fn rel32_to(&mut self, pfx: &[u8], target: usize) -> &mut Self {
        let rel = (target as isize) - ((self.code.len() + pfx.len() + 4) as isize);
        self.raw(pfx).emit(&(rel as i32).to_le_bytes())
    }
    fn call_rel(&mut self, target: usize) -> &mut Self { self.rel32_to(&[0xE8], target) }
    fn jcc_to(&mut self, cc: u8, target: usize) -> &mut Self { self.rel32_to(&[0x0F, cc], target) }
    fn jmp_to(&mut self, target: usize) -> &mut Self { self.rel32_to(&[0xE9], target) }
    fn patch_rel(&mut self, ph: usize, op_len: usize) {
        let rel = ((self.code.len() as isize) - ((ph + op_len + 4) as isize)) as i32;
        if let Some(s) = self.code.get_mut(ph + op_len..ph + op_len + 4) { s.copy_from_slice(&rel.to_le_bytes()); }
    }
    fn patch_jcc(&mut self, ph: usize) { self.patch_rel(ph, 2); }
    fn patch_jmp(&mut self, ph: usize) { self.patch_rel(ph, 1); }
    fn kmovw_k1_eax(&mut self) -> &mut Self { self.emit(&[0xC5, 0xF8, 0x92, 0xC8]) }
    fn ev_rr(&mut self, sp: EvSpec, rvr: [u8; 3], za: [u8; 2], imm: Option<u8>) -> &mut Self {
        let pfx = x86_evex4([rvr[0], 0, rvr[2], rvr[1]], true, [sp.map, sp.w, sp.pp, sp.ll], [za[0], 0, za[1]]);
        self.raw(&pfx).raw(&[sp.op]);
        if let Some(ib) = imm { self.raw(&[x86_modrm(3, rvr[0], rvr[2])]).emit(&[ib]) }
        else { self.emit(&[x86_modrm(3, rvr[0], rvr[2])]) }
    }
    fn ev_mem(&mut self, sp: EvSpec, rv: [u8; 2], m: MemOp, zba: [u8; 3], disp8_scale: i32) -> &mut Self {
        let idx = m.idx.map_or(0, |(i, _)| i);
        let pfx = x86_evex4([rv[0], idx, m.base, rv[1]], false, [sp.map, sp.w, sp.pp, sp.ll], zba);
        self.raw(&pfx).raw(&[sp.op]);
        self.emit_modrm_mem(rv[0], m, disp8_scale)
    }
    fn r512(&mut self, map: u8, pp: u8, op: u8, rvr: [u8; 3]) -> &mut Self {
        self.ev_rr(EvSpec { map, w: 0, pp, ll: 2, op }, rvr, [0, 0], None)
    }
    fn vxorps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 1, 0xEF, [d, a, b]) }
    fn vmovaps(&mut self, d: u8, s: u8) -> &mut Self {
        if d != s { self.r512(1, 0, 0x28, [d, 0, s]) } else { self }
    }
    fn vmovups_ld(&mut self, d: u8, m: MemOp, masked: bool) -> &mut Self {
        let (z, aaa) = if masked { (1, 1) } else { (0, 0) };
        self.ev_mem(EvSpec { map: 1, w: 0, pp: 0, ll: 2, op: 0x10 }, [d, 0], m, [z, 0, aaa], 64)
    }
    fn vmovups_st(&mut self, m: MemOp, s: u8, masked: bool) -> &mut Self {
        self.ev_mem(EvSpec { map: 1, w: 0, pp: 0, ll: 2, op: 0x11 }, [s, 0], m, [0, 0, u8::from(masked)], 64)
    }
    fn vmem_plain(&mut self, spec: [u8; 4], rv: [u8; 2], m: MemOp, bcast: u8, sc: i32) -> &mut Self {
        let [map, pp, ll, op] = spec;
        self.ev_mem(EvSpec { map, w: 0, pp, ll, op }, rv, m, [0, bcast, 0], sc)
    }
    fn vmovss_ld(&mut self, d: u8, m: MemOp) -> &mut Self { self.vmem_plain([1, 2, 0, 0x10], [d, 0], m, 0, 4) }
    fn vmovss_st(&mut self, m: MemOp, s: u8) -> &mut Self { self.vmem_plain([1, 2, 0, 0x11], [s, 0], m, 0, 4) }
    fn vfmadd231ss_mem(&mut self, c: u8, b: u8, m: MemOp) -> &mut Self {
        self.vmem_plain([2, 1, 0, 0xB9], [c, b], m, 0, 4)
    }
    fn rll(&mut self, spec: [u8; 4], rvr: [u8; 3], aaa: u8) -> &mut Self {
        let [map, pp, ll, op] = spec;
        self.ev_rr(EvSpec { map, w: 0, pp, ll, op }, rvr, [0, aaa], None)
    }
    fn vaddss(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.rll([1, 2, 0, 0x58], [d, a, b], 0) }
    fn vrr_imm(&mut self, spec: [u8; 4], rvr: [u8; 3], imm: u8) -> &mut Self {
        let [map, pp, ll, op] = spec;
        self.ev_rr(EvSpec { map, w: 0, pp, ll, op }, rvr, [0, 0], Some(imm))
    }
    fn vextractf32x8(&mut self, dst: u8, src: u8, imm: u8) -> &mut Self {
        self.vrr_imm([3, 1, 2, 0x1B], [src, 0, dst], imm)
    }
    fn vaddps_256(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.rll([1, 0, 1, 0x58], [d, a, b], 0) }
    fn vextractf32x4(&mut self, dst: u8, src: u8, imm: u8) -> &mut Self {
        self.vrr_imm([3, 1, 1, 0x19], [src, 0, dst], imm)
    }
    fn vaddps_128(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.rll([1, 0, 0, 0x58], [d, a, b], 0) }
    fn vshufps_128(&mut self, d: u8, a: u8, b: u8, imm: u8) -> &mut Self {
        self.vrr_imm([1, 0, 0, 0xC6], [d, a, b], imm)
    }
    fn vmovups256_ld(&mut self, d: u8, m: MemOp) -> &mut Self { self.vmem_plain([1, 0, 1, 0x10], [d, 0], m, 0, 32) }
    fn vmovups256_st(&mut self, m: MemOp, s: u8) -> &mut Self { self.vmem_plain([1, 0, 1, 0x11], [s, 0], m, 0, 32) }
    fn vbroadcastss_mem(&mut self, d: u8, m: MemOp) -> &mut Self { self.vmem_plain([2, 1, 2, 0x18], [d, 0], m, 0, 4) }
    fn vbroadcastss_xmm(&mut self, d: u8, s: u8) -> &mut Self { self.r512(2, 1, 0x18, [d, 0, s]) }
    fn vbroadcastss_rip(&mut self, d: u8, bits: u32) -> &mut Self {
        if bits == 0 { return self.vxorps(d, d, d); }
        let idx = self.f32_pool.iter().position(|&c| c == bits).unwrap_or_else(|| {
            self.f32_pool.push(bits);
            self.f32_pool.len() - 1
        });
        let pfx = x86_evex4([d, 0, 0, 0], false, [2, 0, 1, 2], [0, 0, 0]);
        self.raw(&pfx).raw(&[0x18, x86_modrm(0, d, 5)]);
        let off = self.code.len();
        self.f32_patches.push((off, idx));
        self.emit(&[0, 0, 0, 0])
    }
    fn vfmadd231ps_rr(&mut self, c: u8, b: u8, a: u8) -> &mut Self { self.r512(2, 1, 0xB8, [c, b, a]) }
    fn vfmadd231ps_bcst(&mut self, c: u8, b: u8, m: MemOp) -> &mut Self {
        self.vmem_plain([2, 1, 2, 0xB8], [c, b], m, 1, 4)
    }
    fn vfnmadd231ps_rr(&mut self, c: u8, b: u8, a: u8) -> &mut Self { self.r512(2, 1, 0xBC, [c, b, a]) }
    fn vaddps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x58, [d, a, b]) }
    fn vsubps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x5C, [d, a, b]) }
    fn vmulps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x59, [d, a, b]) }
    fn vdivps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x5E, [d, a, b]) }
    fn vmaxps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x5F, [d, a, b]) }
    fn vminps(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 0, 0x5D, [d, a, b]) }
    fn vsqrtps(&mut self, d: u8, a: u8) -> &mut Self { self.r512(1, 0, 0x51, [d, 0, a]) }
    fn vpandd(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 1, 0xDB, [d, a, b]) }
    fn vpandnd(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 1, 0xDF, [d, a, b]) }
    fn vpxord(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 1, 0xEF, [d, a, b]) }
    fn vpaddd(&mut self, d: u8, a: u8, b: u8) -> &mut Self { self.r512(1, 1, 0xFE, [d, a, b]) }
    fn vpslld_imm(&mut self, d: u8, s: u8, imm: u8) -> &mut Self { self.vrr_imm([1, 1, 2, 0x72], [6, d, s], imm) }
    fn vcvttps2dq(&mut self, d: u8, s: u8) -> &mut Self { self.r512(1, 2, 0x5B, [d, 0, s]) }
    fn vrndscaleps(&mut self, d: u8, s: u8, imm: u8) -> &mut Self { self.vrr_imm([3, 1, 2, 0x08], [d, 0, s], imm) }
    fn vcmpps_k1(&mut self, a: u8, b: u8, pred: u8) -> &mut Self { self.vrr_imm([1, 0, 2, 0xC2], [1, a, b], pred) }
    fn vblendmps_k1(&mut self, d: u8, f_val: u8, t_val: u8) -> &mut Self { self.rll([2, 1, 2, 0x65], [d, f_val, t_val], 1) }
    fn finalize_f32_pool(&mut self) {
        if self.f32_pool.is_empty() { return; }
        self.align(4);
        let base = self.code.len();
        for &c in &self.f32_pool { self.code.extend_from_slice(&c.to_le_bytes()); }
        for &(off, idx) in &self.f32_patches {
            let d = (base.saturating_add(idx.saturating_mul(4)) as isize) - ((off + 4) as isize);
            if let Some(s) = self.code.get_mut(off..off + 4) { s.copy_from_slice(&(d as i32).to_le_bytes()); }
        }
    }
}

// =============================================================================
// Part 1: Direct ELF64 AVX-512 compiler for f32 DAG Kernels (8-lane or 16-lane)
// =============================================================================

#[derive(Clone, Copy)]
enum F32Step {
    Load(u8),
    Store(u8, usize),
    Const(u32),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Fma(usize, usize, usize),
    Max(usize, usize),
    Min(usize, usize),
    Sqrt(usize),
    Neg(usize),
    Abs(usize),
    Cmp(CmpKind, usize, usize),
    Sel(usize, usize, usize),
}

struct F32Plan {
    lanes: u8,
    steps: Vec<(usize, F32Step)>,
    phys: Vec<u8>,
    peak_vecs: usize,
}

fn build_f32_plan(k: &Kernel, budget: RegBudget) -> Res<F32Plan> {
    if k.bufs.is_empty() || k.bufs.len() > 5 { return Err(berr("f32 kernel requires 1..=5 buffers")); }
    let lanes = k.bufs[0].lanes;
    if lanes != 8 && lanes != 16 { return Err(berr("avx512 f32 kernel requires 8 or 16 lanes")); }
    if k.bufs.iter().any(|b| b.elem_ty != ElemTy::F32 || b.lanes != lanes
        || matches!(b.access, BufAccess::LookupTable(_))) {
        return Err(berr("avx512 f32 kernel requires uniform f32 buffers"));
    }
    let mut out_groups: Vec<(u8, Vec<NodeId>)> = Vec::new();
    for b in &k.bufs {
        if b.access != BufAccess::MutNoAlias { continue; }
        let mut g = vec![usize::MAX; usize::from(lanes)];
        for o in &k.outputs { if o.buf == b.id && usize::from(o.lane) < g.len() { g[usize::from(o.lane)] = o.val; } }
        if g.iter().all(|&x| x != usize::MAX) { out_groups.push((b.id, g)); }
    }
    if out_groups.is_empty() { return Err(berr("f32 kernel has no complete output vector")); }
    let mut memo: Vec<(Vec<NodeId>, usize)> = Vec::new();
    let mut steps: Vec<(usize, F32Step)> = Vec::new();
    fn lower_vec(k: &Kernel, ids: &[NodeId], memo: &mut Vec<(Vec<NodeId>, usize)>,
        steps: &mut Vec<(usize, F32Step)>) -> Res<usize> {
        if let Some(&(_, v)) = memo.iter().find(|(m, _)| m.as_slice() == ids) { return Ok(v); }
        let o0 = *k.nodes.get(ids[0]).ok_or(CompileError::InvalidNode(ids[0]))?;
        for &id in &ids[1..] {
            let op = *k.nodes.get(id).ok_or(CompileError::InvalidNode(id))?;
            if std::mem::discriminant(&op) != std::mem::discriminant(&o0) {
                return Err(berr("non-uniform f32 lane expression"));
            }
        }
        let step = match o0 {
            Op::FConst(bits) => {
                if !ids[1..].iter().all(|&i| k.nodes.get(i) == Some(&Op::FConst(bits))) {
                    return Err(berr("non-uniform f32 constant across lanes"));
                }
                F32Step::Const(bits)
            }
            Op::Load { mem, buf, lane: 0 } if k.nodes.get(mem) == Some(&Op::MemInit) => {
                for (li, &id) in ids.iter().enumerate() {
                    if k.nodes.get(id) != Some(&Op::Load { mem, buf, lane: li as u8 }) {
                        return Err(berr("non-contiguous f32 load"));
                    }
                }
                F32Step::Load(buf)
            }
            Op::FAdd(..) | Op::FSub(..) | Op::FMul(..) | Op::FDiv(..) | Op::FMax(..) | Op::FMin(..)
            | Op::FCmp(..) => {
                let (mut a, mut b) = (Vec::with_capacity(ids.len()), Vec::with_capacity(ids.len()));
                for &id in ids {
                    let (ch, _) = k.nodes[id].children();
                    a.push(ch[0]);
                    b.push(ch[1]);
                }
                let (va, vb) = (lower_vec(k, &a, memo, steps)?, lower_vec(k, &b, memo, steps)?);
                match o0 {
                    Op::FAdd(..) => F32Step::Add(va, vb),
                    Op::FSub(..) => F32Step::Sub(va, vb),
                    Op::FMul(..) => F32Step::Mul(va, vb),
                    Op::FDiv(..) => F32Step::Div(va, vb),
                    Op::FMax(..) => F32Step::Max(va, vb),
                    Op::FMin(..) => F32Step::Min(va, vb),
                    Op::FCmp(ck, ..) => F32Step::Cmp(ck, va, vb),
                    _ => return Err(berr("unexpected binary f32 op")),
                }
            }
            Op::FFma(..) | Op::Select { .. } => {
                let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
                for &id in ids {
                    let (ch, _) = k.nodes[id].children();
                    a.push(ch[0]);
                    b.push(ch[1]);
                    c.push(ch[2]);
                }
                let (va, vb, vc) = (lower_vec(k, &a, memo, steps)?,
                    lower_vec(k, &b, memo, steps)?, lower_vec(k, &c, memo, steps)?);
                if matches!(o0, Op::FFma(..)) { F32Step::Fma(va, vb, vc) } else { F32Step::Sel(va, vb, vc) }
            }
            Op::FSqrt(_) | Op::FNeg(_) | Op::FAbs(_) => {
                let a: Vec<NodeId> = ids.iter().map(|&id| k.nodes[id].children().0[0]).collect();
                let va = lower_vec(k, &a, memo, steps)?;
                match o0 {
                    Op::FSqrt(_) => F32Step::Sqrt(va),
                    Op::FNeg(_) => F32Step::Neg(va),
                    _ => F32Step::Abs(va),
                }
            }
            _ => return Err(berr("unsupported op in avx512 f32 kernel")),
        };
        let vid = steps.len();
        steps.push((vid, step));
        memo.push((ids.to_vec(), vid));
        Ok(vid)
    }
    for (bid, g) in out_groups {
        let v = lower_vec(k, &g, &mut memo, &mut steps)?;
        let sid = steps.len();
        steps.push((sid, F32Step::Store(bid, v)));
    }
    let nv = steps.len();
    let mut last = vec![0usize; nv];
    for (si, &(_, st)) in steps.iter().enumerate() {
        let uses: [Option<usize>; 3] = match st {
            F32Step::Store(_, v) | F32Step::Sqrt(v) | F32Step::Neg(v) | F32Step::Abs(v) => [Some(v), None, None],
            F32Step::Add(a, b) | F32Step::Sub(a, b) | F32Step::Mul(a, b) | F32Step::Div(a, b)
            | F32Step::Max(a, b) | F32Step::Min(a, b) | F32Step::Cmp(_, a, b) => [Some(a), Some(b), None],
            F32Step::Fma(a, b, c) | F32Step::Sel(a, b, c) => [Some(a), Some(b), Some(c)],
            F32Step::Load(..) | F32Step::Const(_) => [None, None, None],
        };
        for u in uses.into_iter().flatten() { last[u] = last[u].max(si); }
    }
    let max_v = 30usize;
    let (mut free, mut active, mut phys, mut peak) = ((0..max_v as u8).rev().collect::<Vec<_>>(),
        Vec::<(usize, u8)>::new(), vec![0u8; nv], 0usize);
    for (si, &(vid, st)) in steps.iter().enumerate() {
        if !matches!(st, F32Step::Store(..)) {
            let r = free.pop().ok_or(CompileError::RegisterSpill {
                peak_gprs: k.bufs.len(), max_gprs: budget.max_gprs,
                peak_vecs: active.len() + 1, max_vecs: max_v,
            })?;
            phys[vid] = r;
            active.push((vid, r));
            peak = peak.max(active.len());
        }
        active.retain(|&(av, ar)| {
            let keep = last[av] > si;
            if !keep { free.push(ar); }
            keep
        });
    }
    Ok(F32Plan { lanes, steps, phys, peak_vecs: peak })
}

fn emit_f32_plan_body(e: &mut Avx512Emit, plan: &F32Plan, disp_bytes: i32) {
    const TMP_SIGN: u8 = 30;
    const TMP_SCR: u8 = 31;
    let is_512 = plan.lanes == 16;
    for &(vid, st) in &plan.steps {
        let d = plan.phys[vid];
        match st {
            F32Step::Load(buf) => {
                let m = MemOp::bd(X86_ARGS[usize::from(buf)], disp_bytes);
                if is_512 { e.vmovups_ld(d, m, false); } else { e.vmovups256_ld(d, m); }
            }
            F32Step::Store(buf, v) => {
                let m = MemOp::bd(X86_ARGS[usize::from(buf)], disp_bytes);
                let s = plan.phys[v];
                if is_512 { e.vmovups_st(m, s, false); } else { e.vmovups256_st(m, s); }
            }
            F32Step::Const(bits) => { e.vbroadcastss_rip(d, bits); }
            F32Step::Add(a, b) => { e.vaddps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Sub(a, b) => { e.vsubps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Mul(a, b) => { e.vmulps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Div(a, b) => { e.vdivps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Max(a, b) => { e.vmaxps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Min(a, b) => { e.vminps(d, plan.phys[a], plan.phys[b]); }
            F32Step::Sqrt(a) => { e.vsqrtps(d, plan.phys[a]); }
            F32Step::Neg(a) => {
                e.vbroadcastss_rip(TMP_SIGN, 0x8000_0000).vpxord(d, plan.phys[a], TMP_SIGN);
            }
            F32Step::Abs(a) => {
                e.vbroadcastss_rip(TMP_SIGN, 0x8000_0000).vpandnd(d, TMP_SIGN, plan.phys[a]);
            }
            F32Step::Fma(a, b, c) => {
                let (ra, rb, rc) = (plan.phys[a], plan.phys[b], plan.phys[c]);
                if d == rc { e.vfmadd231ps_rr(d, ra, rb); }
                else if d == ra { e.vmovaps(TMP_SCR, ra).vmovaps(d, rc).vfmadd231ps_rr(d, TMP_SCR, rb); }
                else if d == rb { e.vmovaps(TMP_SCR, rb).vmovaps(d, rc).vfmadd231ps_rr(d, ra, TMP_SCR); }
                else { e.vmovaps(d, rc).vfmadd231ps_rr(d, ra, rb); }
            }
            F32Step::Cmp(ck, a, b) => {
                let pred = match ck {
                    CmpKind::Eq => 0, CmpKind::Lt | CmpKind::Ult => 1, CmpKind::Le | CmpKind::Ule => 2,
                    CmpKind::Ne => 4, CmpKind::Gt | CmpKind::Ugt => 14, CmpKind::Ge | CmpKind::Uge => 13,
                };
                e.vcmpps_k1(plan.phys[a], plan.phys[b], pred)
                    .vxorps(TMP_SCR, TMP_SCR, TMP_SCR)
                    .vbroadcastss_rip(TMP_SIGN, 1.0_f32.to_bits())
                    .vblendmps_k1(d, TMP_SCR, TMP_SIGN);
            }
            F32Step::Sel(c, t, f) => {
                e.vxorps(TMP_SCR, TMP_SCR, TMP_SCR)
                    .vcmpps_k1(plan.phys[c], TMP_SCR, 4)
                    .vblendmps_k1(d, plan.phys[f], plan.phys[t]);
            }
        }
    }
}

pub(crate) fn emit_x86_avx512_f32_module(k: &Kernel, _s: &ScheduleReport,
    budget: RegBudget) -> Res<(RegAlloc, ModuleCode)> {
    let plan = build_f32_plan(k, budget)?;
    let _ = plan.peak_vecs;
    let nb = k.bufs.len();
    let stride = i32::from(plan.lanes) * 4;
    let mut e = Avx512Emit::default();
    emit_f32_plan_body(&mut e, &plan, 0);
    e.vzeroupper().ret();
    let k_size = e.code.len() as u64;
    e.align(16);
    let b_off = e.code.len() as u64;
    let cnt = X86_ARGS[nb];
    let jz = e.test_rr(cnt, cnt).jcc_ph(0x84);
    let jb2 = e.cmp_ri(cnt, 2).jcc_ph(0x82);
    e.align(16);
    let top2 = e.code.len();
    for (i, b) in k.bufs.iter().enumerate() {
        if b.access == BufAccess::ReadOnly { e.prefetcht0(MemOp::bd(X86_ARGS[i], 512)); }
    }
    emit_f32_plan_body(&mut e, &plan, 0);
    emit_f32_plan_body(&mut e, &plan, stride);
    for &arg in X86_ARGS.iter().take(nb) { e.add_ri(arg, stride * 2); }
    e.sub_ri(cnt, 2).cmp_ri(cnt, 2).jcc_to(0x83, top2);
    e.patch_jcc(jb2);
    let jb1 = e.test_rr(cnt, cnt).jcc_ph(0x84);
    emit_f32_plan_body(&mut e, &plan, 0);
    e.patch_jcc(jb1);
    e.patch_jcc(jz);
    e.vzeroupper().ret();
    let b_size = (e.code.len() as u64) - b_off;
    e.finalize_f32_pool();
    let syms = vec![
        ElfSymbol { name: format!("zero_cc_{}", k.name), offset: 0, size: k_size },
        ElfSymbol { name: format!("zero_cc_{}_batch", k.name), offset: b_off, size: b_size },
    ];
    let ra = RegAlloc {
        node_reg: vec![None; k.nodes.len()],
        buf_reg: X86_ARGS[..nb].to_vec(),
        callee_saved: Vec::new(),
        spills: 0,
    };
    Ok((ra, (e.code, syms, Vec::new(), e.insns)))
}

// =============================================================================
// Part 2: Direct ELF64 AVX-512 SGEMM + Fused Epilogue Compiler (Zen 4 Optimal)
// =============================================================================
//
// Internal leaf microkernel ABI (ZERO stack spills in K-loop!):
//   rdi = kc (> 0)
//   rsi = B_ptr
//   rdx = ldb_bytes
//   rcx = C_ptr
//   r8  = A_ptr (a0)
//   r9  = lda_bytes
//   r10 = ldc_bytes
//   r11 = flags: bit 0 = acc_c (1 = load existing C), bits 1..4 = epi_kind (0..4)
//   r12 = 3 * lda_bytes (computed by microkernel prologue in leaf-safe scratch or caller)
//   r13 = E_ptr (for epi_kind != 0)
//   r14 = lde_bytes (for epi_kind != 0)
//   zmm31 = broadcast scalar s (for epi_kind == 3)
//   k1 = tail bitmask (for masked == true)
// Clobbers: rax, rdi, rsi, rcx, r8, r13, zmm0..zmm30. Preserves: rdx, r9, r10, r11, r12, r14, r15, rbx, rbp.

fn a_row_mem(row: usize, disp: i32) -> MemOp {
    match row {
        0 => MemOp::bd(R8, disp),
        1 => MemOp::sib(R8, R9, 0, disp),
        2 => MemOp::sib(R8, R9, 1, disp),
        3 => MemOp::sib(R8, R12, 0, disp),
        4 => MemOp::bd(RAX, disp),
        5 => MemOp::sib(RAX, R9, 0, disp),
        6 => MemOp::sib(RAX, R9, 1, disp),
        _ => MemOp::sib(RAX, R12, 0, disp),
    }
}

fn emit_expf_fast512(e: &mut Avx512Emit, x: u8, t0: u8, t1: u8, t2: u8) {
    // Result in x; clobbers t0, t1, t2. Matches uopc_expf_fast512 bit-for-bit.
    e.vbroadcastss_rip(t0, std::f32::consts::LOG2_E.to_bits()).vmulps(x, x, t0)
        .vbroadcastss_rip(t0, 127.0_f32.to_bits()).vminps(x, x, t0)
        .vbroadcastss_rip(t0, (-126.0_f32).to_bits()).vmaxps(x, x, t0)
        .vrndscaleps(t1, x, 0x08) // n = round_nearest(t)
        .vsubps(x, x, t1)         // f = t - n
        .vbroadcastss_rip(t2, 0.009_618_129_f32.to_bits())
        .vbroadcastss_rip(t0, 0.001_333_355_8_f32.to_bits())
        .vfmadd231ps_rr(t2, x, t0)
        .vbroadcastss_rip(t0, 0.055_504_11_f32.to_bits()).vfmadd231ps_rr(t0, x, t2)
        .vbroadcastss_rip(t2, 0.240_226_5_f32.to_bits()).vfmadd231ps_rr(t2, x, t0)
        .vbroadcastss_rip(t0, std::f32::consts::LN_2.to_bits()).vfmadd231ps_rr(t0, x, t2)
        .vbroadcastss_rip(t2, 1.0_f32.to_bits()).vfmadd231ps_rr(t2, x, t0)
        .vcvttps2dq(t1, t1).vpslld_imm(t1, t1, 23).vpaddd(x, t2, t1);
}

fn emit_gelu_pair512(e: &mut Avx512Emit, z: u8, outs: [u8; 2], tmp: [u8; 4]) {
    let ([out_g, out_d], [t0, t1, t2, t3]) = (outs, tmp);
    // Computes out_g = gelu(z), out_d = gelu'(z) in AVX-512 (exact A&S 7.1.26).
    e.vbroadcastss_rip(t0, std::f32::consts::FRAC_1_SQRT_2.to_bits()).vmulps(t3, z, t0) // t3 = x
        .vbroadcastss_rip(t0, 0x8000_0000).vpandnd(t1, t0, t3)                          // t1 = ax = |x|
        .vbroadcastss_rip(t2, 1.0_f32.to_bits())
        .vbroadcastss_rip(t0, 0.327_591_1_f32.to_bits())
        .vfmadd231ps_rr(t2, t0, t1)                                                     // t2 = 1 + p*ax
        .vbroadcastss_rip(t0, 1.0_f32.to_bits()).vdivps(t2, t0, t2)                     // t2 = t
        .vbroadcastss_rip(out_d, (-1.453_152_1_f32).to_bits())
        .vbroadcastss_rip(t0, 1.061_405_4_f32.to_bits()).vfmadd231ps_rr(out_d, t2, t0)
        .vbroadcastss_rip(t0, 1.421_413_8_f32.to_bits()).vfmadd231ps_rr(t0, t2, out_d)
        .vbroadcastss_rip(out_d, (-0.284_496_72_f32).to_bits()).vfmadd231ps_rr(out_d, t2, t0)
        .vbroadcastss_rip(t0, 0.254_829_6_f32.to_bits()).vfmadd231ps_rr(t0, t2, out_d)
        .vmulps(out_d, t0, t2)
        .vmulps(t1, t1, t1).vbroadcastss_rip(t0, 0x8000_0000).vpxord(t1, t1, t0);
    emit_expf_fast512(e, t1, t0, t2, out_g);
    e.vbroadcastss_rip(out_g, 1.0_f32.to_bits()).vfnmadd231ps_rr(out_g, out_d, t1)
        .vbroadcastss_rip(t0, 0x8000_0000).vpandd(t2, t3, t0).vpxord(out_g, out_g, t2)
        .vbroadcastss_rip(t0, 1.0_f32.to_bits()).vaddps(out_g, out_g, t0)
        .vbroadcastss_rip(t0, 0.5_f32.to_bits()).vmulps(t2, out_g, t0)
        .vmulps(out_g, z, t2)
        .vmulps(t0, z, t1)
        .vbroadcastss_rip(t1, 0.398_942_3_f32.to_bits())
        .vmovaps(out_d, t2).vfmadd231ps_rr(out_d, t0, t1);
}

fn emit_k_step(e: &mut Avx512Emit, mr: usize, ncols: usize, masked: bool, k_disp: i32) {
    let b_base = 24u8;
    let a_ping = 27u8;
    let a_pong = 28u8;
    if ncols == 1 {
        e.vmovups_ld(b_base, MemOp::bd(RSI, 0), masked);
        for r in 0..mr {
            e.vfmadd231ps_bcst(r as u8, b_base, a_row_mem(r, k_disp));
        }
    } else {
        // Load all B columns + software-pipeline A broadcasts 1 row ahead to hide L1d load latency!
        for col in 0..ncols {
            e.vmovups_ld(b_base + col as u8, MemOp::bd(RSI, (col as i32) * 64), false);
        }
        e.vbroadcastss_mem(a_ping, a_row_mem(0, k_disp));
        for r in 0..mr {
            let cur = if (r & 1) == 0 { a_ping } else { a_pong };
            let nxt = if (r & 1) == 0 { a_pong } else { a_ping };
            if r + 1 < mr { e.vbroadcastss_mem(nxt, a_row_mem(r + 1, k_disp)); }
            for col in 0..ncols {
                e.vfmadd231ps_rr((r * ncols + col) as u8, b_base + col as u8, cur);
            }
        }
    }
    e.add_rr(RSI, RDX);
}

fn emit_gemm_microkernel_col1(e: &mut Avx512Emit, mr: usize) -> usize {
    // Ultra-fast 1-column plain microkernel (rem == 1, kind == 0, e.g. N=65 vocab head).
    // Uses 4-byte vmovss + vfmadd231ss: zero masked 512-bit cache-line split stalls!
    e.align(32);
    let entry = e.code.len();
    if mr > 4 { e.lea(RAX, MemOp::sib(R8, R9, 2, 0)); }
    e.emit(&[0x41, 0xF6, 0xC3, 0x01]);
    let j_load = e.jcc_ph(0x85);
    for r in 0..mr { e.vxorps(r as u8, r as u8, r as u8); }
    let j_kstart = e.jmp_ph();
    e.patch_jcc(j_load);
    e.push_r(RCX);
    for r in 0..mr {
        e.vmovss_ld(r as u8, MemOp::bd(RCX, 0));
        if r + 1 < mr { e.add_rr(RCX, R10); }
    }
    e.pop_r(RCX);
    e.patch_jmp(j_kstart);
    let j_tail = e.cmp_ri(RDI, 4).jcc_ph(0x8C);
    e.align(16);
    let k_top = e.code.len();
    for u in 0..4 {
        let breg = if (u & 1) == 0 { 16 } else { 17 };
        e.vmovss_ld(breg, MemOp::bd(RSI, 0));
        for r in 0..mr { e.vfmadd231ss_mem(r as u8, breg, a_row_mem(r, u * 4)); }
        e.add_rr(RSI, RDX);
    }
    e.add_ri(R8, 16);
    if mr > 4 { e.add_ri(RAX, 16); }
    e.sub_ri(RDI, 4).cmp_ri(RDI, 4).jcc_to(0x8D, k_top);
    e.patch_jcc(j_tail);
    let j_done = e.test_rr(RDI, RDI).jcc_ph(0x8E);
    let k_rem = e.code.len();
    e.vmovss_ld(16, MemOp::bd(RSI, 0));
    for r in 0..mr { e.vfmadd231ss_mem(r as u8, 16, a_row_mem(r, 0)); }
    e.add_rr(RSI, RDX).add_ri(R8, 4);
    if mr > 4 { e.add_ri(RAX, 4); }
    e.sub_ri(RDI, 1).jcc_to(0x85, k_rem);
    e.patch_jcc(j_done);
    for r in 0..mr {
        e.vmovss_st(MemOp::bd(RCX, 0), r as u8);
        if r + 1 < mr { e.add_rr(RCX, R10); }
    }
    e.ret();
    entry
}

fn emit_gemm_microkernel(e: &mut Avx512Emit, mr: usize, ncols: usize, masked: bool) -> usize {
    e.align(32);
    let entry = e.code.len();
    if mr > 4 { e.lea(RAX, MemOp::sib(R8, R9, 2, 0)); }
    // Bit 0 of r11: acc_c (0 = zero accumulators, 1 = load C)
    e.emit(&[0x41, 0xF6, 0xC3, 0x01]); // test r11b, 1
    let j_load = e.jcc_ph(0x85);
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            e.vxorps(z, z, z);
        }
    }
    let j_kstart = e.jmp_ph();
    e.patch_jcc(j_load);
    e.push_r(RCX);
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            e.vmovups_ld(z, MemOp::bd(RCX, (col as i32) * 64), masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10); }
    }
    e.pop_r(RCX);
    e.patch_jmp(j_kstart);

    // K-unrolled hot loop: 2x for 8x48 (fits Zen 4 L0 Op-Cache!), 4x for narrower tiles (0 spills!)
    let unroll: i32 = if mr == 8 && ncols == 3 { 2 } else { 4 };
    let j_tail = e.cmp_ri(RDI, unroll).jcc_ph(0x8C); // jl tail
    e.align(32);
    let k_top = e.code.len();
    for u in 0..unroll { emit_k_step(e, mr, ncols, masked, u * 4); }
    e.add_ri(R8, unroll * 4);
    if mr > 4 { e.add_ri(RAX, unroll * 4); }
    e.sub_ri(RDI, unroll).cmp_ri(RDI, unroll).jcc_to(0x8D, k_top); // jge k_top
    e.patch_jcc(j_tail);

    // 1x K-remainder loop (0..3 iterations)
    let j_done = e.test_rr(RDI, RDI).jcc_ph(0x8E); // jle done
    let k_rem = e.code.len();
    emit_k_step(e, mr, ncols, masked, 0);
    e.add_ri(R8, 4);
    if mr > 4 { e.add_ri(RAX, 4); }
    e.sub_ri(RDI, 1).jcc_to(0x85, k_rem); // jnz k_rem
    e.patch_jcc(j_done);

    // Store C (check epi_kind = r11 >> 1)
    e.mov_rr(RAX, R11).emit(&[0x48, 0xD1, 0xE8]); // shr rax, 1
    let j_epi = e.test_rr(RAX, RAX).jcc_ph(0x85);
    for r in 0..mr {
        for col in 0..ncols {
            e.vmovups_st(MemOp::bd(RCX, (col as i32) * 64), (r * ncols + col) as u8, masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10); }
    }
    e.ret();

    // Epilogue store paths (kinds 1..4) -- uses zmm24..zmm30 as scratch, zmm31 = s
    e.patch_jcc(j_epi);
    let j_not1 = e.cmp_ri(RAX, 1).jcc_ph(0x85);
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            let off = (col as i32) * 64;
            e.vmovups_ld(24, MemOp::bd(R13, off), masked).vaddps(z, z, 24).vmovups_st(MemOp::bd(RCX, off), z, masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10).add_rr(R13, R14); }
    }
    e.ret();

    e.patch_jcc(j_not1);
    let j_not3 = e.cmp_ri(RAX, 3).jcc_ph(0x85);
    // UOPC_EPI_FMA: t = s * E; C = C + t (separate mul + add, bit-exact with unfused EW!)
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            let off = (col as i32) * 64;
            e.vmovups_ld(24, MemOp::bd(R13, off), masked).vmulps(24, 31, 24).vaddps(z, z, 24)
                .vmovups_st(MemOp::bd(RCX, off), z, masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10).add_rr(R13, R14); }
    }
    e.ret();

    e.patch_jcc(j_not3);
    let j_not2 = e.cmp_ri(RAX, 2).jcc_ph(0x85);
    // UOPC_EPI_GELU: C = C * gelu'(E)
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            let off = (col as i32) * 64;
            e.vmovups_ld(24, MemOp::bd(R13, off), masked);
            emit_gelu_pair512(e, 24, [25, 26], [27, 28, 29, 30]);
            e.vmulps(z, z, 26).vmovups_st(MemOp::bd(RCX, off), z, masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10).add_rr(R13, R14); }
    }
    e.ret();

    // UOPC_EPI_GELU_FWD (kind == 4): C = gelu(C), E = gelu'(C)
    e.patch_jcc(j_not2);
    for r in 0..mr {
        for col in 0..ncols {
            let z = (r * ncols + col) as u8;
            let off = (col as i32) * 64;
            emit_gelu_pair512(e, z, [25, 26], [27, 28, 29, 30]);
            e.vmovups_st(MemOp::bd(R13, off), 26, masked).vmovups_st(MemOp::bd(RCX, off), 25, masked);
        }
        if r + 1 < mr { e.add_rr(RCX, R10).add_rr(R13, R14); }
    }
    e.ret();
    entry
}

#[derive(Clone, Copy)]
struct MicTable {
    m8: [usize; 5], // [48, 32, 16, mask, col1]
    m4: [usize; 5],
    m1: [usize; 5],
}

// Local frame layout relative to RBP (below 5 pushed callee-saved regs at [rbp-8..-40]):
const L_M: i32 = -48;
const L_N: i32 = -56;
const L_K: i32 = -64;
const L_A: i32 = -72;
const L_LDA_B: i32 = -80;
const L_B: i32 = -88;
const L_LDB_B: i32 = -96;
const L_C: i32 = -104;
const L_LDC_B: i32 = -112;
const L_KIND: i32 = -120;
const L_E: i32 = -128;
const L_LDE_B: i32 = -136;
const L_K0: i32 = -144;
const L_KK: i32 = -152;
const L_FLAGS: i32 = -160;
const L_IEND: i32 = -168;
const L_J: i32 = -176;
const L_NR: i32 = -184;
const L_A_ROW: i32 = -192;
const L_C_ROW: i32 = -200;
const L_E_ROW: i32 = -208;
const L_B_ORIG: i32 = -216;
const L_LDB_ORIG: i32 = -224;
const L_E_ORIG: i32 = -232;
const L_LDE_ORIG: i32 = -240;
const L_PACK_PTR: i32 = -248;
const CORE_FRAME: i32 = 216; // 40 (5 pushes) + 216 = 256 -> rsp 16B aligned!

fn emit_call_mic_m_strip(e: &mut Avx512Emit, mics: &MicTable, nr_idx: usize, packed_b: bool) {
    // Sweeps r15 = 0..L_IEND in steps of 8 -> 4 -> 1.
    // When packed_b == true, B_ptr = [rbp + L_PACK_PTR] (contiguous stride = nr*4), else B + k0*ldb + j*4.
    e.mov_rm64(RBX, MemOp::bd(RBP, L_IEND))
        .mov_rm64(R9, MemOp::bd(RBP, L_LDA_B))
        .lea(R12, MemOp::sib(R9, R9, 1, 0))
        .mov_rm64(R10, MemOp::bd(RBP, L_LDC_B))
        .mov_rm64(R14, MemOp::bd(RBP, L_LDE_B))
        .mov_rm64(RDX, MemOp::bd(RBP, L_LDB_B))
        .mov_rm64(R11, MemOp::bd(RBP, L_FLAGS));
    for (step, target) in [(8i32, mics.m8[nr_idx]), (4i32, mics.m4[nr_idx]), (1i32, mics.m1[nr_idx])] {
        let j_skip = e.mov_rr(RAX, RBX).sub_rr(RAX, R15).cmp_ri(RAX, step).jcc_ph(0x8C);
        let top = e.code.len();
        e.mov_rr(R8, R15).imul_rr(R8, R9)
            .mov_rm64(RAX, MemOp::bd(RBP, L_K0)).shl_ri(RAX, 2).add_rr(R8, RAX)
            .mov_rm64(RAX, MemOp::bd(RBP, L_A)).add_rr(R8, RAX);
        if packed_b {
            e.mov_rm64(RSI, MemOp::bd(RBP, L_PACK_PTR));
        } else {
            e.mov_rm64(RSI, MemOp::bd(RBP, L_K0)).imul_rr(RSI, RDX)
                .mov_rm64(RAX, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
                .mov_rm64(RAX, MemOp::bd(RBP, L_J)).shl_ri(RAX, 2).add_rr(RSI, RAX);
        }
        e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).shl_ri(RAX, 2)
            .mov_rr(RCX, R15).imul_rr(RCX, R10).add_rr(RCX, RAX)
            .mov_rm64(RDI, MemOp::bd(RBP, L_C)).add_rr(RCX, RDI)
            .mov_rr(R13, R15).imul_rr(R13, R14).add_rr(R13, RAX)
            .mov_rm64(RDI, MemOp::bd(RBP, L_E)).add_rr(R13, RDI)
            .mov_rm64(RDI, MemOp::bd(RBP, L_KK))
            .call_rel(target)
            .add_ri(R15, step);
        if step == 8 || step == 1 {
            e.mov_rr(RAX, RBX).sub_rr(RAX, R15).cmp_ri(RAX, step).jcc_to(0x8D, top);
        }
        e.patch_jcc(j_skip);
    }
}

fn emit_pack_b_panel(e: &mut Avx512Emit, ncols: usize) {
    // Packs kk rows x (ncols * 16) floats from B_orig[k0..k0+kk, j..j+nr] into [rbp + L_PACK_PTR]
    // with contiguous row stride = ncols * 64 bytes! Eliminates 4KB L1d set-aliasing & DTLB thrash!
    e.mov_rm64(RDX, MemOp::bd(RBP, L_LDB_ORIG))
        .mov_rm64(RSI, MemOp::bd(RBP, L_K0)).imul_rr(RSI, RDX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_B_ORIG)).add_rr(RSI, RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_J)).shl_ri(RAX, 2).add_rr(RSI, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_PACK_PTR))
        .mov_rm64(RCX, MemOp::bd(RBP, L_KK));
    let top = e.code.len();
    let j_done = e.cmp_ri(RCX, 0).jcc_ph(0x8E);
    for col in 0..ncols {
        let off = (col as i32) * 64;
        e.vmovups_ld(col as u8, MemOp::bd(RSI, off), false)
            .vmovups_st(MemOp::bd(RDI, off), col as u8, false);
    }
    e.add_rr(RSI, RDX).add_ri(RDI, (ncols as i32) * 64).sub_ri(RCX, 1).jmp_to(top);
    e.patch_jcc(j_done);
    e.mov_ri32(RAX, (ncols as i32) * 64).mov_mr64(MemOp::bd(RBP, L_LDB_B), RAX);
}

fn emit_hsum512(e: &mut Avx512Emit, acc: u8, tmp: u8) {
    // Reduces 16 f32 lanes in zmm(acc) into lane 0 of xmm(acc); clobbers zmm(tmp).
    e.vextractf32x8(tmp, acc, 1).vaddps_256(acc, acc, tmp)
        .vextractf32x4(tmp, acc, 1).vaddps_128(acc, acc, tmp)
        .vshufps_128(tmp, acc, acc, 0x4E).vaddps_128(acc, acc, tmp)
        .vshufps_128(tmp, acc, acc, 0x11).vaddss(acc, acc, tmp);
}

fn emit_col1_kvec_pass(e: &mut Avx512Emit) {
    // K-vectorized AVX-512 pass for final single col j = N - 1 (N % 16 == 1, kind == 0, K <= 512).
    // Packs B[0..K, N-1] into 2048B stack buffer ONCE, then dots 4 rows of A along K (16-wide ZMM).
    e.sub_ri(RSP, 2112); // >= 512 * 4 = 2048B, multiple of 16
    // Pack B[0..K, N-1] -> [rsp + 0..K*4]
    e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_ri(RAX, 1).shl_ri(RAX, 2)
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RDX, MemOp::bd(RBP, L_LDB_B))
        .mov_rr(RDI, RSP)
        .mov_rm64(RCX, MemOp::bd(RBP, L_K));
    let pk_top = e.code.len();
    let j_pk_done = e.cmp_ri(RCX, 0).jcc_ph(0x8E);
    e.vmovss_ld(0, MemOp::bd(RSI, 0)).vmovss_st(MemOp::bd(RDI, 0), 0)
        .add_rr(RSI, RDX).add_ri(RDI, 4).sub_ri(RCX, 1).jmp_to(pk_top);
    e.patch_jcc(j_pk_done);

    // Prepare C_col_ptr = C + (N-1)*4 -> r13, A_ptr -> r8, lda_b -> r9, ldc_b -> r10, r12 = 3*lda_b
    e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_ri(RAX, 1).shl_ri(RAX, 2)
        .mov_rm64(R13, MemOp::bd(RBP, L_C)).add_rr(R13, RAX)
        .mov_rm64(R8, MemOp::bd(RBP, L_A))
        .mov_rm64(R9, MemOp::bd(RBP, L_LDA_B))
        .lea(R12, MemOp::sib(R9, R9, 1, 0))
        .mov_rm64(R10, MemOp::bd(RBP, L_LDC_B))
        .mov_rm64(R15, MemOp::bd(RBP, L_M)) // rem_m
        .mov_rm64(RBX, MemOp::bd(RBP, L_K)); // K

    // Compute k1 mask for K % 16 (when K % 16 != 0)
    e.mov_rr(RCX, RBX).and_ri(RCX, 15);
    let j_no_ktail = e.cmp_ri(RCX, 0).jcc_ph(0x84);
    e.mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax();
    e.patch_jcc(j_no_ktail);

    // Process 4 rows of A at a time (zmm0..zmm3 accumulators)
    let j_m4_skip = e.cmp_ri(R15, 4).jcc_ph(0x8C);
    let m4_top = e.code.len();
    for r in 0..4u8 { e.vxorps(r, r, r); }
    e.mov_rr(RSI, RSP).mov_rr(RDI, RBX).mov_ri32(RDX, 0); // rdx = k_byte_off
    let j_k16_skip = e.cmp_ri(RDI, 16).jcc_ph(0x8C);
    let k16_top = e.code.len();
    e.vmovups_ld(8, MemOp::bd(RSI, 0), false)
        .vmovups_ld(9, MemOp::sib(R8, RDX, 0, 0), false).vfmadd231ps_rr(0, 8, 9)
        .lea(RAX, MemOp::sib(R8, R9, 0, 0))
        .vmovups_ld(10, MemOp::sib(RAX, RDX, 0, 0), false).vfmadd231ps_rr(1, 8, 10)
        .lea(RAX, MemOp::sib(R8, R9, 1, 0))
        .vmovups_ld(11, MemOp::sib(RAX, RDX, 0, 0), false).vfmadd231ps_rr(2, 8, 11)
        .lea(RAX, MemOp::sib(R8, R12, 0, 0))
        .vmovups_ld(12, MemOp::sib(RAX, RDX, 0, 0), false).vfmadd231ps_rr(3, 8, 12)
        .add_ri(RSI, 64).add_ri(RDX, 64).sub_ri(RDI, 16).cmp_ri(RDI, 16).jcc_to(0x8D, k16_top);
    e.patch_jcc(j_k16_skip);
    let j_krem_skip = e.cmp_ri(RDI, 0).jcc_ph(0x8E);
    e.vmovups_ld(8, MemOp::bd(RSI, 0), true)
        .vmovups_ld(9, MemOp::sib(R8, RDX, 0, 0), true).vfmadd231ps_rr(0, 8, 9)
        .lea(RAX, MemOp::sib(R8, R9, 0, 0))
        .vmovups_ld(10, MemOp::sib(RAX, RDX, 0, 0), true).vfmadd231ps_rr(1, 8, 10)
        .lea(RAX, MemOp::sib(R8, R9, 1, 0))
        .vmovups_ld(11, MemOp::sib(RAX, RDX, 0, 0), true).vfmadd231ps_rr(2, 8, 11)
        .lea(RAX, MemOp::sib(R8, R12, 0, 0))
        .vmovups_ld(12, MemOp::sib(RAX, RDX, 0, 0), true).vfmadd231ps_rr(3, 8, 12);
    e.patch_jcc(j_krem_skip);
    for r in 0..4u8 {
        emit_hsum512(e, r, 16);
        e.vmovss_st(MemOp::bd(R13, 0), r).add_rr(R13, R10);
    }
    e.lea(R8, MemOp::sib(R8, R9, 2, 0)).sub_ri(R15, 4).cmp_ri(R15, 4).jcc_to(0x8D, m4_top);
    e.patch_jcc(j_m4_skip);

    // Remaining 0..3 rows
    let j_m1_skip = e.cmp_ri(R15, 0).jcc_ph(0x8E);
    let m1_top = e.code.len();
    e.vxorps(0, 0, 0).mov_rr(RSI, RSP).mov_rr(RDI, RBX).mov_ri32(RDX, 0);
    let j_k16_1 = e.cmp_ri(RDI, 16).jcc_ph(0x8C);
    let k16_1 = e.code.len();
    e.vmovups_ld(8, MemOp::bd(RSI, 0), false)
        .vmovups_ld(9, MemOp::sib(R8, RDX, 0, 0), false).vfmadd231ps_rr(0, 8, 9)
        .add_ri(RSI, 64).add_ri(RDX, 64).sub_ri(RDI, 16).cmp_ri(RDI, 16).jcc_to(0x8D, k16_1);
    e.patch_jcc(j_k16_1);
    let j_krem_1 = e.cmp_ri(RDI, 0).jcc_ph(0x8E);
    e.vmovups_ld(8, MemOp::bd(RSI, 0), true)
        .vmovups_ld(9, MemOp::sib(R8, RDX, 0, 0), true).vfmadd231ps_rr(0, 8, 9);
    e.patch_jcc(j_krem_1);
    emit_hsum512(e, 0, 16);
    e.vmovss_st(MemOp::bd(R13, 0), 0).add_rr(R13, R10).add_rr(R8, R9)
        .sub_ri(R15, 1).jcc_to(0x85, m1_top);
    e.patch_jcc(j_m1_skip);
    e.add_ri(RSP, 2112);
}

fn emit_call_mic_row_band_16(e: &mut Avx512Emit, mics: &MicTable, m_step: i32) {
    // Cold-C streaming row-outer path (M >= 1024 && K <= 128): uses 8x16 ({1to16} embedded broadcast,
    // 8 cache-line store bursts < Zen 4's 22-entry MAB -> ZERO RFO stall!) + inline K-vec col1 tail!
    let mt = match m_step { 8 => &mics.m8, 4 => &mics.m4, _ => &mics.m1 };
    e.mov_rm64(R9, MemOp::bd(RBP, L_LDA_B))
        .lea(R12, MemOp::sib(R9, R9, 1, 0))
        .mov_rr(RAX, R15).imul_rr(RAX, R9)
        .mov_rm64(RCX, MemOp::bd(RBP, L_A)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_A_ROW), RAX)
        .mov_rm64(R10, MemOp::bd(RBP, L_LDC_B))
        .mov_rr(RAX, R15).imul_rr(RAX, R10)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_C_ROW), RAX)
        .mov_rm64(R14, MemOp::bd(RBP, L_LDE_B))
        .mov_rr(RAX, R15).imul_rr(RAX, R14)
        .mov_rm64(RCX, MemOp::bd(RBP, L_E)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_E_ROW), RAX)
        .mov_rm64(RDX, MemOp::bd(RBP, L_LDB_B))
        .mov_rm64(R11, MemOp::bd(RBP, L_FLAGS))
        .mov_ri32(RBX, 0);
    // 16-col loop with {1to16} embedded broadcast
    let j_skip16 = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 16).jcc_ph(0x8C);
    let top16 = e.code.len();
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[2])
        .add_ri(RBX, 16)
        .mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 16).jcc_to(0x8D, top16);
    e.patch_jcc(j_skip16);
    // Tail: rem = N - j in 1..=15
    let j_nomask = e.mov_rm64(RCX, MemOp::bd(RBP, L_N)).sub_rr(RCX, RBX).cmp_ri(RCX, 0).jcc_ph(0x8E);
    let j_not_c1 = e.cmp_ri(RCX, 1).jcc_ph(0x85);
    let j_not_plain = e.test_rr(R11, R11).jcc_ph(0x85);
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[4]);
    let j_c1_done = e.jmp_ph();
    e.patch_jcc(j_not_c1);
    e.patch_jcc(j_not_plain);
    e.mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax()
        .mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[3]);
    e.patch_jmp(j_c1_done);
    e.patch_jcc(j_nomask);
}

fn emit_call_mic_row_band(e: &mut Avx512Emit, mics: &MicTable, m_step: i32, skip_col1: bool) {
    // Row-outer fast path with HOISTED row pointers + 64->32+32 split (never falls to 8x16 on pow2!)
    let mt = match m_step { 8 => &mics.m8, 4 => &mics.m4, _ => &mics.m1 };
    e.mov_rm64(R9, MemOp::bd(RBP, L_LDA_B))
        .lea(R12, MemOp::sib(R9, R9, 1, 0))
        .mov_rr(RAX, R15).imul_rr(RAX, R9)
        .mov_rm64(RCX, MemOp::bd(RBP, L_A)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_A_ROW), RAX)
        .mov_rm64(R10, MemOp::bd(RBP, L_LDC_B))
        .mov_rr(RAX, R15).imul_rr(RAX, R10)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_C_ROW), RAX)
        .mov_rm64(R14, MemOp::bd(RBP, L_LDE_B))
        .mov_rr(RAX, R15).imul_rr(RAX, R14)
        .mov_rm64(RCX, MemOp::bd(RBP, L_E)).add_rr(RAX, RCX).mov_mr64(MemOp::bd(RBP, L_E_ROW), RAX)
        .mov_rm64(RDX, MemOp::bd(RBP, L_LDB_B))
        .mov_rm64(R11, MemOp::bd(RBP, L_FLAGS))
        .mov_ri32(RBX, 0);
    // 48-col primary loop (24 ZMM accumulators, 2x K-unroll fits Zen 4 Op-Cache; skip 48 if N==64 or N==65 -> 32+32!)
    let j_skip48 = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 48).jcc_ph(0x8C);
    let j_n64 = e.cmp_ri(RAX, 64).jcc_ph(0x84);
    let j_n65 = e.cmp_ri(RAX, 65).jcc_ph(0x84);
    let top48 = e.code.len();
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[0])
        .add_ri(RBX, 48)
        .mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 48).jcc_to(0x8D, top48);
    e.patch_jcc(j_skip48);
    e.patch_jcc(j_n64);
    e.patch_jcc(j_n65);

    // 32-col loop (handles 32, 64->32+32, 65->32+32+1)
    let j_skip32 = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 32).jcc_ph(0x8C);
    let top32 = e.code.len();
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[1])
        .add_ri(RBX, 32)
        .mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 32).jcc_to(0x8D, top32);
    e.patch_jcc(j_skip32);

    // 16-col step
    let j_skip16 = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX).cmp_ri(RAX, 16).jcc_ph(0x8C);
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
        .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
        .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
        .mov_rm64(RDI, MemOp::bd(RBP, L_K))
        .call_rel(mt[2])
        .add_ri(RBX, 16);
    e.patch_jcc(j_skip16);

    if !skip_col1 {
        let j_nomask = e.mov_rm64(RCX, MemOp::bd(RBP, L_N)).sub_rr(RCX, RBX).cmp_ri(RCX, 0).jcc_ph(0x8E);
        let j_not_c1 = e.cmp_ri(RCX, 1).jcc_ph(0x85);
        let j_not_plain = e.test_rr(R11, R11).jcc_ph(0x85);
        e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
            .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
            .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
            .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
            .mov_rm64(RDI, MemOp::bd(RBP, L_K))
            .call_rel(mt[4]);
        let j_c1_done = e.jmp_ph();
        e.patch_jcc(j_not_c1);
        e.patch_jcc(j_not_plain);
        e.mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax()
            .mov_rr(RAX, RBX).shl_ri(RAX, 2)
            .mov_rm64(R8, MemOp::bd(RBP, L_A_ROW))
            .mov_rm64(RSI, MemOp::bd(RBP, L_B)).add_rr(RSI, RAX)
            .mov_rm64(RCX, MemOp::bd(RBP, L_C_ROW)).add_rr(RCX, RAX)
            .mov_rm64(R13, MemOp::bd(RBP, L_E_ROW)).add_rr(R13, RAX)
            .mov_rm64(RDI, MemOp::bd(RBP, L_K))
            .call_rel(mt[3]);
        e.patch_jmp(j_c1_done);
        e.patch_jcc(j_nomask);
    }
}

fn emit_uopc_gemm_core(e: &mut Avx512Emit, mics: &MicTable) -> usize {
    // SysV entry for sgemm_dispatch_epi(M, N, K, A, lda, B, ldb, C, ldc, kind, E, lde, s):
    //   rdi=M, rsi=N, rdx=K, rcx=A, r8=lda, r9=B, xmm0=s
    //   [rbp+16]=ldb, [rbp+24]=C, [rbp+32]=ldc, [rbp+40]=kind, [rbp+48]=E, [rbp+56]=lde
    e.align(32);
    let entry = e.code.len();
    e.push_r(RBP).mov_rr(RBP, RSP)
        .push_r(RBX).push_r(R12).push_r(R13).push_r(R14).push_r(R15)
        .sub_ri(RSP, CORE_FRAME);
    e.movsxd(RDI, RDI).movsxd(RSI, RSI).movsxd(RDX, RDX).movsxd(R8, R8);
    let j_m0 = e.cmp_ri(RDI, 0).jcc_ph(0x8E);
    let j_n0 = e.cmp_ri(RSI, 0).jcc_ph(0x8E);
    // Broadcast s -> zmm31 (preserved across all microkernels!)
    e.vbroadcastss_xmm(31, 0);
    // Save all args to frame locals
    e.mov_mr64(MemOp::bd(RBP, L_M), RDI)
        .mov_mr64(MemOp::bd(RBP, L_N), RSI)
        .mov_mr64(MemOp::bd(RBP, L_K), RDX)
        .mov_mr64(MemOp::bd(RBP, L_A), RCX)
        .mov_rr(RAX, R8).shl_ri(RAX, 2).mov_mr64(MemOp::bd(RBP, L_LDA_B), RAX)
        .mov_mr64(MemOp::bd(RBP, L_B), R9).mov_mr64(MemOp::bd(RBP, L_B_ORIG), R9)
        .movsxd_rm(RAX, MemOp::bd(RBP, 16)).shl_ri(RAX, 2)
        .mov_mr64(MemOp::bd(RBP, L_LDB_B), RAX).mov_mr64(MemOp::bd(RBP, L_LDB_ORIG), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, 24)).mov_mr64(MemOp::bd(RBP, L_C), RAX)
        .movsxd_rm(RAX, MemOp::bd(RBP, 32)).shl_ri(RAX, 2).mov_mr64(MemOp::bd(RBP, L_LDC_B), RAX)
        .movsxd_rm(RAX, MemOp::bd(RBP, 40)).mov_mr64(MemOp::bd(RBP, L_KIND), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, 48))
        .mov_mr64(MemOp::bd(RBP, L_E), RAX).mov_mr64(MemOp::bd(RBP, L_E_ORIG), RAX)
        .movsxd_rm(RAX, MemOp::bd(RBP, 56)).shl_ri(RAX, 2)
        .mov_mr64(MemOp::bd(RBP, L_LDE_B), RAX).mov_mr64(MemOp::bd(RBP, L_LDE_ORIG), RAX)
        .mov_ri32(RAX, 0).mov_mr64(MemOp::bd(RBP, L_PACK_PTR), RAX);

    // Check aliasing: if C == B, run safe column-panel stack-staged path!
    e.mov_rm64(RAX, MemOp::bd(RBP, L_C)).cmp_rr(RAX, R9);
    let j_no_alias = e.jcc_ph(0x85);
    // Aliasing path (C == B): stage each K x min(16, N-j) panel of B onto stack (16-float stride = 64B/row).
    // Works for any K (dynamic stack alloc with 4KB page-touch probe), 0 malloc, 100% OpenMP thread-safe!
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K)).shl_ri(RAX, 6).add_ri(RAX, 64).and_ri(RAX, !63);
    // Touch stack pages every 4096 bytes before lowering RSP
    e.mov_rr(R10, RSP).sub_rr(R10, RAX);
    let probe_top = e.code.len();
    e.sub_ri(RSP, 4096).cmp_rr(RSP, R10);
    let j_probe_done = e.jcc_ph(0x86); // jbe
    e.mov_mr64(MemOp::bd(RSP, 0), RAX).jmp_to(probe_top);
    e.patch_jcc(j_probe_done);
    e.mov_rr(RSP, R10);
    // Now RSP points to staged B panel of K rows x 16 floats (stride 64 bytes).
    e.mov_ri32(RAX, 0).mov_mr64(MemOp::bd(RBP, L_J), RAX)
        .mov_mr64(MemOp::bd(RBP, L_K0), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_K)).mov_mr64(MemOp::bd(RBP, L_KK), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_KIND)).shl_ri(RAX, 1).mov_mr64(MemOp::bd(RBP, L_FLAGS), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_M)).mov_mr64(MemOp::bd(RBP, L_IEND), RAX);
    let alias_j_top = e.code.len();
    e.mov_rm64(RBX, MemOp::bd(RBP, L_J))
        .mov_rm64(RAX, MemOp::bd(RBP, L_N)).sub_rr(RAX, RBX);
    let j_alias_done = e.cmp_ri(RAX, 0).jcc_ph(0x8E);
    // nr = min(16, N - j) -> r12; set k1 if nr < 16
    let j_alias_lt16 = e.cmp_ri(RAX, 16).jcc_ph(0x8C);
    e.mov_ri32(RAX, 16);
    e.patch_jcc(j_alias_lt16);
    e.mov_mr64(MemOp::bd(RBP, L_NR), RAX);
    // Copy K rows x nr floats from B_orig[:, j..j+nr] into RSP[k*16..]
    e.mov_rr(RCX, RAX).mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax()
        .mov_rm64(RDX, MemOp::bd(RBP, L_LDB_ORIG))
        .mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rm64(RSI, MemOp::bd(RBP, L_B_ORIG)).add_rr(RSI, RAX)
        .mov_rr(RDI, RSP)
        .mov_rm64(R8, MemOp::bd(RBP, L_K));
    let copy_top = e.code.len();
    e.cmp_ri(R8, 0);
    let j_copy_done = e.jcc_ph(0x8E);
    e.vmovups_ld(0, MemOp::bd(RSI, 0), true)
        .vmovups_st(MemOp::bd(RDI, 0), 0, false)
        .add_rr(RSI, RDX).add_ri(RDI, 64).sub_ri(R8, 1).jmp_to(copy_top);
    e.patch_jcc(j_copy_done);
    // Point L_B = RSP - j*4 (so B + j*4 == RSP!), L_LDB_B = 64!
    e.mov_rr(RAX, RBX).shl_ri(RAX, 2)
        .mov_rr(R8, RSP).sub_rr(R8, RAX).mov_mr64(MemOp::bd(RBP, L_B), R8)
        .mov_ri32(R9, 64).mov_mr64(MemOp::bd(RBP, L_LDB_B), R9);
    // If E_orig == B_orig, also point L_E = RSP - j*4, L_LDE_B = 64!
    e.mov_rm64(R10, MemOp::bd(RBP, L_E_ORIG)).mov_rm64(R11, MemOp::bd(RBP, L_B_ORIG)).cmp_rr(R10, R11);
    let j_e_not_b = e.jcc_ph(0x85);
    e.mov_mr64(MemOp::bd(RBP, L_E), R8).mov_mr64(MemOp::bd(RBP, L_LDE_B), R9);
    e.patch_jcc(j_e_not_b);
    // Call m-strip with full16 (idx 2) or masked (idx 3)
    e.mov_ri32(R15, 0)
        .mov_rm64(RAX, MemOp::bd(RBP, L_NR)).cmp_ri(RAX, 16);
    let j_alias_mask = e.jcc_ph(0x8C);
    emit_call_mic_m_strip(e, mics, 2, false);
    let j_alias_next = e.jmp_ph();
    e.patch_jcc(j_alias_mask);
    emit_call_mic_m_strip(e, mics, 3, false);
    e.patch_jmp(j_alias_next);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).add_ri(RAX, 16).mov_mr64(MemOp::bd(RBP, L_J), RAX)
        .jmp_to(alias_j_top);
    e.patch_jcc(j_alias_done);
    // Restore RSP from RBP - 40 at exit
    let j_alias_exit = e.jmp_ph();

    // Non-aliasing fast path!
    e.patch_jcc(j_no_alias);
    // Decide Regime 1 (row-outer: K <= 512 && K * N <= 65536) vs Regime 2 (packed-B KC=256 cache-blocked)
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K));
    let j_blocked_k = e.cmp_ri(RAX, 512).jcc_ph(0x8F); // jg -> blocked
    e.mov_rm64(RCX, MemOp::bd(RBP, L_N)).imul_rr(RAX, RCX);
    let j_blocked_b = e.cmp_ri(RAX, 65536).jcc_ph(0x8F); // jg -> blocked

    // --- REGIME 1: Row-outer ---
    // Sub-regime 1A (cold-C streaming: M >= 1024 && K <= 128 && N >= 128): 8x16 {1to16} avoids 24-line MAB stall!
    e.mov_rm64(RAX, MemOp::bd(RBP, L_KIND)).shl_ri(RAX, 1).mov_mr64(MemOp::bd(RBP, L_FLAGS), RAX);
    let j_not_r1a_m = e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).cmp_ri(RAX, 1024).jcc_ph(0x8C);
    let j_not_r1a_k = e.mov_rm64(RAX, MemOp::bd(RBP, L_K)).cmp_ri(RAX, 128).jcc_ph(0x8F);
    let j_not_r1a_n = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).cmp_ri(RAX, 128).jcc_ph(0x8C);
    e.mov_ri32(R15, 0);
    for m_step in [8i32, 4i32, 1i32] {
        let j_mskip = e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_ph(0x8C);
        let m_top = e.code.len();
        emit_call_mic_row_band_16(e, mics, m_step);
        e.add_ri(R15, m_step);
        if m_step == 8 || m_step == 1 {
            e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_to(0x8D, m_top);
        }
        e.patch_jcc(j_mskip);
    }
    let j_r1a_done = e.jmp_ph();
    e.patch_jcc(j_not_r1a_m);
    e.patch_jcc(j_not_r1a_k);
    e.patch_jcc(j_not_r1a_n);
    // If (N & 15) == 1 && kind == 0 && K >= 16 (e.g. N=65), run K-vectorized AVX-512 for col N-1!
    let j_r1_general_k = e.mov_rm64(RAX, MemOp::bd(RBP, L_FLAGS)).cmp_ri(RAX, 0).jcc_ph(0x85);
    let j_r1_general_k2 = e.mov_rm64(RAX, MemOp::bd(RBP, L_K)).cmp_ri(RAX, 16).jcc_ph(0x8C);
    let j_r1_general_n = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).and_ri(RAX, 15).cmp_ri(RAX, 1).jcc_ph(0x85);
    // Fast N % 16 == 1 path: run full-16 columns with skip_col1=true, then ONE K-vectorized pass for col N-1!
    e.mov_ri32(R15, 0);
    for m_step in [8i32, 4i32, 1i32] {
        let j_mskip = e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_ph(0x8C);
        let m_top = e.code.len();
        emit_call_mic_row_band(e, mics, m_step, true);
        e.add_ri(R15, m_step);
        if m_step == 8 || m_step == 1 {
            e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_to(0x8D, m_top);
        }
        e.patch_jcc(j_mskip);
    }
    emit_col1_kvec_pass(e);
    let j_r1_c1_done = e.jmp_ph();

    e.patch_jcc(j_r1_general_k);
    e.patch_jcc(j_r1_general_k2);
    e.patch_jcc(j_r1_general_n);
    e.mov_ri32(R15, 0);
    for m_step in [8i32, 4i32, 1i32] {
        let j_mskip = e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_ph(0x8C);
        let m_top = e.code.len();
        emit_call_mic_row_band(e, mics, m_step, false);
        e.add_ri(R15, m_step);
        if m_step == 8 || m_step == 1 {
            e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).sub_rr(RAX, R15).cmp_ri(RAX, m_step).jcc_to(0x8D, m_top);
        }
        e.patch_jcc(j_mskip);
    }
    e.patch_jmp(j_r1_c1_done);
    e.patch_jmp(j_r1a_done);
    let j_r1_done = e.jmp_ph();

    // --- REGIME 2: Smart Split: Direct KC=512 (when N <= 128 || M < 128) vs Packed-B KC=256 (large MxN) ---
    e.patch_jcc(j_blocked_k);
    e.patch_jcc(j_blocked_b);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).mov_mr64(MemOp::bd(RBP, L_IEND), RAX)
        .mov_ri32(RAX, 0).mov_mr64(MemOp::bd(RBP, L_K0), RAX);
    // Check if M >= 128 && N >= 128 -> use Stack-Packed-B (KC=256); else Direct K-blocked (KC=512, 0 B-copy!)
    let j_unpacked_m = e.mov_rm64(RAX, MemOp::bd(RBP, L_M)).cmp_ri(RAX, 128).jcc_ph(0x8C);
    let j_unpacked_n = e.mov_rm64(RAX, MemOp::bd(RBP, L_N)).cmp_ri(RAX, 128).jcc_ph(0x8C);

    // --- REGIME 2A: Stack-Packed-B (KC=256, 48 KiB contiguous B panel, 0 malloc!) ---
    e.mov_rr(R10, RSP).sub_ri(R10, 49216).and_ri(R10, !63);
    let pk_probe = e.code.len();
    e.sub_ri(RSP, 4096).cmp_rr(RSP, R10);
    let j_pk_done = e.jcc_ph(0x86);
    e.mov_mr64(MemOp::bd(RSP, 0), RAX).jmp_to(pk_probe);
    e.patch_jcc(j_pk_done);
    e.mov_rr(RSP, R10).mov_mr64(MemOp::bd(RBP, L_PACK_PTR), R10);

    let k_outer_top = e.code.len();
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K0))
        .mov_rm64(RCX, MemOp::bd(RBP, L_K)).sub_rr(RCX, RAX);
    let j_k_outer_done = e.cmp_ri(RCX, 0).jcc_ph(0x8E);
    let j_kk_ok = e.cmp_ri(RCX, 256).jcc_ph(0x8E);
    e.mov_ri32(RCX, 256);
    e.patch_jcc(j_kk_ok);
    e.mov_mr64(MemOp::bd(RBP, L_KK), RCX);
    e.mov_ri32(R8, 0).cmp_ri(RAX, 0);
    let j_k0_zero = e.jcc_ph(0x84);
    e.mov_ri32(R8, 1);
    e.patch_jcc(j_k0_zero);
    e.add_rr(RAX, RCX).mov_rm64(RDX, MemOp::bd(RBP, L_K)).cmp_rr(RAX, RDX);
    let j_not_last_k = e.jcc_ph(0x85);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_KIND)).shl_ri(RAX, 1).add_rr(R8, RAX);
    e.patch_jcc(j_not_last_k);
    e.mov_mr64(MemOp::bd(RBP, L_FLAGS), R8);

    e.mov_ri32(RAX, 0).mov_mr64(MemOp::bd(RBP, L_J), RAX);
    for (idx, nr, ncols) in [(0usize, 48i32, 3usize), (1usize, 32i32, 2usize), (2usize, 16i32, 1usize)] {
        let j_skip = e.mov_rm64(RAX, MemOp::bd(RBP, L_N))
            .mov_rm64(RCX, MemOp::bd(RBP, L_J)).sub_rr(RAX, RCX).cmp_ri(RAX, nr).jcc_ph(0x8C);
        let j_top = e.code.len();
        emit_pack_b_panel(e, ncols);
        e.mov_ri32(R15, 0);
        emit_call_mic_m_strip(e, mics, idx, true);
        e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).add_ri(RAX, nr).mov_mr64(MemOp::bd(RBP, L_J), RAX);
        if nr == 48 {
            e.mov_rm64(RCX, MemOp::bd(RBP, L_N)).sub_rr(RCX, RAX).cmp_ri(RCX, 48).jcc_to(0x8D, j_top);
        }
        e.patch_jcc(j_skip);
    }
    let j_tail_skip = e.mov_rm64(RCX, MemOp::bd(RBP, L_N))
        .mov_rm64(RAX, MemOp::bd(RBP, L_J)).sub_rr(RCX, RAX).cmp_ri(RCX, 0).jcc_ph(0x8E);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_B_ORIG)).mov_mr64(MemOp::bd(RBP, L_B), RAX)
        .mov_rm64(RAX, MemOp::bd(RBP, L_LDB_ORIG)).mov_mr64(MemOp::bd(RBP, L_LDB_B), RAX)
        .mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax()
        .mov_ri32(R15, 0);
    emit_call_mic_m_strip(e, mics, 3, false);
    e.patch_jcc(j_tail_skip);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K0)).mov_rm64(RCX, MemOp::bd(RBP, L_KK)).add_rr(RAX, RCX)
        .mov_mr64(MemOp::bd(RBP, L_K0), RAX).jmp_to(k_outer_top);
    e.patch_jcc(j_k_outer_done);
    let j_r2a_done = e.jmp_ph();

    // --- REGIME 2B: Direct K-blocked (KC=512, zero B-copy overhead; ideal for weight grads K=6144!) ---
    e.patch_jcc(j_unpacked_m);
    e.patch_jcc(j_unpacked_n);
    let ku_top = e.code.len();
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K0))
        .mov_rm64(RCX, MemOp::bd(RBP, L_K)).sub_rr(RCX, RAX);
    let j_ku_done = e.cmp_ri(RCX, 0).jcc_ph(0x8E);
    let j_kku_ok = e.cmp_ri(RCX, 512).jcc_ph(0x8E);
    e.mov_ri32(RCX, 512);
    e.patch_jcc(j_kku_ok);
    e.mov_mr64(MemOp::bd(RBP, L_KK), RCX);
    e.mov_ri32(R8, 0).cmp_ri(RAX, 0);
    let j_ku0_z = e.jcc_ph(0x84);
    e.mov_ri32(R8, 1);
    e.patch_jcc(j_ku0_z);
    e.add_rr(RAX, RCX).mov_rm64(RDX, MemOp::bd(RBP, L_K)).cmp_rr(RAX, RDX);
    let j_ku_nl = e.jcc_ph(0x85);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_KIND)).shl_ri(RAX, 1).add_rr(R8, RAX);
    e.patch_jcc(j_ku_nl);
    e.mov_mr64(MemOp::bd(RBP, L_FLAGS), R8)
        .mov_ri32(RAX, 0).mov_mr64(MemOp::bd(RBP, L_J), RAX);
    // 48 -> 32 -> 16 -> mask with 64->32+32 split!
    let j_u48_sk = e.mov_rm64(RAX, MemOp::bd(RBP, L_N))
        .mov_rm64(RCX, MemOp::bd(RBP, L_J)).sub_rr(RAX, RCX).cmp_ri(RAX, 48).jcc_ph(0x8C);
    let j_u48_64 = e.cmp_ri(RAX, 64).jcc_ph(0x84);
    let u48_top = e.code.len();
    e.mov_ri32(R15, 0);
    emit_call_mic_m_strip(e, mics, 0, false);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).add_ri(RAX, 48).mov_mr64(MemOp::bd(RBP, L_J), RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_N)).sub_rr(RCX, RAX);
    let j_u48_64b = e.cmp_ri(RCX, 64).jcc_ph(0x84);
    e.cmp_ri(RCX, 48).jcc_to(0x8D, u48_top);
    e.patch_jcc(j_u48_sk);
    e.patch_jcc(j_u48_64);
    e.patch_jcc(j_u48_64b);
    let j_u32_sk = e.mov_rm64(RAX, MemOp::bd(RBP, L_N))
        .mov_rm64(RCX, MemOp::bd(RBP, L_J)).sub_rr(RAX, RCX).cmp_ri(RAX, 32).jcc_ph(0x8C);
    let u32_top = e.code.len();
    e.mov_ri32(R15, 0);
    emit_call_mic_m_strip(e, mics, 1, false);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).add_ri(RAX, 32).mov_mr64(MemOp::bd(RBP, L_J), RAX)
        .mov_rm64(RCX, MemOp::bd(RBP, L_N)).sub_rr(RCX, RAX).cmp_ri(RCX, 32).jcc_to(0x8D, u32_top);
    e.patch_jcc(j_u32_sk);
    let j_u16_sk = e.mov_rm64(RAX, MemOp::bd(RBP, L_N))
        .mov_rm64(RCX, MemOp::bd(RBP, L_J)).sub_rr(RAX, RCX).cmp_ri(RAX, 16).jcc_ph(0x8C);
    e.mov_ri32(R15, 0);
    emit_call_mic_m_strip(e, mics, 2, false);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_J)).add_ri(RAX, 16).mov_mr64(MemOp::bd(RBP, L_J), RAX);
    e.patch_jcc(j_u16_sk);
    let j_utail_sk = e.mov_rm64(RCX, MemOp::bd(RBP, L_N))
        .mov_rm64(RAX, MemOp::bd(RBP, L_J)).sub_rr(RCX, RAX).cmp_ri(RCX, 0).jcc_ph(0x8E);
    e.mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax()
        .mov_ri32(R15, 0);
    emit_call_mic_m_strip(e, mics, 3, false);
    e.patch_jcc(j_utail_sk);
    e.mov_rm64(RAX, MemOp::bd(RBP, L_K0)).mov_rm64(RCX, MemOp::bd(RBP, L_KK)).add_rr(RAX, RCX)
        .mov_mr64(MemOp::bd(RBP, L_K0), RAX).jmp_to(ku_top);
    e.patch_jcc(j_ku_done);
    e.patch_jmp(j_r2a_done);

    e.patch_jmp(j_r1_done);
    e.patch_jmp(j_alias_exit);
    e.patch_jcc(j_m0);
    e.patch_jcc(j_n0);
    e.vzeroupper()
        .lea(RSP, MemOp::bd(RBP, -40)) // 5 callee-saved regs pushed after rbp = 40 bytes!
        .pop_r(R15).pop_r(R14).pop_r(R13).pop_r(R12).pop_r(RBX).pop_r(RBP).ret();
    entry
}

fn emit_sgemm_plain_wrapper(e: &mut Avx512Emit, core_off: usize) -> (u64, u64) {
    // Plain sgemm(M, N, K, A, lda, B, ldb, C, ldc) -> calls core with kind=0, E=0, lde=0, s=0.0
    // Incoming stack: [rsp+8]=ldb, [rsp+16]=C, [rsp+24]=ldc
    // Outgoing stack for core: [rsp+0]=ldb, [rsp+8]=C, [rsp+16]=ldc, [rsp+24]=0, [rsp+32]=0, [rsp+40]=0
    e.align(16);
    let off = e.code.len() as u64;
    e.sub_ri(RSP, 56) // 56 + 8(ret) = 64 -> 16B aligned before call!
        .mov_rm64(RAX, MemOp::bd(RSP, 64)).mov_mr64(MemOp::bd(RSP, 0), RAX)   // ldb
        .mov_rm64(RAX, MemOp::bd(RSP, 72)).mov_mr64(MemOp::bd(RSP, 8), RAX)   // C
        .mov_rm64(RAX, MemOp::bd(RSP, 80)).mov_mr64(MemOp::bd(RSP, 16), RAX)  // ldc
        .mov_ri32(RAX, 0)
        .mov_mr64(MemOp::bd(RSP, 24), RAX)                                    // kind = 0
        .mov_mr64(MemOp::bd(RSP, 32), RAX)                                    // E = NULL
        .mov_mr64(MemOp::bd(RSP, 40), RAX)                                    // lde = 0
        .call_rel(core_off)
        .add_ri(RSP, 56).ret();
    (off, (e.code.len() as u64) - off)
}

fn emit_sgemm_epilogue_apply(e: &mut Avx512Emit) -> (u64, u64) {
    // void sgemm_epilogue_apply(int M, int N, int kind, float *C, int ldc, const float *E, int lde, float s)
    // rdi=M, rsi=N, rdx=kind, rcx=C, r8=ldc, r9=E, [rsp+8]=lde, xmm0=s
    e.align(16);
    let off = e.code.len() as u64;
    e.push_r(RBP).mov_rr(RBP, RSP).push_r(RBX).push_r(R12).push_r(R13);
    e.movsxd(RDI, RDI).movsxd(RSI, RSI).movsxd(RDX, RDX).movsxd(R8, R8)
        .movsxd_rm(R10, MemOp::bd(RBP, 16)) // lde
        .shl_ri(R8, 2).shl_ri(R10, 2)       // ldc_b, lde_b
        .vbroadcastss_xmm(31, 0);
    let j_m0 = e.cmp_ri(RDI, 0).jcc_ph(0x8E);
    let j_n0 = e.cmp_ri(RSI, 0).jcc_ph(0x8E);
    let row_top = e.code.len();
    e.mov_rr(R11, RCX).mov_rr(R12, R9).mov_rr(RBX, RSI); // r11=cr, r12=er, rbx=rem_n
    let col_top = e.code.len();
    let j_col_done = e.cmp_ri(RBX, 0).jcc_ph(0x8E);
    // Compute k1 mask for min(16, rbx)
    e.mov_ri32(R13, 16).cmp_rr(RBX, R13);
    let j_full = e.jcc_ph(0x8D);
    e.mov_rr(R13, RBX);
    e.patch_jcc(j_full);
    e.push_r(RCX).mov_rr(RCX, R13).mov_ri32(RAX, 1).shl_cl_r32(RAX).sub_ri(RAX, 1).kmovw_k1_eax().pop_r(RCX);
    e.vmovups_ld(0, MemOp::bd(R11, 0), true);
    let j_not1 = e.cmp_ri(RDX, 1).jcc_ph(0x85);
    e.vmovups_ld(1, MemOp::bd(R12, 0), true).vaddps(0, 0, 1).vmovups_st(MemOp::bd(R11, 0), 0, true);
    let j_next1 = e.jmp_ph();
    e.patch_jcc(j_not1);
    let j_not3 = e.cmp_ri(RDX, 3).jcc_ph(0x85);
    e.vmovups_ld(1, MemOp::bd(R12, 0), true).vmulps(1, 31, 1).vaddps(0, 0, 1).vmovups_st(MemOp::bd(R11, 0), 0, true);
    let j_next3 = e.jmp_ph();
    e.patch_jcc(j_not3);
    let j_not2 = e.cmp_ri(RDX, 2).jcc_ph(0x85);
    e.vmovups_ld(1, MemOp::bd(R12, 0), true);
    emit_gelu_pair512(e, 1, [2, 3], [4, 5, 6, 7]);
    e.vmulps(0, 0, 3).vmovups_st(MemOp::bd(R11, 0), 0, true);
    let j_next2 = e.jmp_ph();
    e.patch_jcc(j_not2);
    let j_not4 = e.cmp_ri(RDX, 4).jcc_ph(0x85);
    emit_gelu_pair512(e, 0, [2, 3], [4, 5, 6, 7]);
    e.vmovups_st(MemOp::bd(R12, 0), 3, true).vmovups_st(MemOp::bd(R11, 0), 2, true);
    e.patch_jcc(j_not4);
    e.patch_jmp(j_next1);
    e.patch_jmp(j_next3);
    e.patch_jmp(j_next2);
    e.add_ri(R11, 64).add_ri(R12, 64).sub_ri(RBX, 16).jmp_to(col_top);
    e.patch_jcc(j_col_done);
    e.add_rr(RCX, R8).add_rr(R9, R10).sub_ri(RDI, 1).jcc_to(0x85, row_top);
    e.patch_jcc(j_m0);
    e.patch_jcc(j_n0);
    e.vzeroupper().pop_r(R13).pop_r(R12).pop_r(RBX).pop_r(RBP).ret();
    (off, (e.code.len() as u64) - off)
}

pub fn compile_tensor_to_elf(tk: &TensorKernel) -> Res<CompiledTensorObject> {
    let mut e = Avx512Emit::default();
    let mics = MicTable {
        m8: [
            emit_gemm_microkernel(&mut e, 8, 3, false),
            emit_gemm_microkernel(&mut e, 8, 2, false),
            emit_gemm_microkernel(&mut e, 8, 1, false),
            emit_gemm_microkernel(&mut e, 8, 1, true),
            emit_gemm_microkernel_col1(&mut e, 8),
        ],
        m4: [
            emit_gemm_microkernel(&mut e, 4, 3, false),
            emit_gemm_microkernel(&mut e, 4, 2, false),
            emit_gemm_microkernel(&mut e, 4, 1, false),
            emit_gemm_microkernel(&mut e, 4, 1, true),
            emit_gemm_microkernel_col1(&mut e, 4),
        ],
        m1: [
            emit_gemm_microkernel(&mut e, 1, 3, false),
            emit_gemm_microkernel(&mut e, 1, 2, false),
            emit_gemm_microkernel(&mut e, 1, 1, false),
            emit_gemm_microkernel(&mut e, 1, 1, true),
            emit_gemm_microkernel_col1(&mut e, 1),
        ],
    };
    let core_off = emit_uopc_gemm_core(&mut e, &mics);
    let core_sz = (e.code.len() - core_off) as u64;
    let (plain_off, plain_sz) = emit_sgemm_plain_wrapper(&mut e, core_off);
    let (epi_app_off, epi_app_sz) = emit_sgemm_epilogue_apply(&mut e);
    e.finalize_f32_pool();

    let mut syms = if tk.export_uopc_abi {
        vec![
            ElfSymbol { name: "sgemm_dispatch_epi".into(), offset: core_off as u64, size: core_sz },
            ElfSymbol { name: "sgemm_direct2_epi".into(), offset: core_off as u64, size: core_sz },
            ElfSymbol { name: "sgemm_dispatch".into(), offset: plain_off, size: plain_sz },
            ElfSymbol { name: "sgemm_direct2".into(), offset: plain_off, size: plain_sz },
            ElfSymbol { name: "sgemm".into(), offset: plain_off, size: plain_sz },
            ElfSymbol { name: "sgemm_epilogue_apply".into(), offset: epi_app_off, size: epi_app_sz },
        ]
    } else {
        vec![ElfSymbol { name: format!("zero_cc_{}", tk.name), offset: plain_off, size: plain_sz }]
    };
    if tk.export_uopc_abi && !syms.iter().any(|s| s.name == tk.name) {
        syms.push(ElfSymbol { name: tk.name.clone(), offset: plain_off, size: plain_sz });
    }
    let elf_bytes = emit_elf64_object(TargetIsa::X86_64, &e.code, &syms, &[])?;
    let sum = verify_elf64_object(&elf_bytes)?;
    if !sum.has_nonexec_gnu_stack || sum.rela_count != 0 {
        return Err(berr("avx512 tensor elf64 airlock verification failed"));
    }
    Ok(CompiledTensorObject {
        name: tk.name.clone(),
        symbols: syms.into_iter().map(|s| s.name).collect(),
        code_bytes: e.code,
        elf_bytes,
        direct_insns: e.insns,
        zero_spills: tk.tile.spills == 0,
        tile: tk.tile,
    })
}

pub fn compile_uopc_sgemm_elf(budget: RegBudget) -> Res<CompiledTensorObject> {
    let tk = TensorKernel {
        name: "sgemm_dispatch".into(),
        epilogue: crate::tensor::EpilogueKind::Dynamic,
        export_uopc_abi: true,
        tile: crate::tensor::plan_gemm_tiles(budget)?,
    };
    compile_tensor_to_elf(&tk)
}
