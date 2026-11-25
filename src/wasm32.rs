use crate::codegen::{schedule_and_verify, RegBudget, ScheduleReport};
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op, LANES};
use crate::{optimize_kernel, rust_fe::parse_rust_kernel};

pub const WASM_MAGIC_VERSION: [u8; 8] = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
pub const WASM_DEFAULT_MEM_PAGES: u32 = 4;
type Res<T> = Result<T, CompileError>;
type LaneIds = [NodeId; LANES];
const ZERO_LANES: LaneIds = [0; LANES];
const E64: &str = "unsupported op in vertical i64x2 SIMD128 lowering";
const CMP_IDX: [u8; 10] = [0, 1, 2, 6, 4, 8, 3, 7, 5, 9];
const ELEM_MEM: [(u8, u8); 7] = [
    (0x29, 0x37), (0x29, 0x37), (0x34, 0x3E), (0x32, 0x3D), (0x35, 0x3E), (0x33, 0x3D), (0x31, 0x3C),
];
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetArch { X86_64, Aarch64, Wasm32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmVectorMode { ScalarI64, HorizontalI32x4, VerticalI64x2 }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmExport { pub name: String, pub kind: u8, pub index: u32 }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmSummary {
    pub type_count: usize, pub func_count: usize, pub mem_min_pages: u32, pub exports: Vec<WasmExport>,
    pub code_body_count: usize, pub total_insns: usize, pub simd_insns: usize, pub div_rem_insns: usize,
    pub has_simd128: bool, pub imports_memory: bool,
}
#[derive(Clone, Debug)]
pub struct CompiledWasm {
    pub kernel_sym: String, pub batch_sym: String, pub wasm_bytes: Vec<u8>,
    pub vector_mode: WasmVectorMode, pub summary: WasmSummary,
}
fn werr(msg: impl Into<String>) -> CompileError { CompileError::BackendError(msg.into()) }
fn cmp_idx(k: CmpKind, u: bool) -> u8 {
    let i = CMP_IDX.get(k as usize).copied().unwrap_or(0);
    if u && i >= 2 { i | 1 } else { i }
}
fn mop(t: ElemTy, s: bool) -> u8 {
    let (ld, st) = ELEM_MEM.get(t as usize).copied().unwrap_or((0x29, 0x37));
    if s { st } else { ld }
}
const fn alu_opc(op: Op<NodeId>) -> Option<(u8, u32, u32, Option<u8>)> {
    Some(match op {
        Op::Add(_, _) => (0x7C, 0xAE, 0xCE, None),
        Op::Sub(_, _) => (0x7D, 0xB1, 0xD1, None),
        Op::Mul(_, _) => (0x7E, 0xB5, 0xD5, None),
        Op::Div(_, _) => (0x7F, 0, 0, None),
        Op::Rem(_, _) => (0x81, 0, 0, None),
        Op::BitAnd(_, _) => (0x83, 0x4E, 0x4E, None),
        Op::BitOr(_, _) => (0x84, 0, 0x50, None),
        Op::BitXor(_, _) => (0x85, 0, 0x51, None),
        Op::ShlVar(_, _) => (0x86, 0, 0, None),
        Op::ShrVarU(_, _) => (0x88, 0, 0, None),
        Op::Shl(_, s) => (0x86, 0xAB, 0xCB, Some(s)),
        Op::Shr(_, s) => (0x87, 0xAC, 0xCC, Some(s)),
        Op::Ushr(_, s) => (0x88, 0xAD, 0xCD, Some(s)),
        _ => return None,
    })
}

#[derive(Default)]
struct WasmBuf(Vec<u8>);
impl WasmBuf {
    fn ops(&mut self, bs: &[u8]) -> &mut Self {
        self.0.extend_from_slice(bs);
        self
    }
    fn op(&mut self, b: u8) -> &mut Self { self.ops(&[b]) }
    fn leb(&mut self, mut v: i64, s: bool) -> &mut Self {
        loop {
            let b = (v & 0x7F) as u8;
            v = if s { v >> 7 } else { ((v as u64) >> 7) as i64 };
            let done = if s { (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0) } else { v == 0 };
            self.op(if done { b } else { b | 0x80 });
            if done { return self; }
        }
    }
    fn uleb(&mut self, v: u32) -> &mut Self { self.leb(i64::from(v), false) }
    fn len_bytes(&mut self, d: &[u8]) -> &mut Self { self.uleb(d.len() as u32).ops(d) }
    fn sec(&mut self, id: u8, d: &[u8]) -> &mut Self { self.op(id).len_bytes(d) }
    fn export(&mut self, n: &str, k: u8, i: u32) -> &mut Self { self.len_bytes(n.as_bytes()).op(k).uleb(i) }
    fn op_u(&mut self, b: u8, v: u32) -> &mut Self { self.op(b).uleb(v) }
    fn get(&mut self, i: u32) -> &mut Self { self.op_u(0x20, i) }
    fn set(&mut self, i: u32) -> &mut Self { self.op_u(0x21, i) }
    fn i32c(&mut self, v: i32) -> &mut Self { self.op(0x41).leb(i64::from(v), true) }
    fn i64c(&mut self, v: i64) -> &mut Self { self.op(0x42).leb(v, true) }
    fn simd(&mut self, sub: u32) -> &mut Self { self.op_u(0xFD, sub) }
    fn smem(&mut self, sub: u32, al: u32, off: u32) -> &mut Self { self.simd(sub).uleb(al).uleb(off) }
    fn mem(&mut self, t: ElemTy, o: u32, s: bool) -> &mut Self { self.op_u(mop(t, s), t.scale_shift() as u32).uleb(o) }
    fn v128(&mut self, v: u64) -> &mut Self { self.simd(0x0C).ops(&v.to_le_bytes()).ops(&v.to_le_bytes()) }
    fn trap(&mut self, d: u32, l: i64) -> &mut Self { self.op_u(0x22, d).i64c(l).ops(&[0x5A, 0x04, 0x40, 0x00, 0x0B]) }
}

fn alloc_ssa_locals(k: &Kernel, sched: &ScheduleReport, base: u32) -> Res<(Vec<Option<u32>>, u32)> {
    let mut last = vec![0usize; k.nodes.len()];
    for (step, &id) in sched.order.iter().enumerate() {
        let (kids, cnt) = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?.children();
        for &kid in kids.iter().take(cnt) { if let Some(ls) = last.get_mut(kid) { *ls = (*ls).max(step) } }
    }
    for &(ostep, oi) in &sched.output_order {
        if let Some(ls) = k.outputs.get(oi).and_then(|o| last.get_mut(o.val)) { *ls = (*ls).max(ostep + 1); }
    }
    let (mut locs, mut free, mut active, mut next) = (vec![None; k.nodes.len()], Vec::new(), Vec::new(), 0u32);
    for (step, &id) in sched.order.iter().enumerate() {
        let op = k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if !matches!(op, Op::MemInit | Op::Const(_) | Op::Store { .. }) {
            let slot = free.pop().unwrap_or(next);
            if slot == next { next = next.saturating_add(1); }
            if let Some(dst) = locs.get_mut(id) { *dst = Some(base.saturating_add(slot)); }
            active.push((id, slot));
        }
        active.retain(|&(aid, s)| {
            let live = last.get(aid).copied().unwrap_or(0) > step;
            if !live { free.push(s); }
            live
        });
    }
    Ok((locs, next))
}
fn can_plan_i64x2(k: &Kernel, sched: &ScheduleReport) -> bool {
    !k.outputs.is_empty() && !k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || b.lanes == 0)
        && sched.order.iter().all(|&id| k.nodes.get(id).copied().is_some_and(|op|
            matches!(op, Op::MemInit | Op::Const(_) | Op::Load { .. } | Op::Store { .. } | Op::Select { .. })
                || matches!(op, Op::Cmp(c, _, _) if !c.is_unsigned()) || alu_opc(op).is_some_and(|t| t.2 != 0)))
}
fn collect_hoisted_consts_i64(k: &Kernel, sched: &ScheduleReport) -> Vec<i64> {
    let mut counts: Vec<(i64, usize)> = Vec::new();
    let uses = sched.order.iter().filter_map(|&id| k.nodes.get(id).copied())
        .map(Op::children).flat_map(|(ks, n)| ks.into_iter().take(n));
    for nid in uses.chain(k.outputs.iter().map(|o| o.val)) {
        if let Some(Op::Const(c)) = k.nodes.get(nid).copied() {
            if let Some(e) = counts.iter_mut().find(|e| e.0 == c) { e.1 += 1 } else { counts.push((c, 1)) }
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts.into_iter().filter(|&(_, cnt)| cnt >= 2).take(16).map(|(c, _)| c).collect()
}

enum DagMode<'a> { Scalar(&'a [Domain]), I64x2(&'a [(i64, u32)], u32, Vec<bool>) }
struct DagEmit<'a> { e: &'a mut WasmBuf, k: &'a Kernel, locs: &'a [Option<u32>], mode: DagMode<'a> }
impl<'a> DagEmit<'a> {
    fn loc(&self, id: NodeId) -> Res<u32> { self.locs.get(id).and_then(|&v| v).ok_or(CompileError::InvalidNode(id)) }
    fn doms(&self) -> Res<&'a [Domain]> { if let DagMode::Scalar(d) = self.mode { Ok(d) } else { Err(werr(E64)) } }
    fn buf_offs(&self, buf: u8, lane: u8) -> Res<(ElemTy, u32, u32)> {
        let b = self.k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
        let o0 = u32::from(lane) * u32::from(b.elem_ty.bytes());
        Ok((b.elem_ty, o0, (b.block_bytes() as u32).saturating_add(o0)))
    }
    fn pconst(&mut self, c: i64) -> &mut WasmBuf {
        let DagMode::I64x2(h, _, _) = self.mode else { return self.e.i64c(c) };
        if let Some(&(_, l)) = h.iter().find(|&&(v, _)| v == c) { self.e.get(l) } else { self.e.v128(c as u64) }
    }
    fn pval(&mut self, id: NodeId) -> Res<&mut WasmBuf> {
        let op = self.k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
        if let Op::Const(c) = op { return Ok(self.pconst(c)); }
        if matches!(op, Op::MemInit) { return Ok(self.pconst(0)); }
        self.e.get(self.loc(id)?);
        if matches!(&self.mode, DagMode::I64x2(_, _, m) if m.get(id) == Some(&true)) { self.pconst(1).simd(0x4E); }
        Ok(self.e)
    }
    fn pvals(&mut self, ids: &[NodeId]) -> Res<&mut WasmBuf> {
        for &id in ids { self.pval(id)?; }
        Ok(self.e)
    }
    fn store(&mut self, buf: u8, lane: u8, val: NodeId) -> Res<()> {
        let ((ty, o0, o1), b) = (self.buf_offs(buf, lane)?, u32::from(buf));
        let DagMode::I64x2(_, sc, _) = self.mode else {
            self.e.get(b);
            self.pval(val)?.mem(ty, o0, true);
            return Ok(());
        };
        self.pval(val)?.set(sc);
        for (li, off) in [(0u8, o0), (1u8, o1)] {
            let e = self.e.get(b).get(sc);
            if matches!(ty, ElemTy::I64 | ElemTy::U64) { e.smem(0x5B, 3, off).op(li); }
            else { e.simd(0x1D).op(li).mem(ty, off, true); }
        }
        Ok(())
    }
    fn drain_outputs(&mut self, sched: &ScheduleReport, idx: &mut usize, max: usize) -> Res<()> {
        while let Some(&(_, oi)) = sched.output_order.get(*idx).filter(|&&(s, _)| s <= max) {
            let o = self.k.outputs.get(oi).ok_or(CompileError::InvalidNode(oi))?;
            self.store(o.buf, o.lane, o.val)?;
            *idx += 1;
        }
        Ok(())
    }
    fn emit(&mut self, sched: &ScheduleReport) -> Res<()> {
        let mut out_idx = 0usize;
        for (step, &id) in sched.order.iter().enumerate() {
            self.drain_outputs(sched, &mut out_idx, step)?;
            let op = self.k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id))?;
            if matches!(op, Op::MemInit | Op::Const(_)) { continue; }
            if let Op::Store { buf, lane, val, .. } = op {
                self.store(buf, lane, val)?;
                continue;
            }
            let dst = self.loc(id)?;
            match op {
                Op::Load { buf, lane, .. } => {
                    let ((ty, o0, o1), b) = (self.buf_offs(buf, lane)?, u32::from(buf));
                    if matches!(self.mode, DagMode::Scalar(_)) { self.e.get(b).mem(ty, o0, false); }
                    else if matches!(ty, ElemTy::I64 | ElemTy::U64) {
                        self.e.get(b).get(b).smem(0x0A, 3, o0).smem(0x57, 3, o1).op(1);
                    } else { self.e.get(b).mem(ty, o0, false).simd(0x12).get(b).mem(ty, o1, false).simd(0x1E).op(1); }
                }
                Op::LoadLut { buf, idx } => {
                    let b = self.k.bufs.get(usize::from(buf)).ok_or(CompileError::InvalidBuffer(buf))?;
                    let BufAccess::LookupTable(len) = b.access else { return Err(CompileError::InvalidBuffer(buf)) };
                    let d = self.doms()?.get(idx).copied().unwrap_or_else(Domain::top);
                    self.e.get(u32::from(buf));
                    self.pval(idx)?;
                    if d.is_empty() || !d.is_nonneg() || d.hi >= len as i64 { self.e.trap(dst, len as i64).get(dst); }
                    self.e.op(0xA7);
                    if b.elem_ty.scale_shift() > 0 { self.e.i32c(i32::from(b.elem_ty.scale_shift())).op(0x74); }
                    self.e.op(0x6A).mem(b.elem_ty, 0, false);
                }
                Op::BoundsCheck { idx, limit } => {
                    self.doms()?;
                    self.pval(idx)?.trap(dst, limit);
                    continue;
                }
                Op::Popcnt(a) | Op::Tzcnt(a) => {
                    self.doms()?;
                    self.pval(a)?.op(if matches!(op, Op::Popcnt(_)) { 0x7B } else { 0x7A });
                }
                Op::Cmp(kind, a, b) => {
                    self.pvals(&[a, b])?;
                    let i = cmp_idx(kind, false);
                    if let DagMode::I64x2(_, _, is_mask) = &mut self.mode {
                        if kind.is_unsigned() { return Err(werr("unsigned i64x2 cmp unsupported")); }
                        self.e.simd(if i < 2 { 0xD6 + u32::from(i) } else { 0xD7 + u32::from(i >> 1) });
                        if let Some(m) = is_mask.get_mut(id) { *m = true; }
                    } else { self.e.ops(&[0x51 + i, 0xAD]); }
                }
                Op::Select { cond, t_val, f_val } => if let DagMode::I64x2(_, _, ref m) = self.mode {
                    let cm = m.get(cond) == Some(&true);
                    self.pvals(&[t_val, f_val])?;
                    if cm { self.e.get(self.loc(cond)?) } else {
                        self.pval(cond)?;
                        self.pconst(0).simd(0xD7)
                    }.simd(0x52);
                } else { self.pvals(&[f_val, t_val, cond])?.ops(&[0x50, 0x1B]); },
                _ => {
                    let (sc, _, v64, sh) = alu_opc(op).ok_or_else(|| werr("unsupported op in wasm lowering"))?;
                    let ([a, b, _], _) = op.children();
                    if let (Op::Rem(_, _), DagMode::Scalar(doms)) = (op, &self.mode) {
                        let has = |n, v| doms.get(n).copied().map_or(true, |d: Domain| d.is_empty() || d.contains(v));
                        if has(a, i64::MIN) && has(b, -1) {
                            self.pval(a)?.i64c(i64::MIN).op(0x51);
                            self.pval(b)?.i64c(-1).ops(&[0x51, 0x71, 0x04, 0x40, 0x00, 0x0B]);
                        }
                    }
                    match (sh, &self.mode) {
                        (Some(s), DagMode::Scalar(_)) => { self.pval(a)?.i64c(i64::from(s)).op(sc); }
                        (Some(s), DagMode::I64x2(..)) => { self.pval(a)?.i32c(i32::from(s)).simd(v64); }
                        (None, DagMode::Scalar(_)) => { self.pvals(&[a, b])?.op(sc); }
                        (None, DagMode::I64x2(..)) if v64 != 0 => { self.pvals(&[a, b])?.simd(v64); }
                        (None, DagMode::I64x2(..)) => return Err(werr(E64)),
                    }
                }
            }
            self.e.set(dst);
        }
        self.drain_outputs(sched, &mut out_idx, usize::MAX)
    }
}
#[derive(Clone, Copy)]
enum I32x4Op { Load(u8, ElemTy), Const(u32), Bin(u32, u32, u32), Sh(u32, u32, u8), Mask01(u32), Sel(u32, u32, u32) }
struct I32x4Plan { steps: Vec<(I32x4Op, bool)>, outputs: Vec<(u8, ElemTy, u32, bool)> }
struct I32x4Ctx<'a> {
    k: &'a Kernel, d: &'a [Domain], nf: &'a [bool], u: bool, memo: Vec<(LaneIds, u32)>, steps: Vec<(I32x4Op, bool)>,
}
impl<'a> I32x4Ctx<'a> {
    fn fit(&self, i: usize) -> bool { self.d.get(i).is_some_and(|d| if self.u { d.fits_u32() } else { d.fits_i32() }) }
    fn is_mask(&self, v: u32) -> bool { self.steps.get(v as usize).is_some_and(|s| s.1) }
    fn push(&mut self, op: I32x4Op, mask: bool) -> u32 {
        let id = self.steps.len() as u32;
        self.steps.push((op, mask));
        id
    }
    fn intern_const(&mut self, val: u32) -> u32 {
        self.steps.iter().position(|&(op, _)| matches!(op, I32x4Op::Const(c) if c == val))
            .map_or_else(|| self.push(I32x4Op::Const(val), false), |i| i as u32)
    }
    fn gather_kids(&self, ids: LaneIds, tag: Op<()>, fit: bool) -> Option<[LaneIds; 3]> {
        let mut out = [ZERO_LANES; 3];
        for (i, &id) in ids.iter().enumerate() {
            let (kids, cnt) = self.k.nodes.get(id).copied().filter(|o| o.map_ids(|_| ()) == tag)?.children();
            for (k, &kid) in kids.iter().take(cnt).enumerate() {
                if fit && !self.fit(kid) { return None; }
                if let Some(s) = out.get_mut(k).and_then(|a| a.get_mut(i)) { *s = kid; }
            }
        }
        Some(out)
    }
    fn match_val(&mut self, ids: LaneIds) -> Option<u32> {
        let v = self.match_lanes(ids)?;
        Some(if self.is_mask(v) { self.push(I32x4Op::Mask01(v), false) } else { v })
    }
    fn match_lanes(&mut self, ids: LaneIds) -> Option<u32> {
        if let Some(&(_, v)) = self.memo.iter().find(|&&(k, _)| k == ids) { return Some(v); }
        if ids.iter().any(|&i| self.nf.get(i) == Some(&true)) && !ids.iter().all(|&i| self.fit(i)) { return None; }
        let first = self.k.nodes.get(ids[0]).copied()?;
        let tag = first.map_ids(|_| ());
        let res = match first {
            Op::Const(c) => ids.iter().all(|&i| self.k.nodes.get(i) == Some(&Op::Const(c)))
                .then(|| self.intern_const(c as u32))?,
            Op::Load { buf, .. } => {
                let ty = self.k.bufs.get(usize::from(buf)).map(|b| b.elem_ty)
                    .filter(|t| matches!(t, ElemTy::I32 | ElemTy::U32 | ElemTy::I16 | ElemTy::U16))?;
                ids.iter().enumerate().all(|(li, &id)| matches!(self.k.nodes.get(id),
                    Some(&Op::Load { mem, buf: b2, lane })
                        if b2 == buf && usize::from(lane) == li && self.k.nodes.get(mem) == Some(&Op::MemInit)
                )).then(|| self.push(I32x4Op::Load(buf, ty), false))?
            }
            Op::Add(_, _) | Op::Sub(_, _) | Op::Mul(_, _) | Op::BitAnd(_, _) => {
                let [la, lb, _] = self.gather_kids(ids, tag, false)?;
                let m32 = |a: &LaneIds| a.iter().all(|&n| self.k.nodes.get(n) == Some(&Op::Const(0xFFFF_FFFF)));
                if matches!(first, Op::BitAnd(_, _)) && (m32(&lb) || m32(&la)) {
                    self.match_val(if m32(&lb) { la } else { lb })?
                } else {
                    let (va, vb) = (self.match_val(la)?, self.match_val(lb)?);
                    self.push(I32x4Op::Bin(alu_opc(first)?.1, va, vb), false)
                }
            }
            Op::Shl(_, s) | Op::Shr(_, s) if s < 32 => {
                let shr = matches!(first, Op::Shr(_, _));
                let va = self.match_val(self.gather_kids(ids, tag, shr && s > 0)?[0])?;
                self.push(I32x4Op::Sh(0xAB + u32::from(shr) * (1 + u32::from(self.u)), va, s), false)
            }
            Op::Cmp(kind, _, _) if !kind.is_unsigned() || self.u => {
                let [la, lb, _] = self.gather_kids(ids, tag, true)?;
                let (va, vb) = (self.match_val(la)?, self.match_val(lb)?);
                self.push(I32x4Op::Bin(0x37 + u32::from(cmp_idx(kind, self.u)), va, vb), true)
            }
            Op::Select { .. } => {
                let [lc, lt, lf] = self.gather_kids(ids, tag, false)?;
                let (vc, vt, vf) = (self.match_lanes(lc)?, self.match_val(lt)?, self.match_val(lf)?);
                let mc = if self.is_mask(vc) { vc } else {
                    let z = self.intern_const(0);
                    self.push(I32x4Op::Bin(0x38, vc, z), true)
                };
                self.push(I32x4Op::Sel(mc, vt, vf), false)
            }
            _ => return None,
        };
        self.memo.push((ids, res));
        Some(res)
    }
}

fn try_plan_i32x4(k: &Kernel, sched: &ScheduleReport, doms: &[Domain]) -> Option<I32x4Plan> {
    if k.outputs.is_empty()
        || k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || usize::from(b.lanes) != LANES)
        || sched.order.iter().any(|&id| matches!(k.nodes.get(id), Some(Op::Store { .. } | Op::LoadLut { .. }
            | Op::BoundsCheck { .. } | Op::Div(_, _) | Op::Rem(_, _)))
            || matches!(k.nodes.get(id), Some(&Op::Load { mem, .. }) if k.nodes.get(mem) != Some(&Op::MemInit)))
    { return None; }
    let mut outs: Vec<_> = k.bufs.iter().filter(|b| b.access == BufAccess::MutNoAlias)
        .map(|b| matches!(b.elem_ty, ElemTy::I32 | ElemTy::U32 | ElemTy::I64)
            .then_some((b.id, b.elem_ty, [usize::MAX; LANES]))).collect::<Option<_>>()?;
    for o in &k.outputs {
        let s = outs.iter_mut().find(|g| g.0 == o.buf)?.2.get_mut(usize::from(o.lane)).filter(|s| **s == usize::MAX)?;
        *s = o.val;
    }
    if outs.is_empty() || outs.iter().any(|g| g.2.contains(&usize::MAX)) { return None; }
    let mut nf = vec![false; k.nodes.len()];
    for g in &outs { if g.1 == ElemTy::I64 { for &v in &g.2 { if let Some(s) = nf.get_mut(v) { *s = true } } } }
    for &id in sched.order.iter().rev() {
        let sl: &[NodeId] = match k.nodes.get(id).copied() {
            Some(Op::Cmp(_, a, b)) => &[a, b],
            Some(Op::Shr(a, s)) if s > 0 => &[a],
            Some(Op::Select { cond, t_val, f_val }) if nf.get(id) == Some(&true) => &[cond, t_val, f_val],
            Some(Op::Select { cond, .. }) => &[cond],
            _ => &[],
        };
        for &n in sl { if let Some(s) = nf.get_mut(n) { *s = true } }
    }
    for u in [true, false] {
        let mut ctx = I32x4Ctx { k, d: doms, nf: &nf, u, memo: Vec::new(), steps: Vec::new() };
        if let Some(outputs) = outs.iter().map(|&(buf, ty, lanes)| {
            let v = ctx.match_val(lanes)?;
            let sgn = ty == ElemTy::I64 && {
                let s = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_i32));
                let u = lanes.iter().all(|&id| doms.get(id).copied().is_some_and(Domain::fits_u32));
                (s || u).then_some(!u)?
            };
            Some((buf, ty, v, sgn))
        }).collect() { return Some(I32x4Plan { steps: ctx.steps, outputs }); }
    }
    None
}
impl I32x4Plan {
    fn emit_consts(&self, e: &mut WasmBuf, v: u32) {
        for (sid, &(op, _)) in self.steps.iter().enumerate() {
            if let I32x4Op::Const(c) = op { e.v128(u64::from(c) * 0x1_0000_0001).set(v + sid as u32); }
        }
    }
    fn emit_both(&self, e: &mut WasmBuf, v: u32) -> Res<()> {
        for h in 0..2u32 {
            for (sid, &(step, _)) in self.steps.iter().enumerate() {
                match step {
                    I32x4Op::Const(_) => continue,
                    I32x4Op::Load(b, t) => {
                        let (s, a) = if t == ElemTy::U16 { (4, 1) } else if t == ElemTy::I16 { (3, 1) } else { (0, 2) };
                        e.get(u32::from(b)).smem(s, a, h * (4 << a))
                    }
                    I32x4Op::Bin(sub, a, b) => e.get(v + a).get(v + b).simd(sub),
                    I32x4Op::Sh(sub, a, s) => e.get(v + a).i32c(i32::from(s)).simd(sub),
                    I32x4Op::Mask01(m) => e.get(v + m).v128(0x1_0000_0001).simd(0x4E),
                    I32x4Op::Sel(mc, t, f) => e.get(v + t).get(v + f).get(v + mc).simd(0x52),
                }.set(v + sid as u32);
            }
            for &(b, t, s, sgn) in &self.outputs {
                if t == ElemTy::I64 {
                    let base = if sgn { 0xC7 } else { 0xC9 };
                    for lh in 0..2 { e.get(b.into()).get(v + s).simd(base + lh).smem(0x0B, 3, h * 32 + lh * 16); }
                } else { e.get(b.into()).get(v + s).smem(0x0B, 2, h * 16); }
            }
        }
        Ok(())
    }
}

fn finish_body(cnt: u32, ty: u8, code: &[u8]) -> Vec<u8> {
    let mut b = WasmBuf::default();
    if cnt > 0 { b.op(1).uleb(cnt).op(ty) } else { b.op(0) }.ops(code).op(0x0B);
    b.0
}
fn emit_batch_loop(e: &mut WasmBuf, k: &Kernel, step: usize, body: impl FnOnce(&mut WasmBuf) -> Res<()>) -> Res<()> {
    let (n, (lim, cmp)) = (k.bufs.len() as u32, if step == 2 { (2, 0x48) } else { (0, 0x4C) });
    e.ops(&[0x02, 0x40, 0x03, 0x40]).get(n).i32c(lim).op(cmp).op_u(0x0D, 1);
    body(e)?;
    for (i, b) in k.bufs.iter().enumerate().filter(|(_, b)| !matches!(b.access, BufAccess::LookupTable(_))) {
        e.get(i as u32).i32c(b.block_bytes().saturating_mul(step) as i32).op(0x6A).set(i as u32);
    }
    e.get(n).i32c(step as i32).op(0x6B).set(n).op_u(0x0C, 0).ops(&[0x0B, 0x0B]);
    Ok(())
}
struct LoweredKernel { ks: String, bs: String, nb: usize, sb: Vec<u8>, bb: Vec<u8>, mode: WasmVectorMode }
fn lower_kernel_pair(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], single_fidx: u32) -> Res<LoweredKernel> {
    let (nb, n_loc) = (k.bufs.len(), k.bufs.len() as u32);
    let (mut sf, mut bf) = (WasmBuf::default(), WasmBuf::default());
    let (sb, bb, mode) = if let Some(plan) = try_plan_i32x4(k, sched, doms) {
        let nv = plan.steps.len() as u32;
        plan.emit_consts(&mut sf, n_loc);
        plan.emit_both(&mut sf, n_loc)?;
        plan.emit_consts(&mut bf, n_loc + 1);
        emit_batch_loop(&mut bf, k, 1, |e| plan.emit_both(e, n_loc + 1))?;
        (finish_body(nv, 0x7B, &sf.0), finish_body(nv, 0x7B, &bf.0), WasmVectorMode::HorizontalI32x4)
    } else {
        let (sl, sn) = alloc_ssa_locals(k, sched, n_loc)?;
        DagEmit { e: &mut sf, k, locs: &sl, mode: DagMode::Scalar(doms) }.emit(sched)?;
        let (sb, (bl, bn)) = (finish_body(sn, 0x7E, &sf.0), alloc_ssa_locals(k, sched, n_loc + 1)?);
        if can_plan_i64x2(k, sched) {
            let hb = n_loc + 1 + bn;
            let hc: Vec<(i64, u32)> = collect_hoisted_consts_i64(k, sched)
                .into_iter().enumerate().map(|(i, c)| (c, hb + i as u32)).collect();
            let sc = hb + hc.len() as u32;
            for &(c, loc) in &hc { bf.v128(c as u64).set(loc); }
            emit_batch_loop(&mut bf, k, 2, |e|
                DagEmit { e, k, locs: &bl, mode: DagMode::I64x2(&hc, sc, vec![false; k.nodes.len()]) }.emit(sched))?;
            bf.get(n_loc).i32c(0).ops(&[0x4A, 0x04, 0x40]);
            for idx in 0..nb { bf.get(idx as u32); }
            bf.op_u(0x10, single_fidx).op(0x0B);
            (sb, finish_body(bn + hc.len() as u32 + 1, 0x7B, &bf.0), WasmVectorMode::VerticalI64x2)
        } else {
            emit_batch_loop(&mut bf, k, 1, |e| DagEmit { e, k, locs: &bl, mode: DagMode::Scalar(doms) }.emit(sched))?;
            (sb, finish_body(bn, 0x7E, &bf.0), WasmVectorMode::ScalarI64)
        }
    };
    Ok(LoweredKernel { ks: format!("zero_cc_{}", k.name), bs: format!("zero_cc_{}_batch", k.name), nb, sb, bb, mode })
}
fn assemble_wasm_module(lowered: &[LoweredKernel], mem_pages: u32, imp: bool) -> Vec<u8> {
    let [mut ts, mut fs, mut es, mut cs] = std::array::from_fn(|_| WasmBuf::default());
    let fn_count = (lowered.len() * 2) as u32;
    for b in [&mut ts, &mut fs, &mut cs] { b.uleb(fn_count); }
    es.uleb(fn_count + u32::from(!imp));
    if !imp { es.export("memory", 2, 0); }
    for (idx, lk) in lowered.iter().enumerate() {
        let f = (idx * 2) as u32;
        for n in [lk.nb, lk.nb + 1] {
            ts.op_u(0x60, n as u32);
            ts.0.resize(ts.0.len() + n, 0x7F);
            ts.op(0x00);
        }
        fs.uleb(f).uleb(f + 1);
        es.export(&lk.ks, 0, f).export(&lk.bs, 0, f + 1);
        cs.len_bytes(&lk.sb).len_bytes(&lk.bb);
    }
    let (mut out, mut ms) = (WasmBuf::default(), WasmBuf::default());
    out.ops(&WASM_MAGIC_VERSION).sec(1, &ts.0);
    if imp { ms.op(1).len_bytes(b"env").len_bytes(b"memory").ops(&[2, 0]).uleb(mem_pages.max(1)); }
    else { ms.ops(&[1, 0]).uleb(mem_pages); }
    if imp { out.sec(2, &ms.0).sec(3, &fs.0) } else { out.sec(3, &fs.0).sec(5, &ms.0) }.sec(7, &es.0).sec(10, &cs.0);
    out.0
}
fn lower_one(k: &Kernel, budget: RegBudget, optimize: bool, fidx: u32) -> Res<LoweredKernel> {
    k.validate()?;
    let opt = if optimize { optimize_kernel(k)? } else { k.clone() };
    lower_kernel_pair(&opt, &schedule_and_verify(&opt, budget)?, &opt.infer_domains(), fidx)
}
pub fn compile_kernel_wasm_opts(k: &Kernel, b: RegBudget, opt: bool) -> Res<CompiledWasm> {
    let lk = lower_one(k, b, opt, 0)?;
    let wasm_bytes = assemble_wasm_module(std::slice::from_ref(&lk), WASM_DEFAULT_MEM_PAGES, false);
    let summary = verify_wasm_module(&wasm_bytes)?;
    Ok(CompiledWasm { kernel_sym: lk.ks, batch_sym: lk.bs, wasm_bytes, vector_mode: lk.mode, summary })
}
pub fn compile_kernel_wasm(k: &Kernel, b: RegBudget) -> Res<CompiledWasm> { compile_kernel_wasm_opts(k, b, true) }
pub fn compile_rust_to_wasm(src: &str, budget: RegBudget) -> Res<CompiledWasm> {
    compile_kernel_wasm(&parse_rust_kernel(src)?, budget)
}
pub fn compile_kernels_wasm_with_memory(ks: &[Kernel], b: RegBudget, imp: bool) -> Res<(Vec<u8>, WasmSummary)> {
    if ks.is_empty() { return Err(werr("cannot emit wasm module with zero kernels")); }
    let lks: Res<Vec<_>> = ks.iter().enumerate().map(|(i, k)| lower_one(k, b, true, (i * 2) as u32)).collect();
    let bytes = assemble_wasm_module(&lks?, if imp { 1 } else { WASM_DEFAULT_MEM_PAGES }, imp);
    let sum = verify_wasm_module(&bytes)?;
    Ok((bytes, sum))
}
pub fn compile_kernels_wasm(ks: &[Kernel], b: RegBudget) -> Res<(Vec<u8>, WasmSummary)> {
    compile_kernels_wasm_with_memory(ks, b, false)
}

struct WasmReader<'a> { b: &'a [u8], p: usize }
impl<'a> WasmReader<'a> {
    fn byte(&mut self) -> Res<u8> {
        let b = self.b.get(self.p).copied().ok_or_else(|| werr("wasm unexpected EOF"))?;
        self.p += 1;
        Ok(b)
    }
    fn slice(&mut self, len: usize, msg: &str) -> Res<&'a [u8]> {
        let end = self.p.checked_add(len).ok_or_else(|| werr(msg))?;
        let s = self.b.get(self.p..end).ok_or_else(|| werr(msg))?;
        self.p = end;
        Ok(s)
    }
    fn leb(&mut self, max: usize, u32_chk: bool) -> Res<u32> {
        let (mut val, mut sh) = (0u32, 0u32);
        for _ in 0..max {
            let b = self.byte()?;
            if u32_chk && sh == 28 && (b & 0xF0) != 0 { return Err(werr("wasm uleb32 overflow")); }
            if u32_chk { val |= u32::from(b & 0x7F) << sh; }
            sh += 7;
            if (b & 0x80) == 0 { return Ok(val); }
        }
        Err(werr("wasm leb exceeds max byte width"))
    }
    fn uleb(&mut self) -> Res<u32> { self.leb(5, true) }
    fn vty(&mut self) -> Res<()> { if matches!(self.byte()?, 0x7B..=0x7F) { Ok(()) } else { Err(werr("bad valtype")) } }
    fn vtys(&mut self) -> Res<()> { (0..self.uleb()?).try_for_each(|_| self.vty()) }
    fn name(&mut self) -> Res<String> {
        let n = self.uleb()? as usize;
        std::str::from_utf8(self.slice(n, "wasm name truncated")?).map(str::to_string).map_err(|_| werr("wasm UTF-8"))
    }
    fn memarg(&mut self, max: u32) -> Res<()> {
        if self.uleb()? > max { Err(werr("wasm memarg align exceeds width")) } else { self.uleb().map(|_| ()) }
    }
    fn mem_limits(&mut self) -> Res<u32> {
        let (f, min) = (self.byte()?, self.uleb()?);
        if f == 0 || (f == 1 && self.uleb()? >= min) { Ok(min) } else { Err(werr("wasm invalid memory limits")) }
    }
    fn skip_op(&mut self, simd: bool, op: u32) -> Res<()> {
        match (simd, op) {
            (false, 0x00 | 0x01 | 0x0F | 0x1A | 0x1B | 0x45..=0x6C | 0x71..=0x7E | 0x83..=0x8A | 0xA7..=0xC4)
            | (true, 0x0F..=0x14 | 0x23..=0x53 | 0x5C..=0xFF) => Ok(()),
            (false, 0x20..=0x24) => self.uleb().map(|_| ()),
            (false, 0x2C | 0x2D | 0x30 | 0x31 | 0x3A | 0x3C) | (true, 0x07) => self.memarg(0),
            (false, 0x2E | 0x2F | 0x32 | 0x33 | 0x3B | 0x3D) | (true, 0x08) => self.memarg(1),
            (false, 0x28 | 0x34..=0x36 | 0x3E) | (true, 0x09) => self.memarg(2),
            (false, 0x29 | 0x37) | (true, 0x01..=0x06 | 0x0A) => self.memarg(3),
            (true, 0x00 | 0x0B) => self.memarg(4),
            (false, 0x41 | 0x42) => self.leb(if op == 0x41 { 5 } else { 10 }, false).map(|_| ()),
            (true, 0x0C | 0x0D) => self.slice(16, "wasm v128 immediate truncated").map(|_| ()),
            (true, 0x15..=0x22) => self.byte().map(|_| ()),
            (true, 0x54..=0x5B) => {
                self.memarg((op - 0x54) & 3)?;
                if self.byte()? > if op & 3 == 3 { 1 } else { u8::MAX } { Err(werr("v128 lane OOB")) } else { Ok(()) }
            }
            _ => Err(werr(format!("wasm unsupported opcode 0x{op:x}"))),
        }
    }
}
fn verify_code_body(body: &[u8], func_count: usize, tot: &mut usize, simd: &mut usize, div: &mut usize) -> Res<()> {
    let mut r = WasmReader { b: body, p: 0 };
    (0..r.uleb()?).try_for_each(|_| r.uleb().and_then(|_| r.vty()))?;
    let mut depth = 1usize;
    while r.p < body.len() {
        let op = r.byte()?;
        *tot += 1;
        match op {
            0x02..=0x04 => if matches!(r.byte()?, 0x40 | 0x7B..=0x7F) { depth += 1 }
                else { return Err(werr("wasm invalid blocktype")); },
            0x05 => if depth < 2 { return Err(werr("wasm stray else outside if block")); },
            0x0B => {
                depth = depth.checked_sub(1).ok_or_else(|| werr("wasm unbalanced end"))?;
                if depth == 0 && r.p != body.len() { return Err(werr("wasm trailing bytes after function end")); }
            }
            0x0C | 0x0D => if (r.uleb()? as usize) >= depth { return Err(werr("wasm branch label exceeds depth")); },
            0x10 => if (r.uleb()? as usize) >= func_count { return Err(werr("wasm call target out of bounds")); },
            0x6D..=0x70 | 0x7F..=0x82 => *div += 1,
            0xFD => {
                *simd += 1;
                let sub = r.uleb()?;
                r.skip_op(true, sub)?;
            }
            _ => r.skip_op(false, u32::from(op))?,
        }
    }
    if depth != 0 { return Err(werr("wasm unbalanced block/loop/if without closing end")); }
    Ok(())
}
pub fn verify_wasm_module(bytes: &[u8]) -> Res<WasmSummary> {
    if bytes.get(0..8) != Some(&WASM_MAGIC_VERSION) { return Err(werr("bad wasm magic or version")); }
    let mut top = WasmReader { b: bytes, p: 8 };
    let (mut prev, mut nt, mut nf, mut nm, mut nc, mut mp, mut imp) = (0u8, 0, 0, 0, 0, 0u32, false);
    let (mut tot, mut simd, mut div, mut exports) = (0, 0, 0, Vec::new());
    while top.p < bytes.len() {
        let (sid, slen) = (top.byte()?, top.uleb()? as usize);
        if sid != 0 && (sid <= prev || sid > 12) { return Err(werr("wasm out-of-order or duplicate section id")); }
        if sid != 0 { prev = sid; }
        let payload = top.slice(slen, "wasm section truncated")?;
        let mut r = WasmReader { b: payload, p: 0 };
        match sid {
            1 => {
                nt = r.uleb()? as usize;
                (0..nt).try_for_each(|_| if r.byte()? != 0x60 { Err(werr("wasm bad functype")) }
                    else { (0..2).try_for_each(|_| r.vtys()) })?;
            }
            2 | 5 => {
                let n = r.uleb()? as usize;
                if sid == 5 && (imp || n == 0) { return Err(werr("wasm invalid memory section")); }
                for _ in 0..n {
                    if sid == 2 && (r.name()? != "env" || r.name()? != "memory" || r.byte()? != 2) {
                        return Err(werr("wasm import section only permits env.memory"));
                    }
                    let p = r.mem_limits()?;
                    if nm == 0 { mp = p; }
                    nm += 1;
                }
                imp |= sid == 2 && n > 0;
            }
            3 => {
                nf = r.uleb()? as usize;
                (0..nf).try_for_each(|_| if (r.uleb()? as usize) < nt { Ok(()) } else { Err(werr("func type OOB")) })?;
            }
            7 => for _ in 0..r.uleb()? {
                let (name, k, i) = (r.name()?, r.byte()?, r.uleb()?);
                if !((k == 0 && (i as usize) < nf) || (k == 2 && (i as usize) < nm) || matches!(k, 1 | 3)) {
                    return Err(werr("wasm export index out of range"));
                }
                exports.push(WasmExport { name, kind: k, index: i });
            },
            10 => {
                nc = r.uleb()? as usize;
                if nc != nf { return Err(werr("wasm code body count mismatches functions")); }
                for _ in 0..nc {
                    let blen = r.uleb()? as usize;
                    verify_code_body(r.slice(blen, "wasm code body truncated")?, nf, &mut tot, &mut simd, &mut div)?;
                }
            }
            0 => r.p = payload.len(),
            _ => return Err(werr(format!("wasm unexpected section id {sid}"))),
        }
        if r.p != payload.len() { return Err(werr("wasm section payload length mismatch")); }
    }
    if nt == 0 || nf == 0 || nm == 0 || nc != nf { return Err(werr("wasm module missing mandatory section")); }
    if !imp && !exports.iter().any(|e| e.kind == 2 && e.name == "memory") { return Err(werr("wasm no memory export")); }
    Ok(WasmSummary {
        type_count: nt, func_count: nf, mem_min_pages: mp, exports, code_body_count: nc, total_insns: tot,
        simd_insns: simd, div_rem_insns: div, has_simd128: simd > 0, imports_memory: imp,
    })
}
pub fn verify_wasm_module_with_opts(bytes: &[u8], imp: bool) -> Res<WasmSummary> {
    let s = verify_wasm_module(bytes)?;
    if s.imports_memory == imp { Ok(s) }
    else { Err(werr(if imp { "wasm expected imported env.memory" } else { "wasm unexpectedly imported memory" })) }
}
