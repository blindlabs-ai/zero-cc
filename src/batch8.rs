//! Interval-driven `u32x8` narrowing and split-width `(lo32, hi32)` AVX2 batch backend.
use crate::codegen::ScheduleReport;
use crate::ir::{BufAccess, CmpKind as C, CompileError, Domain, ElemTy, Kernel, NodeId, Op};
use crate::isa::{RegAlloc, X86Emit};
const VEX_LD: [u8; 6] = [1, 1, 2, 0, 0, 0x6F];
const VEX_ST: [u8; 6] = [1, 1, 2, 0, 0, 0x7F];
const SHIFT_SUB: [u8; 4] = [6, 2, 4, 0];
const LD_N: [(ElemTy, u8); 3] = [(ElemTy::U8, 0x31), (ElemTy::U16, 0x33), (ElemTy::I16, 0x23)];
const BIN_VEX: [(u8, u8); 16] = [
    (1, 0xFE), (1, 0xFA), (2, 0x40), (1, 0xDB), (1, 0xDF), (1, 0xEB), (1, 0xEF), (2, 0x3B),
    (2, 0x3F), (1, 0x76), (1, 0x66), (2, 0x47), (2, 0x45), (0, 0), (0, 0), (0, 0),
];
const CMP_U: [C; 10] = [C::Eq, C::Ne, C::Ult, C::Ule, C::Ugt, C::Uge, C::Ult, C::Ule, C::Ugt, C::Uge];
const CMP_FLIP: [C; 10] = [C::Eq, C::Ne, C::Gt, C::Ge, C::Lt, C::Le, C::Ugt, C::Uge, C::Ult, C::Ule];
const CMP_NOT: [C; 10] = [C::Ne, C::Eq, C::Ge, C::Gt, C::Le, C::Lt, C::Uge, C::Ugt, C::Ule, C::Ult];
fn ct(tbl: &[C; 10], k: C) -> C { tbl.get(k as usize).copied().unwrap_or(k) }
fn is64(ty: ElemTy) -> bool { matches!(ty, ElemTy::U64 | ElemTy::I64) }
fn b8_err(m: &str) -> CompileError { CompileError::BackendError(m.into()) }
fn put<T>(v: &mut [T], i: usize, x: T) { if let Some(s) = v.get_mut(i) { *s = x } }
fn clr(flags: &mut [bool], id: usize) -> bool { flags.get_mut(id).is_some_and(|m| std::mem::replace(m, false)) }
fn bump(n: &mut usize) -> usize { std::mem::replace(n, n.saturating_add(1)) }
fn dec0(x: &mut usize) -> bool { std::mem::replace(x, x.saturating_sub(1)) <= 1 }
fn sym_c(k: &Kernel, i: usize) -> Option<u64> { if let &Op::Const(c) = k.nodes.get(i)? { Some(c as _) } else { None } }
fn pw<T>(a: usize, b: usize, mut f: impl FnMut(usize) -> Option<T>) -> Option<(usize, T)> {
    f(b).map(|v| (a, v)).or_else(|| f(a).map(|v| (b, v))) }
fn pc32(k: &Kernel, a: usize, b: usize) -> Option<(usize, u32)> { pw(a, b, |c| u32::try_from(sym_c(k, c)?).ok()) }
fn ld_buf(k: &Kernel, id: NodeId) -> Option<u8> {
    let &Op::Load { buf, mem, .. } = k.nodes.get(id)? else { return None };
    (k.nodes.get(mem) == Some(&Op::MemInit)).then_some(buf) }
#[derive(Clone, Copy, PartialEq, Eq)] enum MulHi { General, Pos01, NegMask }
#[derive(Clone, Copy, PartialEq, Eq)] enum ShOp { Shl, Shr, Sar }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bin { Add, Sub, Mul, And, Andn, Or, Xor, MinU, MaxU, CmpEq, CmpGtS, ShlVar, ShrVar }
#[derive(Clone, Copy, Debug)] struct NetGuard { net: u8, pref: u8, rej: Option<u32> }
#[derive(Clone, Copy, Debug)] pub(crate) struct B8FastPath {
    addr: u8, desc: u8, out: u8, addr_zm: u64, dhi_zm: u32, dlo_max: u32, ng: Option<NetGuard>, ao: [(u64, u64); 2], }
#[derive(Clone, Copy)] enum Step {
    LdN(usize, u8, ElemTy), Ld64(usize, usize, u8), Bin(Bin, usize, usize, usize), Sh(ShOp, usize, usize, u8),
    Bld(usize, usize, usize, usize), MulHi(usize, usize, Option<usize>, usize, u8, MulHi),
    Pop64(usize, usize, Option<usize>, usize, usize), St1(usize, Option<usize>, u8, ElemTy), St2([usize; 4], u8), }
impl Step {
    fn defs(&self) -> ([usize; 2], usize) { match *self {
        Self::Ld64(lo, hi, _) => ([lo, hi], 2),
        Self::St1(..) | Self::St2(..) => ([0, 0], 0),
        Self::LdN(d, ..) | Self::Bin(_, d, ..) | Self::Sh(_, d, ..) | Self::Bld(d, ..)
        | Self::MulHi(d, ..) | Self::Pop64(d, ..) => ([d, 0], 1), }}
    fn uses(&self) -> ([usize; 4], usize) { match *self {
        Self::LdN(..) | Self::Ld64(..) => ([0; 4], 0),
        Self::Sh(_, _, s, _) => ([s, 0, 0, 0], 1),
        Self::Bin(_, _, a, b) => ([a, b, 0, 0], 2),
        Self::Bld(_, f, t, m) => ([f, t, m, 0], 3),
        Self::MulHi(_, lo, hi, c, ..) => ([lo, c, hi.unwrap_or(0), 0], 2 + usize::from(hi.is_some())),
        Self::Pop64(_, lo, hi, m, mul) => ([lo, m, mul, hi.unwrap_or(0)], 3 + usize::from(hi.is_some())),
        Self::St1(lo, hi, ..) => ([lo, hi.unwrap_or(0), 0, 0], 1 + usize::from(hi.is_some())),
        Self::St2(vals, _) => (vals, 4), }}
    fn reuse_dying(&self) -> bool { !matches!(self, Self::MulHi(..) | Self::Pop64(..)) } }
#[derive(Clone, Copy)] enum Val { Const(u64), Lo32(usize), Hi32(usize), Mask32(usize), Split64(usize, usize) }
fn vz(v: Val, hi: bool) -> bool { match v {
    Val::Const(c) => if hi { (c >> 32) == 0 } else { (c as u32) == 0 },
    Val::Lo32(_) | Val::Mask32(_) => hi,
    Val::Hi32(_) => !hi,
    Val::Split64(..) => false, }}
pub(crate) struct Batch8Plan {
    consts_v: Vec<(usize, u32)>, pub(crate) hoisted: Vec<(u8, u32)>,
    vreg_phys: Vec<Option<u8>>, steps: Vec<Step>, pub(crate) fastpath: Option<B8FastPath>, }
fn analyze_bool_masks(k: &Kernel, sched: &ScheduleReport) -> Vec<bool> {
    let (is01, mut ok) = (|x| matches!(k.nodes.get(x), Some(Op::Const(0 | 1))), vec![false; k.nodes.len()]);
    for &id in &sched.order { ok[id] = match k.nodes.get(id).copied() {
        Some(Op::Cmp(..)) => true,
        Some(Op::BitAnd(a, b) | Op::BitOr(a, b)) => ok[a] && ok[b],
        Some(Op::Select { t_val: t, f_val: f, .. }) => (ok[t] || is01(t)) && (ok[f] || is01(f)),
        _ => false, }}
    for o in &k.outputs { clr(&mut ok, o.val); }
    while sched.order.iter().fold(false, |chg, &id| chg | match k.nodes.get(id).copied() {
        Some(Op::BitAnd(a, b) | Op::BitOr(a, b)) if ok[id] => (!ok[a] || !ok[b]) && clr(&mut ok, id),
        Some(Op::Select { t_val: t, f_val: f, .. }) if ok[id] =>
            ((!is01(t) && !ok[t]) || (!is01(f) && !ok[f])) && clr(&mut ok, id),
        Some(Op::Select { t_val: t, f_val: f, .. }) => clr(&mut ok, t) | clr(&mut ok, f),
        Some(op) => op.children().0[..op.children().1].iter().fold(false, |c, &kid| c | clr(&mut ok, kid)),
        None => false, }) {}
    ok }
fn analyze_demanded_lo32(k: &Kernel, sched: &ScheduleReport) -> Vec<bool> {
    let (mut hi, u32c) = (vec![false; k.nodes.len()], |x| sym_c(k, x).is_some_and(|c| c <= u32::MAX as u64));
    for o in &k.outputs {
        if k.bufs.get(o.buf as usize).map_or(true, |b| is64(b.elem_ty)) { put(&mut hi, o.val, true) }}
    for &id in sched.order.iter().rev() { match (k.nodes.get(id).copied(), hi[id]) {
        (Some(Op::BitAnd(a, b)), true) => (hi[a], hi[b]) = (hi[a] | !u32c(b), hi[b] | !u32c(a)),
        (Some(Op::Shl(a, 0..=31) | Op::ShlVar(a, _)), true)
        | (Some(Op::Ushr(a, _) | Op::Shr(a, _) | Op::Popcnt(a) | Op::Tzcnt(a) | Op::ShrVarU(a, _)), _) => hi[a] = true,
        (Some(Op::Select { cond, t_val: t, f_val: f }), n) => (hi[cond], hi[t], hi[f]) = (true, hi[t] | n, hi[f] | n),
        (Some(Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b)), true)
        | (Some(Op::Cmp(_, a, b)), _) => (hi[a], hi[b]) = (true, true),
        _ => {} }}
    hi.into_iter().map(|h| !h).collect() }
fn find_fused_barrett_muls(k: &Kernel, sched: &ScheduleReport, dm: &[Domain]) -> Vec<bool> {
    let (mut uses, mut skip) = (vec![0usize; k.nodes.len()], vec![false; k.nodes.len()]);
    for (ks, n) in sched.order.iter().filter_map(|&id| k.nodes.get(id).copied().map(Op::children)) {
        for &kid in &ks[..n] { if let Some(u) = uses.get_mut(kid) { *u += 1 } } }
    for o in &k.outputs { if let Some(u) = uses.get_mut(o.val) { *u += 1 } }
    for &id in &sched.order {
        let Some(&Op::Shr(a, 32..) | &Op::Ushr(a, 32..)) = k.nodes.get(id) else { continue };
        let Some(&Op::Mul(x, c)) = k.nodes.get(a).filter(|_| uses[a] == 1 && dm[a].is_nonneg()) else { continue };
        if pc32(k, x, c).is_some_and(|(v, _)| dm[v].is_nonneg()) { skip[a] = true } }
    skip }
struct B8Builder<'a> {
    k: &'a Kernel, dm: &'a [Domain], ism: &'a [bool], dlo: &'a [bool],
    vals: Vec<Option<Val>>, steps: Vec<Step>, consts: Vec<(usize, u32)>, next_v: usize, }
impl<'a> B8Builder<'a> {
    fn val(&self, i: NodeId) -> Option<Val> { self.vals.get(i).copied().flatten() }
    fn dom(&self, i: NodeId) -> Domain { self.dm.get(i).copied().unwrap_or_else(Domain::top) }
    fn is_dlo(&self, i: NodeId) -> bool { self.dlo.get(i).copied().unwrap_or(false) }
    fn is_m(&self, i: NodeId) -> bool { self.ism.get(i).copied().unwrap_or(false) }
    fn fits_lo(&self, i: NodeId) -> bool { self.is_dlo(i) || self.dom(i).fits_u32() }
    fn p1(&mut self, f: impl FnOnce(usize) -> Step) -> usize {
        let d = bump(&mut self.next_v);
        self.steps.push(f(d));
        d }
    fn c32(&mut self, val: u32) -> usize { b8_ensure_const(&mut self.consts, &mut self.next_v, val) }
    fn bin(&mut self, op: Bin, a: usize, b: usize) -> usize { self.p1(|d| Step::Bin(op, d, a, b)) }
    fn bin_c(&mut self, op: Bin, a: usize, c: u32) -> usize {
        let cv = self.c32(c);
        self.bin(op, a, cv) }
    fn bin_h(&mut self, op: Bin, va: Val, vb: Val, hi: bool) -> usize {
        let (ra, rb) = (self.h(va, hi), self.h(vb, hi));
        self.bin(op, ra, rb) }
    fn sh(&mut self, o: ShOp, x: usize, s: u8) -> usize { if s == 0 { x } else { self.p1(|d| Step::Sh(o, d, x, s)) } }
    fn sh_h(&mut self, op: ShOp, v: Val, hi: bool, s: u8) -> usize {
        let r = self.h(v, hi);
        self.sh(op, r, s) }
    fn reg(&mut self, v: usize) -> usize {
        if self.consts.iter().any(|&(c, _)| c == v) { self.p1(|d| Step::Sh(ShOp::Shl, d, v, 0)) } else { v } }
    fn inv(&mut self, m: usize) -> usize { self.bin_c(Bin::Xor, m, u32::MAX) }
    fn h(&mut self, v: Val, hi: bool) -> usize { match (v, hi) {
        (Val::Const(c), _) => self.c32(if hi { (c >> 32) as u32 } else { c as u32 }),
        (Val::Lo32(r) | Val::Split64(r, _), false) | (Val::Hi32(r) | Val::Split64(_, r), true) => r,
        (Val::Mask32(m), false) => self.sh(ShOp::Shr, m, 31),
        _ => self.c32(0), }}
    fn vmask(&mut self, v: Val) -> usize {
        let r = match v {
            Val::Mask32(m) => return m,
            Val::Const(c) => return self.c32(if c != 0 { u32::MAX } else { 0 }),
            Val::Lo32(r) | Val::Hi32(r) => r,
            Val::Split64(lo, hi) => self.bin(Bin::Or, lo, hi), };
        let eq = self.bin_c(Bin::CmpEq, r, 0);
        self.inv(eq) }
    fn pop(&mut self, xl: usize, xh: Option<usize>) -> usize {
        let (lo, hi, mf, m1) = (self.reg(xl), xh.map(|h| self.reg(h)), self.c32(0x0F0F_0F0F), self.c32(0x0101_0101));
        self.p1(|d| Step::Pop64(d, lo, hi, mf, m1)) }
    fn mul_hi(&mut self, vx: Val, c_val: u32, s: u8, mode: MulHi) -> (usize, usize, usize) {
        let (lo, hi, c) = (self.h(vx, false), (!vz(vx, true)).then(|| self.h(vx, true)), self.c32(c_val));
        (lo, c, self.p1(|d| Step::MulHi(d, lo, hi, c, s, mode))) }
    fn cmp32(&mut self, k: C, a: usize, vb: Val, hi: bool) -> usize {
        let (k2, b) = match vb {
            Val::Const(raw) => match (k, if hi { (raw >> 32) as u32 } else { raw as u32 }) {
                (C::Ult, 0) | (C::Ugt, u32::MAX) => return self.c32(0),
                (C::Uge, 0) | (C::Ule, u32::MAX) => return self.c32(u32::MAX),
                (C::Ult, 1) | (C::Ule, 0) => (C::Eq, self.c32(0)),
                (C::Uge, 1) | (C::Ugt, 0) => (C::Ne, self.c32(0)),
                (C::Ult, c) => (C::Ule, self.c32(c - 1)),
                (C::Ugt, c) => (C::Uge, self.c32(c + 1)),
                (_, c) => (k, self.c32(c)), },
            _ => (k, self.h(vb, hi)), };
        let neg = matches!(k2, C::Ne | C::Ult | C::Ugt | C::Le | C::Ge);
        let m = match if neg { ct(&CMP_NOT, k2) } else { k2 } {
            C::Eq => self.bin(Bin::CmpEq, a, b),
            C::Gt => self.bin(Bin::CmpGtS, a, b),
            C::Lt => self.bin(Bin::CmpGtS, b, a),
            k0 => {
                let mm = self.bin(if k0 == C::Ule { Bin::MinU } else { Bin::MaxU }, a, b);
                self.bin(Bin::CmpEq, mm, a) }};
        if neg { self.inv(m) } else { m } }
    fn cmp_split64_u(&mut self, k: C, va: Val, vb: Val) -> usize {
        let (la, ha) = (self.h(va, false), self.h(va, true));
        if matches!(k, C::Eq | C::Ne) {
            let (m1, m2) = (self.cmp32(C::Eq, ha, vb, true), self.cmp32(C::Eq, la, vb, false));
            let eq = self.bin(Bin::And, m1, m2);
            return if k == C::Ne { self.inv(eq) } else { eq };}
        if let Val::Const(c) = vb {
            if (c as u32) == 0 && matches!(k, C::Uge | C::Ult) { return self.cmp32(k, ha, vb, true); }
            if (c >> 32) == 0 && matches!(k, C::Ult | C::Ule) {
                let (hz, lm) = (self.bin_c(Bin::CmpEq, ha, 0), self.cmp32(k, la, vb, false));
                return self.bin(Bin::And, hz, lm);}}
        let hk = if matches!(k, C::Ult | C::Ule) { C::Ult } else { C::Ugt };
        let (hs, q, l) = (self.cmp32(hk, ha, vb, true), self.cmp32(C::Eq, ha, vb, true), self.cmp32(k, la, vb, false));
        let eq_ok = self.bin(Bin::And, q, l);
        self.bin(Bin::Or, hs, eq_ok) }
    fn lower_add_sub(&mut self, id: NodeId, a: NodeId, b: NodeId, s: bool) -> Option<Val> {
        let (va, vb) = (self.val(a)?, self.val(b)?);
        let (la, lb, op) = (self.h(va, false), self.h(vb, false), if s { Bin::Sub } else { Bin::Add });
        let (lo, za, zb) = (self.bin(op, la, lb), vz(va, true), vz(vb, true));
        if self.fits_lo(id) && za && zb { return Some(Val::Lo32(lo)); }
        let pos = |v| if let Val::Const(c) = v { (c > 0 && c <= u32::MAX as u64).then_some(c) } else { None };
        let c_opt = if s { pos(vb) } else { pos(vb).or_else(|| pos(va)) };
        let (lhs, rhs) = if s { (la, lb) } else { (lo, la) };
        let cb = self.cmp32(C::Ult, lhs, c_opt.map_or(Val::Lo32(rhs), Val::Const), false);
        let hb = if zb { self.h(va, true) } else if !s && za { self.h(vb, true) } else { self.bin_h(op, va, vb, true) };
        Some(Val::Split64(lo, self.bin(if s { Bin::Add } else { Bin::Sub }, hb, cb))) }
    fn bh(&mut self, bop: Bin, va: Val, vb: Val, hi: bool) -> Option<usize> {
        let (za, zb) = (vz(va, hi), vz(vb, hi));
        let ones = |v| matches!(v, Val::Const(c) if (if hi { c >> 32 } else { c }) as u32 == u32::MAX);
        if (bop == Bin::And && (za || zb)) || (za && zb) { return None; }
        if (bop == Bin::And && ones(vb)) || zb { return Some(self.h(va, hi)); }
        if (bop == Bin::And && ones(va)) || za { return Some(self.h(vb, hi)); }
        Some(self.bin_h(bop, va, vb, hi)) }
    fn lower_bw(&mut self, id: NodeId, bop: Bin, a: NodeId, b: NodeId) -> Option<Val> {
        let (va, vb) = (self.val(a)?, self.val(b)?);
        if self.is_m(id) {
            let (ma, mb) = (self.vmask(va), self.vmask(vb));
            return Some(Val::Mask32(self.bin(bop, ma, mb)));}
        Some(match (self.bh(bop, va, vb, false), (!self.is_dlo(id)).then(|| self.bh(bop, va, vb, true)).flatten()) {
            (None, None) => Val::Const(0),
            (Some(l), None) => Val::Lo32(l),
            (None, Some(h)) => Val::Hi32(h),
            (Some(l), Some(h)) => Val::Split64(l, h), }) }
    fn lower_cmp(&mut self, id: NodeId, kind: C, a: NodeId, b: NodeId) -> Option<Val> {
        let (mut va, mut vb) = (self.val(a)?, self.val(b)?);
        let mut e = if self.dom(a).is_nonneg() && self.dom(b).is_nonneg() { ct(&CMP_U, kind) } else { kind };
        let u_or_eq = e.is_unsigned() || matches!(e, C::Eq | C::Ne);
        if matches!(va, Val::Const(_)) && !matches!(vb, Val::Const(_)) { (va, vb, e) = (vb, va, ct(&CMP_FLIP, e)) }
        let mask = if vz(va, true) && vz(vb, true) {
            if !u_or_eq && (!self.dom(a).fits_i32() || !self.dom(b).fits_i32()) { return None; }
            let la = self.h(va, false);
            self.cmp32(e, la, vb, false)
        } else if u_or_eq { self.cmp_split64_u(e, va, vb) } else { return None };
        Some(if self.is_m(id) { Val::Mask32(mask) } else { Val::Lo32(self.sh(ShOp::Shr, mask, 31)) }) }
    fn lower_select(&mut self, id: NodeId, c: NodeId, t: NodeId, f: NodeId) -> Option<Val> {
        let (vc, vt, vf) = (self.val(c)?, self.val(t)?, self.val(f)?);
        let (cm, m, lo1) = (self.vmask(vc), self.is_m(id), (vz(vt, true) && vz(vf, true)) || self.is_dlo(id));
        let mut sel = |hi: bool| {
            let (rt, rf) = if m { (self.vmask(vt), self.vmask(vf)) } else { (self.h(vt, hi), self.h(vf, hi)) };
            let z = |v| if m { matches!(v, Val::Const(0)) } else { vz(v, hi) };
            if z(vf) { self.bin(Bin::And, cm, rt) } else if z(vt) { self.bin(Bin::Andn, cm, rf) }
            else { self.p1(|d| Step::Bld(d, rf, rt, cm)) } };
        let dl = sel(false);
        Some(if m { Val::Mask32(dl) } else if lo1 { Val::Lo32(dl) } else { Val::Split64(dl, sel(true)) }) }
    fn lower_node(&mut self, id: NodeId, skip_mul: &[bool]) -> Option<()> {
        if skip_mul.get(id).copied().unwrap_or(false) { return Some(()); }
        let op = *self.k.nodes.get(id)?;
        let val = match op {
            Op::MemInit | Op::Store { .. } => Val::Const(0),
            Op::Const(c) => Val::Const(c as u64),
            Op::Load { .. } => {
                let buf = ld_buf(self.k, id)?;
                let bd = self.k.bufs.get(buf as usize).filter(|b| b.lanes == 1)?;
                if is64(bd.elem_ty) {
                    let hi = bump(&mut self.next_v);
                    Val::Split64(self.p1(|lo| Step::Ld64(lo, hi, buf)), hi)
                } else {
                    let lo = self.p1(|d| Step::LdN(d, buf, bd.elem_ty));
                    let sgn = matches!(bd.elem_ty, ElemTy::I32 | ElemTy::I16) && !bd.domain.is_nonneg();
                    if sgn && !self.is_dlo(id) { Val::Split64(lo, self.sh(ShOp::Sar, lo, 31)) }
                    else { Val::Lo32(lo) }}}
            Op::Add(a, b) | Op::Sub(a, b) => self.lower_add_sub(id, a, b, matches!(op, Op::Sub(..)))?,
            Op::Mul(a, b) => {
                let (va, vb) = (self.val(a)?, self.val(b)?);
                if self.is_dlo(id) || [id, a, b].into_iter().all(|x| self.dom(x).fits_u32()) {
                    Val::Lo32(self.bin_h(Bin::Mul, va, vb, false))
                } else if let Some((xi, cv)) = pc32(self.k, a, b) {
                    let (lo, c, hi) = self.mul_hi(self.val(xi)?, cv, 0, MulHi::General);
                    Val::Split64(self.bin(Bin::Mul, lo, c), hi)
                } else { return None } }
            Op::Popcnt(a) => match self.val(a)? {
                Val::Const(c) => Val::Const(u64::from(c.count_ones())),
                va => {
                    let ho = vz(va, false) && !vz(va, true);
                    let (lo, hi) = (self.h(va, ho), (!ho && !vz(va, true)).then(|| self.h(va, true)));
                    Val::Lo32(self.pop(lo, hi)) }},
            Op::Tzcnt(a) => match self.val(a)? {
                Val::Const(c) => Val::Const(u64::from(c.trailing_zeros())),
                va => {
                    let lo = self.h(va, false);
                    let (xm1, bor) = (self.bin_c(Bin::Sub, lo, 1), self.bin_c(Bin::CmpEq, lo, 0));
                    let (tl, th) = (self.bin(Bin::Andn, lo, xm1), if vz(va, true) { bor } else {
                        let hi = self.h(va, true);
                        let xh = self.bin(Bin::Add, hi, bor);
                        self.bin(Bin::Andn, hi, xh) });
                    Val::Lo32(self.pop(tl, Some(th))) }},
            Op::BitAnd(a, b) => self.lower_bw(id, Bin::And, a, b)?,
            Op::BitOr(a, b) => self.lower_bw(id, Bin::Or, a, b)?,
            Op::BitXor(a, b) => self.lower_bw(id, Bin::Xor, a, b)?,
            Op::Shl(a, s) => {
                let va = self.val(a)?;
                let la = self.h(va, false);
                if s == 0 { va } else if s >= 32 { Val::Hi32(self.sh(ShOp::Shl, la, s - 32)) }
                else if self.fits_lo(id) { Val::Lo32(self.sh(ShOp::Shl, la, s)) } else {
                    let (lo, cy) = (self.sh(ShOp::Shl, la, s), self.sh(ShOp::Shr, la, 32 - s));
                    let hi = if vz(va, true) { cy } else {
                        let hp = self.sh_h(ShOp::Shl, va, true, s);
                        self.bin(Bin::Or, hp, cy) };
                    Val::Split64(lo, hi) }}
            Op::Ushr(a, s) | Op::Shr(a, s) => {
                if matches!(op, Op::Shr(..)) && !self.dom(a).is_nonneg() { return None; }
                if s >= 32 && skip_mul.get(a).copied().unwrap_or(false) {
                    let Op::Mul(mx, mc) = *self.k.nodes.get(a)? else { return None };
                    let (xi, cv) = pc32(self.k, mx, mc)?;
                    let m = if self.dom(xi).hi < (1i64 << 33) { MulHi::Pos01 } else { MulHi::General };
                    Val::Lo32(self.mul_hi(self.val(xi)?, cv, s - 32, m).2)
                } else {
                    let va = self.val(a)?;
                    if s == 0 { va } else if s >= 32 { Val::Lo32(self.sh_h(ShOp::Shr, va, true, s - 32)) }
                    else if vz(va, true) { Val::Lo32(self.sh_h(ShOp::Shr, va, false, s)) } else {
                        let (la, ha) = (self.h(va, false), self.h(va, true));
                        let (lp, hp) = (self.sh(ShOp::Shr, la, s), self.sh(ShOp::Shl, ha, 32 - s));
                        let lo = self.bin(Bin::Or, lp, hp);
                        if self.fits_lo(id) { Val::Lo32(lo) } else { Val::Split64(lo, self.sh(ShOp::Shr, ha, s)) }}}}
            Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                let (sl, wlo) = (matches!(op, Op::ShlVar(..)), self.fits_lo(id));
                if sl && !wlo { return None; }
                let (va, vb) = (self.val(a)?, self.val(b)?);
                let (la, lb) = (self.h(va, false), self.h(vb, false));
                let s = if self.dom(b).is_nonneg() && self.dom(b).hi < 64 { lb } else { self.bin_c(Bin::And, lb, 63) };
                if sl || vz(va, true) { Val::Lo32(self.bin(if sl { Bin::ShlVar } else { Bin::ShrVar }, la, s)) } else {
                    let (ha, c32) = (self.h(va, true), self.c32(32));
                    let (si, sh_hi) = (self.bin(Bin::Sub, c32, s), self.bin(Bin::Sub, s, c32));
                    let (hl, hr) = (self.bin(Bin::ShlVar, ha, si), self.bin(Bin::ShrVar, ha, sh_hi));
                    let hp = self.bin(Bin::Or, hl, hr);
                    let lo = if vz(va, false) { hp } else {
                        let lp = self.bin(Bin::ShrVar, la, s);
                        self.bin(Bin::Or, lp, hp) };
                    if wlo { Val::Lo32(lo) } else { Val::Split64(lo, self.bin(Bin::ShrVar, ha, s)) } }}
            Op::Cmp(kind, a, b) => self.lower_cmp(id, kind, a, b)?,
            Op::Select { cond, t_val, f_val } => self.lower_select(id, cond, t_val, f_val)?,
            _ => return None, };
        put(&mut self.vals, id, Some(val));
        Some(()) } }
type B8RegAlloc = (Vec<Option<u8>>, Vec<(u8, u32)>);
fn alloc_regs(steps: &[Step], consts: &[(usize, u32)], next_v: usize, max_vecs: usize) -> Option<B8RegAlloc> {
    let (mut last_use, mut uses) = (vec![0usize; next_v], vec![0usize; next_v]);
    for (i, st) in steps.iter().enumerate() {
        let (us, nu) = st.uses();
        for &u in &us[..nu] { if u < next_v { (last_use[u], uses[u]) = (i, uses[u] + 1) } } }
    let pool = max_vecs.min(13) as u8;
    let (mut free, mut active): (Vec<u8>, Vec<(usize, u8)>) = ((0..pool).rev().collect(), Vec::new());
    let (mut phys, mut used_mask) = (vec![None; next_v], 0u16);
    let expire = |act: &mut Vec<(usize, u8)>, fr: &mut Vec<u8>, s: usize| act.retain(|&(av, ar)| {
        if last_use[av] <= s { fr.push(ar) }
        last_use[av] > s });
    for (i, st) in steps.iter().enumerate() {
        if st.reuse_dying() { expire(&mut active, &mut free, i); }
        let (defs, nd) = st.defs();
        for &d in &defs[..nd] {
            let r = free.pop()?;
            (used_mask, phys[d]) = (used_mask | (1u16 << r), Some(r));
            if last_use[d] > i { active.push((d, r)) } else { free.push(r) } }
        if !st.reuse_dying() { expire(&mut active, &mut free, i);}}
    let mut ranked: Vec<_> = consts.iter().filter(|&&(v, _)| uses.get(v).copied().unwrap_or(0) > 0)
        .map(|&(v, c)| (v, c, uses[v])).collect();
    ranked.sort_by_key(|&(_, c, u)| (std::cmp::Reverse(u), c));
    let hoisted = (0..pool).filter(|&r| (used_mask & (1u16 << r)) == 0).zip(ranked).map(|(r, (v, c, _))| {
        phys[v] = Some(r);
        (r, c) }).collect();
    Some((phys, hoisted)) }
fn schedule_b8_steps(steps: &[Step], next_v: usize) -> Vec<Step> {
    let ns = steps.len();
    if ns <= 2 { return steps.to_vec(); }
    let defs = b8_def_map(steps, next_v);
    let (mut deps, mut cons, mut uuses) = (vec![Vec::new(); ns], vec![Vec::new(); ns], vec![Vec::new(); ns]);
    let (mut base_u, mut su) = (vec![0usize; next_v], vec![1usize; ns]);
    for (i, st) in steps.iter().enumerate() {
        let (us, nu) = st.uses();
        for &v in &us[..nu] { if let Some(ds) = defs.get(v).copied().flatten() {
            if !uuses[i].contains(&v) { uuses[i].push(v) }
            if !deps[i].contains(&ds) { deps[i].push(ds) } }}
        let mut w: Vec<_> = deps[i].iter().map(|&d| su[d]).collect();
        w.sort_unstable_by(|a, b| b.cmp(a));
        su[i] = w.iter().enumerate().fold(1usize, |acc, (j, &s)| acc.max(s + j));
        for &v in &uuses[i] { base_u[v] += 1 }
        for &d in &deps[i] { cons[d].push(i) } }
    let (mut best_order, mut best_peak) = (Vec::new(), usize::MAX);
    for mode in 0..4usize {
        let (mut rem_d, mut rem_u) = (deps.iter().map(Vec::len).collect::<Vec<_>>(), base_u.clone());
        let mut ready: Vec<_> = (0..ns).filter(|&i| rem_d[i] == 0).collect();
        let (mut order, mut live, mut peak) = (Vec::with_capacity(ns), 0usize, 0usize);
        while let Some((pos, &u)) = ready.iter().enumerate().min_by_key(|&(_, &u)| {
            let (ds, nd) = steps[u].defs();
            let dl = ds[..nd].iter().filter(|&&d| rem_u[d] > 0).count() as isize
                - uuses[u].iter().filter(|&&v| rem_u[v] == 1).count() as isize;
            let (ld, is) = (usize::from(matches!(steps[u], Step::LdN(..) | Step::Ld64(..))), usize::MAX - su[u]);
            [(dl, ld, is, u), (dl, ld, su[u], u), (dl, is, ld, u), (dl, ld, is, usize::MAX - u)][mode & 3] }) {
            ready.swap_remove(pos);
            order.push(u);
            let (ds, nd) = steps[u].defs();
            let kl = uuses[u].iter().filter(|&&v| rem_u[v] == 1).count();
            peak = peak.max(if steps[u].reuse_dying() { live.saturating_sub(kl) + nd } else { live + nd });
            live += ds[..nd].iter().filter(|&&d| rem_u[d] > 0).count();
            for &v in &uuses[u] { if dec0(&mut rem_u[v]) { live = live.saturating_sub(1) } }
            for &c in &cons[u] { if dec0(&mut rem_d[c]) { ready.push(c) } } }
        if peak < best_peak { (best_peak, best_order) = (peak, order) } }
    best_order.into_iter().map(|i| steps[i]).collect() }
fn b8_cvals(consts: &[(usize, u32)], next_v: usize) -> Vec<Option<u32>> {
    let mut vals = vec![None; next_v];
    for &(v, c) in consts { put(&mut vals, v, Some(c)) }
    vals }
fn b8_def_map(steps: &[Step], next_v: usize) -> Vec<Option<usize>> {
    let mut defs = vec![None; next_v];
    for (idx, st) in steps.iter().enumerate() {
        let (ds, nd) = st.defs();
        for &d in &ds[..nd] { put(&mut defs, d, Some(idx)) } }
    defs }
fn dstep(v: usize, st: &[Step], d: &[Option<usize>]) -> Option<Step> { st.get(d.get(v).copied()??).copied() }
fn b8_u32_bound(v: usize, st: &[Step], defs: &[Option<usize>], cv: &[Option<u32>]) -> Option<(usize, u32, bool)> {
    let oc = |a, b| pw(a, b, |c| cv.get(c).copied().flatten());
    match dstep(v, st, defs)? {
        Step::Bin(Bin::CmpEq, _, a, b) => [(a, b), (b, a)].into_iter().find_map(|(mv, xv)| {
            let Step::Bin(op @ (Bin::MaxU | Bin::MinU), _, ma, mb) = dstep(mv, st, defs)? else { return None };
            let cid = if ma == xv { mb } else if mb == xv { ma } else { return None };
            cv.get(cid).copied().flatten().map(|c| (xv, c, op == Bin::MaxU)) }),
        Step::Bin(Bin::Xor, _, a, b) => {
            let (eq, u32::MAX) = oc(a, b)? else { return None };
            let Step::Bin(Bin::CmpEq, _, ea, eb) = dstep(eq, st, defs)? else { return None };
            oc(ea, eb).filter(|&(_, c)| c == 0).map(|(x, _)| (x, 1, true)) }
        _ => None, }}
fn b8_ensure_const(consts: &mut Vec<(usize, u32)>, next_v: &mut usize, val: u32) -> usize {
    consts.iter().find_map(|&(v, c)| (c == val).then_some(v)).unwrap_or_else(|| {
        let v = bump(next_v);
        consts.push((v, val));
        v }) }
fn peephole_b8_steps(steps: &mut Vec<Step>, consts: &mut Vec<(usize, u32)>, next_v: &mut usize) {
    let (cv, defs) = (b8_cvals(consts, *next_v), b8_def_map(steps, *next_v));
    let meq = |u: usize, v: usize| -> Option<(usize, u32)> {
        let ((x1, c1, g1), (x2, c2, g2)) = (b8_u32_bound(u, steps, &defs, &cv)?, b8_u32_bound(v, steps, &defs, &cv)?);
        (x1 == x2 && c1 == c2 && g1 != g2).then_some((x1, c1)) };
    let mut out1 = Vec::with_capacity(steps.len());
    for &st in steps.iter() {
        let eq_opt = if let Step::Bin(Bin::And, d, a, b) = st {
            meq(a, b).map(|(x, c)| (d, x, c, None)).or_else(|| [(a, b), (b, a)].into_iter().find_map(|(p, iv)| {
                let Step::Bin(Bin::And, _, ia, ib) = dstep(iv, steps, &defs)? else { return None };
                [(ia, ib), (ib, ia)].into_iter().find_map(|(ca, o)| meq(p, ca).map(|(x, c)| (d, x, c, Some(o)))) }))
        } else { None };
        if let Some((dst, x, c, other)) = eq_opt {
            let cv = b8_ensure_const(consts, next_v, c);
            if let Some(ot) = other {
                let ev = bump(next_v);
                out1.extend([Step::Bin(Bin::CmpEq, ev, x, cv), Step::Bin(Bin::And, dst, ev, ot)]);
            } else { out1.push(Step::Bin(Bin::CmpEq, dst, x, cv)); }
        } else { out1.push(st);}}
    let (cv, defs) = (b8_cvals(consts, *next_v), b8_def_map(&out1, *next_v));
    let out2: Vec<_> = out1.iter().map(|&st| match st {
        Step::Bin(Bin::And, d, a, b) => [(b, a), (a, b)].into_iter().find_map(|(ic, vc)| {
            let Step::Bin(Bin::Xor, _, xa, xb) = dstep(ic, &out1, &defs)? else { return None };
            pw(xa, xb, |c| cv.get(c).copied().flatten())
                .filter(|&(_, c)| c == u32::MAX).map(|(y, _)| Step::Bin(Bin::Andn, d, y, vc)) }),
        Step::MulHi(d, lo, Some(xh), c, sh, MulHi::Pos01) => dstep(xh, &out1, &defs).and_then(|s| {
            let Step::Bin(Bin::Sub, _, a, b) = s else { return None };
            (cv.get(a).copied().flatten() == Some(0)).then_some(Step::MulHi(d, lo, Some(b), c, sh, MulHi::NegMask)) }),
        _ => None,
    }.unwrap_or(st)).collect();
    let (mut live, mut kept) = (vec![false; *next_v], Vec::with_capacity(out2.len()));
    for &st in out2.iter().rev() {
        let (ds, nd) = st.defs();
        if nd == 0 || ds[..nd].iter().any(|&d| live.get(d).copied().unwrap_or(false)) {
            let (us, nu) = st.uses();
            for &u in &us[..nu] { put(&mut live, u, true) }
            kept.push(st);}}
    kept.reverse();
    *steps = kept;}
#[derive(Clone, Copy, PartialEq, Eq)] enum SE { C(u64), B(u8, u64, u64), DH, NS, NC, Pref, PS, PM, NX, Any }
#[derive(Clone, Copy)] struct Sym(u64, u64, SE);
impl Sym {
    fn c(v: u64) -> Self { Self(v, v, SE::C(v)) }
    fn any() -> Self { Self(0, u64::MAX, SE::Any) }
    fn r(lo: u64, hi: u64) -> Self { if lo == hi { Self::c(lo) } else { Self(lo, hi, SE::Any) } } }
fn extract_fp(k: &Kernel, out: u8) -> Option<B8FastPath> {
    let p = |a, b| pc32(k, a, b);
    let (d, did, lid, dm) = k.nodes.iter().find_map(|&op| {
        let (Op::Ushr(mid, s @ 32..=63) | Op::Shr(mid, s @ 32..=63)) = op else { return None };
        let &Op::Mul(ma, mb) = k.nodes.get(mid)? else { return None };
        let (aid, m) = p(ma, mb).filter(|&(_, c)| c > 0)?;
        let &Op::Add(aa, ab) = k.nodes.get(aid)? else { return None };
        let (lid, ac) = p(aa, ab).filter(|&(_, c)| c > 0 && c < 0x1_0000)?;
        let &Op::BitAnd(la, lb) = k.nodes.get(lid)? else { return None };
        let (did, u32::MAX) = p(la, lb)? else { return None };
        let (db, lm) = (ld_buf(k, did)?, ac + 1);
        let q = |x: u32| ((u128::from(x) + u128::from(ac)) * u128::from(m)) >> s;
        (q(1) == q(lm)).then_some((db, did, lid, lm)) })?;
    let dhi_zm = k.nodes.iter().filter_map(|&op| {
        let Op::Cmp(C::Eq, ca, cb) = op else { return None };
        let &Op::BitAnd(ba, bb) = k.nodes.get(p(ca, cb).filter(|&(_, c)| c == 0)?.0)? else { return None };
        let (sid, fm) = p(ba, bb).filter(|&(_, c)| c > 0)?;
        matches!(k.nodes.get(sid), Some(&Op::Ushr(x, 32) | &Op::Shr(x, 32)) if x == did).then_some(fm)
    }).fold(0u32, |a, b| a | b);
    let (a, aid) = k.nodes.iter().find_map(|&op| {
        let Op::Add(aa, ab) = op else { return None };
        let o = if aa == lid { ab } else if ab == lid { aa } else { return None };
        ld_buf(k, o).filter(|&b| b != d).map(|b| (b, o)) })?;
    let bound = k.nodes.iter().filter_map(|&op| match op {
        Op::Cmp(C::Ult | C::Lt, ca, cb) if ca == aid => sym_c(k, cb),
        Op::Cmp(C::Ugt | C::Gt, ca, cb) if cb == aid => sym_c(k, ca),
        _ => None,
    }).filter(|&b| b > u64::from(dm) && b <= 0xFFFF_FFFF).min()?;
    let mut p2 = 1u64 << 31;
    while p2 > 1 && (p2 - 1).saturating_add(u64::from(dm)) > bound { p2 >>= 1 }
    if p2 <= u64::from(dm) { return None; }
    let ng = (0..k.nodes.len()).find_map(|pi| {
        let pb = ld_buf(k, pi).filter(|&b| b != a && b != d)?;
        k.bufs.get(pb as usize).filter(|b| b.elem_ty == ElemTy::U8 && b.lanes == 1)?;
        if !k.nodes.iter().any(|&op| matches!(op, Op::Cmp(C::Ule | C::Le, x, y) if x == pi && sym_c(k, y) == Some(32))
            || matches!(op, Op::Cmp(C::Uge | C::Ge, x, y) if y == pi && sym_c(k, x) == Some(32))) { return None; }
        (0..k.nodes.len()).find_map(|ni| {
            let nb = ld_buf(k, ni).filter(|&b| b != a && b != d && b != pb)?;
            k.bufs.get(nb as usize).filter(|b| is64(b.elem_ty) && b.lanes == 1)?;
            let rej = k.nodes.iter().rev().find_map(|&op| {
                let Op::Cmp(C::Ne, x, y) = op else { return None };
                let (l, rc) = p(x, y)?;
                matches!(k.nodes.get(l), Some(&Op::BitAnd(la, lb)) if p(la, lb) == Some((ni, u32::MAX)))
                    .then_some(rc) });
            Some(NetGuard { net: nb, pref: pb, rej }) }) });
    Some(B8FastPath { addr: a, desc: d, out, addr_zm: !(p2 - 1), dhi_zm, dlo_max: dm, ng, ao: [(u64::MAX, 0); 2] }) }
fn sym_cmp_eval(kind: C, a: Sym, b: Sym, ng: Option<NetGuard>) -> Sym {
    if kind == C::Ne && ng.and_then(|g| g.rej).is_some_and(|r|
        [(a.2, b.2), (b.2, a.2)].contains(&(SE::NS, SE::C(u64::from(r))))) { return Sym::c(1); }
    let eff = if a.1 <= i64::MAX as u64 && b.1 <= i64::MAX as u64 { ct(&CMP_U, kind) } else { kind };
    let (k, x, y) = if eff == C::Ugt { (C::Ult, b, a) } else if eff == C::Uge { (C::Ule, b, a) } else { (eff, a, b) };
    match k {
        C::Eq | C::Ne if x.0 == x.1 && y.0 == y.1 => Sym::c(u64::from((x.0 == y.0) ^ (k == C::Ne))),
        C::Eq | C::Ne if x.1 < y.0 || y.1 < x.0 => Sym::c(u64::from(k == C::Ne)),
        C::Ult if x.1 < y.0 || x.0 >= y.1 => Sym::c(u64::from(x.1 < y.0)),
        C::Ule if x.1 <= y.0 || x.0 > y.1 => Sym::c(u64::from(x.1 <= y.0)),
        _ => Sym::r(0, 1), }}
fn prove_fp(k: &Kernel, sched: &ScheduleReport, mut fp: B8FastPath, lanes: [NodeId; 2]) -> Option<B8FastPath> {
    let mut s: Vec<_> = k.nodes.iter().map(|&o| {
        if let Op::Const(c) = o { Sym::c(c as u64) } else { Sym::any() } }).collect();
    let (ng, m32, dz, dk) = (fp.ng, u64::from(u32::MAX), u64::from(fp.dhi_zm), u64::from(u32::MAX & !fp.dhi_zm));
    let ord_c = |x: SE, y: SE| if matches!(x, SE::C(_)) { (y, x) } else { (x, y) };
    let mk = |o: Option<(u64, u64)>| o.map_or_else(Sym::any, |(l, h)| Sym::r(l, h));
    for &id in &sched.order {
        let Some(&op) = k.nodes.get(id) else { continue };
        s[id] = match op {
            Op::Const(c) => Sym::c(c as u64),
            Op::MemInit | Op::Store { .. } => Sym::c(0),
            Op::Load { buf: b, .. } if ld_buf(k, id).is_some() => {
                let raw = SE::B(b, u64::MAX, 0);
                if b == fp.addr { Sym(0, !fp.addr_zm, raw) }
                else if b == fp.desc { Sym(1, (dk << 32) | u64::from(fp.dlo_max), raw) }
                else if ng.is_some_and(|g| g.net == b) { Sym(0, u64::MAX, raw) }
                else if ng.is_some_and(|g| g.pref == b) { Sym(0, 32, SE::Pref) } else { Sym::any() } }
            Op::Add(a, b) => mk(s[a].0.checked_add(s[b].0).zip(s[a].1.checked_add(s[b].1))),
            Op::Sub(a, b) if ng.is_some() && (s[a].2, s[b].2) == (SE::C(32), SE::Pref) => Sym(0, 32, SE::PS),
            Op::Sub(a, b) if s[b].1 <= s[a].0 => Sym::r(s[a].0 - s[b].1, s[a].1 - s[b].0),
            Op::Mul(a, b) => mk(s[a].0.checked_mul(s[b].0).zip(s[a].1.checked_mul(s[b].1))),
            Op::Ushr(a, sh) | Op::Shr(a, sh) => {
                let sa = s[a];
                if matches!(op, Op::Shr(..)) && sa.1 > i64::MAX as u64 { Sym::any() }
                else if sh >= 64 { Sym::c(0) }
                else if sh == 32 && sa.2 == SE::B(fp.desc, u64::MAX, 0) { Sym(0, dk, SE::DH) }
                else if sh == 32 && ng.is_some_and(|g| sa.2 == SE::B(g.net, u64::MAX, 0)) { Sym(0, m32, SE::NC) }
                else { Sym::r(sa.0 >> sh, sa.1 >> sh) } }
            Op::Shl(a, sh) if matches!(s[a].2, SE::C(_)) => Sym::c(if sh < 64 { s[a].0 << sh } else { 0 }),
            Op::ShlVar(a, b) if (s[a].2, s[b].2) == (SE::C(m32), SE::PS) => Sym(0, m32, SE::PM),
            Op::BitAnd(a, b) => match ord_c(s[a].2, s[b].2) {
                (SE::C(x), SE::C(y)) => Sym::c(x & y),
                (SE::NX, SE::PM) | (SE::PM, SE::NX) => Sym::c(0),
                (SE::PM, SE::C(c)) if c == m32 => Sym(0, m32, SE::PM),
                (SE::DH, SE::C(m)) if (m & !dz) == 0 => Sym::c(0),
                (SE::B(b, am, om), SE::C(m)) => {
                    let lo32 = am == u64::MAX && om == 0 && m == m32;
                    if b == fp.desc && lo32 { Sym(1, u64::from(fp.dlo_max), SE::B(b, m, 0)) }
                    else if ng.is_some_and(|g| g.net == b) && lo32 { Sym(0, m32, SE::NS) }
                    else if b == fp.desc && om == 0 && ((am & m) & !(dz << 32)) == 0 { Sym::c(0) }
                    else { Sym(0, m, SE::B(b, am & m, om & m)) } }
                _ => Sym::r(0, s[a].1.min(s[b].1)), },
            Op::BitOr(a, b) => match ord_c(s[a].2, s[b].2) {
                (SE::C(x), SE::C(y)) => Sym::c(x | y),
                (SE::B(b, am, om), SE::C(c)) => Sym(c, u64::MAX, SE::B(b, am & !c, om | c)),
                (SE::B(b1, a1, o1), SE::B(b2, a2, o2)) if b1 == b2 =>
                    Sym(o1 | o2, u64::MAX, SE::B(b1, (a1 | a2) & !(o1 | o2), o1 | o2)),
                _ => Sym::any(), },
            Op::BitXor(a, b) if matches!((s[a].2, s[b].2), (SE::C(_), SE::C(_))) => Sym::c(s[a].0 ^ s[b].0),
            Op::BitXor(a, b) if matches!((s[a].2, s[b].2), (SE::NS, SE::NC) | (SE::NC, SE::NS)) => Sym(0, m32, SE::NX),
            Op::Cmp(kind, a, b) => sym_cmp_eval(kind, s[a], s[b], ng),
            Op::Select { cond, t_val: t, .. } if s[cond].0 > 0 => s[t],
            Op::Select { cond, f_val: f, .. } if s[cond].1 == 0 => s[f],
            Op::Select { t_val: t, f_val: f, .. } =>
                Sym(s[t].0.min(s[f].0), s[t].1.max(s[f].1), if s[t].2 == s[f].2 { s[t].2 } else { SE::Any }),
            _ => Sym::any(), };}
    let (SE::B(b0, a0, o0), SE::B(b1, a1, o1)) = (s.get(lanes[0])?.2, s.get(lanes[1])?.2) else { return None };
    if b0 != fp.addr || b1 != fp.desc { return None; }
    fp.ao = [(a0, o0), (a1, o1)];
    Some(fp) }
fn ymms(e: &mut X86Emit, ops: &[(u8, u8, u8, u8, u8)]) { for &(m, o, d, a, b) in ops { e.vex_ymm_op(m, o, d, a, b) } }
fn st64_iv(e: &mut X86Emit, base: u8, disp: i32, r: [u8; 5]) {
    ymms(e, &[(1, 0x6C, r[2], r[0], r[1]), (1, 0x6D, r[3], r[0], r[1])]);
    for (imm, off) in [(0x20u8, disp), (0x31, disp + 32)] {
        e.vex_ymm_imm(3, 0x46, r[4], r[2], r[3], imm);
        e.vex_mem_op(VEX_ST, r[4], base, off);}}
pub(crate) fn emit_x86_batch8_fp_consts(e: &mut X86Emit, fp: &B8FastPath) {
    let s = 0xFFFF_FFFF_0000_0000u64 | u64::from(fp.dlo_max.saturating_sub(1));
    for (r, v) in [(6, fp.addr_zm), (8, u64::from(u32::MAX)), (9, s), (10, fp.ao[1].0), (11, fp.ao[1].1)] {
        e.vex_broadcast_rip(r, v as i64);}
    if fp.dhi_zm != 0 { e.vex_broadcast_rip(7, (u64::from(fp.dhi_zm) << 32) as i64); }
    if let Some(ng) = fp.ng {
        for (r, v) in [(12, 32), (13, u32::MAX)] { e.vex_broadcast32_rip(r, v) }
        if let Some(rej) = ng.rej { e.vex_broadcast32_rip(14, rej);}}}
pub(crate) fn emit_x86_batch8_fp_body(e: &mut X86Emit, ra: &RegAlloc, fp: &B8FastPath) -> Result<usize, CompileError> {
    let breg = |b: u8| ra.buf_reg.get(b as usize).copied().ok_or(CompileError::InvalidBuffer(b));
    let (addr_reg, desc_reg, out_reg) = (breg(fp.addr)?, breg(fp.desc)?, breg(fp.out)?);
    if let Some(ng) = fp.ng {
        let (net_reg, pref_reg) = (breg(ng.net)?, breg(ng.pref)?);
        e.vex_mem_op([2, 1, 1, 0, 0, 0x31], 0, pref_reg, 0);
        e.vex_vpermq(0, 0, 0xD8);
        ymms(e, &[(2, 0x3B, 15, 0, 12), (1, 0xEF, 15, 15, 0), (1, 0xFA, 0, 12, 0)]);
        e.vex_ymm_op_w(2, 0, 0x47, 0, 13, 0);
        for (r, off) in [(1u8, 0i32), (2, 32)] { e.vex_mem_op(VEX_LD, r, net_reg, off) }
        for (d, imm) in [(3u8, 0x88u8), (4, 0xDD)] { e.vex_vshufps(d, 1, 2, imm) }
        ymms(e, &[(1, 0xEF, 4, 4, 3), (1, 0xDB, 4, 4, 0), (1, 0xEB, 15, 15, 4)]);
        if ng.rej.is_some() { ymms(e, &[(1, 0x76, 3, 3, 14), (1, 0xEB, 15, 15, 3)]); }
    } else { e.vex_ymm_op(1, 0xEF, 15, 15, 15); }
    for (r0, r1, br, mr, chk) in [(0u8, 1u8, addr_reg, 6u8, true), (2, 3, desc_reg, 7, fp.dhi_zm != 0)] {
        for (r, off) in [(r0, 0), (r1, 32)] { e.vex_mem_op(VEX_LD, r, br, off) }
        if chk { ymms(e, &[(1, 0xEB, 4, r0, r1), (1, 0xDB, 4, 4, mr), (1, 0xEB, 15, 15, 4)]);}}
    for dr in [2, 3] { ymms(e, &[(1, 0xFE, 4, dr, 8), (2, 0x3B, 5, 4, 9), (1, 0xEF, 4, 4, 5), (1, 0xEB, 15, 15, 4)]) }
    e.vex_vptest(15, 15);
    let jnz_miss = e.jcc_placeholder(0x85);
    let [(a0, o0), (a1, o1)] = fp.ao;
    for (en, opc, regs, imm) in [
        (a0 != u64::MAX, 0xDBu8, [0u8, 1], Some(a0)), (o0 != 0, 0xEBu8, [0, 1], Some(o0)),
        (a1 != u64::MAX, 0xDBu8, [2, 3], None), (o1 != 0, 0xEBu8, [2, 3], None), ] {
        if !en { continue; }
        if let Some(c) = imm { e.vex_broadcast_rip(15, c as i64); }
        let rhs = if imm.is_some() { 15 } else if opc == 0xDB { 10 } else { 11 };
        for r in regs { e.vex_ymm_op(1, opc, r, r, rhs) }}
    for (r0, r1, d) in [(0u8, 2u8, 0i32), (1, 3, 64)] { st64_iv(e, out_reg, d, [r0, r1, 4, 5, r0]) }
    Ok(jnz_miss) }
pub(crate) fn try_plan_avx2_batch8(k: &Kernel, s: &ScheduleReport, dm: &[Domain], mv: usize) -> Option<Batch8Plan> {
    if k.outputs.is_empty() || mv < 12 || !k.bufs.iter().any(|b| is64(b.elem_ty)) { return None; }
    if k.bufs.iter().any(|b| {
        matches!((b.access, b.lanes), (BufAccess::LookupTable(_), _) | (BufAccess::ReadOnly, 0 | 2..))
    }) { return None; }
    let (out_buf, n_out) = (k.outputs[0].buf, k.outputs.len());
    let ob = k.bufs.get(out_buf as usize)?;
    let two_u64 = n_out == 2 && ob.lanes == 2 && is64(ob.elem_ty);
    if k.bufs.iter().filter(|b| matches!(b.access, BufAccess::MutNoAlias)).count() != 1
        || !(two_u64 || (n_out == 1 && ob.lanes == 1)) { return None; }
    let (ism, dlo) = (analyze_bool_masks(k, s), analyze_demanded_lo32(k, s));
    let skip_mul = find_fused_barrett_muls(k, s, dm);
    let vals = k.nodes.iter().map(|&o| if let Op::Const(c) = o { Some(Val::Const(c as _)) }
        else { (o == Op::MemInit).then_some(Val::Const(0)) }).collect();
    let mut b = B8Builder { k, dm, ism: &ism, dlo: &dlo, vals, steps: Vec::new(), consts: Vec::new(), next_v: 0 };
    for &id in &s.order { b.lower_node(id, &skip_mul)?; }
    let fp_lanes = if !two_u64 {
        let ov = b.val(k.outputs[0].val)?;
        let (lo, hi) = (b.h(ov, false), is64(ob.elem_ty).then(|| b.h(ov, true)));
        b.steps.push(Step::St1(lo, hi, out_buf, ob.elem_ty));
        None
    } else {
        let &[o0, o1] = k.outputs.as_slice() else { return None };
        if o1.buf != out_buf || (o0.lane | o1.lane) != 1 || o0.lane == o1.lane { return None; }
        let lanes = if o0.lane == 0 { [o0.val, o1.val] } else { [o1.val, o0.val] };
        let (v0, v1) = (b.val(lanes[0])?, b.val(lanes[1])?);
        let vals = [b.h(v0, false), b.h(v0, true), b.h(v1, false), b.h(v1, true)].map(|v| b.reg(v));
        b.steps.push(Step::St2(vals, out_buf));
        Some((out_buf, lanes)) };
    peephole_b8_steps(&mut b.steps, &mut b.consts, &mut b.next_v);
    let ss = schedule_b8_steps(&b.steps, b.next_v);
    let alloc = |st: &[Step]| alloc_regs(st, &b.consts, b.next_v, mv);
    let (steps, (vreg_phys, hoisted)) = match (alloc(&ss), alloc(&b.steps)) {
        (Some(sa), Some(ba)) if sa.1.len() >= ba.1.len() => (ss, sa),
        (_, Some(ba)) => (b.steps, ba),
        (Some(sa), None) => (ss, sa),
        (None, None) => return None, };
    let fastpath = fp_lanes.and_then(|(obuf, lanes)| extract_fp(k, obuf).and_then(|c| prove_fp(k, s, c, lanes)));
    Some(Batch8Plan { consts_v: b.consts, hoisted, vreg_phys, steps, fastpath }) }
fn b8d(p: &Batch8Plan, v: usize) -> Result<u8, CompileError> {
    p.vreg_phys.get(v).and_then(|&r| r).ok_or_else(|| b8_err("batch8 dst")) }
fn b8r(e: &mut X86Emit, p: &Batch8Plan, v: usize, s: u8) -> Result<u8, CompileError> {
    if let Ok(r) = b8d(p, v) { return Ok(r); }
    let &(_, c) = p.consts_v.iter().find(|&&(cv, _)| cv == v).ok_or_else(|| b8_err("batch8 vreg"))?;
    e.vex_broadcast32_rip(s, c);
    Ok(s) }
fn emit_nibble_popcnt(e: &mut X86Emit, d: u8, src: u8, rm0f: u8, acc: bool) {
    e.vex_shift_imm(0x71, 2, 14, src, 4);
    ymms(e, &[(1, 0xDB, 14, 14, rm0f), (2, 0x00, 14, 15, 14)]);
    if acc { e.vex_ymm_op(1, 0xFC, d, d, 14); }
    let lo = if acc { 14 } else { d };
    ymms(e, &[(1, 0xDB, lo, src, rm0f), (2, 0x00, lo, 15, lo), (1, 0xFC, d, d, 14)]);}
pub(crate) fn emit_x86_batch8_body(e: &mut X86Emit, k: &Kernel, ra: &RegAlloc, p: &Batch8Plan, blk: i32)
    -> Result<(), CompileError> {
    let binfo = |buf: u8| -> Result<(u8, i32), CompileError> {
        let br = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
        let bb = k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32;
        Ok((br, blk * bb)) };
    for &st in &p.steps { match st {
        Step::LdN(dst, buf, ty) => {
            let (d, (br, off)) = (b8d(p, dst)?, binfo(buf)?);
            let enc = if matches!(ty, ElemTy::U32 | ElemTy::I32) { VEX_LD }
                else if let Some(&(_, o)) = LD_N.iter().find(|&&(t, _)| t == ty) { [2, 1, 1, 0, 0, o] }
                else { return Err(b8_err("narrow load")) };
            e.vex_mem_op(enc, d, br, off);
            e.vex_vpermq(d, d, 0xD8);}
        Step::Ld64(lo, hi, buf) => {
            let ((dl, dh), (br, off)) = ((b8d(p, lo)?, b8d(p, hi)?), binfo(buf)?);
            for (r, o) in [(14u8, off), (15, off + 32)] { e.vex_mem_op(VEX_LD, r, br, o) }
            for (d, imm) in [(dl, 0x88u8), (dh, 0xDD)] { e.vex_vshufps(d, 14, 15, imm) } }
        Step::Bin(op, dst, a, b) => {
            let (d, x, y, (m, o)) = (b8d(p, dst)?, b8r(e, p, a, 14)?, b8r(e, p, b, 15)?, BIN_VEX[(op as usize) & 15]);
            e.vex_ymm_op(m, o, d, x, y);}
        Step::Sh(op, dst, src, s) => {
            let (d, rs) = (b8d(p, dst)?, b8r(e, p, src, 14)?);
            if s == 0 && d != rs { e.vex_ymm_op(1, 0xEB, d, rs, rs); }
            else if s > 0 { e.vex_shift_imm(0x72, SHIFT_SUB[(op as usize) & 3], d, rs, s);}}
        Step::Bld(dst, fv, tv, mk) => {
            let (d, rf, rt, rm) = (b8d(p, dst)?, b8r(e, p, fv, 13)?, b8r(e, p, tv, 14)?, b8r(e, p, mk, 15)?);
            e.vex_ymm_imm(3, 0x4C, d, rf, rt, rm << 4);}
        Step::MulHi(dst, x_lo, x_hi, c, shift, mode) => {
            let (d, rlo, rc) = (b8d(p, dst)?, b8r(e, p, x_lo, 13)?, b8r(e, p, c, 15)?);
            e.vex_ymm_op(1, 0xF4, 14, rlo, rc);
            for (t, s) in [(14u8, 14u8), (d, rlo)] { e.vex_shift_imm(0x73, 2, t, s, 32) }
            e.vex_ymm_op(1, 0xF4, d, d, rc);
            e.vex_ymm_imm(3, 0x02, d, 14, d, 0xAA);
            if let Some(xh) = x_hi {
                let (rh, (m, o)) = (b8r(e, p, xh, 13)?, if mode == MulHi::General { (2, 0x40) } else { (1, 0xDB) });
                if mode == MulHi::Pos01 { ymms(e, &[(1, 0xEF, 14, 14, 14), (1, 0xFA, 14, 14, rh)]) }
                ymms(e, &[(m, o, 14, if mode == MulHi::Pos01 { 14 } else { rh }, rc), (1, 0xFE, d, d, 14)]);}
            if shift > 0 { e.vex_shift_imm(0x72, 2, d, d, shift);}}
        Step::Pop64(dst, x_lo, x_hi, m0f, m01) => {
            let (d, rlo, rm0f) = (b8d(p, dst)?, b8d(p, x_lo)?, b8r(e, p, m0f, 13)?);
            for (r, c) in [(14, 0x0302_0201_0201_0100u64), (15, 0x0403_0302_0302_0201)] {
                e.vex_broadcast_rip(r, c as i64);}
            e.vex_ymm_op(1, 0x6C, 15, 14, 15);
            emit_nibble_popcnt(e, d, rlo, rm0f, false);
            if let Some(xh) = x_hi { emit_nibble_popcnt(e, d, b8d(p, xh)?, rm0f, true); }
            let rmul = b8r(e, p, m01, 15)?;
            e.vex_ymm_op(2, 0x40, d, d, rmul);
            e.vex_shift_imm(0x72, 2, d, d, 24);}
        Step::St1(lo, hi, buf, ty) => {
            let ((br, off), rl) = (binfo(buf)?, b8r(e, p, lo, 13)?);
            if let Some(hv) = hi {
                let rh = b8r(e, p, hv, 15)?;
                for (o, d) in [(0x62u8, off), (0x6A, off + 32)] {
                    e.vex_ymm_op(1, o, 14, rl, rh);
                    e.vex_mem_op(VEX_ST, 14, br, d);}
            } else if matches!(ty, ElemTy::U32 | ElemTy::I32) {
                e.vex_vpermq(14, rl, 0xD8);
                e.vex_mem_op(VEX_ST, 14, br, off);
            } else { return Err(b8_err("narrow store1")) }}
        Step::St2(v, buf) => {
            let ((br, off), r) = (binfo(buf)?, [b8d(p, v[0])?, b8d(p, v[1])?, b8d(p, v[2])?, b8d(p, v[3])?]);
            for (u, d) in [(0x62u8, off), (0x6A, off + 64)] {
                ymms(e, &[(1, u, 13, r[0], r[1]), (1, u, 14, r[2], r[3])]);
                st64_iv(e, br, d, [13, 14, 15, 13, 14]);}}}}
    Ok(()) }
