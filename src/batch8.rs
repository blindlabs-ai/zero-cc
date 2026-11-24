//! Classical interval-driven `u32x8` narrowing and split-width `(lo32, hi32)` AVX2 batch backend.
//!
//! Processes `8` scalar kernel lanes per `256`-bit `YMM` register (`16` lanes per 2x-unrolled
//! loop iteration), keeping boolean predicates in native all-ones/zeros SIMD masks, eliminating
//! upper-32 zero lanes via `Domain::fits_u32()`, and lowering 64-bit arithmetic/comparisons and
//! Barrett high-multiplies into exact carry/borrow `u32x8` operations.

use crate::codegen::ScheduleReport;
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op};
use crate::isa::{RegAlloc, X86Emit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum B8OutLayout {
    Lane1(u8, ElemTy),
    Lanes2U64(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MulHiMode {
    General,
    Pos01,
    NegMask,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct B8NetGuard {
    pub(crate) net_buf: u8,
    pub(crate) prefix_buf: u8,
    pub(crate) prefix_max: u32,
    pub(crate) reject_src_const: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct B8FastPath {
    pub(crate) addr_buf: u8,
    pub(crate) addr_zero_mask: u64,
    pub(crate) desc_buf: u8,
    pub(crate) desc_hi_zero_mask: u32,
    pub(crate) desc_lo_min: u32,
    pub(crate) desc_lo_max: u32,
    pub(crate) net_guard: Option<B8NetGuard>,
    pub(crate) out_buf: u8,
    pub(crate) out0_and: u64,
    pub(crate) out0_or: u64,
    pub(crate) out1_and: u64,
    pub(crate) out1_or: u64,
}

#[derive(Clone, Copy, Debug)]
enum B8Step {
    LoadNarrow { dst: usize, buf: u8, ty: ElemTy, permute: bool },
    LoadSplit64 { lo: usize, hi: usize, buf: u8 },
    Mov { dst: usize, src: usize },
    Add32 { dst: usize, a: usize, b: usize },
    Sub32 { dst: usize, a: usize, b: usize },
    And32 { dst: usize, a: usize, b: usize },
    Andn32 { dst: usize, mask_inv: usize, val: usize },
    Or32 { dst: usize, a: usize, b: usize },
    Xor32 { dst: usize, a: usize, b: usize },
    MinU32 { dst: usize, a: usize, b: usize },
    MaxU32 { dst: usize, a: usize, b: usize },
    CmpEq32 { dst: usize, a: usize, b: usize },
    CmpGtS32 { dst: usize, a: usize, b: usize },
    ShlImm32 { dst: usize, src: usize, s: u8 },
    ShrImm32 { dst: usize, src: usize, s: u8 },
    ShlVar32 { dst: usize, a: usize, b: usize },
    ShrVar32 { dst: usize, a: usize, b: usize },
    Blend32 { dst: usize, f_val: usize, t_val: usize, mask: usize },
    MulHiShr { dst: usize, x_lo: usize, x_hi: Option<usize>, c: usize, shift: u8, hi_mode: MulHiMode },
    Store1 { lo: usize, hi: Option<usize>, buf: u8, ty: ElemTy, unpermute: bool },
    Store2U64 { o0_lo: usize, o0_hi: usize, o1_lo: usize, o1_hi: usize, buf: u8 },
}

impl B8Step {
    fn defs(&self) -> ([usize; 2], usize) {
        match *self {
            Self::LoadSplit64 { lo, hi, .. } => ([lo, hi], 2),
            Self::Store1 { .. } | Self::Store2U64 { .. } => ([0, 0], 0),
            Self::LoadNarrow { dst, .. }
            | Self::Mov { dst, .. }
            | Self::Add32 { dst, .. }
            | Self::Sub32 { dst, .. }
            | Self::And32 { dst, .. }
            | Self::Andn32 { dst, .. }
            | Self::Or32 { dst, .. }
            | Self::Xor32 { dst, .. }
            | Self::MinU32 { dst, .. }
            | Self::MaxU32 { dst, .. }
            | Self::CmpEq32 { dst, .. }
            | Self::CmpGtS32 { dst, .. }
            | Self::ShlImm32 { dst, .. }
            | Self::ShrImm32 { dst, .. }
            | Self::ShlVar32 { dst, .. }
            | Self::ShrVar32 { dst, .. }
            | Self::Blend32 { dst, .. }
            | Self::MulHiShr { dst, .. } => ([dst, 0], 1),
        }
    }

    fn for_each_use(&self, mut f: impl FnMut(usize)) {
        match *self {
            Self::LoadNarrow { .. } | Self::LoadSplit64 { .. } => {}
            Self::Mov { src, .. } | Self::ShlImm32 { src, .. } | Self::ShrImm32 { src, .. } => f(src),
            Self::Add32 { a, b, .. }
            | Self::Sub32 { a, b, .. }
            | Self::And32 { a, b, .. }
            | Self::Or32 { a, b, .. }
            | Self::Xor32 { a, b, .. }
            | Self::MinU32 { a, b, .. }
            | Self::MaxU32 { a, b, .. }
            | Self::CmpEq32 { a, b, .. }
            | Self::CmpGtS32 { a, b, .. }
            | Self::ShlVar32 { a, b, .. }
            | Self::ShrVar32 { a, b, .. } => { f(a); f(b); }
            Self::Andn32 { mask_inv, val, .. } => { f(mask_inv); f(val); }
            Self::Blend32 { f_val, t_val, mask, .. } => { f(f_val); f(t_val); f(mask); }
            Self::MulHiShr { x_lo, x_hi, c, .. } => { f(x_lo); if let Some(h) = x_hi { f(h); } f(c); }
            Self::Store1 { lo, hi, .. } => { f(lo); if let Some(h) = hi { f(h); } }
            Self::Store2U64 { o0_lo, o0_hi, o1_lo, o1_hi, .. } => { f(o0_lo); f(o0_hi); f(o1_lo); f(o1_hi); }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum B8Val {
    Const(u64),
    Lo32(usize),
    Hi32(usize),
    Mask32(usize),
    Split64(usize, usize),
}

pub(crate) struct Batch8Plan {
    consts_v: Vec<(usize, u32)>,
    pub(crate) hoisted: Vec<(u8, u32)>,
    vreg_phys: Vec<Option<u8>>,
    steps: Vec<B8Step>,
    pub(crate) fastpath: Option<B8FastPath>,
}

fn analyze_bool_masks(k: &Kernel, sched: &ScheduleReport) -> Vec<bool> {
    let n = k.nodes.len();
    let mut is_bool_prod = vec![false; n];
    for &id in &sched.order {
        let Some(&op) = k.nodes.get(id) else { continue; };
        is_bool_prod[id] = match op {
            Op::Cmp(_, _, _) => true,
            Op::BitAnd(a, b) | Op::BitOr(a, b) => is_bool_prod[a] && is_bool_prod[b],
            Op::Select { t_val, f_val, .. } => {
                let b_or_01 = |x: NodeId| {
                    is_bool_prod[x] || matches!(k.nodes.get(x), Some(Op::Const(0 | 1)))
                };
                b_or_01(t_val) && b_or_01(f_val)
            }
            _ => false,
        };
    }
    let mut mask_ok = is_bool_prod.clone();
    for o in &k.outputs {
        if let Some(m) = mask_ok.get_mut(o.val) { *m = false; }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &sched.order {
            let Some(&op) = k.nodes.get(id) else { continue; };
            let self_mask = mask_ok[id];
            match op {
                Op::BitAnd(_, _) | Op::BitOr(_, _) if self_mask => {}
                Op::Select { cond: _, t_val, f_val } if self_mask => {
                    for x in [t_val, f_val] {
                        if !matches!(k.nodes.get(x), Some(Op::Const(0 | 1))) && !mask_ok[x] {
                            mask_ok[id] = false;
                            changed = true;
                        }
                    }
                }
                Op::Select { cond: _, t_val, f_val } => {
                    for x in [t_val, f_val] {
                        if mask_ok[x] { mask_ok[x] = false; changed = true; }
                    }
                }
                _ => {
                    let (kids, cnt) = op.children();
                    for &kid in &kids[..cnt] {
                        if mask_ok.get(kid).copied().unwrap_or(false) {
                            mask_ok[kid] = false;
                            changed = true;
                        }
                    }
                }
            }
        }
        for &id in &sched.order {
            if !mask_ok[id] { continue; }
            if let Some(Op::BitAnd(a, b) | Op::BitOr(a, b)) = k.nodes.get(id).copied() {
                if !mask_ok[a] || !mask_ok[b] { mask_ok[id] = false; changed = true; }
            }
        }
    }
    mask_ok
}

fn analyze_demanded_lo32(k: &Kernel, sched: &ScheduleReport) -> Vec<bool> {
    let n = k.nodes.len();
    let mut needs_hi = vec![false; n];
    for o in &k.outputs {
        let ty = k.bufs.get(o.buf as usize).map_or(ElemTy::U64, |b| b.elem_ty);
        if matches!(ty, ElemTy::I64 | ElemTy::U64) {
            if let Some(h) = needs_hi.get_mut(o.val) { *h = true; }
        }
    }
    for &id in sched.order.iter().rev() {
        let Some(&op) = k.nodes.get(id) else { continue; };
        let hi = needs_hi[id];
        match op {
            Op::BitAnd(a, b) => {
                let ca = matches!(k.nodes.get(a), Some(&Op::Const(c)) if (c as u64) <= u32::MAX as u64);
                let cb = matches!(k.nodes.get(b), Some(&Op::Const(c)) if (c as u64) <= u32::MAX as u64);
                if hi && !cb { needs_hi[a] = true; }
                if hi && !ca { needs_hi[b] = true; }
            }
            Op::Shl(a, s) => { if hi && s < 32 { needs_hi[a] = true; } }
            Op::Ushr(a, _) | Op::Shr(a, _) => { needs_hi[a] = true; }
            Op::Cmp(_, a, b) => { needs_hi[a] = true; needs_hi[b] = true; }
            Op::Select { cond, t_val, f_val } => {
                needs_hi[cond] = true;
                if hi { needs_hi[t_val] = true; needs_hi[f_val] = true; }
            }
            Op::Mul(a, b) => { needs_hi[a] = true; needs_hi[b] = true; }
            Op::Add(a, b) | Op::Sub(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => {
                if hi { needs_hi[a] = true; needs_hi[b] = true; }
            }
            Op::ShlVar(a, _) | Op::ShrVarU(a, _) => {
                if hi { needs_hi[a] = true; }
            }
            _ => {}
        }
    }
    needs_hi.into_iter().map(|h| !h).collect()
}

struct B8Builder<'a> {
    k: &'a Kernel,
    doms: &'a [Domain],
    is_mask: &'a [bool],
    demanded_lo32: &'a [bool],
    use_perm: bool,
    vals: Vec<Option<B8Val>>,
    steps: Vec<B8Step>,
    consts: Vec<(usize, u32)>,
    next_v: usize,
}

impl<'a> B8Builder<'a> {
    fn alloc_v(&mut self) -> usize {
        let v = self.next_v;
        self.next_v += 1;
        v
    }

    fn const32(&mut self, val: u32) -> usize {
        b8_ensure_const(&mut self.consts, &mut self.next_v, val)
    }

    fn emit(&mut self, s: B8Step) { self.steps.push(s); }

    fn val_lo32(&mut self, v: B8Val) -> usize {
        match v {
            B8Val::Const(c) => self.const32(c as u32),
            B8Val::Lo32(r) | B8Val::Split64(r, _) => r,
            B8Val::Hi32(_) => self.const32(0),
            B8Val::Mask32(m) => {
                let d = self.alloc_v();
                self.emit(B8Step::ShrImm32 { dst: d, src: m, s: 31 });
                d
            }
        }
    }

    fn val_hi32(&mut self, v: B8Val) -> usize {
        match v {
            B8Val::Const(c) => self.const32((c >> 32) as u32),
            B8Val::Hi32(r) | B8Val::Split64(_, r) => r,
            B8Val::Lo32(_) | B8Val::Mask32(_) => self.const32(0),
        }
    }

    fn is_zero_hi(&self, v: B8Val) -> bool {
        match v {
            B8Val::Const(c) => (c >> 32) == 0,
            B8Val::Lo32(_) | B8Val::Mask32(_) => true,
            B8Val::Hi32(_) | B8Val::Split64(_, _) => false,
        }
    }

    fn is_zero_lo(&self, v: B8Val) -> bool {
        match v {
            B8Val::Const(c) => (c as u32) == 0,
            B8Val::Hi32(_) => true,
            B8Val::Lo32(_) | B8Val::Mask32(_) | B8Val::Split64(_, _) => false,
        }
    }

    fn val_mask(&mut self, v: B8Val) -> usize {
        match v {
            B8Val::Mask32(m) => m,
            B8Val::Const(c) => self.const32(if c != 0 { u32::MAX } else { 0 }),
            B8Val::Lo32(r) | B8Val::Hi32(r) => {
                let z = self.const32(0);
                let eq = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: eq, a: r, b: z });
                let ones = self.const32(u32::MAX);
                let d = self.alloc_v();
                self.emit(B8Step::Xor32 { dst: d, a: eq, b: ones });
                d
            }
            B8Val::Split64(lo, hi) => {
                let comb = self.alloc_v();
                self.emit(B8Step::Or32 { dst: comb, a: lo, b: hi });
                self.val_mask(B8Val::Lo32(comb))
            }
        }
    }

    fn invert_mask(&mut self, m: usize) -> usize {
        let ones = self.const32(u32::MAX);
        let d = self.alloc_v();
        self.emit(B8Step::Xor32 { dst: d, a: m, b: ones });
        d
    }

    fn cmp_u32(&mut self, kind: CmpKind, a: usize, b: usize) -> usize {
        match kind {
            CmpKind::Eq => {
                let d = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: d, a, b });
                d
            }
            CmpKind::Ne => {
                let eq = self.cmp_u32(CmpKind::Eq, a, b);
                self.invert_mask(eq)
            }
            CmpKind::Ule => {
                let mn = self.alloc_v();
                self.emit(B8Step::MinU32 { dst: mn, a, b });
                self.cmp_u32(CmpKind::Eq, mn, a)
            }
            CmpKind::Uge => {
                let mx = self.alloc_v();
                self.emit(B8Step::MaxU32 { dst: mx, a, b });
                self.cmp_u32(CmpKind::Eq, mx, a)
            }
            CmpKind::Ult => {
                let ge = self.cmp_u32(CmpKind::Uge, a, b);
                self.invert_mask(ge)
            }
            CmpKind::Ugt => {
                let le = self.cmp_u32(CmpKind::Ule, a, b);
                self.invert_mask(le)
            }
            CmpKind::Gt => {
                let d = self.alloc_v();
                self.emit(B8Step::CmpGtS32 { dst: d, a, b });
                d
            }
            CmpKind::Lt => {
                let d = self.alloc_v();
                self.emit(B8Step::CmpGtS32 { dst: d, a: b, b: a });
                d
            }
            CmpKind::Le => {
                let gt = self.cmp_u32(CmpKind::Gt, a, b);
                self.invert_mask(gt)
            }
            CmpKind::Ge => {
                let lt = self.cmp_u32(CmpKind::Lt, a, b);
                self.invert_mask(lt)
            }
        }
    }

    fn cmp_u32_const(&mut self, kind: CmpKind, a: usize, c: u32) -> usize {
        match (kind, c) {
            (CmpKind::Ult, 0) => self.const32(0),
            (CmpKind::Ult, 1) => { let z = self.const32(0); self.cmp_u32(CmpKind::Eq, a, z) }
            (CmpKind::Uge, 0) => self.const32(u32::MAX),
            (CmpKind::Uge, 1) => { let z = self.const32(0); self.cmp_u32(CmpKind::Ne, a, z) }
            (CmpKind::Ugt, 0) => { let z = self.const32(0); self.cmp_u32(CmpKind::Ne, a, z) }
            (CmpKind::Ule, 0) => { let z = self.const32(0); self.cmp_u32(CmpKind::Eq, a, z) }
            (CmpKind::Ult, _) => {
                let cm1 = self.const32(c - 1);
                self.cmp_u32(CmpKind::Ule, a, cm1)
            }
            (CmpKind::Ugt, u32::MAX) => self.const32(0),
            (CmpKind::Ugt, _) => {
                let cp1 = self.const32(c + 1);
                self.cmp_u32(CmpKind::Uge, a, cp1)
            }
            (CmpKind::Ule, u32::MAX) => self.const32(u32::MAX),
            _ => {
                let cv = self.const32(c);
                self.cmp_u32(kind, a, cv)
            }
        }
    }

    fn cmp_split64_u_const(&mut self, kind: CmpKind, lo: usize, hi: usize, c: u64) -> usize {
        let (clo, chi) = (c as u32, (c >> 32) as u32);
        match (kind, chi, clo) {
            (CmpKind::Eq, _, _) => {
                let m1 = self.cmp_u32_const(CmpKind::Eq, hi, chi);
                let m2 = self.cmp_u32_const(CmpKind::Eq, lo, clo);
                let d = self.alloc_v();
                self.emit(B8Step::And32 { dst: d, a: m1, b: m2 });
                d
            }
            (CmpKind::Ne, _, _) => {
                let eq = self.cmp_split64_u_const(CmpKind::Eq, lo, hi, c);
                self.invert_mask(eq)
            }
            (CmpKind::Uge, _, 0) => self.cmp_u32_const(CmpKind::Uge, hi, chi),
            (CmpKind::Ult, _, 0) => self.cmp_u32_const(CmpKind::Ult, hi, chi),
            (CmpKind::Ult, 0, _) => {
                let hz = self.cmp_u32_const(CmpKind::Eq, hi, 0);
                let lm = self.cmp_u32_const(CmpKind::Ult, lo, clo);
                let d = self.alloc_v();
                self.emit(B8Step::And32 { dst: d, a: hz, b: lm });
                d
            }
            (CmpKind::Ule, 0, _) => {
                let hz = self.cmp_u32_const(CmpKind::Eq, hi, 0);
                let lm = self.cmp_u32_const(CmpKind::Ule, lo, clo);
                let d = self.alloc_v();
                self.emit(B8Step::And32 { dst: d, a: hz, b: lm });
                d
            }
            (CmpKind::Ule, _, 0) => {
                let hlt = self.cmp_u32_const(CmpKind::Ult, hi, chi);
                let heq = self.cmp_u32_const(CmpKind::Eq, hi, chi);
                let lz = self.cmp_u32_const(CmpKind::Eq, lo, 0);
                let eq_ok = self.alloc_v();
                self.emit(B8Step::And32 { dst: eq_ok, a: heq, b: lz });
                let d = self.alloc_v();
                self.emit(B8Step::Or32 { dst: d, a: hlt, b: eq_ok });
                d
            }
            (CmpKind::Ult | CmpKind::Ule, _, _) => {
                let hlt = self.cmp_u32_const(CmpKind::Ult, hi, chi);
                let heq = self.cmp_u32_const(CmpKind::Eq, hi, chi);
                let lok = self.cmp_u32_const(kind, lo, clo);
                let eq_ok = self.alloc_v();
                self.emit(B8Step::And32 { dst: eq_ok, a: heq, b: lok });
                let d = self.alloc_v();
                self.emit(B8Step::Or32 { dst: d, a: hlt, b: eq_ok });
                d
            }
            (CmpKind::Ugt | CmpKind::Uge | CmpKind::Lt | CmpKind::Le | CmpKind::Gt | CmpKind::Ge, _, _) => {
                let hgt = self.cmp_u32_const(CmpKind::Ugt, hi, chi);
                let heq = self.cmp_u32_const(CmpKind::Eq, hi, chi);
                let lok = self.cmp_u32_const(kind, lo, clo);
                let eq_ok = self.alloc_v();
                self.emit(B8Step::And32 { dst: eq_ok, a: heq, b: lok });
                let d = self.alloc_v();
                self.emit(B8Step::Or32 { dst: d, a: hgt, b: eq_ok });
                d
            }
        }
    }

    fn lower_add(&mut self, id: NodeId, a: NodeId, b: NodeId) -> Option<B8Val> {
        let (va, vb) = (self.vals[a]?, self.vals[b]?);
        if (self.doms[id].fits_u32() || self.demanded_lo32[id]) && self.is_zero_hi(va) && self.is_zero_hi(vb) {
            let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
            let d = self.alloc_v();
            self.emit(B8Step::Add32 { dst: d, a: la, b: lb });
            return Some(B8Val::Lo32(d));
        }
        let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
        let lo = self.alloc_v();
        self.emit(B8Step::Add32 { dst: lo, a: la, b: lb });
        let carry_mask = match (va, vb) {
            (_, B8Val::Const(c)) if (c >> 32) == 0 && (c as u32) > 0 => {
                let cm1 = self.const32((c as u32) - 1);
                let mn = self.alloc_v();
                self.emit(B8Step::MinU32 { dst: mn, a: lo, b: cm1 });
                let d = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: d, a: mn, b: lo });
                d
            }
            (B8Val::Const(c), _) if (c >> 32) == 0 && (c as u32) > 0 => {
                let cm1 = self.const32((c as u32) - 1);
                let mn = self.alloc_v();
                self.emit(B8Step::MinU32 { dst: mn, a: lo, b: cm1 });
                let d = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: d, a: mn, b: lo });
                d
            }
            _ => {
                let mx = self.alloc_v();
                self.emit(B8Step::MaxU32 { dst: mx, a: lo, b: la });
                let nc = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: nc, a: mx, b: lo });
                self.invert_mask(nc)
            }
        };
        let hi = match (self.is_zero_hi(va), self.is_zero_hi(vb)) {
            (true, true) => {
                let z = self.const32(0);
                let d = self.alloc_v();
                self.emit(B8Step::Sub32 { dst: d, a: z, b: carry_mask });
                d
            }
            (false, true) => {
                let ha = self.val_hi32(va);
                let d = self.alloc_v();
                self.emit(B8Step::Sub32 { dst: d, a: ha, b: carry_mask });
                d
            }
            (true, false) => {
                let hb = self.val_hi32(vb);
                let d = self.alloc_v();
                self.emit(B8Step::Sub32 { dst: d, a: hb, b: carry_mask });
                d
            }
            (false, false) => {
                let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
                let hs = self.alloc_v();
                self.emit(B8Step::Add32 { dst: hs, a: ha, b: hb });
                let d = self.alloc_v();
                self.emit(B8Step::Sub32 { dst: d, a: hs, b: carry_mask });
                d
            }
        };
        Some(B8Val::Split64(lo, hi))
    }

    fn lower_sub(&mut self, id: NodeId, a: NodeId, b: NodeId) -> Option<B8Val> {
        let (va, vb) = (self.vals[a]?, self.vals[b]?);
        if (self.doms[id].fits_u32() || self.demanded_lo32[id]) && self.is_zero_hi(va) && self.is_zero_hi(vb) {
            let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
            let d = self.alloc_v();
            self.emit(B8Step::Sub32 { dst: d, a: la, b: lb });
            return Some(B8Val::Lo32(d));
        }
        let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
        let lo = self.alloc_v();
        self.emit(B8Step::Sub32 { dst: lo, a: la, b: lb });
        let borrow_mask = match vb {
            B8Val::Const(c) if (c >> 32) == 0 && (c as u32) > 0 => {
                let cm1 = self.const32((c as u32) - 1);
                let mn = self.alloc_v();
                self.emit(B8Step::MinU32 { dst: mn, a: la, b: cm1 });
                let d = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: d, a: mn, b: la });
                d
            }
            _ => {
                let mn = self.alloc_v();
                self.emit(B8Step::MinU32 { dst: mn, a: la, b: lb });
                let nb = self.alloc_v();
                self.emit(B8Step::CmpEq32 { dst: nb, a: mn, b: lb });
                self.invert_mask(nb)
            }
        };
        let hd = if self.is_zero_hi(vb) {
            self.val_hi32(va)
        } else {
            let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
            let d = self.alloc_v();
            self.emit(B8Step::Sub32 { dst: d, a: ha, b: hb });
            d
        };
        let hi = self.alloc_v();
        self.emit(B8Step::Add32 { dst: hi, a: hd, b: borrow_mask });
        Some(B8Val::Split64(lo, hi))
    }

    fn lower_bitwise(&mut self, id: NodeId, op: Op<NodeId>, a: NodeId, b: NodeId) -> Option<B8Val> {
        let (va, vb) = (self.vals[a]?, self.vals[b]?);
        if self.is_mask[id] {
            let (ma, mb) = (self.val_mask(va), self.val_mask(vb));
            let d = self.alloc_v();
            let step = match op {
                Op::BitAnd(_, _) => B8Step::And32 { dst: d, a: ma, b: mb },
                Op::BitOr(_, _) => B8Step::Or32 { dst: d, a: ma, b: mb },
                _ => B8Step::Xor32 { dst: d, a: ma, b: mb },
            };
            self.emit(step);
            return Some(B8Val::Mask32(d));
        }
        match op {
            Op::BitAnd(_, _) => {
                if let B8Val::Const(0xFFFF_FFFF) = vb { return Some(B8Val::Lo32(self.val_lo32(va))); }
                if let B8Val::Const(0xFFFF_FFFF) = va { return Some(B8Val::Lo32(self.val_lo32(vb))); }
                let zlo = self.is_zero_lo(va) || self.is_zero_lo(vb);
                let zhi = self.is_zero_hi(va) || self.is_zero_hi(vb) || self.demanded_lo32[id];
                if zlo && zhi { return Some(B8Val::Const(0)); }
                if zhi {
                    let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                    let d = self.alloc_v();
                    self.emit(B8Step::And32 { dst: d, a: la, b: lb });
                    return Some(B8Val::Lo32(d));
                }
                if zlo {
                    let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
                    let d = self.alloc_v();
                    self.emit(B8Step::And32 { dst: d, a: ha, b: hb });
                    return Some(B8Val::Hi32(d));
                }
                let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
                let (dl, dh) = (self.alloc_v(), self.alloc_v());
                self.emit(B8Step::And32 { dst: dl, a: la, b: lb });
                self.emit(B8Step::And32 { dst: dh, a: ha, b: hb });
                Some(B8Val::Split64(dl, dh))
            }
            Op::BitOr(_, _) => {
                if self.is_zero_lo(va) && self.is_zero_hi(vb) {
                    return Some(B8Val::Split64(self.val_lo32(vb), self.val_hi32(va)));
                }
                if self.is_zero_hi(va) && self.is_zero_lo(vb) {
                    return Some(B8Val::Split64(self.val_lo32(va), self.val_hi32(vb)));
                }
                let zhi = self.is_zero_hi(va) && self.is_zero_hi(vb);
                let zlo = self.is_zero_lo(va) && self.is_zero_lo(vb);
                if zhi || self.demanded_lo32[id] {
                    let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                    let d = self.alloc_v();
                    self.emit(B8Step::Or32 { dst: d, a: la, b: lb });
                    return Some(B8Val::Lo32(d));
                }
                if zlo {
                    let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
                    let d = self.alloc_v();
                    self.emit(B8Step::Or32 { dst: d, a: ha, b: hb });
                    return Some(B8Val::Hi32(d));
                }
                let dl = if self.is_zero_lo(va) { self.val_lo32(vb) }
                    else if self.is_zero_lo(vb) { self.val_lo32(va) }
                    else { let (la, lb) = (self.val_lo32(va), self.val_lo32(vb)); let d = self.alloc_v(); self.emit(B8Step::Or32 { dst: d, a: la, b: lb }); d };
                let dh = if self.is_zero_hi(va) { self.val_hi32(vb) }
                    else if self.is_zero_hi(vb) { self.val_hi32(va) }
                    else { let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb)); let d = self.alloc_v(); self.emit(B8Step::Or32 { dst: d, a: ha, b: hb }); d };
                Some(B8Val::Split64(dl, dh))
            }
            Op::BitXor(_, _) => {
                if (self.is_zero_hi(va) && self.is_zero_hi(vb)) || self.demanded_lo32[id] {
                    let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                    let d = self.alloc_v();
                    self.emit(B8Step::Xor32 { dst: d, a: la, b: lb });
                    return Some(B8Val::Lo32(d));
                }
                let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                let (ha, hb) = (self.val_hi32(va), self.val_hi32(vb));
                let (dl, dh) = (self.alloc_v(), self.alloc_v());
                self.emit(B8Step::Xor32 { dst: dl, a: la, b: lb });
                self.emit(B8Step::Xor32 { dst: dh, a: ha, b: hb });
                Some(B8Val::Split64(dl, dh))
            }
            _ => None,
        }
    }

    fn lower_cmp(&mut self, id: NodeId, kind: CmpKind, a: NodeId, b: NodeId) -> Option<B8Val> {
        let (va, vb) = (self.vals[a]?, self.vals[b]?);
        let mask = if self.is_zero_hi(va) && self.is_zero_hi(vb) {
            let eff_kind = if self.doms[a].is_nonneg() && self.doms[b].is_nonneg() {
                match kind {
                    CmpKind::Lt => CmpKind::Ult,
                    CmpKind::Le => CmpKind::Ule,
                    CmpKind::Gt => CmpKind::Ugt,
                    CmpKind::Ge => CmpKind::Uge,
                    other => other,
                }
            } else {
                kind
            };
            if !eff_kind.is_unsigned()
                && !matches!(eff_kind, CmpKind::Eq | CmpKind::Ne)
                && (!self.doms[a].fits_i32() || !self.doms[b].fits_i32())
            {
                return None;
            }
            match (va, vb) {
                (_, B8Val::Const(c)) => {
                    let la = self.val_lo32(va);
                    self.cmp_u32_const(eff_kind, la, c as u32)
                }
                (B8Val::Const(c), _) => {
                    let lb = self.val_lo32(vb);
                    let flipped = match eff_kind {
                        CmpKind::Eq => CmpKind::Eq, CmpKind::Ne => CmpKind::Ne,
                        CmpKind::Ult => CmpKind::Ugt, CmpKind::Ule => CmpKind::Uge,
                        CmpKind::Ugt => CmpKind::Ult, CmpKind::Uge => CmpKind::Ule,
                        CmpKind::Lt => CmpKind::Gt, CmpKind::Le => CmpKind::Ge,
                        CmpKind::Gt => CmpKind::Lt, CmpKind::Ge => CmpKind::Le,
                    };
                    self.cmp_u32_const(flipped, lb, c as u32)
                }
                _ => {
                    let (la, lb) = (self.val_lo32(va), self.val_lo32(vb));
                    self.cmp_u32(eff_kind, la, lb)
                }
            }
        } else if kind.is_unsigned() || matches!(kind, CmpKind::Eq | CmpKind::Ne) {
            match (va, vb) {
                (B8Val::Split64(lo, hi), B8Val::Const(c)) => self.cmp_split64_u_const(kind, lo, hi, c),
                (B8Val::Const(c), B8Val::Split64(lo, hi)) => {
                    let flipped = match kind {
                        CmpKind::Eq => CmpKind::Eq, CmpKind::Ne => CmpKind::Ne,
                        CmpKind::Ult => CmpKind::Ugt, CmpKind::Ule => CmpKind::Uge,
                        CmpKind::Ugt => CmpKind::Ult, CmpKind::Uge => CmpKind::Ule,
                        _ => return None,
                    };
                    self.cmp_split64_u_const(flipped, lo, hi, c)
                }
                _ => return None,
            }
        } else {
            return None;
        };
        if self.is_mask[id] {
            Some(B8Val::Mask32(mask))
        } else {
            let d = self.alloc_v();
            self.emit(B8Step::ShrImm32 { dst: d, src: mask, s: 31 });
            Some(B8Val::Lo32(d))
        }
    }

    fn lower_select(&mut self, id: NodeId, cond: NodeId, t_val: NodeId, f_val: NodeId) -> Option<B8Val> {
        let (vc, vt, vf) = (self.vals[cond]?, self.vals[t_val]?, self.vals[f_val]?);
        let cm = self.val_mask(vc);
        if self.is_mask[id] {
            let (mt, mf) = (self.val_mask(vt), self.val_mask(vf));
            let d = self.alloc_v();
            if matches!(vf, B8Val::Const(0)) {
                self.emit(B8Step::And32 { dst: d, a: cm, b: mt });
            } else if matches!(vt, B8Val::Const(0)) {
                self.emit(B8Step::Andn32 { dst: d, mask_inv: cm, val: mf });
            } else {
                self.emit(B8Step::Blend32 { dst: d, f_val: mf, t_val: mt, mask: cm });
            }
            return Some(B8Val::Mask32(d));
        }
        let sel_half = |this: &mut Self, th: usize, fh: usize, t_zero: bool, f_zero: bool| -> usize {
            let d = this.alloc_v();
            if f_zero {
                this.emit(B8Step::And32 { dst: d, a: cm, b: th });
            } else if t_zero {
                this.emit(B8Step::Andn32 { dst: d, mask_inv: cm, val: fh });
            } else {
                this.emit(B8Step::Blend32 { dst: d, f_val: fh, t_val: th, mask: cm });
            }
            d
        };
        if (self.is_zero_hi(vt) && self.is_zero_hi(vf)) || self.demanded_lo32[id] {
            let (tl, fl) = (self.val_lo32(vt), self.val_lo32(vf));
            return Some(B8Val::Lo32(sel_half(self, tl, fl, self.is_zero_lo(vt), self.is_zero_lo(vf))));
        }
        let (tl, fl) = (self.val_lo32(vt), self.val_lo32(vf));
        let (th, fh) = (self.val_hi32(vt), self.val_hi32(vf));
        let dl = sel_half(self, tl, fl, self.is_zero_lo(vt), self.is_zero_lo(vf));
        let dh = sel_half(self, th, fh, self.is_zero_hi(vt), self.is_zero_hi(vf));
        Some(B8Val::Split64(dl, dh))
    }

    fn lower_node(&mut self, id: NodeId, skip_mul: &[bool]) -> Option<()> {
        if skip_mul[id] { return Some(()); }
        let op = *self.k.nodes.get(id)?;
        let val = match op {
            Op::MemInit | Op::Store { .. } => B8Val::Const(0),
            Op::Const(c) => B8Val::Const(c as u64),
            Op::Load { buf, mem, .. } => {
                if self.k.nodes.get(mem).copied() != Some(Op::MemInit) { return None; }
                let bdecl = self.k.bufs.get(buf as usize)?;
                if bdecl.lanes != 1 { return None; }
                match bdecl.elem_ty {
                    ElemTy::U8 | ElemTy::U16 | ElemTy::U32 => {
                        let d = self.alloc_v();
                        self.emit(B8Step::LoadNarrow { dst: d, buf, ty: bdecl.elem_ty, permute: self.use_perm });
                        B8Val::Lo32(d)
                    }
                    ElemTy::I32 | ElemTy::I16 if bdecl.domain.is_nonneg() => {
                        let d = self.alloc_v();
                        self.emit(B8Step::LoadNarrow { dst: d, buf, ty: bdecl.elem_ty, permute: self.use_perm });
                        B8Val::Lo32(d)
                    }
                    ElemTy::U64 | ElemTy::I64 => {
                        let (lo, hi) = (self.alloc_v(), self.alloc_v());
                        self.emit(B8Step::LoadSplit64 { lo, hi, buf });
                        B8Val::Split64(lo, hi)
                    }
                    _ => return None,
                }
            }
            Op::Add(a, b) => self.lower_add(id, a, b)?,
            Op::Sub(a, b) => self.lower_sub(id, a, b)?,
            Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => self.lower_bitwise(id, op, a, b)?,
            Op::Shl(a, s) => {
                let va = self.vals[a]?;
                if s == 0 { va }
                else if s >= 32 {
                    let la = self.val_lo32(va);
                    if s == 32 { B8Val::Hi32(la) }
                    else {
                        let d = self.alloc_v();
                        self.emit(B8Step::ShlImm32 { dst: d, src: la, s: s - 32 });
                        B8Val::Hi32(d)
                    }
                } else if self.doms[id].fits_u32() || self.demanded_lo32[id] {
                    let la = self.val_lo32(va);
                    let d = self.alloc_v();
                    self.emit(B8Step::ShlImm32 { dst: d, src: la, s });
                    B8Val::Lo32(d)
                } else {
                    return None;
                }
            }
            Op::Ushr(a, s) | Op::Shr(a, s) => {
                if matches!(op, Op::Shr(_, _)) && !self.doms[a].is_nonneg() { return None; }
                if s >= 32 && skip_mul[a] {
                    let Op::Mul(mx, mc) = *self.k.nodes.get(a)? else { return None; };
                    let (x_id, c_val) = match (self.k.nodes.get(mx), self.k.nodes.get(mc)) {
                        (_, Some(&Op::Const(c))) if c >= 0 && c <= u32::MAX as i64 => (mx, c as u32),
                        (Some(&Op::Const(c)), _) if c >= 0 && c <= u32::MAX as i64 => (mc, c as u32),
                        _ => return None,
                    };
                    let vx = self.vals[x_id]?;
                    let x_lo = self.val_lo32(vx);
                    let x_hi = if self.is_zero_hi(vx) { None } else { Some(self.val_hi32(vx)) };
                    let c_v = self.const32(c_val);
                    let hi_mode = if self.doms[x_id].hi < (1i64 << 33) { MulHiMode::Pos01 } else { MulHiMode::General };
                    let d = self.alloc_v();
                    self.emit(B8Step::MulHiShr { dst: d, x_lo, x_hi, c: c_v, shift: s - 32, hi_mode });
                    B8Val::Lo32(d)
                } else {
                    let va = self.vals[a]?;
                    if s == 0 { va }
                    else if s >= 32 {
                        let ha = self.val_hi32(va);
                        if s == 32 { B8Val::Lo32(ha) }
                        else {
                            let d = self.alloc_v();
                            self.emit(B8Step::ShrImm32 { dst: d, src: ha, s: s - 32 });
                            B8Val::Lo32(d)
                        }
                    } else if self.is_zero_hi(va) {
                        let la = self.val_lo32(va);
                        let d = self.alloc_v();
                        self.emit(B8Step::ShrImm32 { dst: d, src: la, s });
                        B8Val::Lo32(d)
                    } else {
                        return None;
                    }
                }
            }
            Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
                if !self.demanded_lo32[id] && !self.doms[id].fits_u32() { return None; }
                let (va, vb) = (self.vals[a]?, self.vals[b]?);
                if matches!(op, Op::ShrVarU(_, _)) && !self.is_zero_hi(va) { return None; }
                let la = self.val_lo32(va);
                let lb = self.val_lo32(vb);
                let sh = if self.doms[b].is_nonneg() && self.doms[b].hi < 64 {
                    lb
                } else {
                    let m63 = self.const32(63);
                    let d = self.alloc_v();
                    self.emit(B8Step::And32 { dst: d, a: lb, b: m63 });
                    d
                };
                let d = self.alloc_v();
                let step = if matches!(op, Op::ShlVar(_, _)) {
                    B8Step::ShlVar32 { dst: d, a: la, b: sh }
                } else {
                    B8Step::ShrVar32 { dst: d, a: la, b: sh }
                };
                self.emit(step);
                B8Val::Lo32(d)
            }
            Op::Cmp(kind, a, b) => self.lower_cmp(id, kind, a, b)?,
            Op::Select { cond, t_val, f_val } => self.lower_select(id, cond, t_val, f_val)?,
            _ => return None,
        };
        self.vals[id] = Some(val);
        Some(())
    }
}

fn find_fused_barrett_muls(k: &Kernel, sched: &ScheduleReport, doms: &[Domain]) -> Vec<bool> {
    let n = k.nodes.len();
    let mut uses = vec![0usize; n];
    for &id in &sched.order {
        if let Some(&op) = k.nodes.get(id) {
            let (kids, cnt) = op.children();
            for &kid in &kids[..cnt] { if let Some(u) = uses.get_mut(kid) { *u += 1; } }
        }
    }
    for o in &k.outputs { if let Some(u) = uses.get_mut(o.val) { *u += 1; } }
    let mut skip = vec![false; n];
    for &id in &sched.order {
        let Some(&op) = k.nodes.get(id) else { continue; };
        let (a, s) = match op { Op::Shr(a, s) | Op::Ushr(a, s) if s >= 32 => (a, s), _ => continue };
        let _ = s;
        if uses[a] != 1 || !doms[a].is_nonneg() { continue; }
        if let Some(&Op::Mul(mx, mc)) = k.nodes.get(a) {
            let ok = match (k.nodes.get(mx), k.nodes.get(mc)) {
                (_, Some(&Op::Const(c))) => c >= 0 && c <= u32::MAX as i64 && doms[mx].is_nonneg(),
                (Some(&Op::Const(c)), _) => c >= 0 && c <= u32::MAX as i64 && doms[mc].is_nonneg(),
                _ => false,
            };
            if ok { skip[a] = true; }
        }
    }
    skip
}

type B8RegAlloc = (Vec<Option<u8>>, Vec<(u8, u32)>);

fn alloc_b8_regs(steps: &[B8Step], consts: &[(usize, u32)], next_v: usize, max_vecs: usize) -> Option<B8RegAlloc> {
    let mut is_const = vec![false; next_v];
    let mut const_uses = vec![0usize; next_v];
    for &(v, _) in consts { is_const[v] = true; }
    let mut last_use = vec![0usize; next_v];
    for (i, st) in steps.iter().enumerate() {
        st.for_each_use(|u| {
            last_use[u] = last_use[u].max(i);
            if is_const[u] { const_uses[u] += 1; }
        });
    }
    let pool_limit = max_vecs.min(13);
    let mut free: Vec<u8> = (0..pool_limit as u8).rev().collect();
    let mut phys: Vec<Option<u8>> = vec![None; next_v];
    let mut active: Vec<(usize, u8)> = Vec::new();
    let mut used_mask = 0u16;
    for (i, st) in steps.iter().enumerate() {
        let (defs, nd) = st.defs();
        for &d in &defs[..nd] {
            let r = free.pop()?;
            used_mask |= 1u16 << r;
            phys[d] = Some(r);
            if last_use[d] > i { active.push((d, r)); } else { free.push(r); }
        }
        let mut j = 0usize;
        while j < active.len() {
            let (av, ar) = active[j];
            if last_use[av] <= i { active.swap_remove(j); free.push(ar); } else { j += 1; }
        }
    }
    let mut ranked: Vec<(usize, u32, usize)> = consts.iter()
        .filter(|&&(v, _)| const_uses[v] > 0)
        .map(|&(v, c)| (v, c, const_uses[v]))
        .collect();
    ranked.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    let mut hoisted = Vec::new();
    for r in 0..pool_limit as u8 {
        if (used_mask & (1u16 << r)) == 0 {
            if let Some(&(v, c, _)) = ranked.get(hoisted.len()) {
                phys[v] = Some(r);
                hoisted.push((r, c));
            }
        }
    }
    Some((phys, hoisted))
}

fn b8_const_val(consts: &[(usize, u32)], next_v: usize) -> Vec<Option<u32>> {
    let mut vals = vec![None; next_v];
    for &(v, c) in consts {
        if let Some(slot) = vals.get_mut(v) { *slot = Some(c); }
    }
    vals
}

fn b8_def_map(steps: &[B8Step], next_v: usize) -> Vec<Option<usize>> {
    let mut defs = vec![None; next_v];
    for (idx, st) in steps.iter().enumerate() {
        let (ds, nd) = st.defs();
        for &d in &ds[..nd] {
            if let Some(slot) = defs.get_mut(d) { *slot = Some(idx); }
        }
    }
    defs
}

fn b8_u32_bound(v: usize, steps: &[B8Step], defs: &[Option<usize>], cvals: &[Option<u32>]) -> Option<(usize, u32, bool)> {
    let st = *steps.get((*defs.get(v)?)?)?;
    match st {
        B8Step::CmpEq32 { a, b, .. } => {
            for (m_v, x_v) in [(a, b), (b, a)] {
                let Some(&Some(midx)) = defs.get(m_v) else { continue; };
                match *steps.get(midx)? {
                    B8Step::MaxU32 { a: x1, b: cv, .. } if x1 == x_v => {
                        if let Some(c) = cvals.get(cv).copied().flatten() { return Some((x_v, c, true)); }
                    }
                    B8Step::MaxU32 { a: cv, b: x1, .. } if x1 == x_v => {
                        if let Some(c) = cvals.get(cv).copied().flatten() { return Some((x_v, c, true)); }
                    }
                    B8Step::MinU32 { a: x1, b: cv, .. } if x1 == x_v => {
                        if let Some(c) = cvals.get(cv).copied().flatten() { return Some((x_v, c, false)); }
                    }
                    B8Step::MinU32 { a: cv, b: x1, .. } if x1 == x_v => {
                        if let Some(c) = cvals.get(cv).copied().flatten() { return Some((x_v, c, false)); }
                    }
                    _ => {}
                }
            }
            None
        }
        B8Step::Xor32 { a, b, .. } => {
            let eq_v = if cvals.get(b).copied().flatten() == Some(u32::MAX) { a }
                else if cvals.get(a).copied().flatten() == Some(u32::MAX) { b }
                else { return None; };
            let eq_st = *steps.get((*defs.get(eq_v)?)?)?;
            if let B8Step::CmpEq32 { a: ea, b: eb, .. } = eq_st {
                if cvals.get(eb).copied().flatten() == Some(0) { return Some((ea, 1, true)); }
                if cvals.get(ea).copied().flatten() == Some(0) { return Some((eb, 1, true)); }
            }
            None
        }
        _ => None,
    }
}

fn b8_ensure_const(consts: &mut Vec<(usize, u32)>, next_v: &mut usize, val: u32) -> usize {
    if let Some(&(v, _)) = consts.iter().find(|&&(_, c)| c == val) { return v; }
    let v = *next_v;
    *next_v += 1;
    consts.push((v, val));
    v
}

fn peephole_b8_steps(steps: &mut Vec<B8Step>, consts: &mut Vec<(usize, u32)>, next_v: &mut usize) {
    // Pass 1: Collapse `And32(x >= c, x <= c)` (direct or nested) into `CmpEq32(x, c)`.
    let cvals = b8_const_val(consts, *next_v);
    let defs = b8_def_map(steps, *next_v);
    let mut out1 = Vec::with_capacity(steps.len());
    for &st in steps.iter() {
        let mut rewritten = false;
        if let B8Step::And32 { dst, a, b } = st {
            if let (Some((x1, c1, g1)), Some((x2, c2, g2))) = (
                b8_u32_bound(a, steps, &defs, &cvals),
                b8_u32_bound(b, steps, &defs, &cvals),
            ) {
                if x1 == x2 && c1 == c2 && g1 != g2 {
                    let cv = b8_ensure_const(consts, next_v, c1);
                    out1.push(B8Step::CmpEq32 { dst, a: x1, b: cv });
                    rewritten = true;
                }
            }
            if !rewritten {
                for (outer_v, inner_v) in [(a, b), (b, a)] {
                    let Some((x1, c1, g1)) = b8_u32_bound(outer_v, steps, &defs, &cvals) else { continue; };
                    let Some(&Some(iidx)) = defs.get(inner_v) else { continue; };
                    if let Some(&B8Step::And32 { a: ia, b: ib, .. }) = steps.get(iidx) {
                        for (cand, other) in [(ia, ib), (ib, ia)] {
                            if let Some((x2, c2, g2)) = b8_u32_bound(cand, steps, &defs, &cvals) {
                                if x1 == x2 && c1 == c2 && g1 != g2 {
                                    let cv = b8_ensure_const(consts, next_v, c1);
                                    let eq_v = *next_v;
                                    *next_v += 1;
                                    out1.push(B8Step::CmpEq32 { dst: eq_v, a: x1, b: cv });
                                    out1.push(B8Step::And32 { dst, a: eq_v, b: other });
                                    rewritten = true;
                                    break;
                                }
                            }
                        }
                    }
                    if rewritten { break; }
                }
            }
        }
        if !rewritten { out1.push(st); }
    }
    *steps = out1;

    // Pass 2: Fold `And32(x, Xor32(y, -1)) -> Andn32(y, x)` and direct negative-mask `MulHiShr`.
    let cvals = b8_const_val(consts, *next_v);
    let defs = b8_def_map(steps, *next_v);
    let mut out2 = Vec::with_capacity(steps.len());
    for &st in steps.iter() {
        let next_st = match st {
            B8Step::And32 { dst, a, b } => {
                let mut folded = None;
                for (inv_cand, val_cand) in [(b, a), (a, b)] {
                    if let Some(&Some(idx)) = defs.get(inv_cand) {
                        if let Some(&B8Step::Xor32 { a: xa, b: xb, .. }) = steps.get(idx) {
                            if cvals.get(xb).copied().flatten() == Some(u32::MAX) {
                                folded = Some(B8Step::Andn32 { dst, mask_inv: xa, val: val_cand });
                                break;
                            }
                            if cvals.get(xa).copied().flatten() == Some(u32::MAX) {
                                folded = Some(B8Step::Andn32 { dst, mask_inv: xb, val: val_cand });
                                break;
                            }
                        }
                    }
                }
                folded.unwrap_or(st)
            }
            B8Step::MulHiShr { dst, x_lo, x_hi: Some(xh), c, shift, hi_mode: MulHiMode::Pos01 } => {
                let mut folded = None;
                if let Some(&Some(idx)) = defs.get(xh) {
                    if let Some(&B8Step::Sub32 { a: sa, b: sb, .. }) = steps.get(idx) {
                        if cvals.get(sa).copied().flatten() == Some(0) {
                            folded = Some(B8Step::MulHiShr {
                                dst, x_lo, x_hi: Some(sb), c, shift, hi_mode: MulHiMode::NegMask,
                            });
                        }
                    }
                }
                folded.unwrap_or(st)
            }
            other => other,
        };
        out2.push(next_st);
    }

    // Pass 3: Backward dead-code elimination rooted at side-effecting stores.
    let mut live = vec![false; *next_v];
    let mut kept = Vec::with_capacity(out2.len());
    for &st in out2.iter().rev() {
        let is_store = matches!(st, B8Step::Store1 { .. } | B8Step::Store2U64 { .. });
        let (ds, nd) = st.defs();
        let def_live = ds[..nd].iter().any(|&d| live.get(d).copied().unwrap_or(false));
        if is_store || def_live {
            st.for_each_use(|u| {
                if let Some(slot) = live.get_mut(u) { *slot = true; }
            });
            kept.push(st);
        }
    }
    kept.reverse();
    *steps = kept;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SymExpr {
    Const(u64),
    BufBits { buf: u8, and_mask: u64, or_mask: u64 },
    DescHi(u8),
    NetSrc(u8),
    NetCidr(u8),
    Prefix(u8),
    PrefixShift(u8),
    PrefixMask(u8),
    NetXor(u8),
    Any,
}

#[derive(Clone, Copy)]
struct SymInfo {
    lo: u64,
    hi: u64,
    expr: SymExpr,
}

impl SymInfo {
    fn from_const(c: u64) -> Self { Self { lo: c, hi: c, expr: SymExpr::Const(c) } }
    fn any() -> Self { Self { lo: 0, hi: u64::MAX, expr: SymExpr::Any } }
    fn range(lo: u64, hi: u64) -> Self {
        if lo == hi { Self::from_const(lo) } else { Self { lo, hi, expr: SymExpr::Any } }
    }
}

fn sym_const_of(k: &Kernel, id: NodeId) -> Option<u64> {
    match k.nodes.get(id) {
        Some(&Op::Const(c)) => Some(c as u64),
        _ => None,
    }
}

fn sym_load_buf(k: &Kernel, id: NodeId) -> Option<u8> {
    match k.nodes.get(id) {
        Some(&Op::Load { buf, mem, .. }) if k.nodes.get(mem).copied() == Some(Op::MemInit) => Some(buf),
        _ => None,
    }
}

fn extract_fastpath_candidate(k: &Kernel, out_buf: u8) -> Option<B8FastPath> {
    // 1. Find a Barrett `(lo + add_c) * m >> s` feeding off `load_desc & 0xFFFF_FFFF`.
    let mut desc_info: Option<(u8, NodeId, NodeId, u32, u32)> = None;
    for (id, &op) in k.nodes.iter().enumerate() {
        let (mul_id, s) = match op { Op::Ushr(a, s) | Op::Shr(a, s) if (32..64).contains(&s) => (a, s), _ => continue };
        let Some(&Op::Mul(ma, mb)) = k.nodes.get(mul_id) else { continue; };
        let (add_id, m) = match (sym_const_of(k, ma), sym_const_of(k, mb)) {
            (_, Some(c)) if c > 0 && c <= u32::MAX as u64 => (ma, c),
            (Some(c), _) if c > 0 && c <= u32::MAX as u64 => (mb, c),
            _ => continue,
        };
        let Some(&Op::Add(aa, ab)) = k.nodes.get(add_id) else { continue; };
        let (len_id, add_c) = match (sym_const_of(k, aa), sym_const_of(k, ab)) {
            (_, Some(c)) if c > 0 && c < 0x1_0000 => (aa, c),
            (Some(c), _) if c > 0 && c < 0x1_0000 => (ab, c),
            _ => continue,
        };
        let Some(&Op::BitAnd(la, lb)) = k.nodes.get(len_id) else { continue; };
        let desc_load_id = match (sym_const_of(k, la), sym_const_of(k, lb)) {
            (_, Some(0xFFFF_FFFF)) => la,
            (Some(0xFFFF_FFFF), _) => lb,
            _ => continue,
        };
        let Some(desc_buf) = sym_load_buf(k, desc_load_id) else { continue; };
        let q1 = ((1u128 + u128::from(add_c)) * u128::from(m)) >> s;
        let lo_max = (add_c + 1) as u32;
        let q_max = ((u128::from(lo_max) + u128::from(add_c)) * u128::from(m)) >> s;
        if q1 == q_max && lo_max >= 1 {
            desc_info = Some((desc_buf, desc_load_id, len_id, 1, lo_max));
            let _ = id;
            break;
        }
    }
    let (desc_buf, desc_load_id, len_id, desc_lo_min, desc_lo_max) = desc_info?;

    // 2. Find optional `(desc >> 32) & flag_mask == 0` guard mask.
    let mut desc_hi_zero_mask = 0u32;
    for &op in &k.nodes {
        let Op::Cmp(CmpKind::Eq, ca, cb) = op else { continue; };
        let masked_id = match (sym_const_of(k, ca), sym_const_of(k, cb)) {
            (_, Some(0)) => ca,
            (Some(0), _) => cb,
            _ => continue,
        };
        if let Some(&Op::BitAnd(ba, bb)) = k.nodes.get(masked_id) {
            let (sh_id, fm) = match (sym_const_of(k, ba), sym_const_of(k, bb)) {
                (_, Some(c)) if c > 0 && c <= u32::MAX as u64 => (ba, c as u32),
                (Some(c), _) if c > 0 && c <= u32::MAX as u64 => (bb, c as u32),
                _ => continue,
            };
            if matches!(k.nodes.get(sh_id), Some(&Op::Ushr(x, 32) | &Op::Shr(x, 32)) if x == desc_load_id) {
                desc_hi_zero_mask |= fm;
            }
        }
    }

    // 3. Find primary 64-bit address buffer paired with `len_id` in `Add(addr, len)` and `addr < bound`.
    let mut addr_buf_cand: Option<(u8, NodeId)> = None;
    for &op in &k.nodes {
        let Op::Add(aa, ab) = op else { continue; };
        let other = if aa == len_id { ab } else if ab == len_id { aa } else { continue };
        if let Some(abuf) = sym_load_buf(k, other) {
            if abuf != desc_buf {
                addr_buf_cand = Some((abuf, other));
                break;
            }
        }
    }
    let (addr_buf, addr_load_id) = addr_buf_cand?;
    let mut addr_bound: Option<u64> = None;
    for &op in &k.nodes {
        let Op::Cmp(kind, ca, cb) = op else { continue; };
        let bound = match (kind, ca == addr_load_id, cb == addr_load_id) {
            (CmpKind::Ult | CmpKind::Lt, true, false) => sym_const_of(k, cb),
            (CmpKind::Ugt | CmpKind::Gt, false, true) => sym_const_of(k, ca),
            _ => None,
        };
        if let Some(b) = bound {
            if b > u64::from(desc_lo_max) && b <= 0xFFFF_FFFF {
                addr_bound = Some(addr_bound.map_or(b, |prev| prev.min(b)));
            }
        }
    }
    let bound = addr_bound?;
    let mut p2 = 1u64 << 31;
    while p2 > 1 && (p2 - 1).saturating_add(u64::from(desc_lo_max)) > bound {
        p2 >>= 1;
    }
    if p2 <= u64::from(desc_lo_max) { return None; }
    let addr_zero_mask = !(p2 - 1);

    // 4. Find optional `(net_buf, prefix_buf)` CIDR/meta guard from IR.
    let mut net_guard = None;
    for (p_idx, &p_op) in k.nodes.iter().enumerate() {
        let Some(pbuf) = sym_load_buf(k, p_idx) else { continue; };
        let _ = p_op;
        if pbuf == addr_buf || pbuf == desc_buf { continue; }
        let Some(pb) = k.bufs.get(pbuf as usize) else { continue; };
        if pb.elem_ty != ElemTy::U8 || pb.lanes != 1 { continue; }
        let mut prefix_max = None;
        for &op in &k.nodes {
            let Op::Cmp(kind, ca, cb) = op else { continue; };
            let c = match (kind, ca == p_idx, cb == p_idx) {
                (CmpKind::Ule | CmpKind::Le, true, false) => sym_const_of(k, cb),
                (CmpKind::Uge | CmpKind::Ge, false, true) => sym_const_of(k, ca),
                _ => None,
            };
            if c == Some(32) { prefix_max = Some(32u32); break; }
        }
        let Some(pmax) = prefix_max else { continue; };
        for (n_idx, _) in k.nodes.iter().enumerate() {
            let Some(nbuf) = sym_load_buf(k, n_idx) else { continue; };
            if nbuf == addr_buf || nbuf == desc_buf || nbuf == pbuf { continue; }
            let Some(nb) = k.bufs.get(nbuf as usize) else { continue; };
            if !matches!(nb.elem_ty, ElemTy::U64 | ElemTy::I64) || nb.lanes != 1 { continue; }
            let mut reject_src = None;
            for &op in &k.nodes {
                let Op::Cmp(CmpKind::Ne, ca, cb) = op else { continue; };
                for (lhs, rhs) in [(ca, cb), (cb, ca)] {
                    let Some(rc) = sym_const_of(k, rhs) else { continue; };
                    if rc > u32::MAX as u64 { continue; }
                    if let Some(&Op::BitAnd(la, lb)) = k.nodes.get(lhs) {
                        let is_src = (la == n_idx && sym_const_of(k, lb) == Some(0xFFFF_FFFF))
                            || (lb == n_idx && sym_const_of(k, la) == Some(0xFFFF_FFFF));
                        if is_src { reject_src = Some(rc as u32); }
                    }
                }
            }
            net_guard = Some(B8NetGuard { net_buf: nbuf, prefix_buf: pbuf, prefix_max: pmax, reject_src_const: reject_src });
            break;
        }
        if net_guard.is_some() { break; }
    }

    Some(B8FastPath {
        addr_buf,
        addr_zero_mask,
        desc_buf,
        desc_hi_zero_mask,
        desc_lo_min,
        desc_lo_max,
        net_guard,
        out_buf,
        out0_and: u64::MAX,
        out0_or: 0,
        out1_and: u64::MAX,
        out1_or: 0,
    })
}

fn sym_cmp_eval(kind: CmpKind, a: SymInfo, b: SymInfo, ng: Option<B8NetGuard>) -> SymInfo {
    if matches!(kind, CmpKind::Ne) {
        if let Some(g) = ng {
            if let Some(rej) = g.reject_src_const {
                let m = (a.expr == SymExpr::NetSrc(g.net_buf) && b.expr == SymExpr::Const(u64::from(rej)))
                    || (b.expr == SymExpr::NetSrc(g.net_buf) && a.expr == SymExpr::Const(u64::from(rej)));
                if m { return SymInfo::from_const(1); }
            }
        }
    }
    let nonneg = a.hi <= i64::MAX as u64 && b.hi <= i64::MAX as u64;
    let eff = match kind {
        CmpKind::Lt if nonneg => CmpKind::Ult,
        CmpKind::Le if nonneg => CmpKind::Ule,
        CmpKind::Gt if nonneg => CmpKind::Ugt,
        CmpKind::Ge if nonneg => CmpKind::Uge,
        other => other,
    };
    let res = match eff {
        CmpKind::Eq => {
            if a.lo == a.hi && b.lo == b.hi { Some(u64::from(a.lo == b.lo)) }
            else if a.hi < b.lo || b.hi < a.lo { Some(0) }
            else { None }
        }
        CmpKind::Ne => {
            if a.lo == a.hi && b.lo == b.hi { Some(u64::from(a.lo != b.lo)) }
            else if a.hi < b.lo || b.hi < a.lo { Some(1) }
            else { None }
        }
        CmpKind::Ult => if a.hi < b.lo { Some(1) } else if a.lo >= b.hi { Some(0) } else { None },
        CmpKind::Ule => if a.hi <= b.lo { Some(1) } else if a.lo > b.hi { Some(0) } else { None },
        CmpKind::Ugt => if a.lo > b.hi { Some(1) } else if a.hi <= b.lo { Some(0) } else { None },
        CmpKind::Uge => if a.lo >= b.hi { Some(1) } else if a.hi < b.lo { Some(0) } else { None },
        _ => None,
    };
    res.map_or_else(|| SymInfo::range(0, 1), SymInfo::from_const)
}

fn prove_fastpath_on_ir(k: &Kernel, sched: &ScheduleReport, mut fp: B8FastPath, lanes: [NodeId; 2]) -> Option<B8FastPath> {
    let mut sym = vec![SymInfo::any(); k.nodes.len()];
    for (idx, &op) in k.nodes.iter().enumerate() {
        if let Op::Const(c) = op { sym[idx] = SymInfo::from_const(c as u64); }
    }
    let ng = fp.net_guard;
    for &id in &sched.order {
        let Some(&op) = k.nodes.get(id) else { continue; };
        let info = match op {
            Op::Const(c) => SymInfo::from_const(c as u64),
            Op::MemInit | Op::Store { .. } => SymInfo::from_const(0),
            Op::Load { buf, mem, .. } => {
                if k.nodes.get(mem).copied() != Some(Op::MemInit) {
                    SymInfo::any()
                } else if buf == fp.addr_buf {
                    SymInfo { lo: 0, hi: !fp.addr_zero_mask, expr: SymExpr::BufBits { buf, and_mask: u64::MAX, or_mask: 0 } }
                } else if buf == fp.desc_buf {
                    let hi_max = u64::from(u32::MAX & !fp.desc_hi_zero_mask);
                    SymInfo {
                        lo: u64::from(fp.desc_lo_min),
                        hi: (hi_max << 32) | u64::from(fp.desc_lo_max),
                        expr: SymExpr::BufBits { buf, and_mask: u64::MAX, or_mask: 0 },
                    }
                } else if ng.is_some_and(|g| g.net_buf == buf) {
                    SymInfo { lo: 0, hi: u64::MAX, expr: SymExpr::BufBits { buf, and_mask: u64::MAX, or_mask: 0 } }
                } else if let Some(g) = ng.filter(|g| g.prefix_buf == buf) {
                    SymInfo { lo: 0, hi: u64::from(g.prefix_max), expr: SymExpr::Prefix(buf) }
                } else {
                    SymInfo::any()
                }
            }
            Op::Add(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                match (sa.lo.checked_add(sb.lo), sa.hi.checked_add(sb.hi)) {
                    (Some(lo), Some(hi)) => SymInfo::range(lo, hi),
                    _ => SymInfo::any(),
                }
            }
            Op::Sub(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                if let (SymExpr::Const(32), SymExpr::Prefix(pb)) = (sa.expr, sb.expr) {
                    if let Some(g) = ng.filter(|g| g.prefix_buf == pb && g.prefix_max <= 32) {
                        SymInfo { lo: u64::from(32 - g.prefix_max), hi: 32, expr: SymExpr::PrefixShift(pb) }
                    } else { SymInfo::any() }
                } else if sb.hi <= sa.lo {
                    SymInfo::range(sa.lo - sb.hi, sa.hi - sb.lo)
                } else {
                    SymInfo::any()
                }
            }
            Op::Mul(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                match (sa.lo.checked_mul(sb.lo), sa.hi.checked_mul(sb.hi)) {
                    (Some(lo), Some(hi)) => SymInfo::range(lo, hi),
                    _ => SymInfo::any(),
                }
            }
            Op::Ushr(a, s) | Op::Shr(a, s) => {
                let sa = sym[a];
                if matches!(op, Op::Shr(_, _)) && sa.hi > i64::MAX as u64 {
                    SymInfo::any()
                } else if s >= 64 {
                    SymInfo::from_const(0)
                } else if s == 32 && matches!(sa.expr, SymExpr::BufBits { buf, and_mask: u64::MAX, or_mask: 0 } if buf == fp.desc_buf) {
                    SymInfo { lo: 0, hi: u64::from(u32::MAX & !fp.desc_hi_zero_mask), expr: SymExpr::DescHi(fp.desc_buf) }
                } else if s == 32 && ng.is_some_and(|g| matches!(sa.expr, SymExpr::BufBits { buf, and_mask: u64::MAX, or_mask: 0 } if buf == g.net_buf)) {
                    SymInfo { lo: 0, hi: u64::from(u32::MAX), expr: SymExpr::NetCidr(ng.map_or(0, |g| g.net_buf)) }
                } else {
                    SymInfo::range(sa.lo >> s, sa.hi >> s)
                }
            }
            Op::Shl(a, s) => {
                let sa = sym[a];
                if let SymExpr::Const(c) = sa.expr {
                    SymInfo::from_const(if s < 64 { c << s } else { 0 })
                } else {
                    SymInfo::any()
                }
            }
            Op::ShlVar(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                if let (SymExpr::Const(0xFFFF_FFFF), SymExpr::PrefixShift(pb)) = (sa.expr, sb.expr) {
                    SymInfo { lo: 0, hi: u64::from(u32::MAX), expr: SymExpr::PrefixMask(pb) }
                } else {
                    SymInfo::any()
                }
            }
            Op::BitAnd(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                match (sa.expr, sb.expr) {
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymInfo::from_const(x & y),
                    (SymExpr::NetXor(nb), SymExpr::PrefixMask(pb)) | (SymExpr::PrefixMask(pb), SymExpr::NetXor(nb))
                        if ng.is_some_and(|g| g.net_buf == nb && g.prefix_buf == pb) =>
                    {
                        SymInfo::from_const(0)
                    }
                    (SymExpr::PrefixMask(pb), SymExpr::Const(0xFFFF_FFFF))
                    | (SymExpr::Const(0xFFFF_FFFF), SymExpr::PrefixMask(pb)) => {
                        SymInfo { lo: 0, hi: u64::from(u32::MAX), expr: SymExpr::PrefixMask(pb) }
                    }
                    (SymExpr::DescHi(db), SymExpr::Const(m)) | (SymExpr::Const(m), SymExpr::DescHi(db))
                        if db == fp.desc_buf && (m & !u64::from(fp.desc_hi_zero_mask)) == 0 =>
                    {
                        SymInfo::from_const(0)
                    }
                    (SymExpr::BufBits { buf, and_mask, or_mask }, SymExpr::Const(m))
                    | (SymExpr::Const(m), SymExpr::BufBits { buf, and_mask, or_mask }) => {
                        if buf == fp.desc_buf && and_mask == u64::MAX && or_mask == 0 && m == 0xFFFF_FFFF {
                            SymInfo {
                                lo: u64::from(fp.desc_lo_min),
                                hi: u64::from(fp.desc_lo_max),
                                expr: SymExpr::BufBits { buf, and_mask: m, or_mask: 0 },
                            }
                        } else if ng.is_some_and(|g| g.net_buf == buf) && and_mask == u64::MAX && or_mask == 0 && m == 0xFFFF_FFFF {
                            SymInfo { lo: 0, hi: 0xFFFF_FFFF, expr: SymExpr::NetSrc(buf) }
                        } else if buf == fp.desc_buf && or_mask == 0 && ((and_mask & m) & !(u64::from(fp.desc_hi_zero_mask) << 32)) == 0 {
                            SymInfo::from_const(0)
                        } else {
                            SymInfo { lo: 0, hi: m, expr: SymExpr::BufBits { buf, and_mask: and_mask & m, or_mask: or_mask & m } }
                        }
                    }
                    _ => SymInfo::range(0, sa.hi.min(sb.hi)),
                }
            }
            Op::BitOr(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                match (sa.expr, sb.expr) {
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymInfo::from_const(x | y),
                    (SymExpr::BufBits { buf, and_mask, or_mask }, SymExpr::Const(c))
                    | (SymExpr::Const(c), SymExpr::BufBits { buf, and_mask, or_mask }) => {
                        SymInfo { lo: c, hi: u64::MAX, expr: SymExpr::BufBits { buf, and_mask: and_mask & !c, or_mask: or_mask | c } }
                    }
                    (
                        SymExpr::BufBits { buf: b1, and_mask: a1, or_mask: o1 },
                        SymExpr::BufBits { buf: b2, and_mask: a2, or_mask: o2 },
                    ) if b1 == b2 => {
                        let or_m = o1 | o2;
                        SymInfo { lo: or_m, hi: u64::MAX, expr: SymExpr::BufBits { buf: b1, and_mask: (a1 | a2) & !or_m, or_mask: or_m } }
                    }
                    _ => SymInfo::any(),
                }
            }
            Op::BitXor(a, b) => {
                let (sa, sb) = (sym[a], sym[b]);
                match (sa.expr, sb.expr) {
                    (SymExpr::Const(x), SymExpr::Const(y)) => SymInfo::from_const(x ^ y),
                    (SymExpr::NetSrc(b1), SymExpr::NetCidr(b2)) | (SymExpr::NetCidr(b2), SymExpr::NetSrc(b1)) if b1 == b2 => {
                        SymInfo { lo: 0, hi: u64::from(u32::MAX), expr: SymExpr::NetXor(b1) }
                    }
                    _ => SymInfo::any(),
                }
            }
            Op::Cmp(kind, a, b) => sym_cmp_eval(kind, sym[a], sym[b], ng),
            Op::Select { cond, t_val, f_val } => {
                let sc = sym[cond];
                if sc.lo > 0 { sym[t_val] }
                else if sc.hi == 0 { sym[f_val] }
                else {
                    let (st, sf) = (sym[t_val], sym[f_val]);
                    let expr = if st.expr == sf.expr { st.expr } else { SymExpr::Any };
                    SymInfo { lo: st.lo.min(sf.lo), hi: st.hi.max(sf.hi), expr }
                }
            }
            _ => SymInfo::any(),
        };
        sym[id] = info;
    }

    let SymExpr::BufBits { buf: b0, and_mask: a0, or_mask: o0 } = sym[lanes[0]].expr else { return None; };
    let SymExpr::BufBits { buf: b1, and_mask: a1, or_mask: o1 } = sym[lanes[1]].expr else { return None; };
    if b0 != fp.addr_buf || b1 != fp.desc_buf { return None; }
    fp.out0_and = a0;
    fp.out0_or = o0;
    fp.out1_and = a1;
    fp.out1_or = o1;
    Some(fp)
}

fn try_plan_fastpath_superblock(k: &Kernel, sched: &ScheduleReport, out_buf: u8, lanes: [NodeId; 2]) -> Option<B8FastPath> {
    let cand = extract_fastpath_candidate(k, out_buf)?;
    prove_fastpath_on_ir(k, sched, cand, lanes)
}

pub(crate) fn emit_x86_batch8_fp_consts(e: &mut X86Emit, fp: &B8FastPath) {
    e.vex_broadcast_rip(6, fp.addr_zero_mask as i64);
    if fp.desc_hi_zero_mask != 0 {
        e.vex_broadcast_rip(7, (u64::from(fp.desc_hi_zero_mask) << 32) as i64);
    }
    e.vex_broadcast_rip(8, u64::from(0u32.wrapping_sub(fp.desc_lo_min)) as i64);
    let span = fp.desc_lo_max.saturating_sub(fp.desc_lo_min);
    e.vex_broadcast_rip(9, (0xFFFF_FFFF_0000_0000u64 | u64::from(span)) as i64);
    e.vex_broadcast_rip(10, fp.out1_and as i64);
    e.vex_broadcast_rip(11, fp.out1_or as i64);
    if let Some(ng) = fp.net_guard {
        e.vex_broadcast32_rip(12, ng.prefix_max);
        e.vex_broadcast32_rip(13, u32::MAX);
        if let Some(rej) = ng.reject_src_const {
            e.vex_broadcast32_rip(14, rej);
        }
    }
}

pub(crate) fn emit_x86_batch8_fp_body(e: &mut X86Emit, ra: &RegAlloc, fp: &B8FastPath) -> Result<usize, CompileError> {
    let addr_reg = *ra.buf_reg.get(fp.addr_buf as usize).ok_or(CompileError::InvalidBuffer(fp.addr_buf))?;
    let desc_reg = *ra.buf_reg.get(fp.desc_buf as usize).ok_or(CompileError::InvalidBuffer(fp.desc_buf))?;
    let out_reg = *ra.buf_reg.get(fp.out_buf as usize).ok_or(CompileError::InvalidBuffer(fp.out_buf))?;

    // Accumulate all guard failure masks into ymm15 (0 iff all 8 lanes match the fast-path domain).
    if let Some(ng) = fp.net_guard {
        let net_reg = *ra.buf_reg.get(ng.net_buf as usize).ok_or(CompileError::InvalidBuffer(ng.net_buf))?;
        let pref_reg = *ra.buf_reg.get(ng.prefix_buf as usize).ok_or(CompileError::InvalidBuffer(ng.prefix_buf))?;
        e.vex_mem_op([2, 1, 1, 0, 0, 0x31], 0, pref_reg, 0); // vpmovzxbd ymm0, [prefix]
        e.vex_vpermq(0, 0, 0xD8);
        e.vex_ymm_op(2, 0x3B, 15, 0, 12); // vpminud ymm15, ymm0, ymm12 (32)
        e.vex_ymm_op(1, 0xEF, 15, 15, 0); // vpxor ymm15, ymm15, ymm0
        e.vex_ymm_op(1, 0xFA, 0, 12, 0);  // vpsubd ymm0, ymm12, ymm0 (32 - prefix)
        e.vex_ymm_op_w(2, 0, 0x47, 0, 13, 0); // vpsllvd ymm0, ymm13 (-1), ymm0
        e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 1, net_reg, 0);
        e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 2, net_reg, 32);
        e.vex_vshufps(3, 1, 2, 0x88); // src_ip
        e.vex_vshufps(4, 1, 2, 0xDD); // cidr_net
        e.vex_ymm_op(1, 0xEF, 4, 4, 3); // src ^ cidr
        e.vex_ymm_op(1, 0xDB, 4, 4, 0); // (src ^ cidr) & mask
        e.vex_ymm_op(1, 0xEB, 15, 15, 4);
        if ng.reject_src_const.is_some() {
            e.vex_ymm_op(1, 0x76, 3, 3, 14); // vpcmpeqd ymm3, src_ip, rej
            e.vex_ymm_op(1, 0xEB, 15, 15, 3);
        }
    } else {
        e.vex_ymm_op(1, 0xEF, 15, 15, 15);
    }

    // Load 8 x u64 addr into ymm0..ymm1 and check `(addr & addr_zero_mask) == 0`.
    e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 0, addr_reg, 0);
    e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 1, addr_reg, 32);
    e.vex_ymm_op(1, 0xEB, 4, 0, 1);
    e.vex_ymm_op(1, 0xDB, 4, 4, 6);
    e.vex_ymm_op(1, 0xEB, 15, 15, 4);

    // Load 8 x u64 desc into ymm2..ymm3 and check flags + `desc_lo_min <= lo32 <= desc_lo_max`.
    e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 2, desc_reg, 0);
    e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 3, desc_reg, 32);
    if fp.desc_hi_zero_mask != 0 {
        e.vex_ymm_op(1, 0xEB, 4, 2, 3);
        e.vex_ymm_op(1, 0xDB, 4, 4, 7);
        e.vex_ymm_op(1, 0xEB, 15, 15, 4);
    }
    for dreg in [2u8, 3u8] {
        e.vex_ymm_op(1, 0xFE, 4, dreg, 8); // lo - min in even dwords, hi unchanged
        e.vex_ymm_op(2, 0x3B, 5, 4, 9);    // min_u32(t, span) in even dwords, hi unchanged
        e.vex_ymm_op(1, 0xEF, 4, 4, 5);    // 0 iff min <= lo <= max
        e.vex_ymm_op(1, 0xEB, 15, 15, 4);
    }

    // Single branch: fall back to general 8-wide body if any lane violated any guard.
    e.vex_vptest(15, 15);
    let jnz_miss = e.jcc_placeholder(0x85);

    // Proven fast-path transform directly on loaded ymm0..ymm3 (zero deinterleave!).
    if fp.out0_and != u64::MAX {
        e.vex_broadcast_rip(15, fp.out0_and as i64);
        e.vex_ymm_op(1, 0xDB, 0, 0, 15);
        e.vex_ymm_op(1, 0xDB, 1, 1, 15);
    }
    if fp.out0_or != 0 {
        e.vex_broadcast_rip(15, fp.out0_or as i64);
        e.vex_ymm_op(1, 0xEB, 0, 0, 15);
        e.vex_ymm_op(1, 0xEB, 1, 1, 15);
    }
    if fp.out1_and != u64::MAX {
        e.vex_ymm_op(1, 0xDB, 2, 2, 10);
        e.vex_ymm_op(1, 0xDB, 3, 3, 10);
    }
    if fp.out1_or != 0 {
        e.vex_ymm_op(1, 0xEB, 2, 2, 11);
        e.vex_ymm_op(1, 0xEB, 3, 3, 11);
    }

    // Interleave [out0, out1] pairs and store 8 x [u64; 2] (128 bytes).
    for &(r0, r1, disp) in &[(0u8, 2u8, 0i32), (1u8, 3u8, 64i32)] {
        e.vex_ymm_op(1, 0x6C, 4, r0, r1);
        e.vex_ymm_op(1, 0x6D, 5, r0, r1);
        e.vex_ymm_imm(3, 0x46, r0, 4, 5, 0x20);
        e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], r0, out_reg, disp);
        e.vex_ymm_imm(3, 0x46, r0, 4, 5, 0x31);
        e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], r0, out_reg, disp + 32);
    }
    Ok(jnz_miss)
}

pub(crate) fn try_plan_avx2_batch8(k: &Kernel, sched: &ScheduleReport, doms: &[Domain], max_vecs: usize) -> Option<Batch8Plan> {
    if k.outputs.is_empty() || max_vecs < 12 { return None; }
    let mut has_u64 = false;
    let mut mut_bufs = 0usize;
    for b in &k.bufs {
        match b.access {
            BufAccess::LookupTable(_) => return None,
            BufAccess::ReadOnly => {
                if b.lanes != 1 { return None; }
                if matches!(b.elem_ty, ElemTy::U64 | ElemTy::I64) { has_u64 = true; }
            }
            BufAccess::MutNoAlias => {
                mut_bufs += 1;
                if matches!(b.elem_ty, ElemTy::U64 | ElemTy::I64) { has_u64 = true; }
            }
        }
    }
    if mut_bufs != 1 || !has_u64 { return None; }
    let out_buf = k.outputs[0].buf;
    let ob = k.bufs.get(out_buf as usize)?;
    let out_layout = if k.outputs.len() == 1 && ob.lanes == 1 {
        B8OutLayout::Lane1(out_buf, ob.elem_ty)
    } else if k.outputs.len() == 2 && ob.lanes == 2 && matches!(ob.elem_ty, ElemTy::U64 | ElemTy::I64) {
        B8OutLayout::Lanes2U64(out_buf)
    } else {
        return None;
    };

    let is_mask = analyze_bool_masks(k, sched);
    let demanded_lo32 = analyze_demanded_lo32(k, sched);
    let skip_mul = find_fused_barrett_muls(k, sched, doms);
    let mut b = B8Builder {
        k, doms, is_mask: &is_mask, demanded_lo32: &demanded_lo32, use_perm: true,
        vals: vec![None; k.nodes.len()], steps: Vec::new(), consts: Vec::new(), next_v: 0,
    };
    for (idx, &op) in k.nodes.iter().enumerate() {
        match op {
            Op::Const(c) => b.vals[idx] = Some(B8Val::Const(c as u64)),
            Op::MemInit => b.vals[idx] = Some(B8Val::Const(0)),
            _ => {}
        }
    }
    for &id in &sched.order { b.lower_node(id, &skip_mul)?; }

    let mut fp_lanes: Option<(u8, [NodeId; 2])> = None;
    match out_layout {
        B8OutLayout::Lane1(buf, ty) => {
            let ov = b.vals[k.outputs[0].val]?;
            let lo = b.val_lo32(ov);
            let hi = if matches!(ty, ElemTy::U64 | ElemTy::I64) { Some(b.val_hi32(ov)) } else { None };
            b.emit(B8Step::Store1 { lo, hi, buf, ty, unpermute: true });
        }
        B8OutLayout::Lanes2U64(buf) => {
            let mut lanes = [usize::MAX; 2];
            for o in &k.outputs {
                if o.buf != buf || o.lane >= 2 || lanes[usize::from(o.lane)] != usize::MAX { return None; }
                lanes[usize::from(o.lane)] = o.val;
            }
            fp_lanes = Some((buf, lanes));
            let (v0, v1) = (b.vals[lanes[0]]?, b.vals[lanes[1]]?);
            let raw = [b.val_lo32(v0), b.val_hi32(v0), b.val_lo32(v1), b.val_hi32(v1)];
            let mut m = [0usize; 4];
            for (idx, &v) in raw.iter().enumerate() {
                m[idx] = if b.consts.iter().any(|&(cv, _)| cv == v) {
                    let d = b.alloc_v();
                    b.emit(B8Step::Mov { dst: d, src: v });
                    d
                } else {
                    v
                };
            }
            b.emit(B8Step::Store2U64 { o0_lo: m[0], o0_hi: m[1], o1_lo: m[2], o1_hi: m[3], buf });
        }
    }

    peephole_b8_steps(&mut b.steps, &mut b.consts, &mut b.next_v);
    let (vreg_phys, hoisted) = alloc_b8_regs(&b.steps, &b.consts, b.next_v, max_vecs)?;
    let fastpath = fp_lanes.and_then(|(obuf, lanes)| try_plan_fastpath_superblock(k, sched, obuf, lanes));
    Some(Batch8Plan { consts_v: b.consts, hoisted, vreg_phys, steps: b.steps, fastpath })
}

fn b8_reg(e: &mut X86Emit, plan: &Batch8Plan, v: usize, scratch: u8) -> Result<u8, CompileError> {
    if let Some(Some(r)) = plan.vreg_phys.get(v) { return Ok(*r); }
    if let Some(&(_, c)) = plan.consts_v.iter().find(|&&(cv, _)| cv == v) {
        e.vex_broadcast32_rip(scratch, c);
        return Ok(scratch);
    }
    Err(CompileError::BackendError("unallocated batch8 vreg".into()))
}

fn b8_dst(plan: &Batch8Plan, v: usize) -> Result<u8, CompileError> {
    plan.vreg_phys.get(v).and_then(|&r| r).ok_or_else(|| CompileError::BackendError("unallocated batch8 dst".into()))
}

pub(crate) fn emit_x86_batch8_body(e: &mut X86Emit, k: &Kernel, ra: &RegAlloc, plan: &Batch8Plan, base_blk: i32) -> Result<(), CompileError> {
    for &st in &plan.steps {
        match st {
            B8Step::LoadNarrow { dst, buf, ty, permute } => {
                let d = b8_dst(plan, dst)?;
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * (k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32);
                match ty {
                    ElemTy::U8 => e.vex_mem_op([2, 1, 1, 0, 0, 0x31], d, breg, off),
                    ElemTy::U16 => e.vex_mem_op([2, 1, 1, 0, 0, 0x33], d, breg, off),
                    ElemTy::I16 => e.vex_mem_op([2, 1, 1, 0, 0, 0x23], d, breg, off),
                    ElemTy::U32 | ElemTy::I32 => e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], d, breg, off),
                    _ => return Err(CompileError::BackendError("invalid narrow load".into())),
                }
                if permute { e.vex_vpermq(d, d, 0xD8); }
            }
            B8Step::LoadSplit64 { lo, hi, buf } => {
                let (dl, dh) = (b8_dst(plan, lo)?, b8_dst(plan, hi)?);
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * 8;
                e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 14, breg, off);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x6F], 15, breg, off + 32);
                e.vex_vshufps(dl, 14, 15, 0x88);
                e.vex_vshufps(dh, 14, 15, 0xDD);
            }
            B8Step::Mov { dst, src } => {
                let (d, s) = (b8_dst(plan, dst)?, b8_reg(e, plan, src, 14)?);
                if d != s { e.vex_ymm_op(1, 0xEB, d, s, s); }
            }
            B8Step::Add32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0xFE, d, ra_, rb_); }
            B8Step::Sub32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0xFA, d, ra_, rb_); }
            B8Step::And32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0xDB, d, ra_, rb_); }
            B8Step::Andn32 { dst, mask_inv, val } => { let (d, rm, rv) = (b8_dst(plan, dst)?, b8_reg(e, plan, mask_inv, 14)?, b8_reg(e, plan, val, 15)?); e.vex_ymm_op(1, 0xDF, d, rm, rv); }
            B8Step::Or32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0xEB, d, ra_, rb_); }
            B8Step::Xor32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0xEF, d, ra_, rb_); }
            B8Step::MinU32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(2, 0x3B, d, ra_, rb_); }
            B8Step::MaxU32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(2, 0x3F, d, ra_, rb_); }
            B8Step::CmpEq32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0x76, d, ra_, rb_); }
            B8Step::CmpGtS32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op(1, 0x66, d, ra_, rb_); }
            B8Step::ShlImm32 { dst, src, s } => { let (d, rs) = (b8_dst(plan, dst)?, b8_reg(e, plan, src, 14)?); e.vex_shift_imm(0x72, 6, d, rs, s); }
            B8Step::ShrImm32 { dst, src, s } => { let (d, rs) = (b8_dst(plan, dst)?, b8_reg(e, plan, src, 14)?); e.vex_shift_imm(0x72, 2, d, rs, s); }
            B8Step::ShlVar32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op_w(2, 0, 0x47, d, ra_, rb_); }
            B8Step::ShrVar32 { dst, a, b } => { let (d, ra_, rb_) = (b8_dst(plan, dst)?, b8_reg(e, plan, a, 14)?, b8_reg(e, plan, b, 15)?); e.vex_ymm_op_w(2, 0, 0x45, d, ra_, rb_); }
            B8Step::Blend32 { dst, f_val, t_val, mask } => {
                let d = b8_dst(plan, dst)?;
                let rf = b8_reg(e, plan, f_val, 13)?;
                let rt = b8_reg(e, plan, t_val, 14)?;
                let rm = b8_reg(e, plan, mask, 15)?;
                e.vex_ymm_imm(3, 0x4C, d, rf, rt, rm << 4);
            }
            B8Step::MulHiShr { dst, x_lo, x_hi, c, shift, hi_mode } => {
                let d = b8_dst(plan, dst)?;
                let rlo = b8_reg(e, plan, x_lo, 13)?;
                let rc = b8_reg(e, plan, c, 15)?;
                e.vex_ymm_op(1, 0xF4, 14, rlo, rc);
                e.vex_shift_imm(0x73, 2, 14, 14, 32);
                e.vex_shift_imm(0x73, 2, d, rlo, 32);
                e.vex_ymm_op(1, 0xF4, d, d, rc);
                e.vex_ymm_imm(3, 0x02, d, 14, d, 0xAA);
                if let Some(xh) = x_hi {
                    let rhi = b8_reg(e, plan, xh, 13)?;
                    match hi_mode {
                        MulHiMode::NegMask => {
                            e.vex_ymm_op(1, 0xDB, 14, rhi, rc);
                        }
                        MulHiMode::Pos01 => {
                            e.vex_ymm_op(1, 0xEF, 14, 14, 14);
                            e.vex_ymm_op(1, 0xFA, 14, 14, rhi);
                            e.vex_ymm_op(1, 0xDB, 14, 14, rc);
                        }
                        MulHiMode::General => {
                            e.vex_ymm_op(2, 0x40, 14, rhi, rc);
                        }
                    }
                    e.vex_ymm_op(1, 0xFE, d, d, 14);
                }
                if shift > 0 { e.vex_shift_imm(0x72, 2, d, d, shift); }
            }
            B8Step::Store1 { lo, hi, buf, ty, unpermute } => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * (k.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.block_bytes() as i32);
                let rl = b8_reg(e, plan, lo, 13)?;
                if let Some(hv) = hi {
                    let rh = b8_reg(e, plan, hv, 15)?;
                    e.vex_ymm_op(1, 0x62, 14, rl, rh);
                    e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off);
                    e.vex_ymm_op(1, 0x6A, 14, rl, rh);
                    e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off + 32);
                } else if matches!(ty, ElemTy::U32 | ElemTy::I32) {
                    if unpermute {
                        e.vex_vpermq(14, rl, 0xD8);
                        e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off);
                    } else {
                        e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], rl, breg, off);
                    }
                } else {
                    return Err(CompileError::BackendError("unsupported narrow store1".into()));
                }
            }
            B8Step::Store2U64 { o0_lo, o0_hi, o1_lo, o1_hi, buf } => {
                let breg = *ra.buf_reg.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                let off = base_blk * 16;
                let (r0l, r0h) = (b8_dst(plan, o0_lo)?, b8_dst(plan, o0_hi)?);
                let (r1l, r1h) = (b8_dst(plan, o1_lo)?, b8_dst(plan, o1_hi)?);
                // Exits 0..4: vpunpckldq recovers natural order [0, 1, 2, 3].
                e.vex_ymm_op(1, 0x62, 13, r0l, r0h);
                e.vex_ymm_op(1, 0x62, 14, r1l, r1h);
                e.vex_ymm_op(1, 0x6C, 15, 13, 14);
                e.vex_ymm_op(1, 0x6D, 13, 13, 14);
                e.vex_ymm_imm(3, 0x46, 14, 15, 13, 0x20);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off);
                e.vex_ymm_imm(3, 0x46, 14, 15, 13, 0x31);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off + 32);
                // Exits 4..8: vpunpckhdq recovers natural order [4, 5, 6, 7].
                e.vex_ymm_op(1, 0x6A, 13, r0l, r0h);
                e.vex_ymm_op(1, 0x6A, 14, r1l, r1h);
                e.vex_ymm_op(1, 0x6C, 15, 13, 14);
                e.vex_ymm_op(1, 0x6D, 13, 13, 14);
                e.vex_ymm_imm(3, 0x46, 14, 15, 13, 0x20);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off + 64);
                e.vex_ymm_imm(3, 0x46, 14, 15, 13, 0x31);
                e.vex_mem_op([1, 1, 2, 0, 0, 0x7F], 14, breg, off + 96);
            }
        }
    }
    Ok(())
}
