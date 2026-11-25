use std::collections::{BTreeMap, BTreeSet};
pub const LANES: usize = 8;
pub type NodeId = usize;
pub type IrOp = Op<NodeId>;
define_errors! { CompileError,
    InvalidBuffer(u8) | Self::InvalidBuffer(b) => "invalid buffer {b}";
    InvalidLane(u8) | Self::InvalidLane(l) => "invalid lane {l}";
    InvalidNode(NodeId) | Self::InvalidNode(n) => "invalid node {n}";
    WriteToReadOnlyBuf(u8) | Self::WriteToReadOnlyBuf(b) => "write to read-only buffer {b}";
    InvalidShift(u8) | Self::InvalidShift(s) => "invalid shift {s}";
    OutOfBounds { idx: i64, limit: i64 } | Self::OutOfBounds { idx, limit } => "index {idx} outside 0..{limit}";
    DivByZero | Self::DivByZero => "division or remainder by zero/overflow";
    RegisterSpill { peak_gprs: usize, max_gprs: usize, peak_vecs: usize, max_vecs: usize } | Self::RegisterSpill {
        peak_gprs: a, max_gprs: b, peak_vecs: c, max_vecs: d } => "register spill: gprs {a}/{b}, vecs {c}/{d}";
    BackendError(String) | Self::BackendError(msg) => "backend error: {msg}";
    ParseError(String) | Self::ParseError(msg) => "parse error: {msg}";
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CmpKind { Eq, Ne, Lt, Le, Gt, Ge, Ult, Ule, Ugt, Uge }
impl CmpKind {
    const SYMS: [&'static str; 10] = ["==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">="];
    const SIGNED: [Self; 10] = [Self::Eq, Self::Ne, Self::Lt, Self::Le, Self::Gt, Self::Ge,
        Self::Lt, Self::Le, Self::Gt, Self::Ge];
    pub const fn eval(self, a: i64, b: i64) -> bool { match self {
        Self::Eq => a == b,
        Self::Ne => a != b,
        Self::Lt => a < b,
        Self::Le => a <= b,
        Self::Gt => a > b,
        Self::Ge => a >= b,
        Self::Ult => (a as u64) < (b as u64),
        Self::Ule => (a as u64) <= (b as u64),
        Self::Ugt => (a as u64) > (b as u64),
        Self::Uge => (a as u64) >= (b as u64),
    }}
    pub const fn is_unsigned(self) -> bool { matches!(self, Self::Ult | Self::Ule | Self::Ugt | Self::Uge) }
    pub const fn sym(self) -> &'static str {
        let i = self as usize;
        if i < Self::SYMS.len() { Self::SYMS[i] } else { "==" }}
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElemTy { #[default] I64, U64, I32, I16, U32, U16, U8 }
impl ElemTy {
    const ALL: [(Self, &'static str, u8, i64, i64); 7] = [
        (Self::I64, "i64", 3, i64::MIN, i64::MAX), (Self::U64, "u64", 3, i64::MIN, i64::MAX),
        (Self::I32, "i32", 2, i32::MIN as _, i32::MAX as _), (Self::I16, "i16", 1, i16::MIN as _, i16::MAX as _),
        (Self::U32, "u32", 2, 0, u32::MAX as _), (Self::U16, "u16", 1, 0, u16::MAX as _),
        (Self::U8, "u8", 0, 0, u8::MAX as _)];
    const fn meta(self) -> (Self, &'static str, u8, i64, i64) {
        let i = self as usize;
        if i < Self::ALL.len() { Self::ALL[i] } else { Self::ALL[0] }}
    pub const fn scale_shift(self) -> u8 { self.meta().2 }
    pub const fn bytes(self) -> u8 { 1u8 << self.scale_shift() }
    pub const fn rust_name(self) -> &'static str { self.meta().1 }
    pub fn from_rust_name(name: &str) -> Option<Self> { Self::ALL.iter().find(|t| t.1 == name).map(|t| t.0) }
    pub const fn natural_domain(self) -> Domain { Domain { lo: self.meta().3, hi: self.meta().4, modulus: 1, rem: 0 } }
    pub const fn cast_i64(self, v: i64) -> i64 { match self {
        Self::I64 | Self::U64 => v,
        Self::I32 => (v as i32) as i64,
        Self::I16 => (v as i16) as i64,
        Self::U32 => (v as u32) as i64,
        Self::U16 => (v as u16) as i64,
        Self::U8 => (v as u8) as i64,
    }}
    pub fn push_le_elem(self, o: &mut Vec<u8>, v: i64) { o.extend_from_slice(&v.to_le_bytes()[..self.bytes().into()]); }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum BufAccess { ReadOnly, MutNoAlias, LookupTable(usize) }
#[derive(Clone, Debug, PartialEq, Eq)] pub struct BufDecl {
    pub id: u8, pub name: String, pub access: BufAccess, pub domain: Domain, pub elem_ty: ElemTy, pub lanes: u8 }
impl BufDecl { pub const fn block_bytes(&self) -> usize { (self.elem_ty.bytes() as usize) * (self.lanes as usize) } }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Domain { pub lo: i64, pub hi: i64, pub modulus: i64, pub rem: i64 }
const fn gcd_u128(a: u128, b: u128) -> u128 { if b == 0 { a } else { gcd_u128(b, a % b) } }
fn gcd_i64(a: i64, b: i64) -> i64 {
    let g = gcd_u128(a.unsigned_abs() as u128, b.unsigned_abs() as u128);
    if g == 0 || g > i64::MAX as u128 { 1 } else { g as i64 }}
const fn ext_gcd_step(or: i128, r: i128, os: i128, s: i128) -> (i128, i128) {
    if r == 0 { if or < 0 { (-or, -os) } else { (or, os) } }
    else { ext_gcd_step(r, or - (or / r) * r, s, os - (or / r) * s) }}
const BS: [u8; 7] = [20, 24, 28, 30, 32, 36, 40];
const fn barrett_step(du: u128, xm: u128, mq: u128, i: usize, fb: Option<(i64, u8)>) -> Option<(i64, u8)> {
    if i >= BS.len() { return fb; }
    let (s, ts) = (BS[i], 1u128 << BS[i]);
    let m = ts.div_ceil(du);
    let lhs = mq.saturating_mul(m.saturating_mul(du).saturating_sub(ts)).saturating_add((du - 1).saturating_mul(m));
    if lhs < ts && xm.saturating_mul(m) <= i64::MAX as u128 && m <= i64::MAX as u128 {
        if s <= 32 && m <= u32::MAX as u128 { return Some((m as i64, s)); }
        return barrett_step(du, xm, mq, i + 1, if fb.is_some() { fb } else { Some((m as i64, s)) });
    }
    barrett_step(du, xm, mq, i + 1, fb)
}
pub const fn barrett_params(d: i64, max_x: i64) -> Option<(i64, u8)> {
    if d < 3 || (d as u64).is_power_of_two() || max_x < 0 { None }
    else { barrett_step(d as u128, max_x as u128, (max_x as u128) / (d as u128), 0, None) }}
impl Domain {
    pub const fn empty() -> Self { Self { lo: 1, hi: 0, modulus: 1, rem: 0 } }
    pub const fn is_empty(self) -> bool { self.lo > self.hi }
    pub const fn top() -> Self { Self { lo: i64::MIN, hi: i64::MAX, modulus: 1, rem: 0 } }
    pub const fn range(lo: i64, hi: i64) -> Self {
        if lo <= hi { Self { lo, hi, modulus: 1, rem: 0 } } else { Self::empty() }}
    pub fn congruent(lo: i64, hi: i64, modulus: i64, rem: i64) -> Self {
        let m = modulus.max(1);
        Self { lo, hi, modulus: m, rem: rem.rem_euclid(m) }.tighten() }
    pub const fn exact(v: i64) -> Self { Self { lo: v, hi: v, modulus: 1, rem: 0 } }
    pub const fn is_const(self) -> Option<i64> { if self.lo == self.hi { Some(self.lo) } else { None } }
    pub const fn is_nonneg(self) -> bool { !self.is_empty() && self.lo >= 0 }
    pub const fn fits_u32(self) -> bool { !self.is_empty() && self.lo >= 0 && self.hi <= u32::MAX as i64 }
    pub const fn fits_i32(self) -> bool { !self.is_empty() && self.lo >= i32::MIN as i64 && self.hi <= i32::MAX as i64 }
    pub fn contains(self, v: i64) -> bool {
        v >= self.lo && v <= self.hi && (self.modulus <= 1 || v.rem_euclid(self.modulus) == self.rem) }
    pub fn rem_mod(self, d: i64) -> Option<i64> {
        if d <= 0 || self.is_empty() { None }
        else if let Some(c) = self.is_const() { Some(c.rem_euclid(d)) }
        else if self.modulus > 1 && self.modulus % d == 0 { Some(self.rem.rem_euclid(d)) } else { None }}
    pub fn tighten(self) -> Self {
        if self.is_empty() { return Self::empty(); }
        if self.modulus <= 1 { return Self { lo: self.lo, hi: self.hi, modulus: 1, rem: 0 }; }
        let (m, r) = (i128::from(self.modulus), i128::from(self.rem.rem_euclid(self.modulus)));
        let (lo, hi) = (i128::from(self.lo), i128::from(self.hi));
        let (nlo, nhi) = (lo + (r - lo.rem_euclid(m)).rem_euclid(m), hi - (hi.rem_euclid(m) - r).rem_euclid(m));
        if nlo > nhi || nlo > i128::from(i64::MAX) || nhi < i128::from(i64::MIN) { Self::empty() }
        else { Self { lo: nlo as i64, hi: nhi as i64, modulus: self.modulus, rem: r as i64 } }}
    pub fn try_meet(self, other: Self) -> Option<Self> {
        if self.is_empty() || other.is_empty() { return None; }
        let (lo, hi) = (self.lo.max(other.lo), self.hi.min(other.hi));
        if lo > hi { return None; }
        let (m1, m2) = (i128::from(self.modulus.max(1)), i128::from(other.modulus.max(1)));
        let (r1, r2) = (i128::from(self.rem).rem_euclid(m1), i128::from(other.rem).rem_euclid(m2));
        let ((g, u), diff) = (ext_gcd_step(m1, m2, 1, 0), r2 - r1);
        if diff.rem_euclid(g) != 0 { return None; }
        let (m2_g, lcm) = (m2 / g, m1.saturating_mul(m2 / g));
        let (modulus, rem) = if lcm <= i128::from(i64::MAX) {
            let k = ((diff / g).rem_euclid(m2_g) * u.rem_euclid(m2_g)).rem_euclid(m2_g);
            (lcm as i64, (r1 + m1 * k).rem_euclid(lcm) as i64)
        } else if m2 > m1 { (m2 as i64, r2 as i64) } else { (m1 as i64, r1 as i64) };
        Some(Self { lo, hi, modulus, rem }.tighten()).filter(|t| !t.is_empty())
    }
    pub fn meet(self, other: Self) -> Self { self.try_meet(other).unwrap_or_else(Self::empty) }
    pub fn join(self, other: Self) -> Self {
        if self.is_empty() { return other; }
        if other.is_empty() { return self; }
        let (lo, hi) = (self.lo.min(other.lo), self.hi.max(other.hi));
        let mr = |d: Self| if d.lo == d.hi { (0u128, d.lo) } else { (d.modulus.max(1) as u128, d.rem) };
        let ((m1, r1), (m2, r2)) = (mr(self), mr(other));
        let g = gcd_u128(gcd_u128(m1, m2), (i128::from(r1) - i128::from(r2)).unsigned_abs());
        if g > 1 && g <= i64::MAX as u128 { Self::congruent(lo, hi, g as i64, r1) } else { Self::range(lo, hi) }}
    fn from_i128(lo: i128, hi: i128, m: i64, r: i64) -> Self {
        if lo >= i128::from(i64::MIN) && hi <= i128::from(i64::MAX) && lo <= hi {
            Self::congruent(lo as i64, hi as i64, m, r) } else { Self::top() }}
    pub fn eval_add_sub(self, rhs: Self, is_sub: bool) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let f = |a: i64, b: i64| if is_sub { i128::from(a) - i128::from(b) } else { i128::from(a) + i128::from(b) };
        let (lo, hi) = if is_sub { (f(self.lo, rhs.hi), f(self.hi, rhs.lo)) }
            else { (f(self.lo, rhs.lo), f(self.hi, rhs.hi)) };
        let (m, r128) = if let Some(c) = rhs.is_const() { (self.modulus.max(1), f(self.rem, c)) }
            else if let Some(c) = self.is_const() { (rhs.modulus.max(1), f(c, rhs.rem)) }
            else { (gcd_i64(self.modulus, rhs.modulus), f(self.rem, rhs.rem)) };
        Self::from_i128(lo, hi, m, r128.rem_euclid(i128::from(m)) as i64)
    }
    fn scale_cong(d: Self, c: i128) -> (i64, i64) {
        let (ac, mo) = (c.abs(), i128::from(d.modulus.max(1)).saturating_mul(c.abs()));
        if c == 0 { (1, 0) }
        else if mo > 1 && mo <= i128::from(i64::MAX) { (mo as i64, (i128::from(d.rem) * c).rem_euclid(mo) as i64) }
        else if ac > 1 && ac <= i128::from(i64::MAX) { (ac as i64, 0) } else { (1, 0) }}
    pub fn eval_mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let [a, b, c, d] = [i128::from(self.lo) * i128::from(rhs.lo), i128::from(self.lo) * i128::from(rhs.hi),
            i128::from(self.hi) * i128::from(rhs.lo), i128::from(self.hi) * i128::from(rhs.hi)];
        let (lo, hi) = (a.min(b).min(c.min(d)), a.max(b).max(c.max(d)));
        let (m, r) = if let Some(k) = rhs.is_const() { Self::scale_cong(self, i128::from(k)) }
            else if let Some(k) = self.is_const() { Self::scale_cong(rhs, i128::from(k)) }
            else {
                let (m1, r1, m2, r2) = (i128::from(self.modulus.max(1)), i128::from(self.rem),
                    i128::from(rhs.modulus.max(1)), i128::from(rhs.rem));
                let g1 = gcd_u128(m1.saturating_mul(r2).unsigned_abs(), m2.saturating_mul(r1).unsigned_abs());
                let g = gcd_u128(m1.saturating_mul(m2) as u128, g1);
                if g > 1 && g <= i64::MAX as u128 { (g as i64, (r1 * r2).rem_euclid(g as i128) as i64) } else { (1, 0) }
            };
        Self::from_i128(lo, hi, m, r)
    }
    pub fn eval_square(self) -> Self {
        if self.is_empty() { return Self::empty(); }
        let (a, b) = (i128::from(self.lo) * i128::from(self.lo), i128::from(self.hi) * i128::from(self.hi));
        let (lo, hi) = if self.lo <= 0 && self.hi >= 0 { (0, a.max(b)) } else { (a.min(b), a.max(b)) };
        let m64 = self.modulus.max(1);
        let (m, r) = (i128::from(m64), i128::from(self.rem.rem_euclid(m64)));
        let mo = (m as u128).saturating_mul(gcd_u128((m + 2 * r) as u128, (2 * m) as u128));
        let (mi, ri) = if mo > 1 && mo <= i64::MAX as u128 { (mo as i64, (r * r).rem_euclid(mo as i128) as i64) }
            else { (1, 0) };
        Self::from_i128(lo, hi, mi, ri)
    }
    pub fn eval_div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() }
        else { rhs.is_const().filter(|&d| d > 0).map_or(Self::top(), |d| Self::range(self.lo / d, self.hi / d)) }}
    pub fn eval_rem(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() }
        else if let Some(d) = rhs.is_const().filter(|&d| d > 0) {
            if self.is_nonneg() { self.rem_mod(d).map_or_else(|| Self::range(0, self.hi.min(d - 1)), Self::exact) }
            else { Self::range(-(d - 1), d - 1) }
        } else { Self::top() }}
    pub fn eval_shl(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); }
        if s >= 63 { return Self::top(); }
        let (m, r) = Self::scale_cong(self, 1i128 << s);
        Self::from_i128(i128::from(self.lo) << s, i128::from(self.hi) << s, m, r) }
    pub fn eval_shr(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); }
        if s >= 64 { return Self::top(); }
        let ex = self.is_nonneg() && s > 0 && s < 63 && self.modulus > 1 && (self.modulus % (1i64 << s) == 0);
        let (m, r) = if ex { (self.modulus >> s, self.rem >> s) } else { (1, 0) };
        Self::from_i128(i128::from(self.lo >> s), i128::from(self.hi >> s), m, r) }
    pub fn eval_and(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) { return Self::exact(a & b); }
        let (lo, hi) = if self.is_nonneg() && rhs.is_nonneg() { (0, self.hi.min(rhs.hi)) }
            else if rhs.is_nonneg() { (0, rhs.hi) } else if self.is_nonneg() { (0, self.hi) }
            else { (i64::MIN, i64::MAX) };
        let p2 = |d: Self, c: i64| (d.modulus > 1 && (d.modulus as u64).is_power_of_two())
            .then(|| (d.modulus, d.rem & (c & (d.modulus - 1))));
        let (m, r) = rhs.is_const().and_then(|c| p2(self, c))
            .or_else(|| self.is_const().and_then(|c| p2(rhs, c))).unwrap_or((1, 0));
        Self::from_i128(i128::from(lo), i128::from(hi), m, r)
    }
    fn eval_or_xor(self, rhs: Self, is_xor: bool) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() }
        else if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) {
            Self::exact(if is_xor { a ^ b } else { a | b })
        } else if self.is_nonneg() && rhs.is_nonneg() {
            let v = self.hi | rhs.hi;
            let hi = if v <= 0 { 0 } else if v >= 1i64 << 62 { i64::MAX }
                else { (1i64 << (64 - v.leading_zeros())) - 1 };
            Self::range(if is_xor { 0 } else { self.lo.max(rhs.lo) }, hi)
        } else { Self::top() }}
    pub fn eval_or(self, rhs: Self) -> Self { self.eval_or_xor(rhs, false) }
    pub fn eval_xor(self, rhs: Self) -> Self { self.eval_or_xor(rhs, true) }
    pub fn eval_ushr(self, s: u8) -> Self {
        if self.is_empty() { Self::empty() } else if s >= 64 { Self::top() }
        else if let Some(c) = self.is_const() { Self::exact(((c as u64) >> s) as i64) }
        else if s == 0 { self } else if self.is_nonneg() { self.eval_shr(s) }
        else { Self::range(0, (u64::MAX >> s) as i64) }}
    fn eval_shift_var(self, rhs: Self, is_shr: bool) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() }
        else if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) {
            let (u, sh) = (a as u64, (b as u32) & 63);
            Self::exact((if is_shr { u.wrapping_shr(sh) } else { u.wrapping_shl(sh) }) as i64)
        } else if let Some(b) = rhs.is_const() {
            let sh = (b as u8) & 63;
            if is_shr { self.eval_ushr(sh) } else { self.eval_shl(sh) }
        } else if is_shr && self.is_nonneg() { Self::range(0, self.hi) }
        else if is_shr && rhs.lo >= 1 && rhs.hi <= 63 { Self::range(0, (u64::MAX >> (rhs.lo as u32)) as i64) }
        else { Self::top() }}
    pub fn eval_shl_var(self, rhs: Self) -> Self { self.eval_shift_var(rhs, false) }
    pub fn eval_shr_var_u(self, rhs: Self) -> Self { self.eval_shift_var(rhs, true) }
    pub fn eval_cmp(self, kind: CmpKind, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) { return Self::exact(kind.eval(a, b) as i64); }
        let same = (self.is_nonneg() && rhs.is_nonneg()) || (self.hi < 0 && rhs.hi < 0);
        let ord = |t: bool, f: bool| if t { Some(true) } else if f { Some(false) } else { None };
        match kind {
            CmpKind::Eq | CmpKind::Ne => self.try_meet(rhs).is_none().then_some(kind == CmpKind::Ne),
            CmpKind::Lt => ord(self.hi < rhs.lo, self.lo >= rhs.hi),
            CmpKind::Le => ord(self.hi <= rhs.lo, self.lo > rhs.hi),
            CmpKind::Gt => ord(self.lo > rhs.hi, self.hi <= rhs.lo),
            CmpKind::Ge => ord(self.lo >= rhs.hi, self.hi < rhs.lo),
            _ if same => return self.eval_cmp(CmpKind::SIGNED.get(kind as usize).copied().unwrap_or(kind), rhs),
            _ if self.is_nonneg() && rhs.hi < 0 => Some(matches!(kind, CmpKind::Ult | CmpKind::Ule)),
            _ if self.hi < 0 && rhs.is_nonneg() => Some(matches!(kind, CmpKind::Ugt | CmpKind::Uge)),
            _ => None,
        }.map_or(Self::range(0, 1), |b| Self::exact(i64::from(b)))
    }
}
define_ops! { f,
    MemInit | Self::MemInit => Op::MemInit;
    Const(i64) | Self::Const(c) => Op::Const(c);
    Load { mem: Id, buf: u8, lane: u8 } | Self::Load { mem, buf, lane } => Op::Load { mem: f(mem), buf, lane };
    Store { mem: Id, buf: u8, lane: u8, val: Id }
        | Self::Store { mem, buf, lane, val } => Op::Store { mem: f(mem), buf, lane, val: f(val) };
    LoadLut { buf: u8, idx: Id } | Self::LoadLut { buf, idx } => Op::LoadLut { buf, idx: f(idx) };
    BoundsCheck { idx: Id, limit: i64 } | Self::BoundsCheck { idx, limit } => Op::BoundsCheck { idx: f(idx), limit };
    Add(Id, Id) | Self::Add(a, b) => Op::Add(f(a), f(b));
    Sub(Id, Id) | Self::Sub(a, b) => Op::Sub(f(a), f(b));
    Mul(Id, Id) | Self::Mul(a, b) => Op::Mul(f(a), f(b));
    Div(Id, Id) | Self::Div(a, b) => Op::Div(f(a), f(b));
    Rem(Id, Id) | Self::Rem(a, b) => Op::Rem(f(a), f(b));
    Shl(Id, u8) | Self::Shl(a, s) => Op::Shl(f(a), s);
    Shr(Id, u8) | Self::Shr(a, s) => Op::Shr(f(a), s);
    Ushr(Id, u8) | Self::Ushr(a, s) => Op::Ushr(f(a), s);
    ShlVar(Id, Id) | Self::ShlVar(a, b) => Op::ShlVar(f(a), f(b));
    ShrVarU(Id, Id) | Self::ShrVarU(a, b) => Op::ShrVarU(f(a), f(b));
    BitAnd(Id, Id) | Self::BitAnd(a, b) => Op::BitAnd(f(a), f(b));
    BitOr(Id, Id) | Self::BitOr(a, b) => Op::BitOr(f(a), f(b));
    BitXor(Id, Id) | Self::BitXor(a, b) => Op::BitXor(f(a), f(b));
    Popcnt(Id) | Self::Popcnt(a) => Op::Popcnt(f(a));
    Tzcnt(Id) | Self::Tzcnt(a) => Op::Tzcnt(f(a));
    Cmp(CmpKind, Id, Id) | Self::Cmp(k, a, b) => Op::Cmp(k, f(a), f(b));
    Select { cond: Id, t_val: Id, f_val: Id }
        | Self::Select { cond, t_val, f_val } => Op::Select { cond: f(cond), t_val: f(t_val), f_val: f(f_val) };
}
impl<Id: Copy> Op<Id> {
    pub fn eval_domain(self, dom: impl Fn(Id) -> Domain, buf_dom: impl Fn(u8) -> Domain,
        is_init: impl Fn(Id) -> bool) -> Domain where Id: PartialEq { match self {
        Self::MemInit | Self::Store { .. } => Domain::top(),
        Self::Const(c) => Domain::exact(c),
        Self::Load { mem, buf, .. } => if is_init(mem) { buf_dom(buf) } else { Domain::top() },
        Self::LoadLut { buf, .. } => buf_dom(buf),
        Self::BoundsCheck { idx, limit: l } => dom(idx).meet(Domain::range(0, l.saturating_sub(1))),
        Self::Add(a, b) => dom(a).eval_add_sub(dom(b), false),
        Self::Sub(a, b) => if a == b { Domain::exact(0) } else { dom(a).eval_add_sub(dom(b), true) },
        Self::Mul(a, b) => if a == b { dom(a).eval_square() } else { dom(a).eval_mul(dom(b)) },
        Self::Div(a, b) => dom(a).eval_div(dom(b)),
        Self::Rem(a, b) => dom(a).eval_rem(dom(b)),
        Self::Shl(a, s) => dom(a).eval_shl(s),
        Self::Shr(a, s) => dom(a).eval_shr(s),
        Self::Ushr(a, s) => dom(a).eval_ushr(s),
        Self::ShlVar(a, b) => dom(a).eval_shl_var(dom(b)),
        Self::ShrVarU(a, b) => dom(a).eval_shr_var_u(dom(b)),
        Self::BitAnd(a, b) => dom(a).eval_and(dom(b)),
        Self::BitOr(a, b) => dom(a).eval_or(dom(b)),
        Self::BitXor(a, b) => if a == b { Domain::exact(0) } else { dom(a).eval_xor(dom(b)) },
        Self::Popcnt(a) | Self::Tzcnt(a) => {
            let (da, pop) = (dom(a), matches!(self, Self::Popcnt(_)));
            if da.is_empty() { Domain::empty() }
            else if let Some(c) = da.is_const() {
                Domain::exact(i64::from(if pop { (c as u64).count_ones() } else { (c as u64).trailing_zeros() }))
            } else {
                Domain::range(0, if pop && da.fits_u32() { 32 } else if !pop && !da.contains(0) { 63 } else { 64 }) }
        }
        Self::Cmp(k, a, b) => if a == b {
            Domain::exact(i64::from(matches!(k, CmpKind::Eq | CmpKind::Le | CmpKind::Ge | CmpKind::Ule | CmpKind::Uge)))
        } else { dom(a).eval_cmp(k, dom(b)) },
        Self::Select { cond, t_val, f_val } => match dom(cond) {
            cd if cd.is_empty() => Domain::empty(),
            cd if cd.is_const() == Some(0) => dom(f_val),
            cd if !cd.contains(0) => dom(t_val),
            _ => dom(t_val).join(dom(f_val)),
        },
    }}
    pub fn hw_cost(self, dom: impl Fn(Id) -> Domain, is_init: impl Fn(Id) -> bool) -> (u64, u64) { match self {
        Self::MemInit | Self::Const(_) => (0, 0),
        Self::Load { mem, .. } => if is_init(mem) { (4, 1) } else { (12, 2) },
        Self::Store { .. } | Self::BoundsCheck { .. } => (10, 2),
        Self::LoadLut { .. } => (5, 1),
        Self::Select { .. } => (2, 1),
        Self::Add(_, _) | Self::Sub(_, _) | Self::Shl(_, _) | Self::Shr(_, _) | Self::Ushr(_, _)
            | Self::BitAnd(_, _) | Self::BitOr(_, _) | Self::BitXor(_, _) | Self::Cmp(_, _, _) => (1, 1),
        Self::ShlVar(_, _) | Self::ShrVarU(_, _) | Self::Popcnt(_) | Self::Tzcnt(_) => (2, 1),
        Self::Mul(a, b) => if dom(a).fits_u32() && dom(b).fits_u32() { (4, 1) } else { (10, 2) },
        Self::Div(_, _) | Self::Rem(_, _) => (36, 3),
    }}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct OutputWrite { pub buf: u8, pub lane: u8, pub val: NodeId }
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)] pub struct KernelCost {
    pub total_latency: u64, pub total_nodes: usize, pub bounds_checks: usize, pub divs_rems: usize, pub mem_ops: usize }
#[derive(Clone, Debug, Default)] pub struct Kernel {
    pub name: String, pub bufs: Vec<BufDecl>, pub nodes: Vec<IrOp>,
    pub outputs: Vec<OutputWrite>, pub proven_domains: Vec<Domain> }
fn checked_div_rem(lhs: i64, rhs: i64, is_rem: bool) -> Result<i64, CompileError> {
    if rhs == 0 || (lhs == i64::MIN && rhs == -1) { Err(CompileError::DivByZero) }
    else { Ok(if is_rem { lhs % rhs } else { lhs / rhs }) }}
impl Kernel {
    pub fn new(name: impl Into<String>) -> Self { Self { name: name.into(), ..Self::default() } }
    pub fn add_buf(&mut self, name: impl Into<String>, access: BufAccess, domain: Domain) -> u8 {
        self.add_typed_buf(name, access, ElemTy::I64, LANES as u8, domain) }
    pub fn add_typed_buf(&mut self, name: impl Into<String>, access: BufAccess, elem_ty: ElemTy,
        lanes: u8, domain: Domain) -> u8 {
        let (id, tight) = (self.bufs.len() as u8, domain.meet(elem_ty.natural_domain()));
        let eff = if tight.is_empty() && !domain.is_empty() { elem_ty.natural_domain() } else { tight };
        self.bufs.push(BufDecl { id, name: name.into(), access, domain: eff, elem_ty, lanes });
        id
    }
    pub fn push(&mut self, op: IrOp) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(op);
        id
    }
    pub fn add_output(&mut self, buf: u8, lane: u8, val: NodeId) { self.outputs.push(OutputWrite { buf, lane, val }); }
    fn check_buf_lane(&self, buf: u8, lane: u8, req_mut: bool) -> Result<(), CompileError> {
        let b = self.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
        if req_mut && b.access != BufAccess::MutNoAlias { return Err(CompileError::WriteToReadOnlyBuf(buf)); }
        if !req_mut && matches!(b.access, BufAccess::LookupTable(_)) { return Err(CompileError::InvalidBuffer(buf)); }
        if lane >= b.lanes || usize::from(b.lanes) > LANES { return Err(CompileError::InvalidLane(lane)); }
        Ok(())
    }
    pub fn validate(&self) -> Result<(), CompileError> {
        for b in &self.bufs {
            if !matches!(b.access, BufAccess::LookupTable(_)) && (b.lanes == 0 || usize::from(b.lanes) > LANES) {
                return Err(CompileError::InvalidLane(b.lanes)); }}
        for (id, &op) in self.nodes.iter().enumerate() {
            let (kids, cnt) = op.children();
            if let Some(&k) = kids[..cnt].iter().find(|&&k| k >= id) { return Err(CompileError::InvalidNode(k)); }
            match op {
                Op::Load { buf, lane, .. } => self.check_buf_lane(buf, lane, false)?,
                Op::Store { buf, lane, .. } => self.check_buf_lane(buf, lane, true)?,
                Op::LoadLut { buf, .. } if !matches!(self.bufs.get(buf as usize).map(|b| b.access),
                    Some(BufAccess::LookupTable(_))) => return Err(CompileError::InvalidBuffer(buf)),
                Op::BoundsCheck { limit, .. } if limit <= 0 => return Err(CompileError::OutOfBounds { idx: 0, limit }),
                Op::Shl(_, s) | Op::Shr(_, s) | Op::Ushr(_, s) if s >= 64 => return Err(CompileError::InvalidShift(s)),
                _ => {}
            }
        }
        for o in &self.outputs {
            self.check_buf_lane(o.buf, o.lane, true)?;
            if o.val >= self.nodes.len() { return Err(CompileError::InvalidNode(o.val)); }
        }
        Ok(())
    }
    pub fn infer_domains(&self) -> Vec<Domain> {
        let mut doms = Vec::with_capacity(self.nodes.len());
        for (id, &op) in self.nodes.iter().enumerate() {
            let d = op.eval_domain(|k| doms.get(k).copied().unwrap_or_else(Domain::top),
                |b| self.bufs.get(b as usize).map_or_else(Domain::top, |x| x.domain),
                |k| matches!(self.nodes.get(k), Some(Op::MemInit)));
            doms.push(self.proven_domains.get(id).copied().map_or(d, |p| d.meet(p)));
        }
        doms
    }
    pub fn cost(&self) -> KernelCost {
        let (doms, mut c) = (self.infer_domains(), KernelCost { total_nodes: self.nodes.len(), ..Default::default() });
        for &op in &self.nodes {
            let (lat, reg) = op.hw_cost(|i| doms.get(i).copied().unwrap_or_else(Domain::top),
                |i| matches!(self.nodes.get(i), Some(Op::MemInit)));
            c.total_latency += lat + reg;
            match op {
                Op::BoundsCheck { .. } => c.bounds_checks += 1,
                Op::Div(_, _) | Op::Rem(_, _) => c.divs_rems += 1,
                Op::Load { .. } | Op::Store { .. } | Op::LoadLut { .. } => c.mem_ops += 1,
                _ => {}
            }
        }
        c
    }
    pub fn eval(&self, inputs: &[&[i64]]) -> Result<Vec<[i64; LANES]>, CompileError> {
        self.validate()?;
        if inputs.len() != self.bufs.len() { return Err(CompileError::InvalidBuffer(inputs.len() as u8)); }
        let mut init_mem = Vec::with_capacity(self.bufs.len());
        for (decl, &inp) in self.bufs.iter().zip(inputs.iter()) {
            let need = if let BufAccess::LookupTable(n) = decl.access { n } else { usize::from(decl.lanes) };
            if inp.len() < need { return Err(CompileError::OutOfBounds { idx: inp.len() as i64, limit: need as i64 }); }
            let mut arr = [0i64; LANES];
            if !matches!(decl.access, BufAccess::LookupTable(_)) {
                for (sl, &v) in arr.iter_mut().zip(inp.iter()).take(need) { *sl = decl.elem_ty.cast_i64(v); }}
            init_mem.push(arr);
        }
        let (mut vals, mut mems, mut mslot) = (Vec::new(), Vec::new(), Vec::new());
        for &op in &self.nodes {
            let (v, ms) = (|i| vals.get(i).copied().unwrap_or(0), |i| mslot.get(i).copied().unwrap_or(0));
            let (mut slot, val) = (0usize, match op {
                Op::MemInit => { mems.push(init_mem.clone());
                    0 }
                Op::Const(c) => c,
                Op::Load { mem, buf, lane } => mems.get(ms(mem))
                    .and_then(|m| m.get(buf as usize)).and_then(|a| a.get(lane as usize)).copied().unwrap_or(0),
                Op::Store { mem, buf, lane, val } => {
                    let mut n = mems.get(ms(mem)).cloned().unwrap_or_default();
                    let et = self.bufs.get(buf as usize).map_or(ElemTy::I64, |b| b.elem_ty);
                    if let Some(s) = n.get_mut(buf as usize).and_then(|a| a.get_mut(lane as usize)) {
                        *s = et.cast_i64(v(val)); }
                    mems.push(n);
                    0
                }
                Op::LoadLut { buf, idx } => {
                    let b = self.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                    let (lim, iv) = (if let BufAccess::LookupTable(n) = b.access { n as i64 } else { 0 }, v(idx));
                    if iv < 0 || iv >= lim { return Err(CompileError::OutOfBounds { idx: iv, limit: lim }); }
                    b.elem_ty.cast_i64(inputs.get(buf as usize).and_then(|s| s.get(iv as usize)).copied().unwrap_or(0))
                }
                Op::BoundsCheck { idx, limit } => { let iv = v(idx);
                    if iv < 0 || iv >= limit { return Err(CompileError::OutOfBounds { idx: iv, limit }); }
                    iv }
                Op::Add(a, b) => v(a).wrapping_add(v(b)),
                Op::Sub(a, b) => v(a).wrapping_sub(v(b)),
                Op::Mul(a, b) => v(a).wrapping_mul(v(b)),
                Op::Div(a, b) => checked_div_rem(v(a), v(b), false)?,
                Op::Rem(a, b) => checked_div_rem(v(a), v(b), true)?,
                Op::Shl(a, s) => v(a).wrapping_shl(u32::from(s)),
                Op::Shr(a, s) => v(a).wrapping_shr(u32::from(s)),
                Op::Ushr(a, s) => ((v(a) as u64).wrapping_shr(u32::from(s))) as i64,
                Op::ShlVar(a, b) => ((v(a) as u64).wrapping_shl((v(b) as u32) & 63)) as i64,
                Op::ShrVarU(a, b) => ((v(a) as u64).wrapping_shr((v(b) as u32) & 63)) as i64,
                Op::BitAnd(a, b) => v(a) & v(b),
                Op::BitOr(a, b) => v(a) | v(b),
                Op::BitXor(a, b) => v(a) ^ v(b),
                Op::Popcnt(a) => i64::from((v(a) as u64).count_ones()),
                Op::Tzcnt(a) => i64::from((v(a) as u64).trailing_zeros()),
                Op::Cmp(k, a, b) => i64::from(k.eval(v(a), v(b))),
                Op::Select { cond, t_val, f_val } => if v(cond) != 0 { v(t_val) } else { v(f_val) },
            });
            if matches!(op, Op::MemInit | Op::Store { .. }) { slot = mems.len() - 1; }
            vals.push(val);
            mslot.push(slot);
        }
        let mut out = init_mem;
        for o in &self.outputs {
            let et = self.bufs.get(o.buf as usize).map_or(ElemTy::I64, |b| b.elem_ty);
            if let Some(s) = out.get_mut(o.buf as usize).and_then(|a| a.get_mut(o.lane as usize)) {
                *s = et.cast_i64(vals.get(o.val).copied().unwrap_or(0)); }}
        Ok(out)
    }
}
fn narrow_channel_val(f: &mut Kernel, et: ElemTy, val: NodeId) -> NodeId { match et {
    ElemTy::I64 | ElemTy::U64 => val,
    ElemTy::U32 | ElemTy::U16 | ElemTy::U8 => { let m = f.push(Op::Const((1i64 << (et.bytes() * 8)) - 1));
        f.push(Op::BitAnd(val, m)) }
    ElemTy::I32 | ElemTy::I16 => { let (sh, s) = (64 - et.bytes() * 8, f.push(Op::Shl(val, 64 - et.bytes() * 8)));
        f.push(Op::Shr(s, sh)) }
}}
pub fn fuse_kernels(name: impl Into<String>, stages: &[&Kernel]) -> Result<Kernel, CompileError> {
    fn bnames(st: &Kernel, acc: BufAccess) -> impl Iterator<Item = String> + '_ {
        st.bufs.iter().filter(move |b| b.access == acc).map(|b| b.name.clone()) }
    if stages.is_empty() { return Err(CompileError::ParseError("cannot fuse empty stage list".into())); }
    for st in stages { st.validate()?; }
    let (mut read_after, mut produced, mo) = (BTreeSet::new(), BTreeSet::new(), BufAccess::MutNoAlias);
    for (idx, st) in stages.iter().enumerate() {
        read_after.extend(bnames(st, BufAccess::ReadOnly).filter(|n| produced.contains(n)));
        if idx + 1 < stages.len() { produced.extend(bnames(st, mo)); }
    }
    let louts: BTreeSet<_> = stages.last().map_or_else(BTreeSet::new, |st| bnames(st, mo).collect());
    let mut fused = Kernel::new(name);
    let (mem0, mut ext_bufs, mut ssa_bus) = (fused.push(Op::MemInit), BTreeMap::new(), BTreeMap::new());
    for (s_idx, st) in stages.iter().enumerate() {
        let mut bmap = Vec::with_capacity(st.bufs.len());
        for b in &st.bufs {
            let is_int = read_after.contains(&b.name) && (b.access == BufAccess::ReadOnly
                || (b.access == mo && (s_idx + 1 < stages.len() || !louts.contains(&b.name))));
            if is_int { bmap.push(None); }
            else if let Some(&eid) = ext_bufs.get(&b.name) {
                let eb = fused.bufs.get(eid as usize).ok_or(CompileError::InvalidBuffer(eid))?;
                if eb.elem_ty != b.elem_ty || eb.lanes != b.lanes || eb.access != b.access {
                    return Err(CompileError::ParseError(format!("incompatible buffer `{}`", b.name))); }
                bmap.push(Some(eid));
            } else {
                let eid = fused.add_typed_buf(b.name.clone(), b.access, b.elem_ty, b.lanes, b.domain);
                ext_bufs.insert(b.name.clone(), eid);
                bmap.push(Some(eid));
            }
        }
        let (mut nmap, map_buf) = (Vec::with_capacity(st.nodes.len()),
            |b: u8| bmap.get(b as usize).copied().flatten().ok_or(CompileError::InvalidBuffer(b)));
        for &op in &st.nodes {
            let map_id = |k| nmap.get(k).copied().ok_or(CompileError::InvalidNode(k));
            let mapped = match op {
                Op::MemInit => mem0,
                Op::Load { mem, buf, lane } => {
                    let bname = &st.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?.name;
                    if let Some(&val) = ssa_bus.get(&(bname.clone(), lane)) { val } else {
                        let nm = if matches!(st.nodes.get(mem), Some(Op::MemInit)) { mem0 } else { map_id(mem)? };
                        fused.push(Op::Load { mem: nm, buf: map_buf(buf)?, lane }) }
                }
                Op::Store { mem, buf, lane, val } => { let (nm, nv) = (map_id(mem)?, map_id(val)?);
                    bmap.get(buf as usize).copied().flatten()
                        .map_or(nm, |eb| fused.push(Op::Store { mem: nm, buf: eb, lane, val: nv })) }
                Op::LoadLut { buf, idx } => fused.push(Op::LoadLut { buf: map_buf(buf)?, idx: map_id(idx)? }),
                other => fused.push(other.map_ids(|k| nmap.get(k).copied().unwrap_or(k))),
            };
            nmap.push(mapped);
        }
        for o in &st.outputs {
            let b = st.bufs.get(o.buf as usize).ok_or(CompileError::InvalidBuffer(o.buf))?;
            let v = nmap.get(o.val).copied().ok_or(CompileError::InvalidNode(o.val))?;
            if read_after.contains(&b.name) {
                let nv = narrow_channel_val(&mut fused, b.elem_ty, v);
                ssa_bus.insert((b.name.clone(), o.lane), nv); }
            if let Some(eb) = bmap.get(o.buf as usize).copied().flatten() { fused.add_output(eb, o.lane, v); }
        }
    }
    fused.validate()?;
    Ok(fused)
}
