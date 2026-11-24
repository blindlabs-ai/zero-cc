pub const LANES: usize = 8;
pub type NodeId = usize;
pub type IrOp = Op<NodeId>;

define_errors! { CompileError;
    InvalidBuffer(u8) | Self::InvalidBuffer(b) => "invalid buffer {b}";
    InvalidLane(u8) | Self::InvalidLane(l) => "invalid lane {l}";
    InvalidNode(NodeId) | Self::InvalidNode(n) => "invalid node {n}";
    WriteToReadOnlyBuf(u8) | Self::WriteToReadOnlyBuf(b) => "write to read-only buffer {b}";
    InvalidShift(u8) | Self::InvalidShift(s) => "invalid shift {s}";
    OutOfBounds { idx: i64, limit: i64 } | Self::OutOfBounds { idx, limit } => "index {idx} outside 0..{limit}";
    DivByZero | Self::DivByZero => "division or remainder by zero/overflow";
    RegisterSpill { peak_gprs: usize, max_gprs: usize, peak_vecs: usize, max_vecs: usize }
        | Self::RegisterSpill { peak_gprs, max_gprs, peak_vecs, max_vecs } => "register spill: gprs {peak_gprs}/{max_gprs}, vecs {peak_vecs}/{max_vecs}";
    BackendError(String) | Self::BackendError(msg) => "backend error: {msg}";
    ParseError(String) | Self::ParseError(msg) => "parse error: {msg}";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CmpKind { Eq, Ne, Lt, Le, Gt, Ge }
impl CmpKind {
    pub const fn eval(self, a: i64, b: i64) -> bool { match self { Self::Eq => a == b, Self::Ne => a != b, Self::Lt => a < b, Self::Le => a <= b, Self::Gt => a > b, Self::Ge => a >= b } }
    pub const fn sym(self) -> &'static str { match self { Self::Eq => "==", Self::Ne => "!=", Self::Lt => "<", Self::Le => "<=", Self::Gt => ">", Self::Ge => ">=" } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufAccess { ReadOnly, MutNoAlias, LookupTable(usize) }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufDecl { pub id: u8, pub name: String, pub access: BufAccess, pub domain: Domain }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Domain { pub lo: i64, pub hi: i64, pub modulus: i64, pub rem: i64 }

const fn gcd_u128(a: u128, b: u128) -> u128 { if b == 0 { a } else { gcd_u128(b, a % b) } }
fn gcd_i64(a: i64, b: i64) -> i64 { let g = gcd_u128(a.unsigned_abs() as u128, b.unsigned_abs() as u128); if g == 0 || g > i64::MAX as u128 { 1 } else { g as i64 } }
const fn ext_gcd_step(or: i128, r: i128, os: i128, s: i128) -> (i128, i128) {
    if r == 0 { if or < 0 { (-or, -os) } else { (or, os) } } else { let q = or / r; ext_gcd_step(r, or - q * r, s, os - q * s) }
}
const fn ext_gcd_i128(a: i128, b: i128) -> (i128, i128) { ext_gcd_step(a, b, 1, 0) }

const BARRETT_SHIFTS: [u8; 7] = [20, 24, 28, 30, 32, 36, 40];
const fn barrett_step(du: u128, x_max: u128, max_q: u128, idx: usize, fb: Option<(i64, u8)>) -> Option<(i64, u8)> {
    if idx >= BARRETT_SHIFTS.len() { return fb; }
    let (s, two_s) = (BARRETT_SHIFTS[idx], 1u128 << BARRETT_SHIFTS[idx]);
    let m = two_s.div_ceil(du);
    let lhs = max_q.saturating_mul(m.saturating_mul(du).saturating_sub(two_s)).saturating_add((du - 1).saturating_mul(m));
    if lhs < two_s && x_max.saturating_mul(m) <= i64::MAX as u128 && m <= i64::MAX as u128 {
        if s <= 32 && m <= u32::MAX as u128 { return Some((m as i64, s)); }
        return barrett_step(du, x_max, max_q, idx + 1, match fb { Some(_) => fb, None => Some((m as i64, s)) });
    }
    barrett_step(du, x_max, max_q, idx + 1, fb)
}
/// Exact non-negative Barrett constants `(m, shift)`: `((x * m) >> shift) == x / d` for `0 <= x <= max_x`.
pub const fn barrett_params(d: i64, max_x: i64) -> Option<(i64, u8)> {
    if d < 3 || (d as u64).is_power_of_two() || max_x < 0 { None } else { barrett_step(d as u128, max_x as u128, (max_x as u128) / (d as u128), 0, None) }
}

impl Domain {
    pub const fn empty() -> Self { Self { lo: 1, hi: 0, modulus: 1, rem: 0 } }
    pub const fn is_empty(self) -> bool { self.lo > self.hi }
    pub const fn top() -> Self { Self { lo: i64::MIN, hi: i64::MAX, modulus: 1, rem: 0 } }
    pub const fn range(lo: i64, hi: i64) -> Self { if lo <= hi { Self { lo, hi, modulus: 1, rem: 0 } } else { Self::empty() } }
    pub fn congruent(lo: i64, hi: i64, modulus: i64, rem: i64) -> Self {
        let m = if modulus > 1 { modulus } else { 1 };
        Self { lo, hi, modulus: m, rem: rem.rem_euclid(m) }.tighten()
    }
    pub const fn exact(v: i64) -> Self { Self { lo: v, hi: v, modulus: 1, rem: 0 } }
    pub const fn is_const(self) -> Option<i64> { if self.lo == self.hi { Some(self.lo) } else { None } }
    pub const fn is_nonneg(self) -> bool { !self.is_empty() && self.lo >= 0 }
    pub const fn fits_u32(self) -> bool { !self.is_empty() && self.lo >= 0 && self.hi <= u32::MAX as i64 }
    pub fn contains(self, v: i64) -> bool { v >= self.lo && v <= self.hi && (self.modulus <= 1 || v.rem_euclid(self.modulus) == self.rem) }
    pub fn rem_mod(self, d: i64) -> Option<i64> {
        if d <= 0 || self.is_empty() { None }
        else if let Some(c) = self.is_const() { Some(c.rem_euclid(d)) }
        else if self.modulus > 1 && self.modulus % d == 0 { Some(self.rem.rem_euclid(d)) } else { None }
    }
    pub fn tighten(self) -> Self {
        if self.is_empty() { return Self::empty(); } else if self.modulus <= 1 { return Self { lo: self.lo, hi: self.hi, modulus: 1, rem: 0 }; }
        let (m, r, lo128, hi128) = (i128::from(self.modulus), i128::from(self.rem.rem_euclid(self.modulus)), i128::from(self.lo), i128::from(self.hi));
        let (nlo, nhi) = (lo128 + (r - lo128.rem_euclid(m)).rem_euclid(m), hi128 - (hi128.rem_euclid(m) - r).rem_euclid(m));
        if nlo > nhi || nlo > i128::from(i64::MAX) || nhi < i128::from(i64::MIN) { Self::empty() } else { Self { lo: nlo as i64, hi: nhi as i64, modulus: self.modulus, rem: r as i64 } }
    }
    pub fn try_meet(self, other: Self) -> Option<Self> {
        if self.is_empty() || other.is_empty() { return None; }
        let (lo, hi) = (self.lo.max(other.lo), self.hi.min(other.hi));
        if lo > hi { return None; }
        let (m1, m2) = (i128::from(self.modulus.max(1)), i128::from(other.modulus.max(1)));
        let (r1, r2) = (i128::from(self.rem).rem_euclid(m1), i128::from(other.rem).rem_euclid(m2));
        let ((g, u), diff) = (ext_gcd_i128(m1, m2), r2 - r1);
        if diff.rem_euclid(g) != 0 { return None; }
        let (m2_g, lcm) = (m2 / g, m1.saturating_mul(m2 / g));
        let (modulus, rem) = if lcm <= i128::from(i64::MAX) { (lcm as i64, (r1 + m1 * ((diff / g).rem_euclid(m2_g) * u.rem_euclid(m2_g)).rem_euclid(m2_g)).rem_euclid(lcm) as i64) }
            else if m2 > m1 { (m2 as i64, r2 as i64) } else { (m1 as i64, r1 as i64) };
        let t = Self { lo, hi, modulus, rem }.tighten();
        if t.is_empty() { None } else { Some(t) }
    }
    pub fn meet(self, other: Self) -> Self { self.try_meet(other).unwrap_or_else(Self::empty) }
    pub fn join(self, other: Self) -> Self {
        if self.is_empty() { return other; } else if other.is_empty() { return self; }
        let (lo, hi) = (self.lo.min(other.lo), self.hi.max(other.hi));
        let (m1, r1) = if self.lo == self.hi { (0u128, self.lo) } else { (self.modulus.max(1) as u128, self.rem) };
        let (m2, r2) = if other.lo == other.hi { (0u128, other.lo) } else { (other.modulus.max(1) as u128, other.rem) };
        let g = gcd_u128(gcd_u128(m1, m2), (i128::from(r1) - i128::from(r2)).unsigned_abs());
        if g > 1 && g <= i64::MAX as u128 { Self::congruent(lo, hi, g as i64, r1) } else { Self::range(lo, hi) }
    }
    fn from_i128(lo: i128, hi: i128, modulus: i64, rem: i64) -> Self {
        if lo >= i128::from(i64::MIN) && hi <= i128::from(i64::MAX) && lo <= hi { Self::congruent(lo as i64, hi as i64, modulus, rem) } else { Self::top() }
    }
    pub fn eval_add_sub(self, rhs: Self, is_sub: bool) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let (lo, hi) = if is_sub { (i128::from(self.lo) - i128::from(rhs.hi), i128::from(self.hi) - i128::from(rhs.lo)) }
            else { (i128::from(self.lo) + i128::from(rhs.lo), i128::from(self.hi) + i128::from(rhs.hi)) };
        let (m, r128) = if let Some(c) = rhs.is_const() {
            (self.modulus.max(1), if is_sub { i128::from(self.rem) - i128::from(c) } else { i128::from(self.rem) + i128::from(c) })
        } else if let Some(c) = self.is_const() {
            (rhs.modulus.max(1), if is_sub { i128::from(c) - i128::from(rhs.rem) } else { i128::from(c) + i128::from(rhs.rem) })
        } else {
            (gcd_i64(self.modulus, rhs.modulus), if is_sub { i128::from(self.rem) - i128::from(rhs.rem) } else { i128::from(self.rem) + i128::from(rhs.rem) })
        };
        Self::from_i128(lo, hi, m, r128.rem_euclid(i128::from(m)) as i64)
    }
    fn mul_const_cong(d: Self, c: i64) -> (i64, i64) {
        if c == 0 { return (1, 0); }
        let (ac, m_out) = (i128::from(c).abs(), i128::from(d.modulus.max(1)).saturating_mul(i128::from(c).abs()));
        if m_out > 1 && m_out <= i128::from(i64::MAX) { (m_out as i64, (i128::from(d.rem) * i128::from(c)).rem_euclid(m_out) as i64) }
        else if ac > 1 && ac <= i128::from(i64::MAX) { (ac as i64, 0) } else { (1, 0) }
    }
    pub fn eval_mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let c = [i128::from(self.lo) * i128::from(rhs.lo), i128::from(self.lo) * i128::from(rhs.hi),
            i128::from(self.hi) * i128::from(rhs.lo), i128::from(self.hi) * i128::from(rhs.hi)];
        let (lo, hi) = (c.into_iter().min().unwrap_or(i128::from(i64::MIN)), c.into_iter().max().unwrap_or(i128::from(i64::MAX)));
        let (m, r) = if let Some(k) = rhs.is_const() { Self::mul_const_cong(self, k) }
        else if let Some(k) = self.is_const() { Self::mul_const_cong(rhs, k) }
        else {
            let (m1, r1, m2, r2) = (i128::from(self.modulus.max(1)), i128::from(self.rem), i128::from(rhs.modulus.max(1)), i128::from(rhs.rem));
            let g = gcd_u128(m1.saturating_mul(m2) as u128, gcd_u128(m1.saturating_mul(r2).unsigned_abs(), m2.saturating_mul(r1).unsigned_abs()));
            if g > 1 && g <= i64::MAX as u128 { (g as i64, (r1 * r2).rem_euclid(g as i128) as i64) } else { (1, 0) }
        };
        Self::from_i128(lo, hi, m, r)
    }
    pub fn eval_square(self) -> Self {
        if self.is_empty() { return Self::empty(); }
        let (a, b) = (i128::from(self.lo) * i128::from(self.lo), i128::from(self.hi) * i128::from(self.hi));
        let (lo, hi) = if self.lo <= 0 && self.hi >= 0 { (0, a.max(b)) } else { (a.min(b), a.max(b)) };
        let m = i128::from(self.modulus.max(1));
        let r = i128::from(self.rem).rem_euclid(m);
        let m_out = (m as u128).saturating_mul(gcd_u128((m + 2 * r) as u128, (2 * m) as u128));
        let (mi, ri) = if m_out > 1 && m_out <= i64::MAX as u128 { (m_out as i64, (r * r).rem_euclid(m_out as i128) as i64) } else { (1, 0) };
        Self::from_i128(lo, hi, mi, ri)
    }
    pub fn eval_div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() } else { rhs.is_const().filter(|&d| d > 0).map_or(Self::top(), |d| Self::range(self.lo / d, self.hi / d)) }
    }
    pub fn eval_rem(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let Some(d) = rhs.is_const().filter(|&d| d > 0) else { return Self::top(); };
        if self.is_nonneg() { self.rem_mod(d).map_or_else(|| Self::range(0, self.hi.min(d - 1)), Self::exact) } else { Self::range(-(d - 1), d - 1) }
    }
    pub fn eval_shl(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); } else if s >= 63 { return Self::top(); }
        let (shift_m, m_out) = (1i128 << s, i128::from(self.modulus.max(1)).saturating_mul(1i128 << s));
        let (m, r) = if m_out > 1 && m_out <= i128::from(i64::MAX) { (m_out as i64, ((i128::from(self.rem) << s).rem_euclid(m_out)) as i64) }
        else if shift_m > 1 && shift_m <= i128::from(i64::MAX) { (shift_m as i64, 0) } else { (1, 0) };
        Self::from_i128(i128::from(self.lo) << s, i128::from(self.hi) << s, m, r)
    }
    pub fn eval_shr(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); } else if s >= 64 { return Self::top(); }
        let (m, r) = if self.is_nonneg() && s > 0 && s < 63 && self.modulus > 1 && (self.modulus % (1i64 << s) == 0) { (self.modulus >> s, self.rem >> s) } else { (1, 0) };
        Self::from_i128(i128::from(self.lo >> s), i128::from(self.hi >> s), m, r)
    }
    pub fn eval_and(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { Self::empty() }
        else if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) { Self::exact(a & b) }
        else if self.is_nonneg() && rhs.is_nonneg() { Self::range(0, self.hi.min(rhs.hi)) }
        else if rhs.is_nonneg() { Self::range(0, rhs.hi) } else if self.is_nonneg() { Self::range(0, self.hi) } else { Self::top() }
    }
    pub fn eval_cmp(self, kind: CmpKind, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) { return Self::exact(i64::from(kind.eval(a, b))); }
        let known = match kind {
            CmpKind::Eq => self.try_meet(rhs).is_none().then_some(false), CmpKind::Ne => self.try_meet(rhs).is_none().then_some(true),
            CmpKind::Lt => if self.hi < rhs.lo { Some(true) } else if self.lo >= rhs.hi { Some(false) } else { None },
            CmpKind::Le => if self.hi <= rhs.lo { Some(true) } else if self.lo > rhs.hi { Some(false) } else { None },
            CmpKind::Gt => if self.lo > rhs.hi { Some(true) } else if self.hi <= rhs.lo { Some(false) } else { None },
            CmpKind::Ge => if self.lo >= rhs.hi { Some(true) } else if self.hi < rhs.lo { Some(false) } else { None },
        };
        known.map_or(Self::range(0, 1), |b| Self::exact(i64::from(b)))
    }
}

define_ops! { f;
    MemInit | Self::MemInit => Op::MemInit;
    Const(i64) | Self::Const(c) => Op::Const(c);
    Load { mem: Id, buf: u8, lane: u8 } | Self::Load { mem, buf, lane } => Op::Load { mem: f(mem), buf, lane };
    Store { mem: Id, buf: u8, lane: u8, val: Id } | Self::Store { mem, buf, lane, val } => Op::Store { mem: f(mem), buf, lane, val: f(val) };
    LoadLut { buf: u8, idx: Id } | Self::LoadLut { buf, idx } => Op::LoadLut { buf, idx: f(idx) };
    BoundsCheck { idx: Id, limit: i64 } | Self::BoundsCheck { idx, limit } => Op::BoundsCheck { idx: f(idx), limit };
    Add(Id, Id) | Self::Add(a, b) => Op::Add(f(a), f(b));
    Sub(Id, Id) | Self::Sub(a, b) => Op::Sub(f(a), f(b));
    Mul(Id, Id) | Self::Mul(a, b) => Op::Mul(f(a), f(b));
    Div(Id, Id) | Self::Div(a, b) => Op::Div(f(a), f(b));
    Rem(Id, Id) | Self::Rem(a, b) => Op::Rem(f(a), f(b));
    Shl(Id, u8) | Self::Shl(a, s) => Op::Shl(f(a), s);
    Shr(Id, u8) | Self::Shr(a, s) => Op::Shr(f(a), s);
    BitAnd(Id, Id) | Self::BitAnd(a, b) => Op::BitAnd(f(a), f(b));
    Cmp(CmpKind, Id, Id) | Self::Cmp(k, a, b) => Op::Cmp(k, f(a), f(b));
    Select { cond: Id, t_val: Id, f_val: Id } | Self::Select { cond, t_val, f_val } => Op::Select { cond: f(cond), t_val: f(t_val), f_val: f(f_val) };
}

impl<Id: Copy> Op<Id> {
    pub fn eval_domain(self, dom: impl Fn(Id) -> Domain, buf_dom: impl Fn(u8) -> Domain, is_init: impl Fn(Id) -> bool) -> Domain where Id: PartialEq {
        match self {
            Self::MemInit | Self::Store { .. } => Domain::top(), Self::Const(c) => Domain::exact(c),
            Self::Load { mem, buf, .. } => if is_init(mem) { buf_dom(buf) } else { Domain::top() },
            Self::LoadLut { buf, .. } => buf_dom(buf),
            Self::BoundsCheck { idx, limit } => if limit > 0 { dom(idx).meet(Domain::range(0, limit - 1)) } else { Domain::empty() },
            Self::Add(a, b) => dom(a).eval_add_sub(dom(b), false),
            Self::Sub(a, b) => if a == b { Domain::exact(0) } else { dom(a).eval_add_sub(dom(b), true) },
            Self::Mul(a, b) => if a == b { dom(a).eval_square() } else { dom(a).eval_mul(dom(b)) },
            Self::Div(a, b) => dom(a).eval_div(dom(b)), Self::Rem(a, b) => dom(a).eval_rem(dom(b)),
            Self::Shl(a, s) => dom(a).eval_shl(s), Self::Shr(a, s) => dom(a).eval_shr(s), Self::BitAnd(a, b) => dom(a).eval_and(dom(b)),
            Self::Cmp(k, a, b) => if a == b { Domain::exact(i64::from(matches!(k, CmpKind::Eq | CmpKind::Le | CmpKind::Ge))) } else { dom(a).eval_cmp(k, dom(b)) },
            Self::Select { cond, t_val, f_val } => {
                let cd = dom(cond);
                if cd.is_empty() { Domain::empty() } else if cd.is_const() == Some(0) { dom(f_val) } else if !cd.contains(0) { dom(t_val) } else { dom(t_val).join(dom(f_val)) }
            }
        }
    }
    pub fn hw_cost(self, dom: impl Fn(Id) -> Domain, is_init: impl Fn(Id) -> bool) -> (u64, u64) {
        match self {
            Self::MemInit | Self::Const(_) => (0, 0), Self::Load { mem, .. } => if is_init(mem) { (4, 1) } else { (12, 2) },
            Self::Store { .. } => (10, 2), Self::LoadLut { .. } => (5, 1), Self::BoundsCheck { .. } => (12, 2), Self::Select { .. } => (2, 1),
            Self::Add(_, _) | Self::Sub(_, _) | Self::Shl(_, _) | Self::Shr(_, _) | Self::BitAnd(_, _) | Self::Cmp(_, _, _) => (1, 1),
            Self::Mul(a, b) => if dom(a).fits_u32() && dom(b).fits_u32() { (4, 1) } else { (10, 2) }, Self::Div(_, _) | Self::Rem(_, _) => (36, 3),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputWrite { pub buf: u8, pub lane: u8, pub val: NodeId }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelCost { pub total_latency: u64, pub total_nodes: usize, pub bounds_checks: usize, pub divs_rems: usize, pub mem_ops: usize }
#[derive(Clone, Debug, Default)]
pub struct Kernel { pub name: String, pub bufs: Vec<BufDecl>, pub nodes: Vec<IrOp>, pub outputs: Vec<OutputWrite>, pub proven_domains: Vec<Domain> }

fn checked_div_rem(lhs: i64, rhs: i64, is_rem: bool) -> Result<i64, CompileError> {
    if rhs == 0 || (lhs == i64::MIN && rhs == -1) { Err(CompileError::DivByZero) } else if is_rem { Ok(lhs % rhs) } else { Ok(lhs / rhs) }
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self { Self { name: name.into(), ..Self::default() } }
    pub fn add_buf(&mut self, name: impl Into<String>, access: BufAccess, domain: Domain) -> u8 { let id = self.bufs.len() as u8; self.bufs.push(BufDecl { id, name: name.into(), access, domain }); id }
    pub fn push(&mut self, op: IrOp) -> NodeId { let id = self.nodes.len(); self.nodes.push(op); id }
    pub fn add_output(&mut self, buf: u8, lane: u8, val: NodeId) { self.outputs.push(OutputWrite { buf, lane, val }); }
    fn check_buf_lane(&self, buf: u8, lane: u8, require_mut: bool) -> Result<(), CompileError> {
        let b = self.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
        if require_mut && b.access != BufAccess::MutNoAlias { return Err(CompileError::WriteToReadOnlyBuf(buf)); }
        if !require_mut && matches!(b.access, BufAccess::LookupTable(_)) { return Err(CompileError::InvalidBuffer(buf)); }
        if usize::from(lane) >= LANES { return Err(CompileError::InvalidLane(lane)); }
        Ok(())
    }
    pub fn validate(&self) -> Result<(), CompileError> {
        for (id, &op) in self.nodes.iter().enumerate() {
            let (kids, cnt) = op.children();
            for &k in &kids[..cnt] { if k >= id { return Err(CompileError::InvalidNode(k)); } }
            match op {
                Op::Load { buf, lane, .. } => self.check_buf_lane(buf, lane, false)?,
                Op::Store { buf, lane, .. } => self.check_buf_lane(buf, lane, true)?,
                Op::LoadLut { buf, .. } if !matches!(self.bufs.get(buf as usize).map(|b| b.access), Some(BufAccess::LookupTable(_))) => return Err(CompileError::InvalidBuffer(buf)),
                Op::BoundsCheck { limit, .. } if limit <= 0 => return Err(CompileError::OutOfBounds { idx: 0, limit }),
                Op::Shl(_, s) | Op::Shr(_, s) if s >= 64 => return Err(CompileError::InvalidShift(s)),
                _ => {}
            }
        }
        for o in &self.outputs { self.check_buf_lane(o.buf, o.lane, true)?; if o.val >= self.nodes.len() { return Err(CompileError::InvalidNode(o.val)); } }
        Ok(())
    }
    pub fn infer_domains(&self) -> Vec<Domain> {
        let mut doms = Vec::with_capacity(self.nodes.len());
        for (id, &op) in self.nodes.iter().enumerate() {
            let d = op.eval_domain(|k| doms[k], |b| self.bufs[b as usize].domain, |k| matches!(self.nodes[k], Op::MemInit));
            doms.push(self.proven_domains.get(id).copied().map_or(d, |p| d.meet(p)));
        }
        doms
    }
    pub fn cost(&self) -> KernelCost {
        let (doms, mut c) = (self.infer_domains(), KernelCost { total_latency: 0, total_nodes: self.nodes.len(), bounds_checks: 0, divs_rems: 0, mem_ops: 0 });
        for &op in &self.nodes {
            let (lat, reg) = op.hw_cost(|id| doms[id], |id| matches!(self.nodes[id], Op::MemInit));
            c.total_latency += lat + reg;
            match op { Op::BoundsCheck { .. } => c.bounds_checks += 1, Op::Div(_, _) | Op::Rem(_, _) => c.divs_rems += 1, Op::Load { .. } | Op::Store { .. } | Op::LoadLut { .. } => c.mem_ops += 1, _ => {} }
        }
        c
    }
    pub fn eval(&self, inputs: &[&[i64]]) -> Result<Vec<[i64; LANES]>, CompileError> {
        self.validate()?;
        if inputs.len() != self.bufs.len() { return Err(CompileError::InvalidBuffer(inputs.len() as u8)); }
        let mut init_mem = Vec::with_capacity(self.bufs.len());
        for (idx, decl) in self.bufs.iter().enumerate() {
            let need = match decl.access { BufAccess::LookupTable(n) => n, _ => LANES };
            if inputs[idx].len() < need { return Err(CompileError::OutOfBounds { idx: inputs[idx].len() as i64, limit: need as i64 }); }
            let mut arr = [0i64; LANES];
            if !matches!(decl.access, BufAccess::LookupTable(_)) { arr.copy_from_slice(&inputs[idx][..LANES]); }
            init_mem.push(arr);
        }
        let (mut vals, mut mems, mut mslot) = (vec![0i64; self.nodes.len()], Vec::new(), vec![0usize; self.nodes.len()]);
        for (id, &op) in self.nodes.iter().enumerate() {
            vals[id] = match op {
                Op::MemInit => { mslot[id] = mems.len(); mems.push(init_mem.clone()); 0 }
                Op::Const(c) => c,
                Op::Load { mem, buf, lane } => mems[mslot[mem]][buf as usize][lane as usize],
                Op::Store { mem, buf, lane, val } => { let mut n = mems[mslot[mem]].clone(); n[buf as usize][lane as usize] = vals[val]; mslot[id] = mems.len(); mems.push(n); 0 }
                Op::LoadLut { buf, idx } => {
                    let (lim, iv) = (match self.bufs[buf as usize].access { BufAccess::LookupTable(n) => n as i64, _ => 0 }, vals[idx]);
                    if iv < 0 || iv >= lim { return Err(CompileError::OutOfBounds { idx: iv, limit: lim }); } inputs[buf as usize][iv as usize]
                }
                Op::BoundsCheck { idx, limit } => { let iv = vals[idx]; if iv < 0 || iv >= limit { return Err(CompileError::OutOfBounds { idx: iv, limit }); } iv }
                Op::Add(a, b) => vals[a].wrapping_add(vals[b]), Op::Sub(a, b) => vals[a].wrapping_sub(vals[b]), Op::Mul(a, b) => vals[a].wrapping_mul(vals[b]),
                Op::Div(a, b) => checked_div_rem(vals[a], vals[b], false)?, Op::Rem(a, b) => checked_div_rem(vals[a], vals[b], true)?,
                Op::Shl(a, s) => vals[a].wrapping_shl(u32::from(s)), Op::Shr(a, s) => vals[a].wrapping_shr(u32::from(s)), Op::BitAnd(a, b) => vals[a] & vals[b],
                Op::Cmp(k, a, b) => i64::from(k.eval(vals[a], vals[b])), Op::Select { cond, t_val, f_val } => if vals[cond] != 0 { vals[t_val] } else { vals[f_val] },
            };
        }
        let mut out = init_mem;
        for o in &self.outputs { out[o.buf as usize][o.lane as usize] = vals[o.val]; }
        Ok(out)
    }
}
