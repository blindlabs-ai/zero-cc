use std::fmt;

pub const LANES: usize = 8;
pub type NodeId = usize;
pub type IrOp = Op<NodeId>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    InvalidBuffer(u8),
    InvalidLane(u8),
    InvalidNode(NodeId),
    WriteToReadOnlyBuf(u8),
    InvalidShift(u8),
    OutOfBounds { idx: i64, limit: i64 },
    DivByZero,
    RegisterSpill { peak_gprs: usize, max_gprs: usize, peak_vecs: usize, max_vecs: usize },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBuffer(b) => write!(f, "invalid buffer {b}"),
            Self::InvalidLane(l) => write!(f, "invalid lane {l}"),
            Self::InvalidNode(n) => write!(f, "invalid node {n}"),
            Self::WriteToReadOnlyBuf(b) => write!(f, "write to read-only buffer {b}"),
            Self::InvalidShift(s) => write!(f, "invalid shift {s}"),
            Self::OutOfBounds { idx, limit } => write!(f, "index {idx} outside 0..{limit}"),
            Self::DivByZero => write!(f, "division or remainder by zero/overflow"),
            Self::RegisterSpill { peak_gprs, max_gprs, peak_vecs, max_vecs } => {
                write!(f, "register spill: gprs {peak_gprs}/{max_gprs}, vecs {peak_vecs}/{max_vecs}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufAccess { ReadOnly, MutNoAlias, LookupTable(usize) }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufDecl { pub id: u8, pub name: String, pub access: BufAccess, pub domain: Domain }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Domain { pub lo: i64, pub hi: i64, pub modulus: i64, pub rem: i64 }

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs(); b = b.abs();
    while b != 0 { let t = a % b; a = b; b = t; }
    a
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let g = gcd_i128(i128::from(a), i128::from(b));
    if g == 0 || g > i128::from(i64::MAX) { 1 } else { g as i64 }
}

fn ext_gcd_i128(a: i128, b: i128) -> (i128, i128) {
    let (mut old_r, mut r, mut old_s, mut s) = (a, b, 1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let (nr, ns) = (old_r - q * r, old_s - q * s);
        old_r = r; r = nr; old_s = s; s = ns;
    }
    if old_r < 0 { (-old_r, -old_s) } else { (old_r, old_s) }
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
        else if self.modulus > 1 && self.modulus % d == 0 { Some(self.rem.rem_euclid(d)) }
        else { None }
    }

    pub fn tighten(self) -> Self {
        if self.is_empty() { return Self::empty(); }
        if self.modulus <= 1 { return Self { lo: self.lo, hi: self.hi, modulus: 1, rem: 0 }; }
        let (m, r) = (i128::from(self.modulus), i128::from(self.rem.rem_euclid(self.modulus)));
        let (lo128, hi128) = (i128::from(self.lo), i128::from(self.hi));
        let new_lo = lo128 + (r - lo128.rem_euclid(m)).rem_euclid(m);
        let new_hi = hi128 - (hi128.rem_euclid(m) - r).rem_euclid(m);
        if new_lo > new_hi || new_lo > i128::from(i64::MAX) || new_hi < i128::from(i64::MIN) { return Self::empty(); }
        Self { lo: new_lo as i64, hi: new_hi as i64, modulus: self.modulus, rem: r as i64 }
    }

    pub fn try_meet(self, other: Self) -> Option<Self> {
        if self.is_empty() || other.is_empty() { return None; }
        let (lo, hi) = (self.lo.max(other.lo), self.hi.min(other.hi));
        if lo > hi { return None; }
        let (m1, m2) = (i128::from(self.modulus.max(1)), i128::from(other.modulus.max(1)));
        let (r1, r2) = (i128::from(self.rem).rem_euclid(m1), i128::from(other.rem).rem_euclid(m2));
        let (g, u) = ext_gcd_i128(m1, m2);
        let diff = r2 - r1;
        if diff.rem_euclid(g) != 0 { return None; }
        let m2_g = m2 / g;
        let lcm = m1.saturating_mul(m2_g);
        let (modulus, rem) = if lcm <= i128::from(i64::MAX) {
            let k = ((diff / g).rem_euclid(m2_g) * u.rem_euclid(m2_g)).rem_euclid(m2_g);
            (lcm as i64, (r1 + m1 * k).rem_euclid(lcm) as i64)
        } else if m2 > m1 { (m2 as i64, r2 as i64) } else { (m1 as i64, r1 as i64) };
        let tightened = Self { lo, hi, modulus, rem }.tighten();
        if tightened.is_empty() { None } else { Some(tightened) }
    }

    pub fn meet(self, other: Self) -> Self { self.try_meet(other).unwrap_or_else(Self::empty) }

    fn from_i128(lo: i128, hi: i128, modulus: i64, rem: i64) -> Self {
        if lo >= i128::from(i64::MIN) && hi <= i128::from(i64::MAX) && lo <= hi {
            let m = if modulus > 1 { modulus } else { 1 };
            Self { lo: lo as i64, hi: hi as i64, modulus: m, rem: rem.rem_euclid(m) }.tighten()
        } else { Self::top() }
    }

    pub fn eval_add_sub(self, rhs: Self, is_sub: bool) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let (lo, hi) = if is_sub {
            (i128::from(self.lo) - i128::from(rhs.hi), i128::from(self.hi) - i128::from(rhs.lo))
        } else {
            (i128::from(self.lo) + i128::from(rhs.lo), i128::from(self.hi) + i128::from(rhs.hi))
        };
        let (m, rem128) = if let Some(c) = rhs.is_const() {
            (self.modulus.max(1), if is_sub { i128::from(self.rem) - i128::from(c) } else { i128::from(self.rem) + i128::from(c) })
        } else if let Some(c) = self.is_const() {
            (rhs.modulus.max(1), if is_sub { i128::from(c) - i128::from(rhs.rem) } else { i128::from(c) + i128::from(rhs.rem) })
        } else {
            (gcd_i64(self.modulus, rhs.modulus), if is_sub { i128::from(self.rem) - i128::from(rhs.rem) } else { i128::from(self.rem) + i128::from(rhs.rem) })
        };
        Self::from_i128(lo, hi, m, rem128.rem_euclid(i128::from(m)) as i64)
    }

    fn mul_const_cong(d: Self, c: i64) -> (i64, i64) {
        if c == 0 { return (1, 0); }
        let ac = i128::from(c).abs();
        let m_out = i128::from(d.modulus.max(1)).saturating_mul(ac);
        if m_out > 1 && m_out <= i128::from(i64::MAX) { (m_out as i64, (i128::from(d.rem) * i128::from(c)).rem_euclid(m_out) as i64) }
        else if ac > 1 && ac <= i128::from(i64::MAX) { (ac as i64, 0) }
        else { (1, 0) }
    }

    pub fn eval_mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let c = [
            i128::from(self.lo) * i128::from(rhs.lo), i128::from(self.lo) * i128::from(rhs.hi),
            i128::from(self.hi) * i128::from(rhs.lo), i128::from(self.hi) * i128::from(rhs.hi),
        ];
        let (lo, hi) = (c.into_iter().min().unwrap_or(i128::from(i64::MIN)), c.into_iter().max().unwrap_or(i128::from(i64::MAX)));
        let (m, r) = if let Some(k) = rhs.is_const() { Self::mul_const_cong(self, k) }
        else if let Some(k) = self.is_const() { Self::mul_const_cong(rhs, k) }
        else {
            let (m1, r1, m2, r2) = (i128::from(self.modulus.max(1)), i128::from(self.rem), i128::from(rhs.modulus.max(1)), i128::from(rhs.rem));
            let g = gcd_i128(m1.saturating_mul(m2), gcd_i128(m1.saturating_mul(r2), m2.saturating_mul(r1)));
            if g > 1 && g <= i128::from(i64::MAX) { (g as i64, (r1 * r2).rem_euclid(g) as i64) } else { (1, 0) }
        };
        Self::from_i128(lo, hi, m, r)
    }

    pub fn eval_square(self) -> Self {
        if self.is_empty() { return Self::empty(); }
        let (a, b) = (i128::from(self.lo) * i128::from(self.lo), i128::from(self.hi) * i128::from(self.hi));
        let (lo, hi) = if self.lo <= 0 && self.hi >= 0 { (0, a.max(b)) } else { (a.min(b), a.max(b)) };
        let (m, r) = (i128::from(self.modulus.max(1)), i128::from(self.rem).rem_euclid(i128::from(self.modulus.max(1))));
        let m_out = m.saturating_mul(gcd_i128(m + 2 * r, 2 * m));
        let (mi, ri) = if m_out > 1 && m_out <= i128::from(i64::MAX) { (m_out as i64, (r * r).rem_euclid(m_out) as i64) } else { (1, 0) };
        Self::from_i128(lo, hi, mi, ri)
    }

    pub fn eval_div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        rhs.is_const().filter(|&d| d > 0).map_or(Self::top(), |d| Self::range(self.lo / d, self.hi / d))
    }

    pub fn eval_rem(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        let Some(d) = rhs.is_const().filter(|&d| d > 0) else { return Self::top(); };
        if self.is_nonneg() { self.rem_mod(d).map_or_else(|| Self::range(0, self.hi.min(d - 1)), Self::exact) }
        else { Self::range(-(d - 1), d - 1) }
    }

    pub fn eval_shl(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); }
        if s >= 63 { return Self::top(); }
        let shift_m = 1i128 << s;
        let m_out = i128::from(self.modulus.max(1)).saturating_mul(shift_m);
        let (m, r) = if m_out > 1 && m_out <= i128::from(i64::MAX) { (m_out as i64, ((i128::from(self.rem) << s).rem_euclid(m_out)) as i64) }
        else if shift_m > 1 && shift_m <= i128::from(i64::MAX) { (shift_m as i64, 0) }
        else { (1, 0) };
        Self::from_i128(i128::from(self.lo) << s, i128::from(self.hi) << s, m, r)
    }

    pub fn eval_shr(self, s: u8) -> Self {
        if self.is_empty() { return Self::empty(); }
        if s >= 64 { return Self::top(); }
        let (m, r) = if self.is_nonneg() && s > 0 && s < 63 && self.modulus > 1 && (self.modulus % (1i64 << s) == 0) {
            (self.modulus >> s, self.rem >> s)
        } else { (1, 0) };
        Self::from_i128(i128::from(self.lo >> s), i128::from(self.hi >> s), m, r)
    }

    pub fn eval_and(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() { return Self::empty(); }
        if let (Some(a), Some(b)) = (self.is_const(), rhs.is_const()) { Self::exact(a & b) }
        else if self.is_nonneg() && rhs.is_nonneg() { Self::range(0, self.hi.min(rhs.hi)) }
        else { Self::top() }
    }
}

/// Exact non-negative Barrett constants `(m, shift)` guaranteeing
/// `((x * m) >> shift) == x / d` for every `0 <= x <= max_x` with zero `i64` overflow.
pub fn barrett_params(d: i64, max_x: i64) -> Option<(i64, u8)> {
    if d < 3 || (d as u64).is_power_of_two() || max_x < 0 { return None; }
    let (du, x_max) = (d as u128, max_x as u128);
    let max_q = x_max / du;
    let mut fallback = None;
    for s in [20u8, 24, 28, 30, 32, 36, 40] {
        let two_s = 1u128 << s;
        let m = two_s.div_ceil(du);
        let err = m.saturating_mul(du).saturating_sub(two_s);
        let lhs = max_q.saturating_mul(err).saturating_add((du - 1).saturating_mul(m));
        if lhs < two_s && x_max.saturating_mul(m) <= i64::MAX as u128 && m <= i64::MAX as u128 {
            if s <= 32 && m <= u128::from(u32::MAX) { return Some((m as i64, s)); }
            if fallback.is_none() { fallback = Some((m as i64, s)); }
        }
    }
    fallback
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op<Id> {
    MemInit,
    Const(i64),
    Load { mem: Id, buf: u8, lane: u8 },
    Store { mem: Id, buf: u8, lane: u8, val: Id },
    LoadLut { buf: u8, idx: Id },
    BoundsCheck { idx: Id, limit: i64 },
    Add(Id, Id), Sub(Id, Id), Mul(Id, Id), Div(Id, Id), Rem(Id, Id),
    Shl(Id, u8), Shr(Id, u8), BitAnd(Id, Id),
}

impl<Id: Copy> Op<Id> {
    pub fn children(self) -> ([Id; 2], usize) where Id: Default {
        match self {
            Self::MemInit | Self::Const(_) => ([Id::default(), Id::default()], 0),
            Self::Load { mem: a, .. } | Self::LoadLut { idx: a, .. } | Self::BoundsCheck { idx: a, .. }
            | Self::Shl(a, _) | Self::Shr(a, _) => ([a, Id::default()], 1),
            Self::Store { mem: a, val: b, .. } | Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b)
            | Self::Div(a, b) | Self::Rem(a, b) | Self::BitAnd(a, b) => ([a, b], 2),
        }
    }

    pub fn map_ids<Out>(self, mut f: impl FnMut(Id) -> Out) -> Op<Out> {
        match self {
            Self::MemInit => Op::MemInit,
            Self::Const(c) => Op::Const(c),
            Self::Load { mem, buf, lane } => Op::Load { mem: f(mem), buf, lane },
            Self::Store { mem, buf, lane, val } => Op::Store { mem: f(mem), buf, lane, val: f(val) },
            Self::LoadLut { buf, idx } => Op::LoadLut { buf, idx: f(idx) },
            Self::BoundsCheck { idx, limit } => Op::BoundsCheck { idx: f(idx), limit },
            Self::Add(a, b) => Op::Add(f(a), f(b)),
            Self::Sub(a, b) => Op::Sub(f(a), f(b)),
            Self::Mul(a, b) => Op::Mul(f(a), f(b)),
            Self::Div(a, b) => Op::Div(f(a), f(b)),
            Self::Rem(a, b) => Op::Rem(f(a), f(b)),
            Self::Shl(a, s) => Op::Shl(f(a), s),
            Self::Shr(a, s) => Op::Shr(f(a), s),
            Self::BitAnd(a, b) => Op::BitAnd(f(a), f(b)),
        }
    }

    pub fn eval_domain(self, dom: impl Fn(Id) -> Domain, buf_dom: impl Fn(u8) -> Domain, is_init: impl Fn(Id) -> bool) -> Domain
    where
        Id: PartialEq,
    {
        match self {
            Self::MemInit | Self::Store { .. } => Domain::top(),
            Self::Const(c) => Domain::exact(c),
            Self::Load { mem, buf, .. } => if is_init(mem) { buf_dom(buf) } else { Domain::top() },
            Self::LoadLut { buf, .. } => buf_dom(buf),
            Self::BoundsCheck { idx, limit } => if limit > 0 { dom(idx).meet(Domain::range(0, limit - 1)) } else { Domain::empty() },
            Self::Add(a, b) => dom(a).eval_add_sub(dom(b), false),
            Self::Sub(a, b) => if a == b { Domain::exact(0) } else { dom(a).eval_add_sub(dom(b), true) },
            Self::Mul(a, b) => if a == b { dom(a).eval_square() } else { dom(a).eval_mul(dom(b)) },
            Self::Div(a, b) => dom(a).eval_div(dom(b)),
            Self::Rem(a, b) => dom(a).eval_rem(dom(b)),
            Self::Shl(a, s) => dom(a).eval_shl(s),
            Self::Shr(a, s) => dom(a).eval_shr(s),
            Self::BitAnd(a, b) => dom(a).eval_and(dom(b)),
        }
    }

    pub fn hw_cost(self, dom: impl Fn(Id) -> Domain, is_init: impl Fn(Id) -> bool) -> (u64, u64) {
        match self {
            Self::MemInit | Self::Const(_) => (0, 0),
            Self::Load { mem, .. } => if is_init(mem) { (4, 1) } else { (12, 2) },
            Self::Store { .. } => (10, 2),
            Self::LoadLut { .. } => (5, 1),
            Self::BoundsCheck { .. } => (12, 2),
            Self::Add(_, _) | Self::Sub(_, _) | Self::Shl(_, _) | Self::Shr(_, _) | Self::BitAnd(_, _) => (1, 1),
            Self::Mul(a, b) => if dom(a).fits_u32() && dom(b).fits_u32() { (4, 1) } else { (10, 2) },
            Self::Div(_, _) | Self::Rem(_, _) => (36, 3),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputWrite { pub buf: u8, pub lane: u8, pub val: NodeId }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelCost {
    pub total_latency: u64, pub total_nodes: usize, pub bounds_checks: usize, pub divs_rems: usize, pub mem_ops: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Kernel {
    pub name: String,
    pub bufs: Vec<BufDecl>,
    pub nodes: Vec<IrOp>,
    pub outputs: Vec<OutputWrite>,
    pub proven_domains: Vec<Domain>,
}

fn checked_div_rem(lhs: i64, rhs: i64, is_rem: bool) -> Result<i64, CompileError> {
    if rhs == 0 || (lhs == i64::MIN && rhs == -1) { Err(CompileError::DivByZero) }
    else if is_rem { Ok(lhs % rhs) } else { Ok(lhs / rhs) }
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self { Self { name: name.into(), ..Self::default() } }
    pub fn add_buf(&mut self, name: impl Into<String>, access: BufAccess, domain: Domain) -> u8 {
        let id = self.bufs.len() as u8;
        self.bufs.push(BufDecl { id, name: name.into(), access, domain });
        id
    }
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
                Op::LoadLut { buf, .. } => {
                    let b = self.bufs.get(buf as usize).ok_or(CompileError::InvalidBuffer(buf))?;
                    if !matches!(b.access, BufAccess::LookupTable(_)) { return Err(CompileError::InvalidBuffer(buf)); }
                }
                Op::Shl(_, s) | Op::Shr(_, s) if s >= 64 => return Err(CompileError::InvalidShift(s)),
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
            let d = op.eval_domain(|k| doms[k], |b| self.bufs[b as usize].domain, |k| matches!(self.nodes[k], Op::MemInit));
            doms.push(self.proven_domains.get(id).copied().map_or(d, |p| d.meet(p)));
        }
        doms
    }

    pub fn cost(&self) -> KernelCost {
        let doms = self.infer_domains();
        let mut c = KernelCost { total_latency: 0, total_nodes: self.nodes.len(), bounds_checks: 0, divs_rems: 0, mem_ops: 0 };
        for &op in &self.nodes {
            let (lat, reg) = op.hw_cost(|id| doms[id], |id| matches!(self.nodes[id], Op::MemInit));
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
        for (idx, decl) in self.bufs.iter().enumerate() {
            let sl = inputs[idx];
            let need = match decl.access { BufAccess::LookupTable(n) => n, _ => LANES };
            if sl.len() < need { return Err(CompileError::OutOfBounds { idx: sl.len() as i64, limit: need as i64 }); }
            let mut arr = [0i64; LANES];
            if !matches!(decl.access, BufAccess::LookupTable(_)) { arr.copy_from_slice(&sl[..LANES]); }
            init_mem.push(arr);
        }
        let (mut vals, mut mems, mut mslot) = (vec![0i64; self.nodes.len()], Vec::new(), vec![0usize; self.nodes.len()]);
        for (id, &op) in self.nodes.iter().enumerate() {
            match op {
                Op::MemInit => { mslot[id] = mems.len(); mems.push(init_mem.clone()); }
                Op::Const(c) => vals[id] = c,
                Op::Load { mem, buf, lane } => vals[id] = mems[mslot[mem]][buf as usize][lane as usize],
                Op::Store { mem, buf, lane, val } => {
                    let mut next = mems[mslot[mem]].clone();
                    next[buf as usize][lane as usize] = vals[val];
                    mslot[id] = mems.len();
                    mems.push(next);
                }
                Op::LoadLut { buf, idx } => {
                    let limit = match self.bufs[buf as usize].access { BufAccess::LookupTable(n) => n as i64, _ => 0 };
                    let iv = vals[idx];
                    if iv < 0 || iv >= limit { return Err(CompileError::OutOfBounds { idx: iv, limit }); }
                    vals[id] = inputs[buf as usize][iv as usize];
                }
                Op::BoundsCheck { idx, limit } => {
                    let iv = vals[idx];
                    if iv < 0 || iv >= limit { return Err(CompileError::OutOfBounds { idx: iv, limit }); }
                    vals[id] = iv;
                }
                Op::Add(a, b) => vals[id] = vals[a].wrapping_add(vals[b]),
                Op::Sub(a, b) => vals[id] = vals[a].wrapping_sub(vals[b]),
                Op::Mul(a, b) => vals[id] = vals[a].wrapping_mul(vals[b]),
                Op::Div(a, b) => vals[id] = checked_div_rem(vals[a], vals[b], false)?,
                Op::Rem(a, b) => vals[id] = checked_div_rem(vals[a], vals[b], true)?,
                Op::Shl(a, s) => vals[id] = vals[a].wrapping_shl(u32::from(s)),
                Op::Shr(a, s) => vals[id] = vals[a].wrapping_shr(u32::from(s)),
                Op::BitAnd(a, b) => vals[id] = vals[a] & vals[b],
            }
        }
        let mut out = init_mem;
        for o in &self.outputs { out[o.buf as usize][o.lane as usize] = vals[o.val]; }
        Ok(out)
    }
}
