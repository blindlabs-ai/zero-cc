use std::collections::{BTreeMap, BTreeSet};
use crate::ir::{barrett_params, CmpKind, CompileError, Domain, Kernel, NodeId, Op, OutputWrite};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct EClassId(pub usize);
pub type ENode = Op<EClassId>;
#[derive(Clone, Debug)] struct EClass { nodes: Vec<ENode>, domain: Domain, parents: Vec<(ENode, EClassId)> }
pub struct EGraph {
    uf: Vec<usize>, classes: Vec<Option<EClass>>, memo: BTreeMap<ENode, EClassId>,
    pending: Vec<EClassId>, buf_domains: Vec<Domain> }
enum Act {
    Alias(EClassId, EClassId), Node(EClassId, ENode), MaskRem(EClassId, EClassId, i64),
    Barrett(EClassId, EClassId, i64, u8, Option<i64>), ShiftPair(EClassId, EClassId, u8, u8, bool),
    Factor(EClassId, EClassId, EClassId, bool, Option<EClassId>, u8) }
impl EGraph {
    pub fn new(buf_domains: Vec<Domain>) -> Self {
        Self { uf: Vec::new(), classes: Vec::new(), memo: BTreeMap::new(), pending: Vec::new(), buf_domains } }
    pub fn find(&self, mut id: EClassId) -> EClassId {
        while self.uf[id.0] != id.0 { id = EClassId(self.uf[id.0]); }
        id }
    fn find_mut(&mut self, mut cur: EClassId) -> EClassId {
        let root = self.find(cur);
        while self.uf[cur.0] != root.0 { cur = EClassId(std::mem::replace(&mut self.uf[cur.0], root.0)); }
        root }
    fn class(&self, id: EClassId) -> Option<&EClass> { self.classes[self.find(id).0].as_ref() }
    fn nodes(&self, id: EClassId) -> &[ENode] { self.class(id).map_or(&[], |c| &c.nodes) }
    pub fn domain(&self, id: EClassId) -> Domain { self.class(id).map_or(Domain::top(), |c| c.domain) }
    fn const_of(&self, id: EClassId) -> Option<i64> { self.domain(id).is_const() }
    fn is_mem_init(&self, id: EClassId) -> bool { self.nodes(id).contains(&Op::MemInit) }
    fn canonicalize(&mut self, node: ENode) -> ENode { node.map_ids(|c| self.find_mut(c)) }
    fn is_bool(&self, id: EClassId) -> bool { matches!(self.domain(id), d if !d.is_empty() && d.lo >= 0 && d.hi <= 1) }
    fn eval_dom(&self, n: ENode) -> Domain {
        n.eval_domain(|c| self.domain(c), |b| self.buf_domains.get(b as usize).copied().unwrap_or_else(Domain::top),
            |c| self.is_mem_init(c)) }
    pub fn add(&mut self, node: ENode) -> EClassId {
        let canon = self.canonicalize(node);
        if let Some(&existing) = self.memo.get(&canon) { return self.find_mut(existing); }
        let (dom, id) = (self.eval_dom(canon), EClassId(self.uf.len()));
        self.uf.push(id.0);
        self.classes.push(Some(EClass { nodes: vec![canon], domain: dom, parents: Vec::new() }));
        self.memo.insert(canon, id);
        let (kids, cnt) = canon.children();
        for &kid in &kids[..cnt] { if let Some(kc) = self.classes[kid.0].as_mut() { kc.parents.push((canon, id)); }}
        id }
    pub fn union(&mut self, a: EClassId, b: EClassId) -> bool {
        let (mut ra, mut rb) = (self.find_mut(a), self.find_mut(b));
        if ra == rb { return false; }
        if ra.0 > rb.0 { std::mem::swap(&mut ra, &mut rb); }
        self.uf[rb.0] = ra.0;
        let Some(cb) = self.classes[rb.0].take() else { return false };
        if let Some(ca) = self.classes[ra.0].as_mut() {
            ca.nodes.extend(cb.nodes);
            ca.parents.extend(cb.parents);
            ca.domain = ca.domain.meet(cb.domain);
        }
        self.pending.push(ra);
        true }
    pub fn rebuild(&mut self) { while !self.pending.is_empty() {
        let mut roots: Vec<_> = std::mem::take(&mut self.pending).into_iter().map(|c| self.find_mut(c)).collect();
        roots.sort();
        roots.dedup();
        for root in roots { self.repair_class(root); }
    }}
    fn repair_class(&mut self, root: EClassId) {
        let Some(cls) = self.classes[root.0].as_mut() else { return };
        let (old_nodes, old_parents) = (std::mem::take(&mut cls.nodes), std::mem::take(&mut cls.parents));
        let mut new_nodes: Vec<ENode> = old_nodes.into_iter().map(|n| self.canonicalize(n)).collect();
        new_nodes.sort();
        new_nodes.dedup();
        if let Some(c) = self.classes[root.0].as_mut() { c.nodes = new_nodes; }
        let mut new_parents: BTreeMap<ENode, EClassId> = BTreeMap::new();
        for (p_node, p_id) in old_parents {
            self.memo.remove(&p_node);
            let (canon, p_root) = (self.canonicalize(p_node), self.find_mut(p_id));
            if let Some(&prev) = self.memo.get(&canon) { self.union(prev, p_root); }
            else { self.memo.insert(canon, p_root); }
            if let Some(prev) = new_parents.insert(canon, p_root) { self.union(prev, p_root); }
            let (d, pr) = (self.eval_dom(canon), self.find_mut(p_root));
            if let Some(pc) = self.classes[pr.0].as_mut().filter(|pc| pc.domain.meet(d) != pc.domain) {
                pc.domain = pc.domain.meet(d);
                self.pending.push(pr);
            }
        }
        let res: Vec<_> = new_parents.into_iter().map(|(pn, pid)| (pn, self.find_mut(pid))).collect();
        let fr = self.find_mut(root);
        if let Some(c) = self.classes[fr.0].as_mut() { c.parents.extend(res); }
    }
    fn apply_rewrites(&mut self) -> usize {
        let (mut acts, mut changed) = (Vec::new(), 0usize);
        for idx in 0..self.classes.len() {
            let Some(cls) = &self.classes[idx] else { continue };
            let cid = EClassId(idx);
            if let Some(c) = cls.domain.is_const().filter(|c| !cls.nodes.contains(&Op::Const(*c))) {
                acts.push(Act::Node(cid, Op::Const(c))); }
            for &node in &cls.nodes { self.match_node(cid, node, &mut acts); }
        }
        for act in acts {
            let (lhs, rhs) = match act {
                Act::Alias(l, r) => (l, r),
                Act::Node(l, op) => (l, self.add(op)),
                Act::MaskRem(l, x, mask) => { let mc = self.add(Op::Const(mask));
                    (l, self.add(Op::BitAnd(x, mc))) }
                Act::Barrett(l, x, m, s, d) => {
                    let mc = self.add(Op::Const(m));
                    let xm = self.add(Op::Mul(x, mc));
                    let q = self.add(Op::Shr(xm, s));
                    if let Some(dv) = d {
                        let dc = self.add(Op::Const(dv));
                        let qd = self.add(Op::Mul(q, dc));
                        (l, self.add(Op::Sub(x, qd)))
                    } else { (l, q) }
                }
                Act::ShiftPair(l, x, s1, s2, is_sub) => {
                    let (t1, t2) = (self.add(Op::Shl(x, s1)), if s2 == 0 { x } else { self.add(Op::Shl(x, s2)) });
                    (l, self.add(if is_sub { Op::Sub(t1, t2) } else { Op::Add(t1, t2) })) }
                Act::Factor(l, u, v, is_sub, mul_c, shl_k) => {
                    let inner = self.add(if is_sub { Op::Sub(u, v) } else { Op::Add(u, v) });
                    (l, self.add(mul_c.map_or(Op::Shl(inner, shl_k), |c| Op::Mul(inner, c)))) }
            };
            if self.union(lhs, rhs) { changed += 1; }
        }
        if changed > 0 { self.rebuild(); }
        changed }
    fn match_node(&self, cid: EClassId, node: ENode, out: &mut Vec<Act>) { match node {
        Op::BoundsCheck { idx, limit } => {
            let d = self.domain(idx);
            if !d.is_empty() && d.lo >= 0 && d.hi < limit { out.push(Act::Alias(cid, idx)); }},
        Op::Load { mem, buf, lane } => for &mn in self.nodes(mem) {
            if let Op::Store { mem: prev, buf: sb, lane: sl, val } = mn {
                out.push(if sb == buf && sl == lane { Act::Alias(cid, val) }
                    else { Act::Node(cid, Op::Load { mem: prev, buf, lane }) }); }},
        Op::Add(a, b) | Op::Sub(a, b) => {
            let is_sub = matches!(node, Op::Sub(_, _));
            if self.const_of(b) == Some(0) { out.push(Act::Alias(cid, a)); }
            else if !is_sub && self.const_of(a) == Some(0) { out.push(Act::Alias(cid, b)); }
            else if is_sub && a == b { out.push(Act::Node(cid, Op::Const(0))); }
            self.match_distributive(cid, a, b, is_sub, out);
        }
        Op::Mul(a, b) => for (x, c) in [(a, b), (b, a)] { self.match_mul_const(cid, x, c, out); },
        Op::Div(x, d) | Op::Rem(x, d) => self.match_div_rem(cid, x, d, matches!(node, Op::Rem(_, _)), out),
        Op::Shl(x, s) => self.match_shift(cid, x, s, 0, out),
        Op::Shr(x, s) => self.match_shift(cid, x, s, 1, out),
        Op::Ushr(x, s) => self.match_shift(cid, x, s, 2, out),
        Op::ShlVar(x, s_id) | Op::ShrVarU(x, s_id) => if let Some(s) = self.const_of(s_id) {
            let sh = (s as u8) & 63;
            out.push(Act::Node(cid, if matches!(node, Op::ShlVar(_, _)) { Op::Shl(x, sh) } else { Op::Ushr(x, sh) }));
        } else if self.const_of(x) == Some(0) { out.push(Act::Node(cid, Op::Const(0))); },
        Op::BitAnd(a, b) => self.match_bitand(cid, a, b, out),
        Op::BitOr(a, b) => {
            let (ca, cb) = (self.const_of(a), self.const_of(b));
            if a == b || cb == Some(0) { out.push(Act::Alias(cid, a)); }
            else if ca == Some(0) { out.push(Act::Alias(cid, b)); }
            else if ca == Some(-1) || cb == Some(-1) { out.push(Act::Node(cid, Op::Const(-1))); }
        }
        Op::BitXor(a, b) => if a == b { out.push(Act::Node(cid, Op::Const(0))); }
            else if self.const_of(b) == Some(0) { out.push(Act::Alias(cid, a)); }
            else if self.const_of(a) == Some(0) { out.push(Act::Alias(cid, b)); },
        Op::Popcnt(a) | Op::Tzcnt(a) => if let Some(c) = self.const_of(a) {
            let v = if matches!(node, Op::Popcnt(_)) { (c as u64).count_ones() } else { (c as u64).trailing_zeros() };
            out.push(Act::Node(cid, Op::Const(i64::from(v)))); },
        Op::Cmp(k, a, b) => for (v, ci) in [(a, b), (b, a)] {
            let c = self.const_of(ci);
            if self.is_bool(v) && ((k == CmpKind::Ne && c == Some(0)) || (k == CmpKind::Eq && c == Some(1))) {
                out.push(Act::Alias(cid, v)); }},
        Op::Select { cond, t_val, f_val } => if let Some(dst) = match self.domain(cond) {
            d if t_val == f_val || (!d.is_empty() && !d.contains(0)) => Some(t_val),
            d if d.is_const() == Some(0) => Some(f_val),
            _ if self.is_bool(cond) && self.const_of(t_val) == Some(1) && self.const_of(f_val) == Some(0) => Some(cond),
            _ => None,
        } { out.push(Act::Alias(cid, dst)); },
        Op::MemInit | Op::Const(_) | Op::Store { .. } | Op::LoadLut { .. } => {}
    }}
    fn match_div_rem(&self, cid: EClassId, x: EClassId, d_id: EClassId, is_rem: bool, out: &mut Vec<Act>) {
        let Some(d) = self.const_of(d_id) else { return };
        let dx = self.domain(x);
        let act = if d == 1 { Some(if is_rem { Act::Node(cid, Op::Const(0)) } else { Act::Alias(cid, x) }) }
            else if d <= 1 || !dx.is_nonneg() { None }
            else if is_rem {
                if let Some(r) = dx.rem_mod(d) { Some(Act::Node(cid, Op::Const(r))) }
                else if dx.hi < d { Some(Act::Alias(cid, x)) }
                else if (d as u64).is_power_of_two() { Some(Act::MaskRem(cid, x, d - 1)) }
                else { barrett_params(d, dx.hi).map(|(m, s)| Act::Barrett(cid, x, m, s, Some(d))) }
            } else if dx.hi < d { Some(Act::Node(cid, Op::Const(0))) }
            else if (d as u64).is_power_of_two() { Some(Act::Node(cid, Op::Shr(x, d.trailing_zeros() as u8))) }
            else { barrett_params(d, dx.hi).map(|(m, s)| Act::Barrett(cid, x, m, s, None)) };
        if let Some(a) = act { out.push(a); }
    }
    fn match_shift(&self, cid: EClassId, x: EClassId, s: u8, mode: u8, out: &mut Vec<Act>) {
        if s == 0 { out.push(Act::Alias(cid, x)); }
        for &xn in self.nodes(x) { match (mode, xn) {
            (0, Op::Shl(b, s0)) if s0.saturating_add(s) < 64 => out.push(Act::Node(cid, Op::Shl(b, s0 + s))),
            (0, Op::BitAnd(u, v)) if s > 0 && s < 64 => for (vid, mid) in [(u, v), (v, u)] {
                let Some(m) = self.const_of(mid) else { continue };
                let mask = ((m as u64) << s) as i64;
                for &vn in self.nodes(vid) {
                    if let Op::Shr(b, k) | Op::Ushr(b, k) = vn { if k == s { out.push(Act::MaskRem(cid, b, mask)); }}
                }
            },
            (1, Op::Shr(b, s0)) if s0.saturating_add(s) < 64 && self.domain(b).is_nonneg() =>
                out.push(Act::Node(cid, Op::Shr(b, s0 + s))),
            (1, Op::Shl(base, s0)) => {
                let db = self.domain(base);
                if db.is_nonneg() && s0 < 63 && ((db.hi as u128) << s0) <= i64::MAX as u128 {
                    out.push(if s0 == s { Act::Alias(cid, base) }
                        else if s0 > s { Act::Node(cid, Op::Shl(base, s0 - s)) }
                        else { Act::Node(cid, Op::Shr(base, s - s0)) }); }}
            (2, Op::Ushr(b, s0)) if s0.saturating_add(s) < 64 => out.push(Act::Node(cid, Op::Ushr(b, s0 + s))),
            _ => {}
        }}
    }
    fn match_bitand(&self, cid: EClassId, a: EClassId, b: EClassId, out: &mut Vec<Act>) {
        if a == b { out.push(Act::Alias(cid, a)); }
        else if self.const_of(a) == Some(0) || self.const_of(b) == Some(0) { out.push(Act::Node(cid, Op::Const(0))); }
        else { for (v, mid) in [(a, b), (b, a)] {
            if let Some(m) = self.const_of(mid).filter(|&m| m >= 0 && ((m as u64) + 1).is_power_of_two()) {
                if matches!(self.domain(v), d if d.is_nonneg() && d.hi <= m) { out.push(Act::Alias(cid, v)); }
            }
        }}
    }
    fn match_mul_const(&self, cid: EClassId, x: EClassId, c_id: EClassId, out: &mut Vec<Act>) {
        let Some(c) = self.const_of(c_id) else { return };
        let (cu, tz) = (c as u64, |u: u64| u.trailing_zeros() as u8);
        if c == 0 { out.push(Act::Node(cid, Op::Const(0))); }
        else if c == 1 { out.push(Act::Alias(cid, x)); }
        else if c > 1 && cu.is_power_of_two() { out.push(Act::Node(cid, Op::Shl(x, tz(cu)))); }
        else if c > 1 && (cu - 1).is_power_of_two() { out.push(Act::ShiftPair(cid, x, tz(cu - 1), 0, false)); }
        else if c > 1 && (cu + 1).is_power_of_two() { out.push(Act::ShiftPair(cid, x, tz(cu + 1), 0, true)); }
        else if c > 1 && cu.count_ones() == 2 {
            out.push(Act::ShiftPair(cid, x, tz(cu ^ (1u64 << tz(cu))), tz(cu), false)); }
    }
    fn match_distributive(&self, cid: EClassId, l: EClassId, r: EClassId, is_sub: bool, out: &mut Vec<Act>) {
        for &ln in self.nodes(l) { for &rn in self.nodes(r) {
            if let (Op::Mul(a1, b1), Op::Mul(a2, b2)) = (ln, rn) {
                let p = if b1 == b2 { Some((a1, a2, b1)) } else if a1 == a2 { Some((b1, b2, a1)) }
                    else if b1 == a2 { Some((a1, b2, b1)) } else if a1 == b2 { Some((b1, a2, a1)) } else { None };
                if let Some((u, v, c)) = p { out.push(Act::Factor(cid, u, v, is_sub, Some(c), 0)); }
            } else if let (Op::Shl(u, k1), Op::Shl(v, k2)) = (ln, rn) {
                if k1 == k2 { out.push(Act::Factor(cid, u, v, is_sub, None, k1)); }
            }
        }}
    }
    fn self_cost(&self, node: ENode) -> u64 {
        let (l, r) = node.hw_cost(|c| self.domain(c), |c| self.is_mem_init(c));
        l + r }
    fn dag_cost(&self, best: &[Option<ENode>], root: usize, node: ENode) -> Option<u64> {
        let (mut seen, mut total, (kids, cnt)) = (BTreeSet::from([root]), self.self_cost(node), node.children());
        let mut stack: Vec<usize> = kids[..cnt].iter().map(|&k| self.find(k).0).collect();
        while let Some(cur) = stack.pop() {
            if cur == root { return None; }
            if !seen.insert(cur) { continue; }
            let bn = best[cur]?;
            total = total.saturating_add(self.self_cost(bn));
            let (bk, bc) = bn.children();
            for &k in &bk[..bc] { stack.push(self.find(k).0); }
        }
        Some(total) }
    fn extract_best(&self) -> Vec<Option<ENode>> {
        let n = self.classes.len();
        let (mut best_cost, mut best_node) = (vec![u64::MAX; n], vec![None; n]);
        for _ in 0..=n {
            let mut updated = false;
            for (idx, cls) in self.classes.iter().enumerate() {
                let Some(cls) = cls else { continue };
                for &node in &cls.nodes {
                    let Some(cost) = self.dag_cost(&best_node, idx, node) else { continue };
                    if cost < best_cost[idx] || (cost == best_cost[idx] && best_node[idx].is_some_and(|b| node < b)) {
                        (best_cost[idx], best_node[idx], updated) = (cost, Some(node), true);
                    }
                }
            }
            if !updated { break; }
        }
        best_node }
}
fn emit_dag(eg: &EGraph, best: &[Option<ENode>], cid: EClassId, memo: &mut BTreeMap<EClassId, NodeId>,
    out: &mut Kernel) -> Result<NodeId, CompileError> {
    let root = eg.find(cid);
    if let Some(&id) = memo.get(&root) { return Ok(id); }
    let node = best[root.0].ok_or(CompileError::InvalidNode(root.0))?;
    let ((kids, cnt), mut mapped) = (node.children(), [0usize; 3]);
    for i in 0..cnt { mapped[i] = emit_dag(eg, best, kids[i], memo, out)?; }
    let (mut it, id) = (mapped[..cnt].iter().copied(), out.nodes.len());
    out.nodes.push(node.map_ids(|_| it.next().unwrap_or(0)));
    out.proven_domains.push(eg.domain(root));
    memo.insert(root, id);
    Ok(id) }
pub fn optimize_kernel(kernel: &Kernel) -> Result<Kernel, CompileError> {
    kernel.validate()?;
    let mut eg = EGraph::new(kernel.bufs.iter().map(|b| b.domain).collect());
    let mut node_map = Vec::with_capacity(kernel.nodes.len());
    for (id, &op) in kernel.nodes.iter().enumerate() {
        let cid = eg.add(op.map_ids(|k| node_map[k]));
        if let Some(&p) = kernel.proven_domains.get(id) {
            let root = eg.find_mut(cid);
            if let Some(c) = eg.classes[root.0].as_mut() { c.domain = c.domain.meet(p); }
        }
        node_map.push(cid);
    }
    for _ in 0..16 { if eg.classes.len() > 4096 || eg.apply_rewrites() == 0 { break; }}
    let (best, mut memo) = (eg.extract_best(), BTreeMap::new());
    let mut opt = Kernel { name: kernel.name.clone(), bufs: kernel.bufs.clone(),
        nodes: Vec::new(), outputs: Vec::with_capacity(kernel.outputs.len()), proven_domains: Vec::new() };
    for o in &kernel.outputs {
        let val = emit_dag(&eg, &best, node_map[o.val], &mut memo, &mut opt)?;
        opt.outputs.push(OutputWrite { buf: o.buf, lane: o.lane, val });
    }
    opt.validate()?;
    Ok(opt) }
