use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{barrett_params, CmpKind, CompileError, Domain, Kernel, NodeId, Op, OutputWrite};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EClassId(pub usize);
pub type ENode = Op<EClassId>;

#[derive(Clone, Debug)]
struct EClass { nodes: Vec<ENode>, domain: Domain, parents: Vec<(ENode, EClassId)> }

pub struct EGraph {
    uf: Vec<usize>, classes: Vec<Option<EClass>>, memo: BTreeMap<ENode, EClassId>,
    pending: Vec<EClassId>, buf_domains: Vec<Domain>,
}

enum Act {
    Alias(EClassId, EClassId), Node(EClassId, ENode),
    MaskRem { l: EClassId, x: EClassId, mask: i64 },
    Barrett { l: EClassId, x: EClassId, m: i64, s: u8, d: Option<i64> },
    ShiftPair { l: EClassId, x: EClassId, s1: u8, s2: u8, is_sub: bool },
    Factor { l: EClassId, u: EClassId, v: EClassId, is_sub: bool, mul_c: Option<EClassId>, shl_k: u8 },
}

impl EGraph {
    pub fn new(buf_domains: Vec<Domain>) -> Self { Self { uf: Vec::new(), classes: Vec::new(), memo: BTreeMap::new(), pending: Vec::new(), buf_domains } }
    pub fn find(&self, mut id: EClassId) -> EClassId { while self.uf[id.0] != id.0 { id = EClassId(self.uf[id.0]); } id }
    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let (root, mut cur) = (self.find(id), id);
        while self.uf[cur.0] != root.0 { let next = EClassId(self.uf[cur.0]); self.uf[cur.0] = root.0; cur = next; } root
    }
    fn class(&self, id: EClassId) -> Option<&EClass> { self.classes[self.find(id).0].as_ref() }
    pub fn domain(&self, id: EClassId) -> Domain { self.class(id).map_or(Domain::top(), |c| c.domain) }
    fn is_mem_init(&self, id: EClassId) -> bool { self.class(id).is_some_and(|c| c.nodes.contains(&Op::MemInit)) }
    fn canonicalize(&mut self, node: ENode) -> ENode { node.map_ids(|c| self.find_mut(c)) }
    fn is_bool(&self, id: EClassId) -> bool { let d = self.domain(id); !d.is_empty() && d.lo >= 0 && d.hi <= 1 }

    pub fn add(&mut self, node: ENode) -> EClassId {
        let canon = self.canonicalize(node);
        if let Some(&existing) = self.memo.get(&canon) { return self.find_mut(existing); }
        let (dom, id) = (canon.eval_domain(|c| self.domain(c), |b| self.buf_domains[b as usize], |c| self.is_mem_init(c)), EClassId(self.uf.len()));
        self.uf.push(id.0); self.classes.push(Some(EClass { nodes: vec![canon], domain: dom, parents: Vec::new() })); self.memo.insert(canon, id);
        let (kids, cnt) = canon.children();
        for &kid in &kids[..cnt] { if let Some(kc) = self.classes[kid.0].as_mut() { kc.parents.push((canon, id)); } }
        id
    }
    pub fn union(&mut self, a: EClassId, b: EClassId) -> bool {
        let (mut ra, mut rb) = (self.find_mut(a), self.find_mut(b));
        if ra == rb { return false; }
        if ra.0 > rb.0 { std::mem::swap(&mut ra, &mut rb); }
        self.uf[rb.0] = ra.0;
        let Some(cb) = self.classes[rb.0].take() else { return false; };
        if let Some(ca) = self.classes[ra.0].as_mut() { ca.nodes.extend(cb.nodes); ca.parents.extend(cb.parents); ca.domain = ca.domain.meet(cb.domain); }
        self.pending.push(ra); true
    }
    pub fn rebuild(&mut self) {
        while !self.pending.is_empty() {
            let mut roots: Vec<EClassId> = std::mem::take(&mut self.pending).into_iter().map(|c| self.find_mut(c)).collect();
            roots.sort(); roots.dedup();
            for root in roots { self.repair_class(root); }
        }
    }
    fn repair_class(&mut self, root: EClassId) {
        let Some(cls) = self.classes[root.0].as_mut() else { return; };
        let (old_nodes, old_parents) = (std::mem::take(&mut cls.nodes), std::mem::take(&mut cls.parents));
        let mut new_nodes: Vec<ENode> = old_nodes.into_iter().map(|n| self.canonicalize(n)).collect();
        new_nodes.sort(); new_nodes.dedup();
        if let Some(c) = self.classes[root.0].as_mut() { c.nodes = new_nodes; }
        let mut new_parents: BTreeMap<ENode, EClassId> = BTreeMap::new();
        for (p_node, p_id) in old_parents {
            self.memo.remove(&p_node);
            let (canon, p_root) = (self.canonicalize(p_node), self.find_mut(p_id));
            if let Some(&prev) = self.memo.get(&canon) { self.union(prev, p_root); } else { self.memo.insert(canon, p_root); }
            if let Some(prev) = new_parents.insert(canon, p_root) { self.union(prev, p_root); }
            let d = canon.eval_domain(|c| self.domain(c), |b| self.buf_domains[b as usize], |c| self.is_mem_init(c));
            let pr = self.find_mut(p_root);
            if let Some(pc) = self.classes[pr.0].as_mut() { let t = pc.domain.meet(d); if t != pc.domain { pc.domain = t; self.pending.push(pr); } }
        }
        let resolved: Vec<(ENode, EClassId)> = new_parents.into_iter().map(|(pn, pid)| (pn, self.find_mut(pid))).collect();
        let fr = self.find_mut(root);
        if let Some(c) = self.classes[fr.0].as_mut() { c.parents.extend(resolved); }
    }
    fn apply_rewrites(&mut self) -> usize {
        let mut acts = Vec::new();
        for idx in 0..self.classes.len() {
            let Some(cls) = &self.classes[idx] else { continue; };
            let cid = EClassId(idx);
            if let Some(c) = cls.domain.is_const() { if !cls.nodes.contains(&Op::Const(c)) { acts.push(Act::Node(cid, Op::Const(c))); } }
            for &node in &cls.nodes { self.match_node(cid, node, &mut acts); }
        }
        let mut changed = 0usize;
        for act in acts {
            let (lhs, rhs) = match act {
                Act::Alias(l, r) => (l, r), Act::Node(l, op) => (l, self.add(op)),
                Act::MaskRem { l, x, mask } => { let mc = self.add(Op::Const(mask)); (l, self.add(Op::BitAnd(x, mc))) }
                Act::Barrett { l, x, m, s, d } => {
                    let mc = self.add(Op::Const(m)); let xm = self.add(Op::Mul(x, mc)); let q = self.add(Op::Shr(xm, s));
                    if let Some(dv) = d { let dc = self.add(Op::Const(dv)); let qd = self.add(Op::Mul(q, dc)); (l, self.add(Op::Sub(x, qd))) } else { (l, q) }
                }
                Act::ShiftPair { l, x, s1, s2, is_sub } => {
                    let (t1, t2) = (self.add(Op::Shl(x, s1)), if s2 == 0 { x } else { self.add(Op::Shl(x, s2)) });
                    (l, self.add(if is_sub { Op::Sub(t1, t2) } else { Op::Add(t1, t2) }))
                }
                Act::Factor { l, u, v, is_sub, mul_c, shl_k } => {
                    let inner = self.add(if is_sub { Op::Sub(u, v) } else { Op::Add(u, v) });
                    (l, self.add(match mul_c { Some(c) => Op::Mul(inner, c), None => Op::Shl(inner, shl_k) }))
                }
            };
            if self.union(lhs, rhs) { changed += 1; }
        }
        if changed > 0 { self.rebuild(); } changed
    }
    fn match_node(&self, cid: EClassId, node: ENode, out: &mut Vec<Act>) {
        match node {
            Op::BoundsCheck { idx, limit } => { let d = self.domain(idx); if !d.is_empty() && d.lo >= 0 && d.hi < limit { out.push(Act::Alias(cid, idx)); } }
            Op::Load { mem, buf, lane } => for &mn in self.class(mem).map_or(&[][..], |c| &c.nodes) {
                if let Op::Store { mem: prev, buf: sb, lane: sl, val } = mn { out.push(if sb == buf && sl == lane { Act::Alias(cid, val) } else { Act::Node(cid, Op::Load { mem: prev, buf, lane }) }); }
            },
            Op::Add(a, b) => {
                if self.domain(b).is_const() == Some(0) { out.push(Act::Alias(cid, a)); } else if self.domain(a).is_const() == Some(0) { out.push(Act::Alias(cid, b)); }
                self.match_distributive(cid, a, b, false, out);
            }
            Op::Sub(a, b) => {
                if self.domain(b).is_const() == Some(0) { out.push(Act::Alias(cid, a)); } else if a == b { out.push(Act::Node(cid, Op::Const(0))); }
                self.match_distributive(cid, a, b, true, out);
            }
            Op::Mul(a, b) => { self.match_mul_const(cid, a, b, out); self.match_mul_const(cid, b, a, out); }
            Op::Div(x, d_id) | Op::Rem(x, d_id) => self.match_div_rem(cid, x, d_id, matches!(node, Op::Rem(_, _)), out),
            Op::Shl(x, s) | Op::Shr(x, s) => self.match_shift(cid, x, s, matches!(node, Op::Shr(_, _)), out),
            Op::Ushr(x, s) => {
                if s == 0 { out.push(Act::Alias(cid, x)); }
                for &xn in self.class(x).map_or(&[][..], |c| &c.nodes) {
                    if let Op::Ushr(base, s0) = xn { if u16::from(s0) + u16::from(s) < 64 { out.push(Act::Node(cid, Op::Ushr(base, s0 + s))); } }
                }
            }
            Op::ShlVar(x, s_id) => {
                if let Some(s) = self.domain(s_id).is_const() { out.push(Act::Node(cid, Op::Shl(x, (s as u8) & 63))); }
                else if self.domain(x).is_const() == Some(0) { out.push(Act::Node(cid, Op::Const(0))); }
            }
            Op::ShrVarU(x, s_id) => {
                if let Some(s) = self.domain(s_id).is_const() { out.push(Act::Node(cid, Op::Ushr(x, (s as u8) & 63))); }
                else if self.domain(x).is_const() == Some(0) { out.push(Act::Node(cid, Op::Const(0))); }
            }
            Op::BitAnd(a, b) => self.match_bitand(cid, a, b, out),
            Op::BitOr(a, b) => {
                if a == b || self.domain(b).is_const() == Some(0) { out.push(Act::Alias(cid, a)); }
                else if self.domain(a).is_const() == Some(0) { out.push(Act::Alias(cid, b)); }
                else if self.domain(a).is_const() == Some(-1) || self.domain(b).is_const() == Some(-1) { out.push(Act::Node(cid, Op::Const(-1))); }
            }
            Op::BitXor(a, b) => {
                if a == b { out.push(Act::Node(cid, Op::Const(0))); }
                else if self.domain(b).is_const() == Some(0) { out.push(Act::Alias(cid, a)); }
                else if self.domain(a).is_const() == Some(0) { out.push(Act::Alias(cid, b)); }
            }
            Op::Cmp(k, a, b) => for (val, c_id) in [(a, b), (b, a)] {
                if let Some(c) = self.domain(c_id).is_const() { if self.is_bool(val) && ((k == CmpKind::Ne && c == 0) || (k == CmpKind::Eq && c == 1)) { out.push(Act::Alias(cid, val)); } }
            },
            Op::Select { cond, t_val, f_val } => {
                let cd = self.domain(cond);
                if t_val == f_val || (!cd.is_empty() && !cd.contains(0)) { out.push(Act::Alias(cid, t_val)); }
                else if cd.is_const() == Some(0) { out.push(Act::Alias(cid, f_val)); }
                else if self.is_bool(cond) && self.domain(t_val).is_const() == Some(1) && self.domain(f_val).is_const() == Some(0) { out.push(Act::Alias(cid, cond)); }
            }
            Op::MemInit | Op::Const(_) | Op::Store { .. } | Op::LoadLut { .. } => {}
        }
    }
    fn match_div_rem(&self, cid: EClassId, x: EClassId, d_id: EClassId, is_rem: bool, out: &mut Vec<Act>) {
        let Some(d) = self.domain(d_id).is_const() else { return; };
        let dx = self.domain(x);
        if d == 1 { out.push(if is_rem { Act::Node(cid, Op::Const(0)) } else { Act::Alias(cid, x) }); }
        else if d > 1 && dx.is_nonneg() {
            if is_rem {
                if let Some(r) = dx.rem_mod(d) { out.push(Act::Node(cid, Op::Const(r))); }
                else if dx.hi < d { out.push(Act::Alias(cid, x)); }
                else if (d as u64).is_power_of_two() { out.push(Act::MaskRem { l: cid, x, mask: d - 1 }); }
                else if let Some((m, s)) = barrett_params(d, dx.hi) { out.push(Act::Barrett { l: cid, x, m, s, d: Some(d) }); }
            } else if dx.hi < d { out.push(Act::Node(cid, Op::Const(0))); }
            else if (d as u64).is_power_of_two() { out.push(Act::Node(cid, Op::Shr(x, d.trailing_zeros() as u8))); }
            else if let Some((m, s)) = barrett_params(d, dx.hi) { out.push(Act::Barrett { l: cid, x, m, s, d: None }); }
        }
    }
    fn match_shift(&self, cid: EClassId, x: EClassId, s: u8, is_shr: bool, out: &mut Vec<Act>) {
        if s == 0 { out.push(Act::Alias(cid, x)); }
        for &xn in self.class(x).map_or(&[][..], |c| &c.nodes) {
            match (is_shr, xn) {
                (false, Op::Shl(base, s0)) if u16::from(s0) + u16::from(s) < 64 => out.push(Act::Node(cid, Op::Shl(base, s0 + s))),
                (true, Op::Shr(base, s0)) if u16::from(s0) + u16::from(s) < 64 && self.domain(base).is_nonneg() => out.push(Act::Node(cid, Op::Shr(base, s0 + s))),
                (true, Op::Shl(base, s0)) => {
                    let db = self.domain(base);
                    if db.is_nonneg() && s0 < 63 && ((db.hi as u128) << s0) <= i64::MAX as u128 {
                        if s0 == s { out.push(Act::Alias(cid, base)); }
                        else if s0 > s { out.push(Act::Node(cid, Op::Shl(base, s0 - s))); }
                        else { out.push(Act::Node(cid, Op::Shr(base, s - s0))); }
                    }
                }
                _ => {}
            }
        }
    }
    fn match_bitand(&self, cid: EClassId, a: EClassId, b: EClassId, out: &mut Vec<Act>) {
        if a == b { out.push(Act::Alias(cid, a)); return; }
        if self.domain(a).is_const() == Some(0) || self.domain(b).is_const() == Some(0) { out.push(Act::Node(cid, Op::Const(0))); return; }
        for (val, mask_id) in [(a, b), (b, a)] {
            if let Some(m) = self.domain(mask_id).is_const().filter(|&m| m >= 0 && ((m as u64) + 1).is_power_of_two()) {
                let dv = self.domain(val); if dv.is_nonneg() && dv.hi <= m { out.push(Act::Alias(cid, val)); }
            }
        }
    }
    fn match_mul_const(&self, cid: EClassId, x: EClassId, c_id: EClassId, out: &mut Vec<Act>) {
        let Some(c) = self.domain(c_id).is_const() else { return; };
        if c == 0 { out.push(Act::Node(cid, Op::Const(0))); } else if c == 1 { out.push(Act::Alias(cid, x)); }
        else if c > 1 {
            let cu = c as u64;
            if cu.is_power_of_two() { out.push(Act::Node(cid, Op::Shl(x, cu.trailing_zeros() as u8))); }
            else if (cu - 1).is_power_of_two() { out.push(Act::ShiftPair { l: cid, x, s1: (cu - 1).trailing_zeros() as u8, s2: 0, is_sub: false }); }
            else if (cu + 1).is_power_of_two() { out.push(Act::ShiftPair { l: cid, x, s1: (cu + 1).trailing_zeros() as u8, s2: 0, is_sub: true }); }
            else if cu.count_ones() == 2 {
                let s2 = cu.trailing_zeros() as u8;
                out.push(Act::ShiftPair { l: cid, x, s1: (cu ^ (1u64 << s2)).trailing_zeros() as u8, s2, is_sub: false });
            }
        }
    }
    fn match_distributive(&self, cid: EClassId, l: EClassId, r: EClassId, is_sub: bool, out: &mut Vec<Act>) {
        for &ln in self.class(l).map_or(&[][..], |c| &c.nodes) {
            for &rn in self.class(r).map_or(&[][..], |c| &c.nodes) {
                if let (Op::Mul(a1, b1), Op::Mul(a2, b2)) = (ln, rn) {
                    let pair = if b1 == b2 { Some((a1, a2, b1)) } else if a1 == a2 { Some((b1, b2, a1)) }
                        else if b1 == a2 { Some((a1, b2, b1)) } else if a1 == b2 { Some((b1, a2, a1)) } else { None };
                    if let Some((u, v, c)) = pair { out.push(Act::Factor { l: cid, u, v, is_sub, mul_c: Some(c), shl_k: 0 }); }
                } else if let (Op::Shl(u, k1), Op::Shl(v, k2)) = (ln, rn) {
                    if k1 == k2 { out.push(Act::Factor { l: cid, u, v, is_sub, mul_c: None, shl_k: k1 }); }
                }
            }
        }
    }
    fn self_cost(&self, node: ENode) -> u64 { let (l, r) = node.hw_cost(|c| self.domain(c), |c| self.is_mem_init(c)); l + r }
    fn dag_cost(&self, best: &[Option<ENode>], root: usize, node: ENode) -> Option<u64> {
        let ((mut seen, mut total), (kids, cnt)) = ((BTreeSet::from([root]), self.self_cost(node)), node.children());
        let mut stack: Vec<usize> = kids[..cnt].iter().map(|&k| self.find(k).0).collect();
        while let Some(cur) = stack.pop() {
            if cur == root { return None; }
            if !seen.insert(cur) { continue; }
            let bn = best[cur]?; total = total.saturating_add(self.self_cost(bn));
            let (bk, bc) = bn.children();
            for &k in &bk[..bc] { stack.push(self.find(k).0); }
        }
        Some(total)
    }
    fn extract_best(&self) -> Vec<Option<ENode>> {
        let (n, mut best_cost, mut best_node) = (self.classes.len(), vec![u64::MAX; self.classes.len()], vec![None; self.classes.len()]);
        for _ in 0..=n {
            let mut updated = false;
            for idx in 0..n {
                let Some(cls) = &self.classes[idx] else { continue; };
                for &node in &cls.nodes {
                    if let Some(cost) = self.dag_cost(&best_node, idx, node) {
                        if cost < best_cost[idx] || (cost == best_cost[idx] && best_node[idx].is_some_and(|b| node < b)) { best_cost[idx] = cost; best_node[idx] = Some(node); updated = true; }
                    }
                }
            }
            if !updated { break; }
        }
        best_node
    }
}

fn emit_dag(eg: &EGraph, best: &[Option<ENode>], cid: EClassId, memo: &mut BTreeMap<EClassId, NodeId>, out: &mut Kernel) -> Result<NodeId, CompileError> {
    let root = eg.find(cid);
    if let Some(&id) = memo.get(&root) { return Ok(id); }
    let node = best[root.0].ok_or(CompileError::InvalidNode(root.0))?;
    let ((kids, cnt), mut mapped, mut idx) = (node.children(), [0usize; 3], 0usize);
    for i in 0..cnt { mapped[i] = emit_dag(eg, best, kids[i], memo, out)?; }
    let (ir_op, id) = (node.map_ids(|_| { let id = mapped[idx]; idx += 1; id }), out.nodes.len());
    out.nodes.push(ir_op); out.proven_domains.push(eg.domain(root)); memo.insert(root, id); Ok(id)
}

pub fn optimize_kernel(kernel: &Kernel) -> Result<Kernel, CompileError> {
    kernel.validate()?;
    let (mut eg, mut node_map) = (EGraph::new(kernel.bufs.iter().map(|b| b.domain).collect()), Vec::with_capacity(kernel.nodes.len()));
    for (id, &op) in kernel.nodes.iter().enumerate() {
        let cid = eg.add(op.map_ids(|k| node_map[k]));
        if let Some(&p) = kernel.proven_domains.get(id) { let root = eg.find_mut(cid); if let Some(c) = eg.classes[root.0].as_mut() { c.domain = c.domain.meet(p); } }
        node_map.push(cid);
    }
    for _ in 0..16 { if eg.classes.len() > 4096 || eg.apply_rewrites() == 0 { break; } }
    let (best, mut memo) = (eg.extract_best(), BTreeMap::new());
    let mut opt = Kernel { name: kernel.name.clone(), bufs: kernel.bufs.clone(), nodes: Vec::new(), outputs: Vec::with_capacity(kernel.outputs.len()), proven_domains: Vec::new() };
    for o in &kernel.outputs { let val = emit_dag(&eg, &best, node_map[o.val], &mut memo, &mut opt)?; opt.outputs.push(OutputWrite { buf: o.buf, lane: o.lane, val }); }
    opt.validate()?; Ok(opt)
}
