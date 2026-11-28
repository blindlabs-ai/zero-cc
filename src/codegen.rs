use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use crate::ir::{BufAccess, CompileError, Domain, Kernel, NodeId, Op, OutputWrite, LANES};
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct RegBudget { pub max_gprs: usize, pub max_vecs: usize }
impl Default for RegBudget { fn default() -> Self { Self { max_gprs: 14, max_vecs: 16 } } }
#[derive(Clone, Debug, PartialEq, Eq)] pub struct ScheduleReport {
    pub order: Vec<NodeId>, pub output_order: Vec<(usize, usize)>,
    pub peak_gprs: usize, pub peak_vecs: usize, pub live_nodes: usize,
}
fn at<T: Copy>(s: &[T], i: usize, d: T) -> T { s.get(i).copied().unwrap_or(d) }
fn sl<T>(v: &[Vec<T>], i: usize) -> &[T] { v.get(i).map_or(&[], Vec::as_slice) }
fn set<T>(s: &mut [T], i: usize, v: T) { if let Some(x) = s.get_mut(i) { *x = v; }}
fn inc(s: &mut [usize], i: usize) { if let Some(x) = s.get_mut(i) { *x = x.saturating_add(1); }}
fn push_uniq(v: &mut Vec<usize>, x: usize) { if !v.contains(&x) { v.push(x); }}
fn is_inline(op: Op<NodeId>) -> bool { matches!(op, Op::MemInit | Op::Const(_) | Op::FConst(_)) }
fn live_op(k: &Kernel, r: &[bool], i: usize) -> bool { at(r, i, false) && !is_inline(at(&k.nodes, i, Op::MemInit)) }
fn calc_su(uniq: &[usize], su: &[usize]) -> usize {
    let mut w: Vec<usize> = uniq.iter().map(|&k| at(su, k, 0)).collect();
    w.sort_unstable_by(|a, b| b.cmp(a));
    w.iter().enumerate().fold(1usize, |acc, (j, &s)| acc.max(s + j))
}
struct SchedGraph {
    reach: Vec<bool>, su: Vec<usize>, deps: Vec<Vec<NodeId>>, consumers: Vec<Vec<NodeId>>,
    rem_uses: Vec<usize>, pending_mem_ops: BTreeMap<(u8, u8), usize>,
}
fn build_sched_graph(kernel: &Kernel) -> SchedGraph {
    let n = kernel.nodes.len();
    let (mut reach, mut stack) = (vec![false; n], kernel.outputs.iter().map(|o| o.val).collect::<Vec<_>>());
    while let Some(u) = stack.pop() {
        if at(&reach, u, true) { continue; }
        set(&mut reach, u, true);
        if let Some((kids, cnt)) = kernel.nodes.get(u).map(|o| o.children()) { stack.extend_from_slice(&kids[..cnt]); }
    }
    let (mut su, mut deps, mut cons, mut ru) = (vec![0; n], vec![Vec::new(); n], vec![Vec::new(); n], vec![0; n]);
    let (mut pm, mut mem_loads) = (BTreeMap::new(), BTreeMap::<(NodeId, u8, u8), Vec<NodeId>>::new());
    for id in 0..n {
        if !live_op(kernel, &reach, id) { continue; }
        let (op, mut uniq) = (at(&kernel.nodes, id, Op::MemInit), Vec::new());
        let (kids, cnt) = op.children();
        for &k in &kids[..cnt] { if !is_inline(at(&kernel.nodes, k, Op::MemInit)) { push_uniq(&mut uniq, k); }}
        if let Op::Load { buf, lane, .. } | Op::Store { buf, lane, .. } = op {
            *pm.entry((buf, lane)).or_insert(0) += 1;
        }
        if let Op::Load { mem, buf, lane } = op { mem_loads.entry((mem, buf, lane)).or_default().push(id); }
        else if let Op::Store { mut mem, buf, lane, .. } = op { loop {
            for &ld in mem_loads.get(&(mem, buf, lane)).map_or(&[][..], Vec::as_slice) { push_uniq(&mut uniq, ld); }
            match kernel.nodes.get(mem).copied() {
                Some(Op::Store { mem: p, buf: sb, lane: sl, .. }) if (sb, sl) != (buf, lane) => mem = p,
                _ => break,
            }
        }}
        let s = calc_su(&uniq, &su);
        set(&mut su, id, s);
        for &k in &uniq {
            if let Some(c) = cons.get_mut(k) { c.push(id); }
            inc(&mut ru, k);
        }
        set(&mut deps, id, uniq);
    }
    for o in &kernel.outputs { if !is_inline(at(&kernel.nodes, o.val, Op::MemInit)) { inc(&mut ru, o.val); }}
    SchedGraph { reach, su, deps, consumers: cons, rem_uses: ru, pending_mem_ops: pm }
}
struct SchedState<'a> {
    kernel: &'a Kernel, base: &'a SchedGraph, rem_uses: Vec<usize>, pending_mem_ops: BTreeMap<(u8, u8), usize>,
    sched: Vec<bool>, out_done: Vec<bool>, live: BTreeSet<NodeId>, output_order: Vec<(usize, usize)>,
    peak_ssa: usize, peak_reuse: usize,
}
impl<'a> SchedState<'a> {
    fn new(kernel: &'a Kernel, base: &'a SchedGraph) -> Self { Self {
        kernel, base, rem_uses: base.rem_uses.clone(), pending_mem_ops: base.pending_mem_ops.clone(),
        sched: vec![false; kernel.nodes.len()], out_done: vec![false; kernel.outputs.len()],
        live: BTreeSet::new(), output_order: Vec::new(), peak_ssa: 0, peak_reuse: 0,
    }}
    fn dec_use(&mut self, id: NodeId) { if let Some(ru) = self.rem_uses.get_mut(id) {
        *ru = ru.saturating_sub(1);
        if *ru == 0 { self.live.remove(&id); }
    }}
    fn step(&mut self, u: NodeId, step_len: usize, ready_upd: Option<(&mut Vec<usize>, &mut Vec<NodeId>)>) {
        set(&mut self.sched, u, true);
        let op_u = at(&self.kernel.nodes, u, Op::MemInit);
        if let Op::Load { buf, lane, .. } | Op::Store { buf, lane, .. } = op_u {
            if let Some(cnt) = self.pending_mem_ops.get_mut(&(buf, lane)) { *cnt = cnt.saturating_sub(1); }}
        let (holds, ds) = (!matches!(op_u, Op::Store { .. }) && at(&self.rem_uses, u, 0) > 0, sl(&self.base.deps, u));
        let dying = ds.iter().filter(|&&d| at(&self.rem_uses, d, 0) == 1 && self.live.contains(&d)).count();
        self.peak_ssa = self.peak_ssa.max(self.live.len() + usize::from(holds));
        self.peak_reuse = self.peak_reuse.max(self.live.len().saturating_sub(dying) + usize::from(holds));
        if holds { self.live.insert(u); }
        for &d in ds { self.dec_use(d); }
        if let Some((rem_deps, ready)) = ready_upd { for &c in sl(&self.base.consumers, u) {
            if let Some(rd) = rem_deps.get_mut(c) {
                *rd = rd.saturating_sub(1);
                if *rd == 0 { ready.push(c); }
            }
        }}
        for (oi, o) in self.kernel.outputs.iter().enumerate() {
            let ov_in = is_inline(at(&self.kernel.nodes, o.val, Op::MemInit));
            let clr = self.pending_mem_ops.get(&(o.buf, o.lane)).copied().unwrap_or(0) == 0;
            if !at(&self.out_done, oi, true) && (ov_in || at(&self.sched, o.val, false)) && clr {
                set(&mut self.out_done, oi, true);
                self.output_order.push((step_len, oi));
                if !ov_in { self.dec_use(o.val); }
            }
        }
    }
    fn finish(mut self, order: Vec<NodeId>) -> (ScheduleReport, usize, usize) {
        let len = order.len();
        for i in 0..self.kernel.outputs.len() { if !at(&self.out_done, i, true) { self.output_order.push((len, i)); }}
        let live_nodes = self.sched.iter().filter(|&&s| s).count();
        (ScheduleReport { order, output_order: self.output_order, peak_gprs: self.kernel.bufs.len() + self.peak_ssa,
            peak_vecs: self.peak_ssa.saturating_mul(2), live_nodes }, self.peak_ssa, self.peak_reuse)
    }
}
fn run_greedy_sched(kernel: &Kernel, base: &SchedGraph, mode: u8) -> (ScheduleReport, usize, usize) {
    let mut rem_deps: Vec<usize> = base.deps.iter().map(Vec::len).collect();
    let mut ready: Vec<NodeId> = (0..kernel.nodes.len())
        .filter(|&i| live_op(kernel, &base.reach, i) && at(&rem_deps, i, 1) == 0).collect();
    let (mut st, mut order) = (SchedState::new(kernel, base), Vec::new());
    while !ready.is_empty() {
        let mut best = (0usize, (isize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX));
        for (pos, &u) in ready.iter().enumerate() {
            let op_u = at(&kernel.nodes, u, Op::MemInit);
            let (is_st, is_ld) = (matches!(op_u, Op::Store { .. }), usize::from(matches!(op_u, Op::Load { .. })));
            let (deps_u, cons_u) = (sl(&base.deps, u), sl(&base.consumers, u));
            let dying = deps_u.iter().filter(|&&d| !matches!(kernel.nodes.get(d).copied(), Some(Op::Store { .. }))
                && at(&st.rem_uses, d, 0) == 1).count() as isize;
            let delta = isize::from(!is_st && at(&st.rem_uses, u, 0) > 0) - dying;
            let hit = |c: &usize| base.deps.get(*c).is_some_and(|ds| ds.iter().any(|d| st.live.contains(d)));
            let (helps, h_now) = (usize::from(!cons_u.iter().any(hit)),
                usize::from(!cons_u.iter().any(|c| at(&rem_deps, *c, 0) == 1 && hit(c))));
            let (su_a, su_d) = (at(&base.su, u, 0), usize::MAX - at(&base.su, u, 0));
            let key = match mode {
                0 => (delta, helps, su_d, u, 0),
                1 => (delta, is_ld, h_now, su_d, u),
                2 => (delta, is_ld, h_now, su_a, u),
                3 => (delta, h_now, is_ld, su_a, u),
                4 => (delta, is_ld, h_now, su_d, usize::MAX - u),
                _ => (delta, h_now, is_ld, su_d, usize::MAX - u),
            };
            if key < best.1 { best = (pos, key); }
        }
        let u = ready.swap_remove(best.0);
        order.push(u);
        st.step(u, order.len(), Some((&mut rem_deps, &mut ready)));
    }
    st.finish(order)
}
fn eval_fixed_order(kernel: &Kernel, base: &SchedGraph, order: Vec<NodeId>) -> (ScheduleReport, usize, usize) {
    let mut st = SchedState::new(kernel, base);
    for (step, &u) in order.iter().enumerate() { st.step(u, step + 1, None); }
    st.finish(order)
}
fn dfs_su_visit(u: NodeId, base: &SchedGraph, visited: &mut [bool], order: &mut Vec<NodeId>, cm: u8) {
    if at(visited, u, true) { return; }
    set(visited, u, true);
    if let Some(ds) = base.deps.get(u) {
        let mut kids = ds.clone();
        match cm {
            0 => kids.sort_by(|&a, &b| base.su.get(b).cmp(&base.su.get(a)).then_with(|| a.cmp(&b))),
            1 => kids.sort_by(|&a, &b| base.su.get(a).cmp(&base.su.get(b)).then_with(|| a.cmp(&b))),
            2 => kids.sort_unstable(),
            _ => kids.sort_by(|a, b| b.cmp(a)),
        }
        for k in kids { dfs_su_visit(k, base, visited, order, cm); }
    }
    order.push(u);
}
fn run_su_dfs_sched(kernel: &Kernel, base: &SchedGraph, rev_outs: bool, cm: u8) -> (ScheduleReport, usize, usize) {
    let (n, mut visited, mut order) = (kernel.nodes.len(), vec![false; kernel.nodes.len()], Vec::new());
    let mut seeds: Vec<NodeId> = kernel.outputs.iter().map(|o| o.val).collect();
    if rev_outs { seeds.reverse(); }
    seeds.extend(0..n);
    for id in seeds { if live_op(kernel, &base.reach, id) { dfs_su_visit(id, base, &mut visited, &mut order, cm); }}
    eval_fixed_order(kernel, base, order)
}
pub fn schedule_and_verify(kernel: &Kernel, budget: RegBudget) -> Result<ScheduleReport, CompileError> {
    kernel.validate()?;
    let g = build_sched_graph(kernel);
    let (mut best_rep, ssa0, rssa0) = run_greedy_sched(kernel, &g, 0);
    let mut best_key = (rssa0, ssa0);
    let mut consider = |(rep, ssa, rssa): (ScheduleReport, usize, usize)| {
        if (rssa, ssa) < best_key { (best_rep, best_key) = (rep, (rssa, ssa)); }
    };
    for mode in 1..6u8 { consider(run_greedy_sched(kernel, &g, mode)); }
    for rev in [false, true] { for cm in 0..4u8 { consider(run_su_dfs_sched(kernel, &g, rev, cm)); }}
    consider(eval_fixed_order(kernel, &g, (0..kernel.nodes.len()).filter(|&i| live_op(kernel, &g.reach, i)).collect()));
    if best_rep.peak_gprs > budget.max_gprs || best_rep.peak_vecs > budget.max_vecs {
        return Err(CompileError::RegisterSpill { peak_gprs: best_rep.peak_gprs, max_gprs: budget.max_gprs,
            peak_vecs: best_rep.peak_vecs, max_vecs: budget.max_vecs });
    }
    Ok(best_rep)
}
struct RematCtx<'a> { nodes: &'a [Op<NodeId>], doms: &'a [Domain], map: &'a [NodeId], tsz: &'a [u8] }
fn clone_ro_subgraph(ctx: &RematCtx<'_>, out_n: &mut Vec<Op<NodeId>>, out_d: &mut Vec<Domain>,
    kid: NodeId, max_n: u8, memo: &mut Vec<(NodeId, NodeId)>) -> NodeId {
    if let Some(&(_, nid)) = memo.iter().find(|&&(k, _)| k == kid) { return nid; }
    let sz = at(ctx.tsz, kid, u8::MAX);
    if sz == 0 || sz > max_n { return at(ctx.map, kid, 0); }
    let op = at(ctx.nodes, kid, Op::MemInit).map_ids(|c| match at(ctx.tsz, c, u8::MAX) {
        csz if csz > 0 && csz < sz => clone_ro_subgraph(ctx, out_n, out_d, c, max_n, memo),
        _ => at(ctx.map, c, 0),
    });
    let nid = out_n.len();
    out_n.push(op);
    out_d.push(at(ctx.doms, kid, Domain::top()));
    memo.push((kid, nid));
    nid
}
pub fn remat_readonly_loads(kernel: &Kernel, max_nodes: u8) -> Kernel {
    let (n, mut uses, mut tsz) = (kernel.nodes.len(), vec![0; kernel.nodes.len()], vec![u8::MAX; kernel.nodes.len()]);
    for &op in &kernel.nodes { let (kids, cnt) = op.children();
        for &k in &kids[..cnt] { inc(&mut uses, k); }}
    for o in &kernel.outputs { inc(&mut uses, o.val); }
    for id in 0..n {
        let sz = match at(&kernel.nodes, id, Op::MemInit) {
            Op::Const(_) => 0u8,
            Op::Load { mem, buf, .. } => if kernel.nodes.get(mem) == Some(&Op::MemInit)
                && kernel.bufs.get(buf as usize).is_some_and(|b| matches!(b.access, BufAccess::ReadOnly)) { 1 }
                else { u8::MAX },
            Op::MemInit | Op::Store { .. } | Op::LoadLut { .. } | Op::BoundsCheck { .. } => u8::MAX,
            op => {
                let (kids, cnt) = op.children();
                kids[..cnt].iter().try_fold(1u16, |acc, &k| {
                    let ks = at(&tsz, k, u8::MAX);
                    (ks != u8::MAX).then(|| acc.saturating_add(u16::from(ks)))
                }).filter(|&s| s < u16::from(u8::MAX)).map_or(u8::MAX, |s| s as u8)
            }
        };
        set(&mut tsz, id, sz);
    }
    let can_remat = |k: NodeId| matches!(at(&tsz, k, u8::MAX), sz if sz > 0 && sz <= max_nodes
        && at(&uses, k, 0) >= if sz <= 2 { 2 } else { 3 });
    let (mut out_n, mut out_d) = (Vec::with_capacity(n * 2), Vec::with_capacity(n * 2));
    let (mut map, mut node_clones) = (vec![0usize; n], vec![Vec::new(); n]);
    for id in 0..n {
        let (op, ctx) = (at(&kernel.nodes, id, Op::MemInit),
            RematCtx { nodes: &kernel.nodes, doms: &kernel.proven_domains, map: &map, tsz: &tsz });
        let mut memo = if let Op::Select { cond, .. } = op {
            node_clones.get(cond).cloned().unwrap_or_default()
        } else { Vec::new() };
        let mapped = op.map_ids(|k| if can_remat(k) {
            clone_ro_subgraph(&ctx, &mut out_n, &mut out_d, k, max_nodes, &mut memo)
        } else { at(&map, k, 0) });
        let nid = out_n.len();
        out_n.push(mapped);
        out_d.push(at(&kernel.proven_domains, id, Domain::top()));
        set(&mut map, id, nid);
        set(&mut node_clones, id, memo);
    }
    let ctx = RematCtx { nodes: &kernel.nodes, doms: &kernel.proven_domains, map: &map, tsz: &tsz };
    let outputs = kernel.outputs.iter().map(|o| OutputWrite { buf: o.buf, lane: o.lane,
        val: if can_remat(o.val) { clone_ro_subgraph(&ctx, &mut out_n, &mut out_d, o.val, max_nodes, &mut Vec::new()) }
            else { at(&map, o.val, 0) } }).collect();
    Kernel { name: kernel.name.clone(), bufs: kernel.bufs.clone(), nodes: out_n, outputs, proven_domains: out_d }
}
fn ref_expr(kernel: &Kernel, id: NodeId) -> String { match at(&kernel.nodes, id, Op::MemInit) {
    Op::Const(c) => format!("{c}_i64"),
    Op::FConst(b) => format!("{b}_u32 as i64"),
    Op::MemInit => "0_i64".into(),
    _ => format!("v{id}"),
}}
fn fref(kernel: &Kernel, id: NodeId) -> String { match at(&kernel.nodes, id, Op::MemInit) {
    Op::FConst(b) => format!("f32::from_bits({b}_u32)"),
    _ => format!("f32::from_bits({} as u32)", ref_expr(kernel, id)),
}}
fn emit_op_rhs(kernel: &Kernel, doms: &[Domain], id: NodeId) -> String {
    let r = |x: NodeId| ref_expr(kernel, x);
    let bname = |buf: u8| kernel.bufs.get(usize::from(buf)).map_or("", |b| b.name.as_str());
    match at(&kernel.nodes, id, Op::MemInit) {
        Op::MemInit | Op::Const(_) => ref_expr(kernel, id),
        Op::Load { buf, lane, .. } => format!("{}[{lane}]", bname(buf)),
        Op::Store { val, .. } => r(val),
        Op::LoadLut { buf, idx } => {
            let ix = r(idx);
            let Some(b) = kernel.bufs.get(usize::from(buf)) else { return format!("0_i64 /* {ix} */") };
            if let BufAccess::LookupTable(n) = b.access { if n.is_power_of_two()
                && doms.get(idx).is_some_and(|d| !d.is_empty() && d.lo >= 0 && d.hi < n as i64) {
                return format!("{}[({ix} as usize) & {}]", b.name, n - 1); }}
            format!("{}[{ix} as usize]", b.name)
        }
        Op::BoundsCheck { idx, limit } => format!("{{ assert!({0} >= 0 && {0} < {limit}); {0} }}", r(idx)),
        Op::Add(a, b) => format!("{}.wrapping_add({})", r(a), r(b)),
        Op::Sub(a, b) => format!("{}.wrapping_sub({})", r(a), r(b)),
        Op::Mul(a, b) => if doms.get(a).is_some_and(|d| d.fits_u32()) && doms.get(b).is_some_and(|d| d.fits_u32()) {
            format!("(({} as u32 as u64).wrapping_mul({} as u32 as u64)) as i64", r(a), r(b))
        } else { format!("{}.wrapping_mul({})", r(a), r(b)) },
        Op::Div(a, b) => format!("{} / {}", r(a), r(b)),
        Op::Rem(a, b) => format!("{} % {}", r(a), r(b)),
        Op::Shl(a, s) => format!("{}.wrapping_shl({s})", r(a)),
        Op::Shr(a, s) => if doms.get(a).is_some_and(|d| d.is_nonneg()) { format!("(({} as u64) >> {s}) as i64", r(a)) }
            else { format!("{} >> {s}", r(a)) },
        Op::Ushr(a, s) => format!("(({} as u64) >> {s}) as i64", r(a)),
        Op::ShlVar(a, b) => format!("(({} as u64).wrapping_shl(({} as u32) & 63)) as i64", r(a), r(b)),
        Op::ShrVarU(a, b) => format!("(({} as u64).wrapping_shr(({} as u32) & 63)) as i64", r(a), r(b)),
        Op::BitAnd(a, b) => format!("{} & {}", r(a), r(b)),
        Op::BitOr(a, b) => format!("{} | {}", r(a), r(b)),
        Op::BitXor(a, b) => format!("{} ^ {}", r(a), r(b)),
        Op::Popcnt(a) => format!("(({} as u64).count_ones()) as i64", r(a)),
        Op::Tzcnt(a) => format!("(({} as u64).trailing_zeros()) as i64", r(a)),
        Op::Cmp(k, a, b) => if k.is_unsigned() { format!("(({} as u64) {} ({} as u64)) as i64", r(a), k.sym(), r(b)) }
            else { format!("({} {} {}) as i64", r(a), k.sym(), r(b)) },
        Op::Select { cond: c, t_val: t, f_val: f } => format!("if {} != 0_i64 {{ {} }} else {{ {} }}",
            r(c), r(t), r(f)),
        Op::FConst(b) => format!("{b}_u32 as i64"),
        Op::FAdd(a, b) => format!("({} + {}).to_bits() as i64", fref(kernel, a), fref(kernel, b)),
        Op::FSub(a, b) => format!("({} - {}).to_bits() as i64", fref(kernel, a), fref(kernel, b)),
        Op::FMul(a, b) => format!("({} * {}).to_bits() as i64", fref(kernel, a), fref(kernel, b)),
        Op::FDiv(a, b) => format!("({} / {}).to_bits() as i64", fref(kernel, a), fref(kernel, b)),
        Op::FFma(a, b, c) => format!("{}.mul_add({}, {}).to_bits() as i64",
            fref(kernel, a), fref(kernel, b), fref(kernel, c)),
        Op::FMax(a, b) => format!("{{ let (a, b) = ({}, {}); (if a > b {{ a }} else {{ b }}).to_bits() as i64 }}",
            fref(kernel, a), fref(kernel, b)),
        Op::FMin(a, b) => format!("{{ let (a, b) = ({}, {}); (if a < b {{ a }} else {{ b }}).to_bits() as i64 }}",
            fref(kernel, a), fref(kernel, b)),
        Op::FSqrt(a) => format!("{}.sqrt().to_bits() as i64", fref(kernel, a)),
        Op::FNeg(a) => format!("(-{}).to_bits() as i64", fref(kernel, a)),
        Op::FAbs(a) => format!("{}.abs().to_bits() as i64", fref(kernel, a)),
        Op::FCmp(k, a, b) => format!("({} {} {}) as i64", fref(kernel, a), k.sym(), fref(kernel, b)),
    }
}
fn drain_outputs(s: &mut String, kernel: &Kernel, sched: &ScheduleReport, oc: &mut usize, step: Option<usize>) {
    while let Some(&(st, oi)) = sched.output_order.get(*oc) {
        if step.is_some_and(|want| st != want) { break; }
        if let Some((o, b)) = kernel.outputs.get(oi).and_then(|o| Some((o, kernel.bufs.get(usize::from(o.buf))?))) {
            let _ = writeln!(s, "    {}[{}] = {};", b.name, o.lane, ref_expr(kernel, o.val));
        }
        *oc += 1;
    }
}
pub fn emit_rust(kernel: &Kernel, sched: &ScheduleReport) -> Result<String, CompileError> {
    kernel.validate()?;
    let (doms, mut s, mut oc) = (kernel.infer_domains(), String::with_capacity(2048), 0usize);
    let _ = writeln!(s, "// Generated by zero-cc | live_nodes={} peak_gprs={} peak_vecs={}",
        sched.live_nodes, sched.peak_gprs, sched.peak_vecs);
    let (mut params, mut bparams, mut args, mut lbufs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in &kernel.bufs {
        let d = b.domain;
        if d != Domain::top() && !d.is_empty() {
            let m = if d.modulus > 1 { format!(" mod {} == {}", d.modulus, d.rem) } else { String::new() };
            let _ = writeln!(s, "/// @domain {}: {}..={}{m}", b.name, d.lo, d.hi);
        }
        if let BufAccess::LookupTable(len) = b.access {
            let p = format!("{}: &[i64; {len}]", b.name);
            params.push(p.clone());
            bparams.push(p);
            args.push(b.name.clone());
        } else {
            let m = if matches!(b.access, BufAccess::MutNoAlias) { "mut " } else { "" };
            params.push(format!("{}: &{m}[i64; {LANES}]", b.name));
            bparams.push(format!("{}: &{m}[[i64; {LANES}]]", b.name));
            args.push(format!("&{m}{}[blk]", b.name));
            lbufs.push(b.name.as_str());
        }
    }
    let _ = writeln!(s, "#[inline(always)]\npub fn {}({}) {{", kernel.name, params.join(", "));
    for (step, &id) in sched.order.iter().enumerate() {
        drain_outputs(&mut s, kernel, sched, &mut oc, Some(step));
        if let Some(Op::Store { buf, lane, val, .. }) = kernel.nodes.get(id).copied() {
            if let Some(b) = kernel.bufs.get(usize::from(buf)) {
                let _ = writeln!(s, "    {}[{lane}] = {};", b.name, ref_expr(kernel, val)); }}
        else { let _ = writeln!(s, "    let v{id}: i64 = {};", emit_op_rhs(kernel, &doms, id)); }
    }
    drain_outputs(&mut s, kernel, sched, &mut oc, None);
    let _ = writeln!(s, "}}\n\n#[inline(never)]\npub fn {}_batch({}) {{", kernel.name, bparams.join(", "));
    if let Some(first) = lbufs.first() {
        let len_expr = lbufs[1..].iter().fold(format!("{first}.len()"), |a, &lb| format!("{a}.min({lb}.len())"));
        let _ = writeln!(s, "    let blocks = {len_expr};\n    for blk in 0..blocks {{\n        {}({});\n    }}",
            kernel.name, args.join(", "));
    }
    let _ = writeln!(s, "}}");
    Ok(s)
}
