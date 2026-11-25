use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{BufAccess, CompileError, Domain, Kernel, NodeId, Op, LANES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegBudget { pub max_gprs: usize, pub max_vecs: usize }
impl Default for RegBudget { fn default() -> Self { Self { max_gprs: 14, max_vecs: 16 } } }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleReport {
    pub order: Vec<NodeId>, pub output_order: Vec<(usize, usize)>,
    pub peak_gprs: usize, pub peak_vecs: usize, pub live_nodes: usize,
}

fn is_inline(op: Op<NodeId>) -> bool { matches!(op, Op::MemInit | Op::Const(_)) }

fn calc_su(uniq: &[usize], su: &[usize]) -> usize {
    let mut w: Vec<usize> = uniq.iter().map(|&k| su.get(k).copied().unwrap_or(0)).collect();
    w.sort_unstable_by(|a, b| b.cmp(a));
    w.iter().enumerate().fold(1usize, |acc, (j, &s)| acc.max(s + j))
}

struct SchedGraph {
    reach: Vec<bool>, su: Vec<usize>, deps: Vec<Vec<NodeId>>, consumers: Vec<Vec<NodeId>>,
    rem_uses: Vec<usize>, pending_mem_ops: BTreeMap<(u8, u8), usize>,
}

fn build_sched_graph(kernel: &Kernel) -> SchedGraph {
    let (n, mut reach, mut stack) = (kernel.nodes.len(), vec![false; kernel.nodes.len()], kernel.outputs.iter().map(|o| o.val).collect::<Vec<_>>());
    while let Some(u) = stack.pop() {
        if !reach.get(u).copied().unwrap_or(true) {
            if let Some(r) = reach.get_mut(u) { *r = true; }
            if let Some(&op) = kernel.nodes.get(u) { let (kids, cnt) = op.children(); stack.extend_from_slice(&kids[..cnt]); }
        }
    }
    let (mut su, mut deps, mut consumers, mut rem_uses, mut pending_mem_ops) = (vec![0; n], vec![Vec::new(); n], vec![Vec::new(); n], vec![0; n], BTreeMap::new());
    let mut mem_loads = BTreeMap::<(NodeId, u8, u8), Vec<NodeId>>::new();
    for id in 0..n {
        let Some(&op) = kernel.nodes.get(id) else { continue; };
        if !reach.get(id).copied().unwrap_or(false) || is_inline(op) { continue; }
        let ((kids, cnt), mut uniq) = (op.children(), Vec::new());
        for &k in &kids[..cnt] { if kernel.nodes.get(k).is_some_and(|&kop| !is_inline(kop)) && !uniq.contains(&k) { uniq.push(k); } }
        if let Op::Load { mem, buf, lane } = op {
            *pending_mem_ops.entry((buf, lane)).or_insert(0) += 1;
            mem_loads.entry((mem, buf, lane)).or_default().push(id);
        } else if let Op::Store { mem, buf, lane, .. } = op {
            *pending_mem_ops.entry((buf, lane)).or_insert(0) += 1;
            let mut m = mem;
            loop {
                if let Some(lds) = mem_loads.get(&(m, buf, lane)) {
                    for &ld in lds { if !uniq.contains(&ld) { uniq.push(ld); } }
                }
                match kernel.nodes.get(m).copied() {
                    Some(Op::Store { mem: prev, buf: sb, lane: sl, .. }) if (sb, sl) != (buf, lane) => m = prev,
                    _ => break,
                }
            }
        }
        let s_val = calc_su(&uniq, &su);
        if let Some(s) = su.get_mut(id) { *s = s_val; }
        for &k in &uniq {
            if let Some(c) = consumers.get_mut(k) { c.push(id); }
            if let Some(ru) = rem_uses.get_mut(k) { *ru += 1; }
        }
        if let Some(d) = deps.get_mut(id) { *d = uniq; }
    }
    for o in &kernel.outputs {
        if kernel.nodes.get(o.val).is_some_and(|&op| !is_inline(op)) {
            if let Some(ru) = rem_uses.get_mut(o.val) { *ru += 1; }
        }
    }
    SchedGraph { reach, su, deps, consumers, rem_uses, pending_mem_ops }
}

fn run_greedy_sched(kernel: &Kernel, base: &SchedGraph, mode: u8) -> (ScheduleReport, usize, usize) {
    let n = kernel.nodes.len();
    let (mut rem_uses, mut pending_mem_ops) = (base.rem_uses.clone(), base.pending_mem_ops.clone());
    let mut rem_deps: Vec<usize> = base.deps.iter().map(Vec::len).collect();
    let mut ready: Vec<NodeId> = (0..n).filter(|&i| base.reach.get(i).copied().unwrap_or(false) && kernel.nodes.get(i).is_some_and(|&op| !is_inline(op)) && rem_deps.get(i).copied().unwrap_or(1) == 0).collect();
    let (mut sched, mut out_done) = (vec![false; n], vec![false; kernel.outputs.len()]);
    let (mut live, mut order, mut output_order, mut peak_ssa, mut peak_reuse) = (BTreeSet::new(), Vec::new(), Vec::new(), 0usize, 0usize);
    while !ready.is_empty() {
        let mut best = (0usize, (isize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX));
        for (pos, &u) in ready.iter().enumerate() {
            let op_u = kernel.nodes.get(u).copied().unwrap_or(Op::MemInit);
            let is_store = matches!(op_u, Op::Store { .. });
            let is_load = usize::from(matches!(op_u, Op::Load { .. }));
            let deps_u = base.deps.get(u).map(Vec::as_slice).unwrap_or(&[]);
            let cons_u = base.consumers.get(u).map(Vec::as_slice).unwrap_or(&[]);
            let delta = isize::from(!is_store && rem_uses.get(u).copied().unwrap_or(0) > 0)
                - deps_u.iter().filter(|&&d| !matches!(kernel.nodes.get(d).copied(), Some(Op::Store { .. })) && rem_uses.get(d).copied().unwrap_or(0) == 1).count() as isize;
            let helps = usize::from(!cons_u.iter().any(|&c| base.deps.get(c).is_some_and(|ds| ds.iter().any(|d| live.contains(d)))));
            let helps_now = usize::from(!cons_u.iter().any(|&c| rem_deps.get(c).copied().unwrap_or(0) == 1 && base.deps.get(c).is_some_and(|ds| ds.iter().any(|d| live.contains(d)))));
            let su_asc = base.su.get(u).copied().unwrap_or(0);
            let su_desc = usize::MAX - su_asc;
            let key = match mode {
                0 => (delta, helps, su_desc, u, 0, 0),
                1 => (delta, is_load, helps_now, su_desc, u, 0),
                2 => (delta, is_load, helps_now, su_asc, u, 0),
                3 => (delta, helps_now, is_load, su_asc, u, 0),
                4 => (delta, is_load, helps_now, su_desc, usize::MAX - u, 0),
                _ => (delta, helps_now, is_load, su_desc, usize::MAX - u, 0),
            };
            if key < best.1 { best = (pos, key); }
        }
        let u = ready.swap_remove(best.0);
        if let Some(s) = sched.get_mut(u) { *s = true; }
        order.push(u);
        let op_u = kernel.nodes.get(u).copied().unwrap_or(Op::MemInit);
        if let Op::Load { buf, lane, .. } | Op::Store { buf, lane, .. } = op_u {
            if let Some(cnt) = pending_mem_ops.get_mut(&(buf, lane)) { *cnt = cnt.saturating_sub(1); }
        }
        let holds_reg = !matches!(op_u, Op::Store { .. }) && rem_uses.get(u).copied().unwrap_or(0) > 0;
        let dying = count_dying(base, &rem_uses, &live, u);
        peak_ssa = peak_ssa.max(live.len() + usize::from(holds_reg));
        peak_reuse = peak_reuse.max(live.len().saturating_sub(dying) + usize::from(holds_reg));
        if holds_reg { live.insert(u); }
        if let Some(ds) = base.deps.get(u) {
            for &d in ds {
                if let Some(ru) = rem_uses.get_mut(d) { *ru = ru.saturating_sub(1); if *ru == 0 { live.remove(&d); } }
            }
        }
        if let Some(cs) = base.consumers.get(u) {
            for &c in cs {
                if let Some(rd) = rem_deps.get_mut(c) { *rd = rd.saturating_sub(1); if *rd == 0 { ready.push(c); } }
            }
        }
        for (oi, o) in kernel.outputs.iter().enumerate() {
            let ov_inline = kernel.nodes.get(o.val).is_some_and(|&op| is_inline(op));
            let ov_sched = sched.get(o.val).copied().unwrap_or(false);
            if !out_done.get(oi).copied().unwrap_or(true) && (ov_inline || ov_sched) && pending_mem_ops.get(&(o.buf, o.lane)).copied().unwrap_or(0) == 0 {
                if let Some(done) = out_done.get_mut(oi) { *done = true; }
                output_order.push((order.len(), oi));
                if !ov_inline {
                    if let Some(ru) = rem_uses.get_mut(o.val) { *ru = ru.saturating_sub(1); if *ru == 0 { live.remove(&o.val); } }
                }
            }
        }
    }
    for (oi, _) in kernel.outputs.iter().enumerate() { if !out_done.get(oi).copied().unwrap_or(true) { output_order.push((order.len(), oi)); } }
    let (peak_gprs, peak_vecs) = (kernel.bufs.len() + peak_ssa, peak_ssa.saturating_mul(2));
    (ScheduleReport { order, output_order, peak_gprs, peak_vecs, live_nodes: sched.iter().filter(|&&s| s).count() }, peak_ssa, peak_reuse)
}

fn dfs_su_visit(u: NodeId, base: &SchedGraph, visited: &mut [bool], order: &mut Vec<NodeId>, child_mode: u8) {
    if visited.get(u).copied().unwrap_or(true) { return; }
    if let Some(v) = visited.get_mut(u) { *v = true; }
    if let Some(ds) = base.deps.get(u) {
        let mut kids = ds.clone();
        match child_mode {
            0 => kids.sort_by(|&a, &b| base.su.get(b).cmp(&base.su.get(a)).then_with(|| a.cmp(&b))),
            1 => kids.sort_by(|&a, &b| base.su.get(a).cmp(&base.su.get(b)).then_with(|| a.cmp(&b))),
            2 => kids.sort_unstable(),
            _ => kids.sort_by(|a, b| b.cmp(a)),
        }
        for k in kids { dfs_su_visit(k, base, visited, order, child_mode); }
    }
    order.push(u);
}

fn count_dying(base: &SchedGraph, rem_uses: &[usize], live: &BTreeSet<NodeId>, u: NodeId) -> usize {
    let mut dying = 0usize;
    if let Some(ds) = base.deps.get(u) {
        for &d in ds {
            if rem_uses.get(d).copied().unwrap_or(0) == 1 && live.contains(&d) {
                dying += 1;
            }
        }
    }
    dying
}

fn eval_fixed_order(kernel: &Kernel, base: &SchedGraph, order: Vec<NodeId>) -> (ScheduleReport, usize, usize) {
    let n = kernel.nodes.len();
    let (mut rem_uses, mut pending_mem_ops) = (base.rem_uses.clone(), base.pending_mem_ops.clone());
    let (mut sched, mut out_done) = (vec![false; n], vec![false; kernel.outputs.len()]);
    let (mut live, mut output_order, mut peak_ssa, mut peak_reuse) = (BTreeSet::new(), Vec::new(), 0usize, 0usize);
    for (step, &u) in order.iter().enumerate() {
        if let Some(s) = sched.get_mut(u) { *s = true; }
        let op_u = kernel.nodes.get(u).copied().unwrap_or(Op::MemInit);
        if let Op::Load { buf, lane, .. } | Op::Store { buf, lane, .. } = op_u {
            if let Some(cnt) = pending_mem_ops.get_mut(&(buf, lane)) { *cnt = cnt.saturating_sub(1); }
        }
        let holds_reg = !matches!(op_u, Op::Store { .. }) && rem_uses.get(u).copied().unwrap_or(0) > 0;
        let dying = count_dying(base, &rem_uses, &live, u);
        peak_ssa = peak_ssa.max(live.len() + usize::from(holds_reg));
        peak_reuse = peak_reuse.max(live.len().saturating_sub(dying) + usize::from(holds_reg));
        if holds_reg { live.insert(u); }
        if let Some(ds) = base.deps.get(u) {
            for &d in ds {
                if let Some(ru) = rem_uses.get_mut(d) { *ru = ru.saturating_sub(1); if *ru == 0 { live.remove(&d); } }
            }
        }
        for (oi, o) in kernel.outputs.iter().enumerate() {
            let ov_inline = kernel.nodes.get(o.val).is_some_and(|&op| is_inline(op));
            let ov_sched = sched.get(o.val).copied().unwrap_or(false);
            if !out_done.get(oi).copied().unwrap_or(true) && (ov_inline || ov_sched) && pending_mem_ops.get(&(o.buf, o.lane)).copied().unwrap_or(0) == 0 {
                if let Some(done) = out_done.get_mut(oi) { *done = true; }
                output_order.push((step + 1, oi));
                if !ov_inline {
                    if let Some(ru) = rem_uses.get_mut(o.val) { *ru = ru.saturating_sub(1); if *ru == 0 { live.remove(&o.val); } }
                }
            }
        }
    }
    for (oi, _) in kernel.outputs.iter().enumerate() { if !out_done.get(oi).copied().unwrap_or(true) { output_order.push((order.len(), oi)); } }
    let (peak_gprs, peak_vecs) = (kernel.bufs.len() + peak_ssa, peak_ssa.saturating_mul(2));
    (ScheduleReport { order, output_order, peak_gprs, peak_vecs, live_nodes: sched.iter().filter(|&&s| s).count() }, peak_ssa, peak_reuse)
}

fn run_su_dfs_sched(kernel: &Kernel, base: &SchedGraph, rev_outs: bool, child_mode: u8) -> (ScheduleReport, usize, usize) {
    let n = kernel.nodes.len();
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut out_idxs: Vec<usize> = (0..kernel.outputs.len()).collect();
    if rev_outs { out_idxs.reverse(); }
    for &oi in &out_idxs {
        if let Some(o) = kernel.outputs.get(oi) {
            if base.reach.get(o.val).copied().unwrap_or(false) && kernel.nodes.get(o.val).is_some_and(|&op| !is_inline(op)) {
                dfs_su_visit(o.val, base, &mut visited, &mut order, child_mode);
            }
        }
    }
    for id in 0..n {
        if base.reach.get(id).copied().unwrap_or(false) && kernel.nodes.get(id).is_some_and(|&op| !is_inline(op)) {
            dfs_su_visit(id, base, &mut visited, &mut order, child_mode);
        }
    }
    eval_fixed_order(kernel, base, order)
}

/// Minimum-live-register Sethi-Ullman DAG scheduler with hard zero-spill cliff guard.
pub fn schedule_and_verify(kernel: &Kernel, budget: RegBudget) -> Result<ScheduleReport, CompileError> {
    kernel.validate()?;
    let g = build_sched_graph(kernel);
    let (mut best_rep, mut best_key) = {
        let (rep, ssa, rssa) = run_greedy_sched(kernel, &g, 0);
        (rep, (rssa, ssa))
    };
    for mode in 1..6u8 {
        let (rep, ssa, rssa) = run_greedy_sched(kernel, &g, mode);
        if (rssa, ssa) < best_key { best_rep = rep; best_key = (rssa, ssa); }
    }
    for rev in [false, true] {
        for cm in 0..4u8 {
            let (rep, ssa, rssa) = run_su_dfs_sched(kernel, &g, rev, cm);
            if (rssa, ssa) < best_key { best_rep = rep; best_key = (rssa, ssa); }
        }
    }
    let nat: Vec<NodeId> = (0..kernel.nodes.len())
        .filter(|&id| g.reach.get(id).copied().unwrap_or(false) && kernel.nodes.get(id).is_some_and(|&op| !is_inline(op)))
        .collect();
    let (rep, ssa, rssa) = eval_fixed_order(kernel, &g, nat);
    if (rssa, ssa) < best_key { best_rep = rep; }
    if best_rep.peak_gprs > budget.max_gprs || best_rep.peak_vecs > budget.max_vecs {
        return Err(CompileError::RegisterSpill {
            peak_gprs: best_rep.peak_gprs, max_gprs: budget.max_gprs,
            peak_vecs: best_rep.peak_vecs, max_vecs: budget.max_vecs,
        });
    }
    Ok(best_rep)
}

struct RematCtx<'a> {
    src_nodes: &'a [Op<NodeId>],
    src_doms: &'a [Domain],
    map: &'a [NodeId],
    tree_size: &'a [u8],
}

fn clone_ro_subgraph(
    ctx: &RematCtx<'_>,
    out_nodes: &mut Vec<Op<NodeId>>,
    out_doms: &mut Vec<Domain>,
    kid: NodeId,
    max_nodes: u8,
    memo: &mut Vec<(NodeId, NodeId)>,
) -> NodeId {
    if let Some(&(_, nid)) = memo.iter().find(|&&(k, _)| k == kid) {
        return nid;
    }
    let sz = ctx.tree_size.get(kid).copied().unwrap_or(u8::MAX);
    if sz == 0 || sz > max_nodes {
        return ctx.map.get(kid).copied().unwrap_or(0);
    }
    let op = ctx.src_nodes.get(kid).copied().unwrap_or(Op::MemInit);
    let dom = ctx.src_doms.get(kid).copied().unwrap_or(Domain::top());
    let new_op = op.map_ids(|c| {
        let csz = ctx.tree_size.get(c).copied().unwrap_or(u8::MAX);
        if csz > 0 && csz < sz {
            clone_ro_subgraph(ctx, out_nodes, out_doms, c, max_nodes, memo)
        } else {
            ctx.map.get(c).copied().unwrap_or(0)
        }
    });
    let nid = out_nodes.len();
    out_nodes.push(new_op);
    out_doms.push(dom);
    memo.push((kid, nid));
    nid
}

/// Classical read-only `Op::Load` and bounded pure-subtree rematerialization to cut
/// long live ranges across unrelated DAG subgraphs without emitting stack spills.
pub fn remat_readonly_loads(kernel: &Kernel, max_nodes: u8) -> Kernel {
    let n = kernel.nodes.len();
    let mut uses = vec![0usize; n];
    for &op in &kernel.nodes {
        let (kids, cnt) = op.children();
        for &k in &kids[..cnt] {
            if let Some(u) = uses.get_mut(k) { *u += 1; }
        }
    }
    for o in &kernel.outputs {
        if let Some(u) = uses.get_mut(o.val) { *u += 1; }
    }
    let mut tree_size = vec![u8::MAX; n];
    for id in 0..n {
        let op = kernel.nodes.get(id).copied().unwrap_or(Op::MemInit);
        let sz = match op {
            Op::Const(_) => 0u8,
            Op::Load { mem, buf, .. } => {
                let mem_init = kernel.nodes.get(mem).copied() == Some(Op::MemInit);
                let is_ro = kernel.bufs.get(usize::from(buf)).is_some_and(|b| matches!(b.access, BufAccess::ReadOnly));
                if mem_init && is_ro { 1u8 } else { u8::MAX }
            }
            Op::MemInit | Op::Store { .. } | Op::LoadLut { .. } | Op::BoundsCheck { .. } => u8::MAX,
            _ => {
                let (kids, cnt) = op.children();
                let mut sum = 1u16;
                let mut ok = true;
                for &k in &kids[..cnt] {
                    let ks = tree_size.get(k).copied().unwrap_or(u8::MAX);
                    if ks == u8::MAX { ok = false; break; }
                    sum = sum.saturating_add(u16::from(ks));
                }
                if ok && sum < u16::from(u8::MAX) { sum as u8 } else { u8::MAX }
            }
        };
        if let Some(slot) = tree_size.get_mut(id) { *slot = sz; }
    }
    let can_remat = |k: NodeId| {
        let sz = tree_size.get(k).copied().unwrap_or(u8::MAX);
        let min_u = if sz <= 2 { 2 } else { 3 };
        uses.get(k).copied().unwrap_or(0) >= min_u && sz > 0 && sz <= max_nodes
    };
    let mut out_nodes = Vec::with_capacity(n.saturating_mul(2));
    let mut out_doms = Vec::with_capacity(n.saturating_mul(2));
    let mut map = vec![0usize; n];
    let mut node_clones: Vec<Vec<(NodeId, NodeId)>> = vec![Vec::new(); n];
    for id in 0..n {
        let op = kernel.nodes.get(id).copied().unwrap_or(Op::MemInit);
        let ctx = RematCtx {
            src_nodes: &kernel.nodes,
            src_doms: &kernel.proven_domains,
            map: &map,
            tree_size: &tree_size,
        };
        let mut memo = if let Op::Select { cond, .. } = op {
            node_clones.get(cond).cloned().unwrap_or_default()
        } else {
            Vec::new()
        };
        let mapped = op.map_ids(|k| {
            if can_remat(k) {
                clone_ro_subgraph(&ctx, &mut out_nodes, &mut out_doms, k, max_nodes, &mut memo)
            } else {
                map.get(k).copied().unwrap_or(0)
            }
        });
        let nid = out_nodes.len();
        out_nodes.push(mapped);
        out_doms.push(kernel.proven_domains.get(id).copied().unwrap_or(Domain::top()));
        if let Some(m) = map.get_mut(id) { *m = nid; }
        if let Some(nc) = node_clones.get_mut(id) { *nc = memo; }
    }
    let ctx = RematCtx {
        src_nodes: &kernel.nodes,
        src_doms: &kernel.proven_domains,
        map: &map,
        tree_size: &tree_size,
    };
    let outputs = kernel.outputs.iter().map(|o| {
        let mut memo = Vec::new();
        let val = if can_remat(o.val) {
            clone_ro_subgraph(&ctx, &mut out_nodes, &mut out_doms, o.val, max_nodes, &mut memo)
        } else {
            map.get(o.val).copied().unwrap_or(0)
        };
        crate::ir::OutputWrite { buf: o.buf, lane: o.lane, val }
    }).collect();
    Kernel { name: kernel.name.clone(), bufs: kernel.bufs.clone(), nodes: out_nodes, outputs, proven_domains: out_doms }
}

fn ref_expr(kernel: &Kernel, id: NodeId) -> String {
    match kernel.nodes.get(id).copied().unwrap_or(Op::MemInit) { Op::Const(c) => format!("{c}_i64"), Op::MemInit => String::from("0_i64"), _ => format!("v{id}") }
}

fn emit_op_rhs(kernel: &Kernel, doms: &[Domain], id: NodeId) -> String {
    let bname = |buf: u8| kernel.bufs.get(usize::from(buf)).map_or("", |b| b.name.as_str());
    match kernel.nodes.get(id).copied().unwrap_or(Op::MemInit) {
        Op::MemInit => String::from("0_i64"), Op::Const(c) => format!("{c}_i64"),
        Op::Load { buf, lane, .. } => format!("{}[{lane}]", bname(buf)),
        Op::Store { val, .. } => ref_expr(kernel, val),
        Op::LoadLut { buf, idx } => {
            let ix = ref_expr(kernel, idx);
            if let Some(b) = kernel.bufs.get(usize::from(buf)) {
                if let BufAccess::LookupTable(len) = b.access {
                    if len.is_power_of_two() && doms.get(idx).is_some_and(|d| !d.is_empty() && d.lo >= 0 && d.hi < len as i64) { return format!("{}[({ix} as usize) & {}]", b.name, len - 1); }
                }
                return format!("{}[{ix} as usize]", b.name);
            }
            format!("0_i64 /* {ix} */")
        }
        Op::BoundsCheck { idx, limit } => { let ix = ref_expr(kernel, idx); format!("{{ assert!({ix} >= 0 && {ix} < {limit}); {ix} }}") }
        Op::Add(a, b) => format!("{}.wrapping_add({})", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Sub(a, b) => format!("{}.wrapping_sub({})", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Mul(a, b) => {
            let (ra, rb) = (ref_expr(kernel, a), ref_expr(kernel, b));
            if doms.get(a).is_some_and(|d| d.fits_u32()) && doms.get(b).is_some_and(|d| d.fits_u32()) { format!("(({ra} as u32 as u64).wrapping_mul({rb} as u32 as u64)) as i64") } else { format!("{ra}.wrapping_mul({rb})") }
        }
        Op::Div(a, b) => format!("{} / {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Rem(a, b) => format!("{} % {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Shl(a, s) => format!("{}.wrapping_shl({s})", ref_expr(kernel, a)),
        Op::Shr(a, s) => { let ra = ref_expr(kernel, a); if doms.get(a).is_some_and(|d| d.is_nonneg()) { format!("(({ra} as u64) >> {s}) as i64") } else { format!("{ra} >> {s}") } }
        Op::Ushr(a, s) => format!("(({} as u64) >> {s}) as i64", ref_expr(kernel, a)),
        Op::ShlVar(a, b) => format!("(({} as u64).wrapping_shl(({} as u32) & 63)) as i64", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::ShrVarU(a, b) => format!("(({} as u64).wrapping_shr(({} as u32) & 63)) as i64", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::BitAnd(a, b) => format!("{} & {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::BitOr(a, b) => format!("{} | {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::BitXor(a, b) => format!("{} ^ {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Popcnt(a) => format!("(({} as u64).count_ones()) as i64", ref_expr(kernel, a)),
        Op::Tzcnt(a) => format!("(({} as u64).trailing_zeros()) as i64", ref_expr(kernel, a)),
        Op::Cmp(k, a, b) => if k.is_unsigned() {
            format!("(({} as u64) {} ({} as u64)) as i64", ref_expr(kernel, a), k.sym(), ref_expr(kernel, b))
        } else {
            format!("({} {} {}) as i64", ref_expr(kernel, a), k.sym(), ref_expr(kernel, b))
        },
        Op::Select { cond, t_val, f_val } => format!("if {} != 0_i64 {{ {} }} else {{ {} }}", ref_expr(kernel, cond), ref_expr(kernel, t_val), ref_expr(kernel, f_val)),
    }
}

fn drain_outputs(s: &mut String, kernel: &Kernel, sched: &ScheduleReport, oc: &mut usize, step: Option<usize>) {
    while let Some(&(st, oi)) = sched.output_order.get(*oc) {
        if step.is_some_and(|want| st != want) { break; }
        if let Some(o) = kernel.outputs.get(oi) {
            if let Some(b) = kernel.bufs.get(usize::from(o.buf)) {
                let _ = writeln!(s, "    {}[{}] = {};", b.name, o.lane, ref_expr(kernel, o.val));
            }
        }
        *oc += 1;
    }
}

/// Emit branchless, lane-unrolled Rust source targeting `rustc -C opt-level=3`.
pub fn emit_rust(kernel: &Kernel, sched: &ScheduleReport) -> Result<String, CompileError> {
    kernel.validate()?;
    let (doms, mut s, mut oc) = (kernel.infer_domains(), String::with_capacity(2048), 0usize);
    let _ = writeln!(s, "// Generated by zero-cc | live_nodes={} peak_gprs={} peak_vecs={}", sched.live_nodes, sched.peak_gprs, sched.peak_vecs);
    for b in &kernel.bufs {
        if b.domain != Domain::top() && !b.domain.is_empty() {
            if b.domain.modulus > 1 { let _ = writeln!(s, "/// @domain {}: {}..={} mod {} == {}", b.name, b.domain.lo, b.domain.hi, b.domain.modulus, b.domain.rem); }
            else { let _ = writeln!(s, "/// @domain {}: {}..={}", b.name, b.domain.lo, b.domain.hi); }
        }
    }
    let params: Vec<String> = kernel.bufs.iter().map(|b| match b.access {
        BufAccess::ReadOnly => format!("{}: &[i64; {LANES}]", b.name),
        BufAccess::MutNoAlias => format!("{}: &mut [i64; {LANES}]", b.name),
        BufAccess::LookupTable(len) => format!("{}: &[i64; {len}]", b.name),
    }).collect();
    let _ = writeln!(s, "#[inline(always)]\npub fn {}({}) {{", kernel.name, params.join(", "));
    for (step, &id) in sched.order.iter().enumerate() {
        drain_outputs(&mut s, kernel, sched, &mut oc, Some(step));
        if let Some(Op::Store { buf, lane, val, .. }) = kernel.nodes.get(id).copied() {
            if let Some(b) = kernel.bufs.get(usize::from(buf)) {
                let _ = writeln!(s, "    {}[{lane}] = {};", b.name, ref_expr(kernel, val));
            }
        } else {
            let _ = writeln!(s, "    let v{id}: i64 = {};", emit_op_rhs(kernel, &doms, id));
        }
    }
    drain_outputs(&mut s, kernel, sched, &mut oc, None);
    let _ = writeln!(s, "}}");
    let (mut bparams, mut args, mut lbufs) = (Vec::new(), Vec::new(), Vec::new());
    for b in &kernel.bufs {
        match b.access {
            BufAccess::ReadOnly => { bparams.push(format!("{}: &[[i64; {LANES}]]", b.name)); args.push(format!("&{}[blk]", b.name)); lbufs.push(b.name.as_str()); }
            BufAccess::MutNoAlias => { bparams.push(format!("{}: &mut [[i64; {LANES}]]", b.name)); args.push(format!("&mut {}[blk]", b.name)); lbufs.push(b.name.as_str()); }
            BufAccess::LookupTable(len) => { bparams.push(format!("{}: &[i64; {len}]", b.name)); args.push(b.name.clone()); }
        }
    }
    let _ = writeln!(s, "\n#[inline(never)]\npub fn {}_batch({}) {{", kernel.name, bparams.join(", "));
    if let Some(first) = lbufs.first() {
        let len_expr = lbufs[1..].iter().fold(format!("{first}.len()"), |acc, &lb| format!("{acc}.min({lb}.len())"));
        let _ = writeln!(s, "    let blocks = {len_expr};\n    for blk in 0..blocks {{\n        {}({});\n    }}", kernel.name, args.join(", "));
    }
    let _ = writeln!(s, "}}");
    Ok(s)
}
