use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{BufAccess, CompileError, Domain, Kernel, NodeId, Op, LANES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegBudget { pub max_gprs: usize, pub max_vecs: usize }

impl Default for RegBudget {
    fn default() -> Self { Self { max_gprs: 14, max_vecs: 16 } }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleReport {
    pub order: Vec<NodeId>,
    pub output_order: Vec<(usize, usize)>,
    pub peak_gprs: usize,
    pub peak_vecs: usize,
    pub live_nodes: usize,
}

fn is_inline(op: Op<NodeId>) -> bool { matches!(op, Op::MemInit | Op::Const(_)) }

/// Minimum-live-register Sethi-Ullman DAG scheduler with hard zero-spill cliff guard.
pub fn schedule_and_verify(kernel: &Kernel, budget: RegBudget) -> Result<ScheduleReport, CompileError> {
    kernel.validate()?;
    let n = kernel.nodes.len();
    let mut reach = vec![false; n];
    let mut stack: Vec<NodeId> = kernel.outputs.iter().map(|o| o.val).collect();
    while let Some(u) = stack.pop() {
        if reach[u] { continue; }
        reach[u] = true;
        let (kids, cnt) = kernel.nodes[u].children();
        stack.extend_from_slice(&kids[..cnt]);
    }
    let (mut su, mut deps, mut consumers, mut rem_uses) = (vec![0usize; n], vec![Vec::new(); n], vec![Vec::new(); n], vec![0usize; n]);
    let mut pending_reads: BTreeMap<(u8, u8), usize> = BTreeMap::new();
    for id in 0..n {
        if !reach[id] || is_inline(kernel.nodes[id]) { continue; }
        if let Op::Load { buf, lane, .. } = kernel.nodes[id] {
            *pending_reads.entry((buf, lane)).or_insert(0) += 1;
        }
        let (kids, cnt) = kernel.nodes[id].children();
        let mut uniq = Vec::new();
        for &k in &kids[..cnt] {
            if !is_inline(kernel.nodes[k]) && !uniq.contains(&k) { uniq.push(k); }
        }
        su[id] = match uniq.as_slice() {
            [] => 1,
            [a] => su[*a].max(1),
            [a, b] => if su[*a] == su[*b] { su[*a] + 1 } else { su[*a].max(su[*b]) },
            _ => 1,
        };
        for &k in &uniq { consumers[k].push(id); rem_uses[k] += 1; }
        deps[id] = uniq;
    }
    for o in &kernel.outputs {
        if !is_inline(kernel.nodes[o.val]) { rem_uses[o.val] += 1; }
    }
    let mut rem_deps: Vec<usize> = deps.iter().map(Vec::len).collect();
    let mut ready: Vec<NodeId> = (0..n).filter(|&i| reach[i] && !is_inline(kernel.nodes[i]) && rem_deps[i] == 0).collect();
    let (mut sched, mut out_done) = (vec![false; n], vec![false; kernel.outputs.len()]);
    let (mut live, mut order, mut output_order, mut peak_ssa) = (BTreeSet::new(), Vec::new(), Vec::new(), 0usize);

    while !ready.is_empty() {
        let mut best = (0usize, (isize::MAX, 1usize, 0usize, usize::MAX));
        for (pos, &u) in ready.iter().enumerate() {
            let kills = deps[u].iter().filter(|&&d| rem_uses[d] == 1).count() as isize;
            let delta = isize::from(rem_uses[u] > 0) - kills;
            let helps = usize::from(!consumers[u].iter().any(|&c| deps[c].iter().any(|d| live.contains(d))));
            let key = (delta, helps, usize::MAX - su[u], u);
            if key < best.1 { best = (pos, key); }
        }
        let u = ready.swap_remove(best.0);
        sched[u] = true;
        order.push(u);
        if let Op::Load { buf, lane, .. } = kernel.nodes[u] {
            if let Some(cnt) = pending_reads.get_mut(&(buf, lane)) { *cnt = cnt.saturating_sub(1); }
        }
        peak_ssa = peak_ssa.max(live.len() + usize::from(rem_uses[u] > 0));
        if rem_uses[u] > 0 { live.insert(u); }
        for &d in &deps[u] {
            rem_uses[d] -= 1;
            if rem_uses[d] == 0 { live.remove(&d); }
        }
        for &c in &consumers[u] {
            rem_deps[c] -= 1;
            if rem_deps[c] == 0 { ready.push(c); }
        }
        for (oi, o) in kernel.outputs.iter().enumerate() {
            if out_done[oi] { continue; }
            let val_ready = is_inline(kernel.nodes[o.val]) || sched[o.val];
            if val_ready && pending_reads.get(&(o.buf, o.lane)).copied().unwrap_or(0) == 0 {
                out_done[oi] = true;
                output_order.push((order.len(), oi));
                if !is_inline(kernel.nodes[o.val]) {
                    rem_uses[o.val] -= 1;
                    if rem_uses[o.val] == 0 { live.remove(&o.val); }
                }
            }
        }
    }
    for (oi, o) in kernel.outputs.iter().enumerate() {
        if !out_done[oi] {
            output_order.push((order.len(), oi));
            if !is_inline(kernel.nodes[o.val]) { rem_uses[o.val] = rem_uses[o.val].saturating_sub(1); }
        }
    }
    let (peak_gprs, peak_vecs) = (kernel.bufs.len() + peak_ssa, peak_ssa.saturating_mul(2));
    if peak_gprs > budget.max_gprs || peak_vecs > budget.max_vecs {
        return Err(CompileError::RegisterSpill {
            peak_gprs, max_gprs: budget.max_gprs, peak_vecs, max_vecs: budget.max_vecs,
        });
    }
    Ok(ScheduleReport { order, output_order, peak_gprs, peak_vecs, live_nodes: sched.iter().filter(|&&s| s).count() })
}

fn ref_expr(kernel: &Kernel, id: NodeId) -> String {
    match kernel.nodes[id] { Op::Const(c) => format!("{c}_i64"), Op::MemInit => String::from("0_i64"), _ => format!("v{id}") }
}

fn emit_op_rhs(kernel: &Kernel, doms: &[Domain], id: NodeId) -> String {
    match kernel.nodes[id] {
        Op::MemInit => String::from("0_i64"),
        Op::Const(c) => format!("{c}_i64"),
        Op::Load { buf, lane, .. } => format!("{}[{lane}]", kernel.bufs[buf as usize].name),
        Op::Store { val, .. } => ref_expr(kernel, val),
        Op::LoadLut { buf, idx } => {
            let (b, ix) = (&kernel.bufs[buf as usize], ref_expr(kernel, idx));
            if let BufAccess::LookupTable(len) = b.access {
                if len.is_power_of_two() && doms[idx].lo >= 0 && doms[idx].hi < len as i64 {
                    return format!("{}[({ix} as usize) & {}]", b.name, len - 1);
                }
            }
            format!("{}[{ix} as usize]", b.name)
        }
        Op::BoundsCheck { idx, limit } => {
            let ix = ref_expr(kernel, idx);
            format!("{{ assert!({ix} >= 0 && {ix} < {limit}); {ix} }}")
        }
        Op::Add(a, b) => format!("{}.wrapping_add({})", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Sub(a, b) => format!("{}.wrapping_sub({})", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Mul(a, b) => {
            let (ra, rb) = (ref_expr(kernel, a), ref_expr(kernel, b));
            if doms[a].fits_u32() && doms[b].fits_u32() {
                format!("(({ra} as u32 as u64).wrapping_mul({rb} as u32 as u64)) as i64")
            } else {
                format!("{ra}.wrapping_mul({rb})")
            }
        }
        Op::Div(a, b) => format!("{} / {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Rem(a, b) => format!("{} % {}", ref_expr(kernel, a), ref_expr(kernel, b)),
        Op::Shl(a, s) => format!("{}.wrapping_shl({s})", ref_expr(kernel, a)),
        Op::Shr(a, s) => {
            let ra = ref_expr(kernel, a);
            if doms[a].is_nonneg() { format!("(({ra} as u64) >> {s}) as i64") } else { format!("{ra} >> {s}") }
        }
        Op::BitAnd(a, b) => format!("{} & {}", ref_expr(kernel, a), ref_expr(kernel, b)),
    }
}

/// Emit branchless, lane-unrolled Rust source targeting `rustc -C opt-level=3`.
pub fn emit_rust(kernel: &Kernel, sched: &ScheduleReport) -> Result<String, CompileError> {
    kernel.validate()?;
    let doms = kernel.infer_domains();
    let mut s = String::with_capacity(2048);
    let _ = writeln!(s, "// Generated by zero-cc | live_nodes={} peak_gprs={} peak_vecs={}",
        sched.live_nodes, sched.peak_gprs, sched.peak_vecs);
    let params: Vec<String> = kernel.bufs.iter().map(|b| match b.access {
        BufAccess::ReadOnly => format!("{}: &[i64; {LANES}]", b.name),
        BufAccess::MutNoAlias => format!("{}: &mut [i64; {LANES}]", b.name),
        BufAccess::LookupTable(len) => format!("{}: &[i64; {len}]", b.name),
    }).collect();
    let _ = writeln!(s, "#[inline(always)]\npub fn {}({}) {{", kernel.name, params.join(", "));
    let mut oc = 0usize;
    for (step, &id) in sched.order.iter().enumerate() {
        while oc < sched.output_order.len() && sched.output_order[oc].0 == step {
            let o = kernel.outputs[sched.output_order[oc].1];
            let _ = writeln!(s, "    {}[{}] = {};", kernel.bufs[o.buf as usize].name, o.lane, ref_expr(kernel, o.val));
            oc += 1;
        }
        if !matches!(kernel.nodes[id], Op::Store { .. }) {
            let _ = writeln!(s, "    let v{id}: i64 = {};", emit_op_rhs(kernel, &doms, id));
        }
    }
    while oc < sched.output_order.len() {
        let o = kernel.outputs[sched.output_order[oc].1];
        let _ = writeln!(s, "    {}[{}] = {};", kernel.bufs[o.buf as usize].name, o.lane, ref_expr(kernel, o.val));
        oc += 1;
    }
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
        let mut len_expr = format!("{first}.len()");
        for &lb in &lbufs[1..] { len_expr = format!("{len_expr}.min({lb}.len())"); }
        let _ = writeln!(s, "    let blocks = {len_expr};");
        let _ = writeln!(s, "    for blk in 0..blocks {{\n        {}({});\n    }}", kernel.name, args.join(", "));
    }
    let _ = writeln!(s, "}}");
    Ok(s)
}
