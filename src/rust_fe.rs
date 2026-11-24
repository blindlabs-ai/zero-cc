use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::codegen::RegBudget;
use crate::elf64::{verify_elf64_executable, verify_elf64_object, TargetIsa};
use crate::ir::{
    barrett_params, BufAccess, CmpKind, CompileError, Domain, ElemTy, Kernel, KernelCost, NodeId,
    Op, LANES,
};
use crate::isa::{compile_kernel_direct, compile_standalone_elf_exec};
use crate::{compile_kernel, CompiledKernel};

const MAX_UNROLL: i64 = 16;
const MAX_INLINE_DEPTH: usize = 6;
const MAX_FILE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok { Ident(String), Int(i64), DocDomain(String), Sym(String) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValTy { I64, I32, I16, U64, U32, U16, U8, Usize }

impl ValTy {
    const fn from_elem(e: ElemTy) -> Self {
        match e {
            ElemTy::I64 => Self::I64,
            ElemTy::U64 => Self::U64,
            ElemTy::I32 => Self::I32,
            ElemTy::I16 => Self::I16,
            ElemTy::U32 => Self::U32,
            ElemTy::U16 => Self::U16,
            ElemTy::U8 => Self::U8,
        }
    }
    const fn is_signed(self) -> bool { matches!(self, Self::I64 | Self::I32 | Self::I16) }
    const fn is_u64_cmp(self) -> bool { matches!(self, Self::U64 | Self::Usize) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExprVal {
    Node(NodeId, ValTy),
    Xor(NodeId, NodeId, ValTy),
    MaskedXor { a: NodeId, b: NodeId, cond: NodeId, ty: ValTy },
}

#[derive(Clone, Debug)]
struct HelperFn {
    params: Vec<(String, ElemTy)>,
    body_toks: Vec<Tok>,
}

#[derive(Clone, Debug)]
pub struct ExtractedKernel {
    pub file: String,
    pub name: String,
    pub kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct CompiledCrateKernel {
    pub file: String,
    pub name: String,
    pub original_cost: KernelCost,
    pub optimized_cost: KernelCost,
    pub direct_insns: usize,
    pub direct_share_pct: u8,
    pub exec_share_pct: u8,
    pub zero_spills: bool,
    pub obj_bytes: usize,
    pub exec_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct CrateCompileReport {
    pub root: String,
    pub files_scanned: usize,
    pub functions_seen: usize,
    pub kernels_extracted: usize,
    pub compiled: Vec<CompiledCrateKernel>,
    pub skipped: usize,
}

fn perr(msg: impl Into<String>) -> CompileError { CompileError::ParseError(msg.into()) }
fn lane_u8(v: i64) -> u8 { if !(0..=255).contains(&v) { 255 } else { v as u8 } }

fn parse_int_literal(raw: &str, radix: u32) -> Result<i64, CompileError> {
    let digits: String = raw.chars().filter(|&c| c != '_').collect();
    if digits.is_empty() { return Err(perr(format!("empty int '{raw}'"))); }
    u64::from_str_radix(&digits, radix).map(|v| v as i64).map_err(|_| perr(format!("bad int '{raw}'")))
}

fn lex_rust(src: &str) -> Result<Vec<Tok>, CompileError> {
    if !src.is_ascii() { return Err(perr("non-ASCII Rust source")); }
    let (b, mut i, mut out) = (src.as_bytes(), 0usize, Vec::new());
    while i < b.len() {
        if b[i].is_ascii_whitespace() { i += 1; continue; }
        if b[i..].starts_with(b"///") {
            i += 3; let s = i; while i < b.len() && b[i] != b'\n' { i += 1; }
            if let Some(d) = src[s..i].trim().strip_prefix("@domain").or_else(|| src[s..i].trim().strip_prefix("@zero_domain")) {
                out.push(Tok::DocDomain(d.trim().into()));
            }
        } else if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' { i += 1; }
        } else if b[i..].starts_with(b"/*") {
            i += 2; while i + 1 < b.len() && &b[i..i + 2] != b"*/" { i += 1; }
            if i + 1 >= b.len() { return Err(perr("unterminated block comment")); }
            i += 2;
        } else if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i; while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            out.push(Tok::Ident(src[s..i].into()));
        } else if b[i].is_ascii_digit() {
            let (radix, start) = if b[i] == b'0' && matches!(b.get(i + 1), Some(b'x' | b'X')) {
                i += 2; (16u32, i)
            } else if b[i] == b'0' && matches!(b.get(i + 1), Some(b'b' | b'B')) {
                i += 2; (2u32, i)
            } else {
                (10u32, i)
            };
            while i < b.len() && (b[i] == b'_' || (b[i] as char).is_digit(radix)) { i += 1; }
            let end = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            out.push(Tok::Int(parse_int_literal(&src[start..end], radix)?));
        } else if let Some(s) = ["..=", "..", "==", "!=", ">=", "<=", "&&", "||", "<<", ">>", "->", "::", "#!"]
            .into_iter().find(|s| b[i..].starts_with(s.as_bytes()))
        {
            out.push(Tok::Sym(s.into())); i += s.len();
        } else {
            out.push(Tok::Sym((b[i] as char).to_string())); i += 1;
        }
    }
    Ok(out)
}

fn parse_signed(toks: &[Tok], p: &mut usize) -> Result<i64, CompileError> {
    let neg = toks.get(*p) == Some(&Tok::Sym("-".into()));
    if neg { *p += 1; }
    let Some(Tok::Int(v)) = toks.get(*p) else { return Err(perr("expected integer")); };
    *p += 1;
    if neg { v.checked_neg().ok_or_else(|| perr("int overflow")) } else { Ok(*v) }
}

fn parse_domain_list(toks: &[Tok], out: &mut BTreeMap<String, Domain>) -> Result<(), CompileError> {
    let slice = if toks.first() == Some(&Tok::Sym("(".into())) && toks.last() == Some(&Tok::Sym(")".into())) {
        &toks[1..toks.len() - 1]
    } else {
        toks
    };
    for part in slice.split(|t| *t == Tok::Sym(",".into())).filter(|p| !p.is_empty()) {
        let mut p = 0usize;
        let name = match part.get(p) { Some(Tok::Ident(s)) => { p += 1; s.clone() } _ => return Err(perr("expected domain param")) };
        if matches!(part.get(p), Some(Tok::Sym(s)) if s == ":" || s == "=") { p += 1; } else { return Err(perr("expected ':' or '='")); }
        let lo = parse_signed(part, &mut p)?;
        if part.get(p) != Some(&Tok::Sym("..=".into())) { return Err(perr("expected '..='")); }
        p += 1;
        let hi = parse_signed(part, &mut p)?;
        if lo > hi { return Err(perr("empty domain lo > hi")); }
        let dom = if part.get(p) == Some(&Tok::Ident("mod".into())) {
            p += 1;
            let m = parse_signed(part, &mut p)?;
            if m <= 0 || part.get(p) != Some(&Tok::Sym("==".into())) { return Err(perr("bad mod clause")); }
            p += 1;
            Domain::congruent(lo, hi, m, parse_signed(part, &mut p)?)
        } else {
            Domain::range(lo, hi)
        };
        if p != part.len() { return Err(perr("trailing domain tokens")); }
        out.insert(name, dom);
    }
    Ok(())
}

/// Parse a CLI or API domain specification such as `"raw: -262144..=262144, x: 0..=1023"`.
pub fn parse_domain_spec(spec: &str) -> Result<BTreeMap<String, Domain>, CompileError> {
    let mut out = BTreeMap::new();
    if spec.trim().is_empty() { return Ok(out); }
    parse_domain_list(&lex_rust(spec)?, &mut out)?;
    Ok(out)
}

fn binop_prec(tok: Option<&Tok>) -> Option<(u8, &'static str)> {
    let Tok::Sym(s) = tok? else { return None };
    Some(match s.as_str() {
        "||" => (1, "||"), "&&" => (2, "&&"),
        "==" => (3, "=="), "!=" => (3, "!="), "<" => (3, "<"), "<=" => (3, "<="), ">" => (3, ">"), ">=" => (3, ">="),
        "|" => (4, "|"), "^" => (5, "^"), "&" => (6, "&"),
        "<<" => (7, "<<"), ">>" => (7, ">>"),
        "+" => (8, "+"), "-" => (8, "-"), "*" => (9, "*"), "/" => (9, "/"), "%" => (9, "%"),
        _ => return None,
    })
}

fn elem_from_scalar_name(name: &str) -> Option<(ElemTy, ValTy, Domain)> {
    if name == "usize" {
        return Some((ElemTy::I64, ValTy::Usize, Domain::range(0, i64::MAX)));
    }
    if name == "bool" {
        return Some((ElemTy::U8, ValTy::U8, Domain::range(0, 1)));
    }
    let e = ElemTy::from_rust_name(name)?;
    Some((e, ValTy::from_elem(e), e.natural_domain()))
}

fn disjoint_or_pow2(da: Domain, db: Domain) -> bool {
    if da.is_empty() || db.is_empty() || !db.is_nonneg() || db.hi == i64::MAX { return false; }
    if db.hi == 0 { return true; }
    let h = db.hi as u64;
    let p = h.next_power_of_two();
    let m = if p <= h { p.saturating_mul(2) } else { p };
    (2..=(1u64 << 48)).contains(&m) && da.rem_mod(m as i64) == Some(0)
}

fn refine_clamp_domain(k: &Kernel, doms: &[Domain], op: Op<NodeId>, base: Domain) -> Domain {
    let Op::Select { cond, t_val, f_val } = op else { return base; };
    let Some(&Op::Cmp(kind, a, b)) = k.nodes.get(cond) else { return base; };
    let Some(c) = doms.get(b).copied().and_then(Domain::is_const) else { return base; };
    let mut d = base;
    if t_val == b && f_val == a {
        match kind {
            CmpKind::Gt | CmpKind::Ge => { if d.hi > c { d.hi = c; d.modulus = 1; d.rem = 0; } }
            CmpKind::Lt | CmpKind::Le => { if d.lo < c { d.lo = c; d.modulus = 1; d.rem = 0; } }
            _ => {}
        }
    } else if t_val == a && f_val == b {
        match kind {
            CmpKind::Lt | CmpKind::Le => { if d.hi > c { d.hi = c; d.modulus = 1; d.rem = 0; } }
            CmpKind::Gt | CmpKind::Ge => { if d.lo < c { d.lo = c; d.modulus = 1; d.rem = 0; } }
            _ => {}
        }
    }
    d
}

struct Lowerer {
    toks: Vec<Tok>,
    pos: usize,
    k: Kernel,
    doms: Vec<Domain>,
    init_mem: NodeId,
    mem: NodeId,
    buf_ids: BTreeMap<String, u8>,
    lane_loaded: BTreeSet<u8>,
    vars: BTreeMap<String, (NodeId, ValTy)>,
    consts: BTreeMap<String, i64>,
    final_out: Vec<(u8, u8, NodeId)>,
    helpers: BTreeMap<String, HelperFn>,
    bool_masks: BTreeMap<NodeId, NodeId>,
    inline_depth: usize,
}

impl Lowerer {
    fn new(toks: Vec<Tok>, helpers: BTreeMap<String, HelperFn>) -> Self {
        Self {
            toks, pos: 0, k: Kernel::new(""), doms: Vec::new(), init_mem: 0, mem: 0,
            buf_ids: BTreeMap::new(), lane_loaded: BTreeSet::new(), vars: BTreeMap::new(),
            consts: BTreeMap::new(), final_out: Vec::new(), helpers,
            bool_masks: BTreeMap::new(), inline_depth: 0,
        }
    }
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
    fn eat(&mut self, is_sym: bool, s: &str) -> bool {
        let hit = match self.peek() {
            Some(Tok::Sym(t)) => is_sym && t == s,
            Some(Tok::Ident(t)) => !is_sym && t == s,
            _ => false,
        };
        if hit { self.pos += 1; }
        hit
    }
    fn eat_sym(&mut self, s: &str) -> bool { self.eat(true, s) }
    fn eat_id(&mut self, s: &str) -> bool { self.eat(false, s) }
    fn req_sym(&mut self, s: &str) -> Result<(), CompileError> {
        if self.eat_sym(s) { Ok(()) } else { Err(perr(format!("expected '{s}'"))) }
    }
    fn req_id(&mut self) -> Result<String, CompileError> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => { self.pos += 1; Ok(s) }
            _ => Err(perr("expected identifier")),
        }
    }
    fn push(&mut self, op: Op<NodeId>) -> NodeId {
        let raw = op.eval_domain(
            |i| self.doms.get(i).copied().unwrap_or_else(Domain::top),
            |b| self.k.bufs.get(b as usize).map_or_else(Domain::top, |d| d.domain),
            |i| matches!(self.k.nodes.get(i), Some(Op::MemInit)),
        );
        let d = refine_clamp_domain(&self.k, &self.doms, op, raw);
        let id = self.k.push(op);
        self.doms.push(d);
        if let Op::Sub(a, b) = op {
            if self.doms.get(a).and_then(|x| x.is_const()) == Some(0) {
                if let Some(bd) = self.doms.get(b) {
                    if !bd.is_empty() && bd.lo >= 0 && bd.hi <= 1 {
                        self.bool_masks.insert(id, b);
                    }
                }
            }
        }
        id
    }
    fn dom(&self, id: NodeId) -> Domain { self.doms.get(id).copied().unwrap_or_else(Domain::top) }
    fn as_bool(&mut self, id: NodeId) -> NodeId {
        let d = self.dom(id);
        if !d.is_empty() && d.lo >= 0 && d.hi <= 1 {
            id
        } else {
            let z = self.push(Op::Const(0));
            self.push(Op::Cmp(CmpKind::Ne, id, z))
        }
    }
    fn record_store(&mut self, buf: u8, lane: u8, val: NodeId) {
        self.mem = self.push(Op::Store { mem: self.mem, buf, lane, val });
        if let Some(slot) = self.final_out.iter_mut().find(|o| o.0 == buf && o.1 == lane) {
            slot.2 = val;
        } else {
            self.final_out.push((buf, lane, val));
        }
    }
    fn realize(&mut self, ev: ExprVal) -> Result<(NodeId, ValTy), CompileError> {
        match ev {
            ExprVal::Node(n, t) => Ok((n, t)),
            ExprVal::Xor(a, b, t) => Ok((self.push(Op::BitXor(a, b)), t)),
            ExprVal::MaskedXor { a, b, cond, ty } => {
                let x = self.push(Op::BitXor(a, b));
                let z = self.push(Op::Const(0));
                Ok((self.push(Op::Select { cond, t_val: x, f_val: z }), ty))
            }
        }
    }
    fn parse_node(&mut self, min_p: u8) -> Result<(NodeId, ValTy), CompileError> {
        let ev = self.parse_prec(min_p)?;
        self.realize(ev)
    }
    fn push_shift(&mut self, base: NodeId, ty: ValTy, amt: NodeId, is_shr: bool, _check_ushr: bool) -> Result<NodeId, CompileError> {
        if let Some(s) = self.dom(amt).is_const() {
            if !(0..64).contains(&s) { return Err(CompileError::InvalidShift(s as u8)); }
            if !is_shr { return Ok(self.push(Op::Shl(base, s as u8))); }
            if !ty.is_signed() && !self.dom(base).is_nonneg() {
                return Ok(self.push(Op::Ushr(base, s as u8)));
            }
            return Ok(self.push(Op::Shr(base, s as u8)));
        }
        if !is_shr {
            Ok(self.push(Op::ShlVar(base, amt)))
        } else if !ty.is_signed() {
            Ok(self.push(Op::ShrVarU(base, amt)))
        } else {
            Err(perr("signed variable right shift unsupported in kernel IR"))
        }
    }
    fn push_div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        if let Some(d) = self.dom(rhs).is_const() {
            let dx = self.dom(lhs);
            if d >= 3 && !(d as u64).is_power_of_two() && !dx.is_empty() && dx.lo < 0 {
                let max_abs = dx.lo.unsigned_abs().max(dx.hi.unsigned_abs());
                if max_abs <= i64::MAX as u64 {
                    if let Some((m, s)) = barrett_params(d, max_abs as i64) {
                        let mc = self.push(Op::Const(m));
                        let prod = self.push(Op::Mul(lhs, mc));
                        let q_floor = self.push(Op::Shr(prod, s));
                        let sign = self.push(Op::Shr(lhs, 63));
                        return self.push(Op::Sub(q_floor, sign));
                    }
                }
            }
        }
        self.push(Op::Div(lhs, rhs))
    }
    fn skip_balanced(&mut self, open: &str, close: &str) -> Result<usize, CompileError> {
        self.req_sym(open)?;
        let (start, mut depth) = (self.pos, 1usize);
        while depth > 0 {
            if self.pos >= self.toks.len() { return Err(perr("unclosed delimiter")); }
            if self.eat_sym(open) { depth += 1; } else if self.eat_sym(close) { depth -= 1; } else { self.pos += 1; }
        }
        Ok(start)
    }
    fn try_const(&self, start: usize, end: usize) -> Option<i64> {
        let atom = |p: &mut usize| match self.toks.get(*p)? {
            Tok::Int(v) => { *p += 1; Some(*v) }
            Tok::Ident(s) => { *p += 1; self.consts.get(s).copied() }
            _ => None,
        };
        let mut p = start;
        let mut val = atom(&mut p)?;
        while p < end {
            let op = match self.toks.get(p)? {
                Tok::Sym(s) if s == "+" || s == "-" || s == "*" => { p += 1; s.as_str() }
                _ => return None,
            };
            let rhs = atom(&mut p)?;
            val = match op {
                "+" => val.checked_add(rhs)?,
                "-" => val.checked_sub(rhs)?,
                _ => val.checked_mul(rhs)?,
            };
        }
        Some(val)
    }
    fn parse_assert(&mut self) -> Result<(Option<String>, NodeId), CompileError> {
        self.req_sym("!")?; self.req_sym("(")?;
        let vname = match (self.peek(), self.toks.get(self.pos + 1)) {
            (Some(Tok::Ident(s)), Some(Tok::Sym(t))) if t == ">=" => Some(s.clone()),
            _ => None,
        };
        let (idx, _) = self.parse_node(4)?;
        self.req_sym(">=")?;
        if parse_signed(&self.toks, &mut self.pos)? != 0 { return Err(perr("assert! lower bound must be 0")); }
        self.req_sym("&&")?;
        let _ = self.parse_node(4)?;
        self.req_sym("<")?;
        let limit = parse_signed(&self.toks, &mut self.pos)?;
        if limit <= 0 { return Err(perr("assert! limit must be > 0")); }
        self.req_sym(")")?;
        Ok((vname, self.push(Op::BoundsCheck { idx, limit })))
    }
    fn merge_select(&mut self, cond: NodeId, t_id: NodeId, f_id: NodeId) -> NodeId {
        if t_id == f_id { t_id } else { self.push(Op::Select { cond, t_val: t_id, f_val: f_id }) }
    }
    fn narrow_for_branch(&mut self, cond: NodeId, is_true: bool) -> Option<(NodeId, Domain)> {
        let Op::Cmp(kind, a, b) = *self.k.nodes.get(cond)? else { return None; };
        let c = self.dom(b).is_const()?;
        let old = self.dom(a);
        let mut next = old;
        match (kind, is_true) {
            (CmpKind::Lt, true) => { next.hi = next.hi.min(c.saturating_sub(1)); }
            (CmpKind::Le, true) | (CmpKind::Gt, false) => { next.hi = next.hi.min(c); }
            (CmpKind::Gt, true) => { next.lo = next.lo.max(c.saturating_add(1)); }
            (CmpKind::Ge, true) | (CmpKind::Lt, false) => { next.lo = next.lo.max(c); }
            _ => return None,
        }
        if next.lo <= next.hi && (next.lo != old.lo || next.hi != old.hi) {
            next.modulus = 1; next.rem = 0;
            if let Some(slot) = self.doms.get_mut(a) { *slot = next; }
            Some((a, old))
        } else {
            None
        }
    }
    fn restore_branch(&mut self, saved: Option<(NodeId, Domain)>) {
        if let Some((id, d)) = saved {
            if let Some(slot) = self.doms.get_mut(id) { *slot = d; }
        }
    }
    fn parse_if_stmt(&mut self) -> Result<(), CompileError> {
        let (cond, _) = self.parse_node(1)?;
        let (sv, sm, so) = (self.vars.clone(), self.mem, self.final_out.clone());
        let st = self.narrow_for_branch(cond, true);
        let tr = self.parse_block();
        self.restore_branch(st);
        tr?;
        let (tv, to) = (self.vars.clone(), self.final_out.clone());
        self.vars = sv.clone(); self.mem = sm; self.final_out = so.clone();
        if !self.eat_id("else") { return Err(perr("if statement requires else")); }
        let sf = self.narrow_for_branch(cond, false);
        let fr = if self.eat_id("if") { self.parse_if_stmt() } else { self.parse_block() };
        self.restore_branch(sf);
        fr?;
        let (fv, fo) = (self.vars.clone(), self.final_out.clone());
        self.vars = sv; self.mem = sm; self.final_out = so;
        for (name, &(t_id, ty)) in &tv {
            if let Some(&(f_id, _)) = fv.get(name) {
                if self.vars.contains_key(name) {
                    let v = self.merge_select(cond, t_id, f_id);
                    self.vars.insert(name.clone(), (v, ty));
                }
            }
        }
        let mut keys: Vec<(u8, u8)> = to.iter().chain(&fo).map(|&(b, l, _)| (b, l)).collect();
        keys.sort_unstable(); keys.dedup();
        for (buf, lane) in keys {
            let t_id = to.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2).unwrap_or_else(|| self.push(Op::Load { mem: sm, buf, lane }));
            let f_id = fo.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2).unwrap_or_else(|| self.push(Op::Load { mem: sm, buf, lane }));
            let old = self.final_out.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2);
            if Some(t_id) != old || Some(f_id) != old {
                let m = self.merge_select(cond, t_id, f_id);
                self.record_store(buf, lane, m);
            }
        }
        Ok(())
    }
    fn parse_if_expr_vals(&mut self) -> Result<Vec<(NodeId, ValTy)>, CompileError> {
        if self.eat_id("let") {
            let some_kw = self.req_id()?;
            if some_kw != "Some" { return Err(perr("if-let only supports Some(var)")); }
            self.req_sym("(")?;
            let bind = self.req_id()?;
            self.req_sym(")")?;
            self.req_sym("=")?;
            let (lhs, lty) = self.parse_primary().and_then(|ev| self.realize(ev))?;
            self.req_sym(".")?;
            let meth = self.req_id()?;
            self.req_sym("(")?;
            let (rhs, _) = self.parse_node(1)?;
            self.req_sym(")")?;
            let (val, ok) = match meth.as_str() {
                "checked_add" => {
                    let sum = self.push(Op::Add(lhs, rhs));
                    let cmp = self.push(Op::Cmp(CmpKind::Uge, sum, lhs));
                    (sum, cmp)
                }
                "checked_sub" => {
                    let diff = self.push(Op::Sub(lhs, rhs));
                    let cmp = self.push(Op::Cmp(CmpKind::Uge, lhs, rhs));
                    (diff, cmp)
                }
                _ => return Err(perr(format!("unsupported if-let method '{meth}'"))),
            };
            let saved_vars = self.vars.clone();
            self.vars.insert(bind, (val, lty));
            let tv = self.parse_expr_block()?;
            self.vars = saved_vars;
            if !self.eat_id("else") { return Err(perr("if-let expr requires else")); }
            let fv = if self.eat_id("if") { self.parse_if_expr_vals()? } else { self.parse_expr_block()? };
            if tv.len() != fv.len() { return Err(perr("if-let branch tuple arity mismatch")); }
            return Ok(tv.into_iter().zip(fv).map(|((t, ty), (f, _))| (self.merge_select(ok, t, f), ty)).collect());
        }
        let (cond, _) = self.parse_node(1)?;
        let st = self.narrow_for_branch(cond, true);
        let tv_res = self.parse_expr_block();
        self.restore_branch(st);
        let tv = tv_res?;
        if tv.is_empty() { return Err(perr("empty if branch")); }
        if !self.eat_id("else") { return Err(perr("if expr requires else")); }
        let sf = self.narrow_for_branch(cond, false);
        let fv_res = if self.eat_id("if") { self.parse_if_expr_vals() } else { self.parse_expr_block() };
        self.restore_branch(sf);
        let fv = fv_res?;
        if tv.len() != fv.len() { return Err(perr("if branch tuple arity mismatch")); }
        Ok(tv.into_iter().zip(fv).map(|((t, ty), (f, _))| (self.merge_select(cond, t, f), ty)).collect())
    }
    fn parse_tuple_rhs(&mut self) -> Result<Vec<(NodeId, ValTy)>, CompileError> {
        if self.eat_id("if") { return self.parse_if_expr_vals(); }
        if let Some(Tok::Ident(name)) = self.peek().cloned() {
            if self.toks.get(self.pos + 1) == Some(&Tok::Sym("(".into())) {
                if let Some(helper) = self.helpers.get(&name).cloned() {
                    self.pos += 2;
                    let mut args = Vec::new();
                    while !self.eat_sym(")") {
                        args.push(self.parse_node(1)?);
                        if !self.eat_sym(",") { self.req_sym(")")?; break; }
                    }
                    return self.inline_helper_vals(helper, args);
                }
            }
        }
        if self.peek() == Some(&Tok::Sym("(".into())) {
            let saved = self.pos;
            self.pos += 1;
            let first = self.parse_node(1)?;
            if self.eat_sym(",") {
                let mut vals = vec![first];
                while !self.eat_sym(")") {
                    vals.push(self.parse_node(1)?);
                    if !self.eat_sym(",") { self.req_sym(")")?; break; }
                }
                return Ok(vals);
            }
            self.pos = saved;
        }
        let saved = self.pos;
        if let Ok(lhs_ev) = self.parse_primary() {
            if self.eat_sym(".") {
                if let Some(Tok::Ident(m)) = self.peek().cloned() {
                    if m == "overflowing_add" || m == "overflowing_sub" {
                        self.pos += 1;
                        self.req_sym("(")?;
                        let (rhs, _) = self.parse_node(1)?;
                        self.req_sym(")")?;
                        let (lhs, ty) = self.realize(lhs_ev)?;
                        let (res, ovf) = if m == "overflowing_add" {
                            let s = self.push(Op::Add(lhs, rhs));
                            (s, self.push(Op::Cmp(CmpKind::Ult, s, lhs)))
                        } else {
                            let d = self.push(Op::Sub(lhs, rhs));
                            (d, self.push(Op::Cmp(CmpKind::Ult, lhs, rhs)))
                        };
                        return Ok(vec![(res, ty), (ovf, ValTy::U8)]);
                    }
                }
            }
        }
        self.pos = saved;
        Ok(vec![self.parse_node(1)?])
    }
    fn parse_let_stmt(&mut self) -> Result<(), CompileError> {
        if self.eat_sym("(") {
            let mut names = Vec::new();
            while !self.eat_sym(")") {
                let _ = self.eat_id("mut");
                names.push(self.req_id()?);
                if !self.eat_sym(",") { self.req_sym(")")?; break; }
            }
            if self.eat_sym(":") {
                let _ = self.skip_balanced("(", ")")?;
            }
            self.req_sym("=")?;
            let vals = self.parse_tuple_rhs()?;
            self.req_sym(";")?;
            if vals.len() != names.len() { return Err(perr("tuple let binding arity mismatch")); }
            for (name, val) in names.into_iter().zip(vals) {
                if name != "_" { self.vars.insert(name, val); }
            }
            return Ok(());
        }
        let _ = self.eat_id("mut");
        let name = self.req_id()?;
        if self.eat_sym(":") { let _ = self.req_id()?; }
        self.req_sym("=")?;
        let val = self.parse_node(1)?;
        self.req_sym(";")?;
        self.vars.insert(name, val);
        Ok(())
    }
    fn parse_block(&mut self) -> Result<(), CompileError> {
        self.req_sym("{")?;
        while !self.eat_sym("}") {
            if self.pos >= self.toks.len() { return Err(perr("unclosed block")); }
            if self.eat_id("for") {
                let var = self.req_id()?;
                if !self.eat_id("in") { return Err(perr("expected 'in'")); }
                let lo = parse_signed(&self.toks, &mut self.pos)?;
                let incl = if self.eat_sym("..=") { true } else { self.req_sym("..")?; false };
                let hi = parse_signed(&self.toks, &mut self.pos)?;
                let hi_e = if incl { hi.checked_add(1).ok_or_else(|| perr("loop overflow"))? } else { hi };
                if lo < 0 || hi_e < lo || hi_e - lo > MAX_UNROLL {
                    return Err(perr("for-loop bounds must be constant with 0..=16 trips"));
                }
                let (bpos, mut epos) = (self.pos, self.pos);
                if lo == hi_e { self.skip_balanced("{", "}")?; continue; }
                for v in lo..hi_e {
                    let (sv, sc) = (self.vars.clone(), self.consts.clone());
                    self.consts.insert(var.clone(), v);
                    self.pos = bpos;
                    self.parse_block()?;
                    epos = self.pos;
                    self.vars = sv; self.consts = sc;
                }
                self.pos = epos;
            } else if self.eat_id("if") {
                self.parse_if_stmt()?;
            } else if self.eat_id("let") {
                self.parse_let_stmt()?;
            } else if self.eat_id("assert") {
                let (vname, chk) = self.parse_assert()?;
                self.req_sym(";")?;
                if let Some(n) = vname { self.vars.insert(n, (chk, ValTy::I64)); }
            } else if let Some(Tok::Ident(name)) = self.peek().cloned() {
                if let Some(&buf) = self.buf_ids.get(&name) {
                    self.pos += 1;
                    let lstart = self.skip_balanced("[", "]")?;
                    let lv = self.try_const(lstart, self.pos - 1).ok_or_else(|| perr("store index must be constant lane"))?;
                    let max_lanes = self.k.bufs.get(buf as usize).map_or(LANES as i64, |b| i64::from(b.lanes));
                    if lv < 0 || lv >= max_lanes { return Err(CompileError::InvalidLane(lane_u8(lv))); }
                    self.req_sym("=")?;
                    let (val, _) = self.parse_node(1)?;
                    self.req_sym(";")?;
                    self.record_store(buf, lv as u8, val);
                } else if self.vars.contains_key(&name) {
                    self.pos += 1;
                    self.req_sym("=")?;
                    let val = self.parse_node(1)?;
                    self.req_sym(";")?;
                    self.vars.insert(name, val);
                } else {
                    return Err(perr(format!("unknown target '{name}'")));
                }
            } else {
                return Err(perr("unexpected token in kernel body"));
            }
        }
        Ok(())
    }
    fn parse_expr_block(&mut self) -> Result<Vec<(NodeId, ValTy)>, CompileError> {
        self.req_sym("{")?;
        loop {
            if self.pos >= self.toks.len() { return Err(perr("unclosed expr block")); }
            if self.eat_id("let") {
                self.parse_let_stmt()?;
            } else if self.eat_id("assert") {
                let (vname, chk) = self.parse_assert()?;
                self.req_sym(";")?;
                if let Some(n) = vname { self.vars.insert(n, (chk, ValTy::I64)); }
            } else {
                break;
            }
        }
        let out = self.parse_tuple_rhs()?;
        let _ = self.eat_sym(";");
        self.req_sym("}")?;
        Ok(out)
    }
    fn parse_prec(&mut self, min_p: u8) -> Result<ExprVal, CompileError> {
        let mut lhs = self.parse_cast()?;
        while let Some((p, op)) = binop_prec(self.peek()) {
            if p < min_p { break; }
            self.pos += 1;
            let rhs = self.parse_prec(p + 1)?;
            lhs = self.apply_binop(op, lhs, rhs)?;
        }
        Ok(lhs)
    }
    fn apply_binop(&mut self, op: &str, lhs: ExprVal, rhs: ExprVal) -> Result<ExprVal, CompileError> {
        if op == "^" {
            if let (ExprVal::Node(l, ty), ExprVal::MaskedXor { a, b, cond, .. }) = (lhs, rhs) {
                if l == b { return Ok(ExprVal::Node(self.push(Op::Select { cond, t_val: a, f_val: b }), ty)); }
                if l == a { return Ok(ExprVal::Node(self.push(Op::Select { cond, t_val: b, f_val: a }), ty)); }
            }
            if let (ExprVal::MaskedXor { a, b, cond, .. }, ExprVal::Node(r, ty)) = (lhs, rhs) {
                if r == b { return Ok(ExprVal::Node(self.push(Op::Select { cond, t_val: a, f_val: b }), ty)); }
                if r == a { return Ok(ExprVal::Node(self.push(Op::Select { cond, t_val: b, f_val: a }), ty)); }
            }
            let ((ln, ty), (rn, _)) = (self.realize(lhs)?, self.realize(rhs)?);
            if let (Some(a), Some(b)) = (self.dom(ln).is_const(), self.dom(rn).is_const()) {
                return Ok(ExprVal::Node(self.push(Op::Const(a ^ b)), ty));
            }
            return Ok(ExprVal::Xor(ln, rn, ty));
        }
        if op == "&" {
            if let (ExprVal::Xor(a, b, ty), ExprVal::Node(mask, _)) = (lhs, rhs) {
                if let Some(&cond) = self.bool_masks.get(&mask) {
                    return Ok(ExprVal::MaskedXor { a, b, cond, ty });
                }
            }
            if let (ExprVal::Node(mask, _), ExprVal::Xor(a, b, ty)) = (lhs, rhs) {
                if let Some(&cond) = self.bool_masks.get(&mask) {
                    return Ok(ExprVal::MaskedXor { a, b, cond, ty });
                }
            }
        }
        let ((ln, ty), (rn, rty)) = (self.realize(lhs)?, self.realize(rhs)?);
        let u64_cmp = ty.is_u64_cmp() || rty.is_u64_cmp();
        let id = match op {
            "||" => {
                let (ba, bb) = (self.as_bool(ln), self.as_bool(rn));
                let s = self.push(Op::Add(ba, bb));
                let z = self.push(Op::Const(0));
                self.push(Op::Cmp(CmpKind::Ne, s, z))
            }
            "&&" => {
                let (ba, bb) = (self.as_bool(ln), self.as_bool(rn));
                self.push(Op::BitAnd(ba, bb))
            }
            "==" => self.push(Op::Cmp(CmpKind::Eq, ln, rn)),
            "!=" => self.push(Op::Cmp(CmpKind::Ne, ln, rn)),
            "<" => self.push(Op::Cmp(if u64_cmp { CmpKind::Ult } else { CmpKind::Lt }, ln, rn)),
            "<=" => self.push(Op::Cmp(if u64_cmp { CmpKind::Ule } else { CmpKind::Le }, ln, rn)),
            ">" => self.push(Op::Cmp(if u64_cmp { CmpKind::Ugt } else { CmpKind::Gt }, ln, rn)),
            ">=" => self.push(Op::Cmp(if u64_cmp { CmpKind::Uge } else { CmpKind::Ge }, ln, rn)),
            "|" => match (self.dom(ln).is_const(), self.dom(rn).is_const()) {
                (Some(a), Some(b)) => self.push(Op::Const(a | b)),
                _ if disjoint_or_pow2(self.dom(ln), self.dom(rn)) || disjoint_or_pow2(self.dom(rn), self.dom(ln)) => {
                    self.push(Op::Add(ln, rn))
                }
                _ => self.push(Op::BitOr(ln, rn)),
            },
            "&" => self.push(Op::BitAnd(ln, rn)),
            "<<" | ">>" => self.push_shift(ln, ty, rn, op == ">>", true)?,
            "+" => self.push(Op::Add(ln, rn)),
            "-" => self.push(Op::Sub(ln, rn)),
            "*" => self.push(Op::Mul(ln, rn)),
            "/" => self.push_div(ln, rn),
            _ => self.push(Op::Rem(ln, rn)),
        };
        Ok(ExprVal::Node(id, ty))
    }
    fn parse_cast(&mut self) -> Result<ExprVal, CompileError> {
        let mut expr = self.parse_primary()?;
        while self.eat_sym(".") {
            let (mut n, ty) = self.realize(expr)?;
            let m = self.req_id()?;
            self.req_sym("(")?;
            if self.eat_sym(")") {
                n = match m.as_str() {
                    "wrapping_neg" => { let z = self.push(Op::Const(0)); self.push(Op::Sub(z, n)) }
                    "abs" => {
                        let z = self.push(Op::Const(0));
                        let lt = self.push(Op::Cmp(CmpKind::Lt, n, z));
                        let neg = self.push(Op::Sub(z, n));
                        self.push(Op::Select { cond: lt, t_val: neg, f_val: n })
                    }
                    _ => return Err(perr(format!("unsupported 0-arg method '.{m}'"))),
                };
            } else {
                let (arg, _) = self.parse_node(1)?;
                self.req_sym(")")?;
                n = match m.as_str() {
                    "wrapping_add" => self.push(Op::Add(n, arg)),
                    "wrapping_sub" => self.push(Op::Sub(n, arg)),
                    "wrapping_mul" => self.push(Op::Mul(n, arg)),
                    "wrapping_shl" | "wrapping_shr" => self.push_shift(n, ty, arg, m == "wrapping_shr", false)?,
                    "min" => { let k = if ty.is_u64_cmp() { CmpKind::Ult } else { CmpKind::Lt }; let c = self.push(Op::Cmp(k, n, arg)); self.push(Op::Select { cond: c, t_val: n, f_val: arg }) }
                    "max" => { let k = if ty.is_u64_cmp() { CmpKind::Ugt } else { CmpKind::Gt }; let c = self.push(Op::Cmp(k, n, arg)); self.push(Op::Select { cond: c, t_val: n, f_val: arg }) }
                    "div_ceil" => {
                        let d = self.dom(arg).is_const().ok_or_else(|| perr("div_ceil divisor must be constant"))?;
                        if d <= 0 || !self.dom(n).is_nonneg() { return Err(perr("div_ceil requires non-negative lhs and d > 0")); }
                        let add_c = self.push(Op::Const(d - 1));
                        let sum = self.push(Op::Add(n, add_c));
                        self.push_div(sum, arg)
                    }
                    _ => return Err(perr(format!("unsupported method '.{m}'"))),
                };
            }
            expr = ExprVal::Node(n, ty);
        }
        while self.eat_id("as") {
            let (mut n, _) = self.realize(expr)?;
            let d = self.dom(n);
            let target = self.req_id()?;
            let ty = match target.as_str() {
                "i64" => ValTy::I64,
                "u64" => ValTy::U64,
                "usize" => ValTy::Usize,
                "u32" => {
                    if !d.fits_u32() {
                        let m = self.push(Op::Const(0xFFFF_FFFF));
                        n = self.push(Op::BitAnd(n, m));
                    }
                    ValTy::U32
                }
                "u16" => {
                    if !d.is_nonneg() || d.hi > u16::MAX as i64 {
                        let m = self.push(Op::Const(0xFFFF));
                        n = self.push(Op::BitAnd(n, m));
                    }
                    ValTy::U16
                }
                "u8" => {
                    if !d.is_nonneg() || d.hi > u8::MAX as i64 {
                        let m = self.push(Op::Const(0xFF));
                        n = self.push(Op::BitAnd(n, m));
                    }
                    ValTy::U8
                }
                "i32" => {
                    if d.is_empty() || d.lo < i32::MIN as i64 || d.hi > i32::MAX as i64 {
                        let s = self.push(Op::Shl(n, 32));
                        n = self.push(Op::Shr(s, 32));
                    }
                    ValTy::I32
                }
                "i16" => {
                    if d.is_empty() || d.lo < i16::MIN as i64 || d.hi > i16::MAX as i64 {
                        let s = self.push(Op::Shl(n, 48));
                        n = self.push(Op::Shr(s, 48));
                    }
                    ValTy::I16
                }
                o => return Err(perr(format!("unsupported cast '{o}'"))),
            };
            expr = ExprVal::Node(n, ty);
        }
        Ok(expr)
    }
    fn inline_helper_vals(&mut self, helper: HelperFn, args: Vec<(NodeId, ValTy)>) -> Result<Vec<(NodeId, ValTy)>, CompileError> {
        if self.inline_depth >= MAX_INLINE_DEPTH { return Err(perr("helper inline depth exceeded")); }
        if args.len() != helper.params.len() { return Err(perr("helper arity mismatch")); }
        let saved_toks = std::mem::replace(&mut self.toks, helper.body_toks);
        let saved_pos = std::mem::replace(&mut self.pos, 0);
        let saved_vars = std::mem::take(&mut self.vars);
        for ((pname, pty), (aid, _)) in helper.params.into_iter().zip(args) {
            self.vars.insert(pname, (aid, ValTy::from_elem(pty)));
        }
        self.inline_depth += 1;
        let res = self.parse_expr_block();
        self.inline_depth -= 1;
        self.toks = saved_toks;
        self.pos = saved_pos;
        self.vars = saved_vars;
        let vals = res?;
        if vals.is_empty() { return Err(perr("empty helper body")); }
        Ok(vals)
    }
    fn inline_helper(&mut self, helper: HelperFn, args: Vec<(NodeId, ValTy)>) -> Result<ExprVal, CompileError> {
        let vals = self.inline_helper_vals(helper, args)?;
        if vals.len() != 1 { return Err(perr("tuple helper call unsupported inside scalar expr")); }
        Ok(ExprVal::Node(vals[0].0, vals[0].1))
    }
    fn parse_primary(&mut self) -> Result<ExprVal, CompileError> {
        if self.eat_sym("-") {
            if let Some(&Tok::Int(v)) = self.peek() {
                self.pos += 1;
                return Ok(ExprVal::Node(self.push(Op::Const(v.checked_neg().ok_or_else(|| perr("int overflow"))?)), ValTy::I64));
            }
            let (e, ty) = self.parse_primary().and_then(|ev| self.realize(ev))?;
            let z = self.push(Op::Const(0));
            return Ok(ExprVal::Node(self.push(Op::Sub(z, e)), ty));
        }
        if self.eat_sym("!") {
            let (e, _) = self.parse_primary().and_then(|ev| self.realize(ev))?;
            let z = self.push(Op::Const(0));
            return Ok(ExprVal::Node(self.push(Op::Cmp(CmpKind::Eq, e, z)), ValTy::I64));
        }
        if let Some(&Tok::Int(v)) = self.peek() {
            self.pos += 1;
            return Ok(ExprVal::Node(self.push(Op::Const(v)), ValTy::I64));
        }
        if self.eat_sym("(") {
            let v = self.parse_prec(1)?;
            self.req_sym(")")?;
            return Ok(v);
        }
        if self.eat_id("if") {
            let vals = self.parse_if_expr_vals()?;
            if vals.len() != 1 { return Err(perr("tuple if expr unsupported inside scalar expr")); }
            return Ok(ExprVal::Node(vals[0].0, vals[0].1));
        }
        if self.eat_sym("{") {
            if !self.eat_id("assert") { return Err(perr("block expr must start with assert!")); }
            let (_, chk) = self.parse_assert()?;
            self.req_sym(";")?;
            let _ = self.parse_node(1)?;
            self.req_sym("}")?;
            return Ok(ExprVal::Node(chk, ValTy::I64));
        }
        let name = self.req_id()?;
        if name == "true" || name == "false" {
            return Ok(ExprVal::Node(self.push(Op::Const(i64::from(name == "true"))), ValTy::I64));
        }
        if self.eat_sym("::") {
            let method = self.req_id()?;
            if method == "from" {
                self.req_sym("(")?;
                let (arg, _) = self.parse_node(1)?;
                self.req_sym(")")?;
                let ty = elem_from_scalar_name(&name).map_or(ValTy::I64, |(_, t, _)| t);
                return Ok(ExprVal::Node(arg, ty));
            }
            if method == "MAX" || method == "MIN" {
                let c = match (name.as_str(), method.as_str()) {
                    ("u64", "MAX") => (u64::MAX as i64, ValTy::U64),
                    ("u64", "MIN") => (0, ValTy::U64),
                    ("u32", "MAX") => (u32::MAX as i64, ValTy::U32),
                    ("u32", "MIN") => (0, ValTy::U32),
                    ("u16", "MAX") => (u16::MAX as i64, ValTy::U16),
                    ("u8", "MAX") => (u8::MAX as i64, ValTy::U8),
                    ("i64", "MAX") => (i64::MAX, ValTy::I64),
                    ("i64", "MIN") => (i64::MIN, ValTy::I64),
                    _ => return Err(perr(format!("unsupported const '{name}::{method}'"))),
                };
                return Ok(ExprVal::Node(self.push(Op::Const(c.0)), c.1));
            }
            return Err(perr(format!("unsupported path '{name}::{method}'")));
        }
        if self.peek() == Some(&Tok::Sym("(".into())) {
            if let Some(helper) = self.helpers.get(&name).cloned() {
                self.pos += 1;
                let mut args = Vec::new();
                while !self.eat_sym(")") {
                    args.push(self.parse_node(1)?);
                    if !self.eat_sym(",") { self.req_sym(")")?; break; }
                }
                return self.inline_helper(helper, args);
            }
            return Err(perr(format!("unknown helper function '{name}'")));
        }
        if let Some(&buf) = self.buf_ids.get(&name) {
            let (saved, istart) = (self.pos, self.skip_balanced("[", "]")?);
            let b_elem = self.k.bufs.get(buf as usize).map_or(ElemTy::I64, |b| b.elem_ty);
            let b_lanes = self.k.bufs.get(buf as usize).map_or(LANES, |b| usize::from(b.lanes));
            let b_acc = self.k.bufs.get(buf as usize).map_or(BufAccess::ReadOnly, |b| b.access);
            let ty = ValTy::from_elem(b_elem);
            if !matches!(b_acc, BufAccess::LookupTable(_)) {
                if let Some(lv) = self.try_const(istart, self.pos - 1) {
                    if lv < 0 || lv >= b_lanes as i64 { return Err(CompileError::InvalidLane(lane_u8(lv))); }
                    self.lane_loaded.insert(buf);
                    let m = if b_acc == BufAccess::ReadOnly { self.init_mem } else { self.mem };
                    return Ok(ExprVal::Node(self.push(Op::Load { mem: m, buf, lane: lv as u8 }), ty));
                }
                if b_acc == BufAccess::ReadOnly && !self.lane_loaded.contains(&buf) {
                    if let Some(slot) = self.k.bufs.get_mut(buf as usize) {
                        slot.access = BufAccess::LookupTable(b_lanes);
                    }
                }
            }
            self.pos = saved;
            self.req_sym("[")?;
            let (idx, _) = self.parse_node(1)?;
            self.req_sym("]")?;
            return Ok(ExprVal::Node(self.push(Op::LoadLut { buf, idx }), ty));
        }
        if let Some(&(n, t)) = self.vars.get(&name) { return Ok(ExprVal::Node(n, t)); }
        if let Some(&c) = self.consts.get(&name) { return Ok(ExprVal::Node(self.push(Op::Const(c)), ValTy::I64)); }
        Err(perr(format!("unknown identifier '{name}'")))
    }
}

fn parse_fn_header(low: &mut Lowerer, domains: &mut BTreeMap<String, Domain>) -> Result<String, CompileError> {
    while let Some(tok) = low.peek().cloned() {
        match tok {
            Tok::DocDomain(s) => { parse_domain_list(&lex_rust(&s)?, domains)?; low.pos += 1; }
            Tok::Sym(s) if s == "#!" || s == "#" => {
                low.pos += 1;
                let astart = low.skip_balanced("[", "]")?;
                let attr = &low.toks[astart..low.pos - 1];
                if attr.first() == Some(&Tok::Ident("zero_domain".into())) && attr.len() > 2 {
                    parse_domain_list(&attr[1..], domains)?;
                }
            }
            _ => break,
        }
    }
    if low.eat_id("pub") && low.peek() == Some(&Tok::Sym("(".into())) {
        low.skip_balanced("(", ")")?;
    }
    let _ = low.eat_id("const");
    if !low.eat_id("fn") { return Err(perr("expected 'fn'")); }
    let name = low.req_id()?;
    low.req_sym("(")?;
    Ok(name)
}

fn tighten_with_elem(dom: Domain, elem: ElemTy) -> Domain {
    let nat = elem.natural_domain();
    dom.try_meet(nat).unwrap_or(nat)
}

fn parse_kernel_internal(
    src: &str,
    overrides: &BTreeMap<String, Domain>,
    helpers: &BTreeMap<String, HelperFn>,
    allow_scalar_lift: bool,
) -> Result<Kernel, CompileError> {
    let mut domains = BTreeMap::new();
    let mut low = Lowerer::new(lex_rust(src)?, helpers.clone());
    low.k.name = parse_fn_header(&mut low, &mut domains)?;
    for (k, v) in overrides { domains.insert(k.clone(), *v); }

    let mut scalar_params: Vec<(String, ElemTy, ValTy, Domain)> = Vec::new();
    let mut saw_array = false;
    while !low.eat_sym(")") {
        let pname = low.req_id()?;
        low.req_sym(":")?;
        if low.eat_sym("&") {
            saw_array = true;
            let is_mut = low.eat_id("mut");
            low.req_sym("[")?;
            let ty_name = low.req_id()?;
            let elem_ty = ElemTy::from_rust_name(&ty_name)
                .ok_or_else(|| perr(format!("unsupported buffer element type '{ty_name}'")))?;
            low.req_sym(";")?;
            let len = match low.peek() {
                Some(&Tok::Int(v)) if v > 0 => { low.pos += 1; v as usize }
                _ => return Err(perr("expected array len")),
            };
            low.req_sym("]")?;
            let (access, lanes) = if is_mut {
                if len == 0 || len > LANES { return Err(perr("&mut buffer lanes must be 1..=8")); }
                (BufAccess::MutNoAlias, len as u8)
            } else if len <= LANES {
                (BufAccess::ReadOnly, len as u8)
            } else {
                (BufAccess::LookupTable(len), LANES as u8)
            };
            let dom = domains.get(&pname).copied().map_or_else(|| elem_ty.natural_domain(), |d| tighten_with_elem(d, elem_ty));
            let id = low.k.add_typed_buf(pname.clone(), access, elem_ty, lanes, dom);
            low.buf_ids.insert(pname, id);
        } else {
            if !allow_scalar_lift || saw_array { return Err(perr("expected '&[T; N]' buffer param")); }
            let ty_name = low.req_id()?;
            let (elem_ty, vty, nat) = elem_from_scalar_name(&ty_name)
                .ok_or_else(|| perr(format!("unsupported scalar param type '{ty_name}'")))?;
            let dom = domains.get(&pname).copied().map_or(nat, |d| tighten_with_elem(d, elem_ty));
            scalar_params.push((pname, elem_ty, vty, dom));
        }
        if !low.eat_sym(",") { low.req_sym(")")?; break; }
    }

    if saw_array {
        if !scalar_params.is_empty() { return Err(perr("mixed scalar and array params unsupported")); }
        low.init_mem = low.push(Op::MemInit);
        low.mem = low.init_mem;
        low.parse_block()?;
    } else {
        if !allow_scalar_lift || scalar_params.is_empty() || scalar_params.len() > 4 {
            return Err(perr("scalar lift requires 1..=4 scalar params"));
        }
        low.req_sym("->")?;
        let mut ret_tys = Vec::new();
        if low.eat_sym("(") {
            while !low.eat_sym(")") {
                let tn = low.req_id()?;
                let (e, _, d) = elem_from_scalar_name(&tn).ok_or_else(|| perr(format!("unsupported return type '{tn}'")))?;
                ret_tys.push((e, d));
                if !low.eat_sym(",") { low.req_sym(")")?; break; }
            }
        } else {
            let tn = low.req_id()?;
            let (e, _, d) = elem_from_scalar_name(&tn).ok_or_else(|| perr(format!("unsupported return type '{tn}'")))?;
            ret_tys.push((e, d));
        }
        if ret_tys.is_empty() || ret_tys.len() > 2 || scalar_params.len() + ret_tys.len() > 5 {
            return Err(perr("scalar lift supports 1..=2 returns and <= 5 total buffers"));
        }
        let mut in_bufs = Vec::new();
        for (pname, elem_ty, vty, dom) in scalar_params {
            let bid = low.k.add_typed_buf(pname.clone(), BufAccess::ReadOnly, elem_ty, LANES as u8, dom);
            in_bufs.push((pname, bid, vty));
        }
        let mut out_bufs = Vec::new();
        for (idx, (elem_ty, dom)) in ret_tys.iter().enumerate() {
            let oname = if ret_tys.len() == 1 { "out".into() } else { format!("out{idx}") };
            out_bufs.push(low.k.add_typed_buf(oname, BufAccess::MutNoAlias, *elem_ty, LANES as u8, *dom));
        }
        low.init_mem = low.push(Op::MemInit);
        low.mem = low.init_mem;
        let body_pos = low.pos;
        for lane in 0..LANES as u8 {
            low.vars.clear();
            for (pname, bid, vty) in &in_bufs {
                let ld = low.push(Op::Load { mem: low.init_mem, buf: *bid, lane });
                low.vars.insert(pname.clone(), (ld, *vty));
            }
            low.pos = body_pos;
            let res = low.parse_expr_block()?;
            if res.len() != out_bufs.len() { return Err(perr("scalar return arity mismatch")); }
            for (&obuf, (val, _)) in out_bufs.iter().zip(res) {
                low.record_store(obuf, lane, val);
            }
        }
    }

    for (buf, lane, val) in low.final_out { low.k.add_output(buf, lane, val); }
    low.k.validate()?;
    Ok(low.k)
}

/// Parse a raw safe-Rust `.rs` kernel function (with optional preceding helper functions) into `Kernel` IR.
pub fn parse_rust_kernel(src: &str) -> Result<Kernel, CompileError> {
    let clean = sanitize_ascii_comments(src);
    let fns = extract_top_level_fns(&clean);
    if fns.len() > 1 {
        let mut helpers = BTreeMap::new();
        for (_, snippet) in &fns {
            if let Some((name, h)) = try_parse_helper_fn(snippet) {
                helpers.insert(name, h);
            }
        }
        if !helpers.is_empty() {
            if let Some((_, snippet)) = fns.iter().find(|(n, _)| !helpers.contains_key(n)) {
                return parse_kernel_internal(snippet, &BTreeMap::new(), &helpers, true);
            }
        }
    }
    parse_kernel_internal(src, &BTreeMap::new(), &BTreeMap::new(), false)
}

/// Parse a named entry kernel from a multi-function safe-Rust module, inlining all sibling helpers.
pub fn parse_rust_pipeline(src: &str, entry: &str) -> Result<Kernel, CompileError> {
    let clean = sanitize_ascii_comments(src);
    let fns = extract_top_level_fns(&clean);
    let mut helpers = BTreeMap::new();
    for (name, snippet) in &fns {
        if name != entry {
            if let Some((hname, h)) = try_parse_helper_fn(snippet) {
                helpers.insert(hname, h);
            }
        }
    }
    let (_, snippet) = fns.iter().find(|(n, _)| n == entry)
        .ok_or_else(|| perr(format!("pipeline entry '{entry}' not found")))?;
    parse_kernel_internal(snippet, &BTreeMap::new(), &helpers, true)
}

/// Parse a raw safe-Rust `.rs` kernel function with explicit parameter domain overrides.
pub fn parse_rust_kernel_with_domains(src: &str, overrides: &BTreeMap<String, Domain>) -> Result<Kernel, CompileError> {
    parse_kernel_internal(src, overrides, &BTreeMap::new(), true)
}

/// Compile a raw safe-Rust `.rs` kernel source directly through E-Graph + zero-spill codegen.
pub fn compile_rust_source(src: &str, budget: RegBudget) -> Result<CompiledKernel, CompileError> {
    compile_kernel(&parse_rust_kernel(src)?, budget)
}

fn sanitize_ascii_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    let mut in_line_comment = false;
    let mut block_depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    while let Some(c) = chars.next() {
        if in_line_comment {
            if c == '\n' { in_line_comment = false; out.push('\n'); }
            else if c.is_ascii() { out.push(c); } else { out.push(' '); }
            continue;
        }
        if block_depth > 0 {
            if c == '/' && chars.peek() == Some(&'*') { chars.next(); block_depth += 1; out.push_str("/*"); }
            else if c == '*' && chars.peek() == Some(&'/') { chars.next(); block_depth -= 1; out.push_str("*/"); }
            else if c.is_ascii() { out.push(c); } else { out.push(' '); }
            continue;
        }
        if in_str {
            if esc { esc = false; out.push(if c.is_ascii() { c } else { ' ' }); }
            else if c == '\\' { esc = true; out.push(c); }
            else if c == '"' { in_str = false; out.push(c); }
            else { out.push(if c.is_ascii() { c } else { ' ' }); }
            continue;
        }
        if c == '/' && chars.peek() == Some(&'/') {
            chars.next(); in_line_comment = true; out.push_str("//");
        } else if c == '/' && chars.peek() == Some(&'*') {
            chars.next(); block_depth = 1; out.push_str("/*");
        } else if c == '"' {
            in_str = true; out.push(c);
        } else if c.is_ascii() {
            out.push(c);
        } else {
            out.push(' ');
        }
    }
    out
}

fn skip_ws_and_comments(b: &[u8], mut i: usize) -> usize {
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
        if i + 1 < b.len() && &b[i..i + 2] == b"//" {
            while i < b.len() && b[i] != b'\n' { i += 1; }
        } else if i + 1 < b.len() && &b[i..i + 2] == b"/*" {
            i += 2;
            let mut d = 1usize;
            while i + 1 < b.len() && d > 0 {
                if &b[i..i + 2] == b"/*" { d += 1; i += 2; }
                else if &b[i..i + 2] == b"*/" { d -= 1; i += 2; }
                else { i += 1; }
            }
        } else {
            return i;
        }
    }
}

fn scan_code_byte(b: &[u8], i: usize) -> Option<usize> {
    if i + 1 < b.len() && &b[i..i + 2] == b"//" {
        let mut j = i + 2;
        while j < b.len() && b[j] != b'\n' { j += 1; }
        return Some(j);
    }
    if i + 1 < b.len() && &b[i..i + 2] == b"/*" {
        let mut j = i + 2;
        let mut d = 1usize;
        while j + 1 < b.len() && d > 0 {
            if &b[j..j + 2] == b"/*" { d += 1; j += 2; }
            else if &b[j..j + 2] == b"*/" { d -= 1; j += 2; }
            else { j += 1; }
        }
        return Some(j);
    }
    if b[i] == b'"' {
        let mut j = i + 1;
        while j < b.len() {
            if b[j] == b'\\' { j = (j + 2).min(b.len()); }
            else if b[j] == b'"' { return Some(j + 1); }
            else { j += 1; }
        }
        return Some(b.len());
    }
    if b[i] == b'\'' && i + 2 < b.len() && (b[i + 1] == b'\\' || b[i + 2] == b'\'') {
        let mut j = i + 1;
        if b[j] == b'\\' { j = (j + 2).min(b.len()); } else { j += 1; }
        if j < b.len() && b[j] == b'\'' { return Some(j + 1); }
    }
    None
}

fn find_matching_delim(b: &[u8], open_pos: usize, open: u8, close: u8) -> Option<usize> {
    let mut i = open_pos + 1;
    let mut depth = 1usize;
    while i < b.len() {
        if let Some(next) = scan_code_byte(b, i) { i = next; continue; }
        if b[i] == open { depth += 1; }
        else if b[i] == close {
            depth -= 1;
            if depth == 0 { return Some(i); }
        }
        i += 1;
    }
    None
}

fn extract_top_level_fns(clean: &str) -> Vec<(String, String)> {
    let b = clean.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut pending_prefix = String::new();
    let mut skip_cfg_test = false;
    while i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() { i += 1; }
        if i >= b.len() { break; }
        if i + 2 < b.len() && &b[i..i + 3] == b"///" {
            let s = i;
            while i < b.len() && b[i] != b'\n' { i += 1; }
            pending_prefix.push_str(&clean[s..i]);
            pending_prefix.push('\n');
            continue;
        }
        if let Some(next) = scan_code_byte(b, i) { i = next; continue; }
        if b[i] == b'#' {
            let open = if b.get(i + 1) == Some(&b'!') && b.get(i + 2) == Some(&b'[') { Some(i + 2) }
                else if b.get(i + 1) == Some(&b'[') { Some(i + 1) } else { None };
            if let Some(op) = open {
                if let Some(end) = find_matching_delim(b, op, b'[', b']') {
                    let attr = &clean[i..=end];
                    if attr.contains("cfg(test)") { skip_cfg_test = true; }
                    else if attr.contains("zero_domain") {
                        pending_prefix.push_str(attr);
                        pending_prefix.push('\n');
                    }
                    i = end + 1;
                    continue;
                }
                break;
            }
        }
        let item_start = i;
        let mut is_fn = false;
        let mut fn_name = String::new();
        let mut scan = i;
        while scan < b.len() {
            scan = skip_ws_and_comments(b, scan);
            if scan >= b.len() { break; }
            let ws = scan;
            while scan < b.len() && (b[scan].is_ascii_alphanumeric() || b[scan] == b'_') { scan += 1; }
            let word = &clean[ws..scan];
            if word == "pub" {
                let p = skip_ws_and_comments(b, scan);
                if b.get(p) == Some(&b'(') {
                    if let Some(end) = find_matching_delim(b, p, b'(', b')') { scan = end + 1; continue; }
                }
                continue;
            }
            if word == "const" { continue; }
            if word == "fn" {
                let ns = skip_ws_and_comments(b, scan);
                let mut ne = ns;
                while ne < b.len() && (b[ne].is_ascii_alphanumeric() || b[ne] == b'_') { ne += 1; }
                if ne > ns { is_fn = true; fn_name = clean[ns..ne].to_string(); }
            }
            break;
        }
        let mut j = item_start;
        let mut item_end = b.len();
        while j < b.len() {
            if let Some(next) = scan_code_byte(b, j) { j = next; continue; }
            if b[j] == b'(' {
                if let Some(end) = find_matching_delim(b, j, b'(', b')') { j = end + 1; continue; }
                break;
            }
            if b[j] == b'[' {
                if let Some(end) = find_matching_delim(b, j, b'[', b']') { j = end + 1; continue; }
                break;
            }
            if b[j] == b';' { item_end = j + 1; break; }
            if b[j] == b'{' {
                if let Some(end) = find_matching_delim(b, j, b'{', b'}') {
                    item_end = end + 1;
                    if is_fn && !skip_cfg_test {
                        let snippet = format!("{}{}", pending_prefix, &clean[item_start..=end]);
                        out.push((fn_name.clone(), snippet));
                    }
                }
                break;
            }
            j += 1;
        }
        pending_prefix.clear();
        skip_cfg_test = false;
        i = item_end.max(i + 1);
    }
    out
}

fn try_parse_helper_fn(snippet: &str) -> Option<(String, HelperFn)> {
    let toks = lex_rust(snippet).ok()?;
    let mut low = Lowerer::new(toks, BTreeMap::new());
    let mut domains = BTreeMap::new();
    let name = parse_fn_header(&mut low, &mut domains).ok()?;
    let mut params = Vec::new();
    while !low.eat_sym(")") {
        let pname = low.req_id().ok()?;
        low.req_sym(":").ok()?;
        let tname = low.req_id().ok()?;
        let (elem, _, _) = elem_from_scalar_name(&tname)?;
        params.push((pname, elem));
        if !low.eat_sym(",") { low.req_sym(")").ok()?; break; }
    }
    if params.is_empty() || params.len() > 8 { return None; }
    low.req_sym("->").ok()?;
    if low.eat_sym("(") {
        let mut ret_cnt = 0usize;
        while !low.eat_sym(")") {
            let tn = low.req_id().ok()?;
            let _ = elem_from_scalar_name(&tn)?;
            ret_cnt += 1;
            if !low.eat_sym(",") { low.req_sym(")").ok()?; break; }
        }
        if ret_cnt == 0 || ret_cnt > 4 { return None; }
    } else {
        let ret_name = low.req_id().ok()?;
        let _ = elem_from_scalar_name(&ret_name)?;
    }
    if low.peek() != Some(&Tok::Sym("{".into())) { return None; }
    let body_toks = low.toks[low.pos..].to_vec();
    Some((name, HelperFn { params, body_toks }))
}

/// Extract all parseable kernels (array + lifted scalar, with intra-file helper inlining) from a `.rs` file.
pub fn parse_rust_file_kernels(
    src: &str,
    domain_overrides: &BTreeMap<String, BTreeMap<String, Domain>>,
) -> Vec<Kernel> {
    let clean = sanitize_ascii_comments(src);
    let fns = extract_top_level_fns(&clean);
    let mut helpers = BTreeMap::new();
    for (_, snippet) in &fns {
        if let Some((name, h)) = try_parse_helper_fn(snippet) {
            helpers.insert(name, h);
        }
    }
    let global = domain_overrides.get("*").cloned().unwrap_or_default();
    let mut out = Vec::new();
    for (name, snippet) in fns {
        let mut ov = global.clone();
        if let Some(specific) = domain_overrides.get(&name) {
            for (k, v) in specific { ov.insert(k.clone(), *v); }
        }
        if let Ok(k) = parse_kernel_internal(&snippet, &ov, &helpers, true) {
            out.push(k);
        }
    }
    out
}

fn collect_rs_files(path: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), CompileError> {
    if path.is_file() {
        if path.extension().and_then(|e| e.to_str()) == Some("rs") { out.push(path.to_path_buf()); }
        return Ok(());
    }
    let rd = fs::read_dir(path).map_err(|e| perr(format!("read_dir '{}': {e}", path.display())))?;
    let mut entries = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| perr(format!("dir entry: {e}")))?;
        entries.push(ent.path());
    }
    entries.sort();
    for p in entries {
        if p.is_dir() {
            let skip = p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                matches!(n, "target" | "tests" | "benches" | "examples" | ".git")
            });
            if !skip { collect_rs_files(&p, out)?; }
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
    Ok(())
}

/// Extract all pure numeric kernels from a `.rs` file or crate directory tree.
pub fn extract_crate_kernels(
    path: &Path,
    domain_overrides: &BTreeMap<String, BTreeMap<String, Domain>>,
) -> Result<(usize, usize, Vec<ExtractedKernel>), CompileError> {
    let mut files = Vec::new();
    collect_rs_files(path, &mut files)?;
    let (mut files_scanned, mut functions_seen, mut extracted) = (0usize, 0usize, Vec::new());
    for file in files {
        let Ok(meta) = fs::metadata(&file) else { continue; };
        if meta.len() as usize > MAX_FILE_BYTES { continue; }
        let Ok(src) = fs::read_to_string(&file) else { continue; };
        files_scanned += 1;
        let clean = sanitize_ascii_comments(&src);
        let fns = extract_top_level_fns(&clean);
        functions_seen += fns.len();
        let file_label = file.strip_prefix(path).unwrap_or(&file).display().to_string();
        for kernel in parse_rust_file_kernels(&src, domain_overrides) {
            extracted.push(ExtractedKernel { file: file_label.clone(), name: kernel.name.clone(), kernel });
        }
    }
    Ok((files_scanned, functions_seen, extracted))
}

/// Adjust the register budget for narrow-typed kernels that lower on GPRs only (0 YMM usage).
pub fn kernel_reg_budget(kernel: &Kernel, base: RegBudget) -> RegBudget {
    let uses_avx2_i64 = kernel.bufs.iter().all(|b| b.elem_ty == ElemTy::I64 && usize::from(b.lanes) == LANES);
    if uses_avx2_i64 { base } else { RegBudget { max_gprs: base.max_gprs, max_vecs: base.max_vecs.max(32) } }
}

/// Extract and compile every kernel in a file or crate directory directly to ELF64 `ET_REL` and `ET_EXEC`.
pub fn extract_and_compile_crate(
    path: &Path,
    isa: TargetIsa,
    budget: RegBudget,
    domain_overrides: &BTreeMap<String, BTreeMap<String, Domain>>,
) -> Result<CrateCompileReport, CompileError> {
    let (files_scanned, functions_seen, extracted) = extract_crate_kernels(path, domain_overrides)?;
    let kernels_extracted = extracted.len();
    let (mut compiled, mut skipped) = (Vec::new(), 0usize);
    for ek in extracted {
        let orig = ek.kernel.cost();
        let Ok(opt_k) = crate::egraph::optimize_kernel(&ek.kernel) else { skipped += 1; continue; };
        let opt_cost = opt_k.cost();
        let k_budget = kernel_reg_budget(&ek.kernel, budget);
        let Ok(obj) = compile_kernel_direct(&ek.kernel, k_budget, isa) else { skipped += 1; continue; };
        let Ok(exec) = compile_standalone_elf_exec(&ek.kernel, k_budget, isa, &[]) else { skipped += 1; continue; };
        if verify_elf64_object(&obj.elf_bytes).is_err() || verify_elf64_executable(&exec.elf_bytes).is_err() {
            skipped += 1;
            continue;
        }
        compiled.push(CompiledCrateKernel {
            file: ek.file,
            name: ek.name,
            original_cost: orig,
            optimized_cost: opt_cost,
            direct_insns: obj.direct_insns,
            direct_share_pct: obj.direct_share_pct,
            exec_share_pct: exec.direct_share_pct,
            zero_spills: obj.zero_spills && exec.zero_spills,
            obj_bytes: obj.elf_bytes.len(),
            exec_bytes: exec.elf_bytes.len(),
        });
    }
    Ok(CrateCompileReport {
        root: path.display().to_string(),
        files_scanned,
        functions_seen,
        kernels_extracted,
        compiled,
        skipped,
    })
}
