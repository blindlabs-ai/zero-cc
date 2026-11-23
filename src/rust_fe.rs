use std::collections::{BTreeMap, BTreeSet};

use crate::codegen::RegBudget;
use crate::ir::{BufAccess, CmpKind, CompileError, Domain, Kernel, NodeId, Op, LANES};
use crate::{compile_kernel, CompiledKernel};

const MAX_UNROLL: i64 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok { Ident(String), Int(i64), DocDomain(String), Sym(String) }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValTy { I64, U32, U64, Usize }

fn perr(msg: impl Into<String>) -> CompileError { CompileError::ParseError(msg.into()) }
fn lane_u8(v: i64) -> u8 { if !(0..=255).contains(&v) { 255 } else { v as u8 } }

fn lex_rust(src: &str) -> Result<Vec<Tok>, CompileError> {
    if !src.is_ascii() { return Err(perr("non-ASCII Rust source")); }
    let (b, mut i, mut out) = (src.as_bytes(), 0usize, Vec::new());
    while i < b.len() {
        if b[i].is_ascii_whitespace() { i += 1; continue; }
        if b[i..].starts_with(b"///") {
            i += 3; let s = i; while i < b.len() && b[i] != b'\n' { i += 1; }
            if let Some(d) = src[s..i].trim().strip_prefix("@domain").or_else(|| src[s..i].trim().strip_prefix("@zero_domain")) { out.push(Tok::DocDomain(d.trim().into())); }
        } else if b[i..].starts_with(b"//") { while i < b.len() && b[i] != b'\n' { i += 1; } }
        else if b[i..].starts_with(b"/*") {
            i += 2; while i + 1 < b.len() && &b[i..i + 2] != b"*/" { i += 1; }
            if i + 1 >= b.len() { return Err(perr("unterminated block comment")); } i += 2;
        } else if b[i].is_ascii_alphabetic() || b[i] == b'_' {
            let s = i; while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            out.push(Tok::Ident(src[s..i].into()));
        } else if b[i].is_ascii_digit() {
            let s = i; while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') { i += 1; }
            let d: String = src[s..i].chars().filter(|&c| c != '_').collect();
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            out.push(Tok::Int(d.parse::<i64>().map_err(|_| perr(format!("bad int '{d}'")))?));
        } else if let Some(s) = ["..=", "..", "==", "!=", ">=", "<=", "&&", "<<", ">>", "->", "#!"].into_iter().find(|s| b[i..].starts_with(s.as_bytes())) {
            out.push(Tok::Sym(s.into())); i += s.len();
        } else { out.push(Tok::Sym((b[i] as char).to_string())); i += 1; }
    }
    Ok(out)
}

fn parse_signed(toks: &[Tok], p: &mut usize) -> Result<i64, CompileError> {
    let neg = toks.get(*p) == Some(&Tok::Sym("-".into())); if neg { *p += 1; }
    let Some(Tok::Int(v)) = toks.get(*p) else { return Err(perr("expected integer")); };
    *p += 1; if neg { v.checked_neg().ok_or_else(|| perr("int overflow")) } else { Ok(*v) }
}

fn parse_domain_list(toks: &[Tok], out: &mut BTreeMap<String, Domain>) -> Result<(), CompileError> {
    let slice = if toks.first() == Some(&Tok::Sym("(".into())) && toks.last() == Some(&Tok::Sym(")".into())) { &toks[1..toks.len() - 1] } else { toks };
    for part in slice.split(|t| *t == Tok::Sym(",".into())).filter(|p| !p.is_empty()) {
        let mut p = 0usize;
        let name = match part.get(p) { Some(Tok::Ident(s)) => { p += 1; s.clone() } _ => return Err(perr("expected domain param")) };
        if matches!(part.get(p), Some(Tok::Sym(s)) if s == ":" || s == "=") { p += 1; } else { return Err(perr("expected ':' or '='")); }
        let lo = parse_signed(part, &mut p)?;
        if part.get(p) != Some(&Tok::Sym("..=".into())) { return Err(perr("expected '..='")); }
        p += 1; let hi = parse_signed(part, &mut p)?;
        if lo > hi { return Err(perr("empty domain lo > hi")); }
        let dom = if part.get(p) == Some(&Tok::Ident("mod".into())) {
            p += 1; let m = parse_signed(part, &mut p)?;
            if m <= 0 || part.get(p) != Some(&Tok::Sym("==".into())) { return Err(perr("bad mod clause")); }
            p += 1; Domain::congruent(lo, hi, m, parse_signed(part, &mut p)?)
        } else { Domain::range(lo, hi) };
        if p != part.len() { return Err(perr("trailing domain tokens")); }
        out.insert(name, dom);
    }
    Ok(())
}

fn binop_prec(tok: Option<&Tok>) -> Option<(u8, &'static str)> {
    let Tok::Sym(s) = tok? else { return None };
    Some(match s.as_str() {
        "==" => (1, "=="), "!=" => (1, "!="), "<" => (1, "<"), "<=" => (1, "<="), ">" => (1, ">"), ">=" => (1, ">="),
        "|" => (2, "|"), "^" => (3, "^"), "&" => (4, "&"), "<<" => (5, "<<"), ">>" => (5, ">>"),
        "+" => (6, "+"), "-" => (6, "-"), "*" => (7, "*"), "/" => (7, "/"), "%" => (7, "%"), _ => return None,
    })
}

struct Lowerer {
    toks: Vec<Tok>, pos: usize, k: Kernel, doms: Vec<Domain>, init_mem: NodeId, mem: NodeId,
    buf_ids: BTreeMap<String, u8>, lane_loaded: BTreeSet<u8>,
    vars: BTreeMap<String, (NodeId, ValTy)>, consts: BTreeMap<String, i64>, final_out: Vec<(u8, u8, NodeId)>,
}

impl Lowerer {
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
    fn eat(&mut self, is_sym: bool, s: &str) -> bool {
        let hit = match self.peek() { Some(Tok::Sym(t)) => is_sym && t == s, Some(Tok::Ident(t)) => !is_sym && t == s, _ => false };
        if hit { self.pos += 1; } hit
    }
    fn eat_sym(&mut self, s: &str) -> bool { self.eat(true, s) }
    fn eat_id(&mut self, s: &str) -> bool { self.eat(false, s) }
    fn req_sym(&mut self, s: &str) -> Result<(), CompileError> { if self.eat_sym(s) { Ok(()) } else { Err(perr(format!("expected '{s}'"))) } }
    fn req_id(&mut self) -> Result<String, CompileError> { match self.peek().cloned() { Some(Tok::Ident(s)) => { self.pos += 1; Ok(s) } _ => Err(perr("expected identifier")) } }
    fn push(&mut self, op: Op<NodeId>) -> NodeId {
        let d = op.eval_domain(|i| self.doms[i], |b| self.k.bufs[b as usize].domain, |i| matches!(self.k.nodes[i], Op::MemInit));
        let id = self.k.push(op); self.doms.push(d); id
    }
    fn record_store(&mut self, buf: u8, lane: u8, val: NodeId) {
        self.mem = self.push(Op::Store { mem: self.mem, buf, lane, val });
        if let Some(slot) = self.final_out.iter_mut().find(|o| o.0 == buf && o.1 == lane) { slot.2 = val; } else { self.final_out.push((buf, lane, val)); }
    }
    fn push_shift(&mut self, base: NodeId, ty: ValTy, amt: NodeId, is_shr: bool, check_ushr: bool) -> Result<NodeId, CompileError> {
        let s = self.doms[amt].is_const().ok_or_else(|| perr("shift amount must be constant"))?;
        if !(0..64).contains(&s) { return Err(CompileError::InvalidShift(s as u8)); }
        if check_ushr && is_shr && ty != ValTy::I64 && !self.doms[base].is_nonneg() { return Err(perr("unsigned right shift on possibly-negative i64")); }
        Ok(self.push(if is_shr { Op::Shr(base, s as u8) } else { Op::Shl(base, s as u8) }))
    }
    fn skip_balanced(&mut self, open: &str, close: &str) -> Result<usize, CompileError> {
        self.req_sym(open)?; let (start, mut depth) = (self.pos, 1usize);
        while depth > 0 {
            if self.pos >= self.toks.len() { return Err(perr("unclosed delimiter")); }
            if self.eat_sym(open) { depth += 1; } else if self.eat_sym(close) { depth -= 1; } else { self.pos += 1; }
        }
        Ok(start)
    }
    fn try_const(&self, start: usize, end: usize) -> Option<i64> {
        let atom = |p: &mut usize| match self.toks.get(*p)? {
            Tok::Int(v) => { *p += 1; Some(*v) } Tok::Ident(s) => { *p += 1; self.consts.get(s).copied() } _ => None,
        };
        let mut p = start; let mut val = atom(&mut p)?;
        while p < end {
            let op = match self.toks.get(p)? { Tok::Sym(s) if s == "+" || s == "-" => { p += 1; s.as_str() } _ => return None };
            let rhs = atom(&mut p)?; val = if op == "+" { val.checked_add(rhs)? } else { val.checked_sub(rhs)? };
        }
        Some(val)
    }
    fn parse_assert(&mut self) -> Result<(Option<String>, NodeId), CompileError> {
        self.req_sym("!")?; self.req_sym("(")?;
        let vname = match (self.peek(), self.toks.get(self.pos + 1)) { (Some(Tok::Ident(s)), Some(Tok::Sym(t))) if t == ">=" => Some(s.clone()), _ => None };
        let (idx, _) = self.parse_prec(2)?; self.req_sym(">=")?;
        if parse_signed(&self.toks, &mut self.pos)? != 0 { return Err(perr("assert! lower bound must be 0")); }
        self.req_sym("&&")?; let _ = self.parse_prec(2)?; self.req_sym("<")?;
        let limit = parse_signed(&self.toks, &mut self.pos)?;
        if limit <= 0 { return Err(perr("assert! limit must be > 0")); }
        self.req_sym(")")?; Ok((vname, self.push(Op::BoundsCheck { idx, limit })))
    }
    fn merge_select(&mut self, cond: NodeId, t_id: NodeId, f_id: NodeId) -> NodeId {
        if t_id == f_id { t_id } else { self.push(Op::Select { cond, t_val: t_id, f_val: f_id }) }
    }
    fn parse_if_stmt(&mut self) -> Result<(), CompileError> {
        let ((cond, _), sv, sm, so) = (self.parse_prec(1)?, self.vars.clone(), self.mem, self.final_out.clone());
        self.parse_block()?; let (tv, to) = (self.vars.clone(), self.final_out.clone());
        self.vars = sv.clone(); self.mem = sm; self.final_out = so.clone();
        if !self.eat_id("else") { return Err(perr("if statement requires else")); }
        if self.eat_id("if") { self.parse_if_stmt()?; } else { self.parse_block()?; }
        let (fv, fo) = (self.vars.clone(), self.final_out.clone());
        self.vars = sv; self.mem = sm; self.final_out = so;
        for (name, &(t_id, ty)) in &tv {
            if let Some(&(f_id, _)) = fv.get(name) { if self.vars.contains_key(name) { let v = self.merge_select(cond, t_id, f_id); self.vars.insert(name.clone(), (v, ty)); } }
        }
        let mut keys: Vec<(u8, u8)> = to.iter().chain(&fo).map(|&(b, l, _)| (b, l)).collect();
        keys.sort_unstable(); keys.dedup();
        for (buf, lane) in keys {
            let t_id = to.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2).unwrap_or_else(|| self.push(Op::Load { mem: sm, buf, lane }));
            let f_id = fo.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2).unwrap_or_else(|| self.push(Op::Load { mem: sm, buf, lane }));
            let old = self.final_out.iter().find(|o| o.0 == buf && o.1 == lane).map(|o| o.2);
            if Some(t_id) != old || Some(f_id) != old { let m = self.merge_select(cond, t_id, f_id); self.record_store(buf, lane, m); }
        }
        Ok(())
    }
    fn parse_block(&mut self) -> Result<(), CompileError> {
        self.req_sym("{")?;
        while !self.eat_sym("}") {
            if self.pos >= self.toks.len() { return Err(perr("unclosed block")); }
            if self.eat_id("for") {
                let var = self.req_id()?; if !self.eat_id("in") { return Err(perr("expected 'in'")); }
                let lo = parse_signed(&self.toks, &mut self.pos)?;
                let incl = if self.eat_sym("..=") { true } else { self.req_sym("..")?; false };
                let hi = parse_signed(&self.toks, &mut self.pos)?;
                let hi_e = if incl { hi.checked_add(1).ok_or_else(|| perr("loop overflow"))? } else { hi };
                if lo < 0 || hi_e < lo || hi_e - lo > MAX_UNROLL { return Err(perr("for-loop bounds must be constant with 0..=16 trips")); }
                let (bpos, mut epos) = (self.pos, self.pos);
                if lo == hi_e { self.skip_balanced("{", "}")?; continue; }
                for v in lo..hi_e {
                    let (sv, sc) = (self.vars.clone(), self.consts.clone());
                    self.consts.insert(var.clone(), v); self.pos = bpos; self.parse_block()?; epos = self.pos; self.vars = sv; self.consts = sc;
                }
                self.pos = epos;
            } else if self.eat_id("if") { self.parse_if_stmt()?; }
            else if self.eat_id("let") {
                let _ = self.eat_id("mut"); let name = self.req_id()?; if self.eat_sym(":") { let _ = self.req_id()?; }
                self.req_sym("=")?; let val = self.parse_prec(1)?; self.req_sym(";")?; self.vars.insert(name, val);
            } else if self.eat_id("assert") {
                let (vname, chk) = self.parse_assert()?; self.req_sym(";")?;
                if let Some(n) = vname { self.vars.insert(n, (chk, ValTy::I64)); }
            } else if let Some(Tok::Ident(name)) = self.peek().cloned() {
                let Some(&buf) = self.buf_ids.get(&name) else { return Err(perr(format!("unknown target '{name}'"))); };
                self.pos += 1; let lstart = self.skip_balanced("[", "]")?;
                let lv = self.try_const(lstart, self.pos - 1).ok_or_else(|| perr("store index must be constant lane"))?;
                if lv < 0 || lv >= LANES as i64 { return Err(CompileError::InvalidLane(lane_u8(lv))); }
                self.req_sym("=")?; let (val, _) = self.parse_prec(1)?; self.req_sym(";")?; self.record_store(buf, lv as u8, val);
            } else { return Err(perr("unexpected token in kernel body")); }
        }
        Ok(())
    }
    fn parse_prec(&mut self, min_p: u8) -> Result<(NodeId, ValTy), CompileError> {
        let mut lhs = self.parse_cast()?;
        while let Some((p, op)) = binop_prec(self.peek()) {
            if p < min_p { break; }
            self.pos += 1; let rhs = self.parse_prec(p + 1)?;
            lhs.0 = match op {
                "==" => self.push(Op::Cmp(CmpKind::Eq, lhs.0, rhs.0)), "!=" => self.push(Op::Cmp(CmpKind::Ne, lhs.0, rhs.0)),
                "<" => self.push(Op::Cmp(CmpKind::Lt, lhs.0, rhs.0)), "<=" => self.push(Op::Cmp(CmpKind::Le, lhs.0, rhs.0)),
                ">" => self.push(Op::Cmp(CmpKind::Gt, lhs.0, rhs.0)), ">=" => self.push(Op::Cmp(CmpKind::Ge, lhs.0, rhs.0)),
                "|" | "^" => match (self.doms[lhs.0].is_const(), self.doms[rhs.0].is_const()) {
                    (Some(a), Some(b)) => self.push(Op::Const(if op == "|" { a | b } else { a ^ b })),
                    _ => return Err(perr(format!("non-constant '{op}' unsupported in kernel IR"))),
                },
                "&" => self.push(Op::BitAnd(lhs.0, rhs.0)), "<<" | ">>" => self.push_shift(lhs.0, lhs.1, rhs.0, op == ">>", true)?,
                "+" => self.push(Op::Add(lhs.0, rhs.0)), "-" => self.push(Op::Sub(lhs.0, rhs.0)),
                "*" => self.push(Op::Mul(lhs.0, rhs.0)), "/" => self.push(Op::Div(lhs.0, rhs.0)), _ => self.push(Op::Rem(lhs.0, rhs.0)),
            };
        }
        Ok(lhs)
    }
    fn parse_cast(&mut self) -> Result<(NodeId, ValTy), CompileError> {
        let mut val = self.parse_primary()?;
        while self.eat_sym(".") {
            let m = self.req_id()?; self.req_sym("(")?; let arg = self.parse_prec(1)?; self.req_sym(")")?;
            val.0 = match m.as_str() {
                "wrapping_add" => self.push(Op::Add(val.0, arg.0)), "wrapping_sub" => self.push(Op::Sub(val.0, arg.0)), "wrapping_mul" => self.push(Op::Mul(val.0, arg.0)),
                "wrapping_shl" | "wrapping_shr" => self.push_shift(val.0, val.1, arg.0, m == "wrapping_shr", false)?,
                _ => return Err(perr(format!("unsupported method '.{m}'"))),
            };
        }
        while self.eat_id("as") {
            val.1 = match self.req_id()?.as_str() {
                "i64" => ValTy::I64, "u64" => ValTy::U64, "usize" => ValTy::Usize,
                "u32" => if self.doms[val.0].fits_u32() { ValTy::U32 } else { return Err(perr("cast 'as u32' requires 0..=u32::MAX")); },
                o => return Err(perr(format!("unsupported cast '{o}'"))),
            };
        }
        Ok(val)
    }
    fn parse_primary(&mut self) -> Result<(NodeId, ValTy), CompileError> {
        if self.eat_sym("-") {
            if let Some(&Tok::Int(v)) = self.peek() { self.pos += 1; return Ok((self.push(Op::Const(v.checked_neg().ok_or_else(|| perr("int overflow"))?)), ValTy::I64)); }
            let (e, ty) = self.parse_primary()?; let z = self.push(Op::Const(0)); return Ok((self.push(Op::Sub(z, e)), ty));
        }
        if let Some(&Tok::Int(v)) = self.peek() { self.pos += 1; return Ok((self.push(Op::Const(v)), ValTy::I64)); }
        if self.eat_sym("(") { let v = self.parse_prec(1)?; self.req_sym(")")?; return Ok(v); }
        if self.eat_id("if") {
            let (cond, _) = self.parse_prec(1)?;
            self.req_sym("{")?; let (t_val, ty) = self.parse_prec(1)?; self.req_sym("}")?;
            if !self.eat_id("else") { return Err(perr("if expr requires else")); }
            let (f_val, _) = if self.peek() == Some(&Tok::Ident("if".into())) { self.parse_primary()? } else { self.req_sym("{")?; let v = self.parse_prec(1)?; self.req_sym("}")?; v };
            return Ok((self.push(Op::Select { cond, t_val, f_val }), ty));
        }
        if self.eat_sym("{") {
            if !self.eat_id("assert") { return Err(perr("block expr must start with assert!")); }
            let (_, chk) = self.parse_assert()?; self.req_sym(";")?; let _ = self.parse_prec(1)?; self.req_sym("}")?; return Ok((chk, ValTy::I64));
        }
        let name = self.req_id()?;
        if name == "true" || name == "false" { return Ok((self.push(Op::Const(i64::from(name == "true"))), ValTy::I64)); }
        if let Some(&buf) = self.buf_ids.get(&name) {
            let (saved, istart) = (self.pos, self.skip_balanced("[", "]")?);
            if !matches!(self.k.bufs[buf as usize].access, BufAccess::LookupTable(_)) {
                if let Some(lv) = self.try_const(istart, self.pos - 1) {
                    if lv < 0 || lv >= LANES as i64 { return Err(CompileError::InvalidLane(lane_u8(lv))); }
                    self.lane_loaded.insert(buf);
                    let m = if self.k.bufs[buf as usize].access == BufAccess::ReadOnly { self.init_mem } else { self.mem };
                    return Ok((self.push(Op::Load { mem: m, buf, lane: lv as u8 }), ValTy::I64));
                }
                if self.k.bufs[buf as usize].access == BufAccess::ReadOnly && !self.lane_loaded.contains(&buf) { self.k.bufs[buf as usize].access = BufAccess::LookupTable(LANES); }
            }
            self.pos = saved; self.req_sym("[")?; let (idx, _) = self.parse_prec(1)?; self.req_sym("]")?;
            return Ok((self.push(Op::LoadLut { buf, idx }), ValTy::I64));
        }
        if let Some(&v) = self.vars.get(&name) { return Ok(v); }
        if let Some(&c) = self.consts.get(&name) { return Ok((self.push(Op::Const(c)), ValTy::I64)); }
        Err(perr(format!("unknown identifier '{name}'")))
    }
}

/// Parse a raw safe-Rust `.rs` kernel function into `Kernel` IR.
pub fn parse_rust_kernel(src: &str) -> Result<Kernel, CompileError> {
    let (mut domains, mut low) = (BTreeMap::new(), Lowerer {
        toks: lex_rust(src)?, pos: 0, k: Kernel::new(""), doms: Vec::new(), init_mem: 0, mem: 0,
        buf_ids: BTreeMap::new(), lane_loaded: BTreeSet::new(), vars: BTreeMap::new(), consts: BTreeMap::new(), final_out: Vec::new(),
    });
    while let Some(tok) = low.peek().cloned() {
        match tok {
            Tok::DocDomain(s) => { parse_domain_list(&lex_rust(&s)?, &mut domains)?; low.pos += 1; }
            Tok::Sym(s) if s == "#!" || s == "#" => {
                low.pos += 1; let astart = low.skip_balanced("[", "]")?; let attr = &low.toks[astart..low.pos - 1];
                if attr.first() == Some(&Tok::Ident("zero_domain".into())) && attr.len() > 2 { parse_domain_list(&attr[1..], &mut domains)?; }
            }
            _ => break,
        }
    }
    let _ = low.eat_id("pub"); if !low.eat_id("fn") { return Err(perr("expected 'fn'")); }
    low.k.name = low.req_id()?; low.req_sym("(")?;
    while !low.eat_sym(")") {
        let pname = low.req_id()?; low.req_sym(":")?; low.req_sym("&")?; let is_mut = low.eat_id("mut"); low.req_sym("[")?;
        if !low.eat_id("i64") { return Err(perr("param element type must be i64")); }
        low.req_sym(";")?;
        let len = match low.peek() { Some(&Tok::Int(v)) if v > 0 => { low.pos += 1; v as usize } _ => return Err(perr("expected array len")) };
        low.req_sym("]")?;
        let access = if is_mut { if len != LANES { return Err(perr("&mut buffer must have 8 lanes")); } BufAccess::MutNoAlias } else if len == LANES { BufAccess::ReadOnly } else { BufAccess::LookupTable(len) };
        let id = low.k.add_buf(pname.clone(), access, domains.get(&pname).copied().unwrap_or_else(Domain::top));
        low.buf_ids.insert(pname, id); if !low.eat_sym(",") { low.req_sym(")")?; break; }
    }
    low.init_mem = low.push(Op::MemInit); low.mem = low.init_mem; low.parse_block()?;
    for (buf, lane, val) in low.final_out { low.k.add_output(buf, lane, val); }
    low.k.validate()?; Ok(low.k)
}

/// Compile a raw safe-Rust `.rs` kernel source directly through E-Graph + zero-spill codegen.
pub fn compile_rust_source(src: &str, budget: RegBudget) -> Result<CompiledKernel, CompileError> { compile_kernel(&parse_rust_kernel(src)?, budget) }
