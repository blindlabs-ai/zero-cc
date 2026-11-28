use std::{collections::{BTreeMap, BTreeSet}, fs, mem::{replace, take}, path::{Path, PathBuf}};
use crate::ir::{barrett_params, BufAccess as Ba, CmpKind as Ck, CompileError, Domain, ElemTy as Et, Kernel, KernelCost};
use crate::{codegen::RegBudget, compile_kernel, ir::{NodeId as Nid, Op, LANES}, isa::{compile_kernel_direct,
  compile_standalone_elf_exec, X86VectorMode}, elf64::{verify_elf64_executable, verify_elf64_object, TargetIsa}, CompiledKernel};
type Res<T> = Result<T, CompileError>;
type Vals = Vec<(Nid, Vt)>;
type CMap = BTreeMap<String, i64>;
type DMap = BTreeMap<String, Domain>;
type DOvr = BTreeMap<String, DMap>;
type HMap = BTreeMap<String, HelperFn>;
#[derive(Clone, Debug, PartialEq, Eq)] enum Tok {
  Id(String), Int(i64, Option<Vt>), Float(u32), Doc(String), Sy(String), Other }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] enum Vt { Bool, I64, I32, I16, U64, U32, U16, U8, Usize, F32 }
#[derive(Clone, Copy, PartialEq, Eq)] enum Ev { N(Nid, Vt), Xor(Nid, Nid, Vt), Mxr(Nid, Nid, Nid, Vt) }
#[derive(Clone)] struct HelperFn { params: Vec<(String, Et)>, body: Vec<Tok> }
#[derive(Clone, Debug)] pub struct ExtractedKernel { pub file: String, pub name: String, pub kernel: Kernel }
#[derive(Clone, Debug)] pub struct CompiledCrateKernel { pub file: String, pub name: String,
  pub original_cost: KernelCost, pub optimized_cost: KernelCost, pub vector_mode: X86VectorMode, pub max_lanes: u8,
  pub direct_insns: usize, pub direct_share_pct: u8, pub exec_share_pct: u8, pub zero_spills: bool,
  pub obj_bytes: usize, pub exec_bytes: usize }
#[derive(Clone, Debug)] pub struct CrateCompileReport { pub root: String, pub files_scanned: usize,
  pub functions_seen: usize, pub kernels_extracted: usize, pub compiled: Vec<CompiledCrateKernel>, pub skipped: usize }
const SCALARS: &[(&str, Et, Vt, u8, bool, Vt)] = &[("i64", Et::I64, Vt::I64, 64, true, Vt::U64),
  ("u64", Et::U64, Vt::U64, 64, false, Vt::U64), ("i32", Et::I32, Vt::I32, 32, true, Vt::U32),
  ("u32", Et::U32, Vt::U32, 32, false, Vt::U32), ("i16", Et::I16, Vt::I16, 16, true, Vt::U16),
  ("u16", Et::U16, Vt::U16, 16, false, Vt::U16), ("u8", Et::U8, Vt::U8, 8, false, Vt::U8),
  ("usize", Et::I64, Vt::Usize, 64, false, Vt::Usize), ("bool", Et::U8, Vt::Bool, 1, false, Vt::Bool),
  ("f32", Et::F32, Vt::F32, 32, true, Vt::F32)];
const BINOPS: &[(&str, u8)] = &[("||", 1), ("&&", 2), ("==", 3), ("!=", 3), ("<", 3), ("<=", 3), (">", 3), (">=", 3),
  ("|", 4), ("^", 5), ("&", 6), ("<<", 7), (">>", 7), ("+", 8), ("-", 8), ("*", 9), ("/", 9), ("%", 9)];
const COMP: &[&str] = &["+=", "-=", "*=", "&=", "|=", "^=", "<<=", ">>="];
const CMPS: &[(&str, Ck, Ck)] = &[("==", Ck::Eq, Ck::Eq), ("!=", Ck::Ne, Ck::Ne), ("<", Ck::Lt, Ck::Ult),
  ("<=", Ck::Le, Ck::Ule), (">", Ck::Gt, Ck::Ugt), (">=", Ck::Ge, Ck::Uge)];
const MULTI_SYMS: &[&str] = &["<<=", ">>=", "..=", "+=", "-=", "*=", "&=", "|=", "^=", "..", "==", "!=", ">=", "<=",
  "&&", "||", "<<", ">>", "->", "::", "#!"];
impl Vt {
  fn from_elem(e: Et) -> Self { SCALARS.iter().find(|s| s.1 == e).map_or(Self::I64, |s| s.2) }
  fn inf(self) -> (u8, bool, Vt) { SCALARS.iter().find(|s| s.2 == self).map_or((64, false, self), |s| (s.3, s.4, s.5)) }
  const fn u64_cmp(self) -> bool { matches!(self, Self::U64 | Self::Usize) } }
fn perr(msg: impl Into<String>) -> CompileError { CompileError::ParseError(msg.into()) }
fn skip_w(b: &[u8], i: usize, p: impl Fn(u8) -> bool) -> usize {
  b.get(i..).and_then(|s| s.iter().position(|&c| !p(c))).map_or(b.len(), |k| i + k) }
fn sym(t: Option<&Tok>) -> Option<&str> { if let Some(Tok::Sy(s)) = t { Some(s) } else { None } }
fn id_s(t: Option<&Tok>) -> Option<&str> { if let Some(Tok::Id(s)) = t { Some(s) } else { None } }
fn is01(d: Domain) -> bool { !d.is_empty() && d.lo >= 0 && d.hi <= 1 }
fn chk_lane(lv: i64, m: usize) -> Res<u8> {
  if lv >= 0 && lv < m as i64 { Ok(lv as u8) } else { Err(CompileError::InvalidLane(lv.try_into().unwrap_or(255))) } }
fn cmp_k(op: &str, u: bool) -> Option<Ck> { CMPS.iter().find(|c| c.0 == op).map(|c| if u { c.2 } else { c.1 }) }
fn comp_op(t: Option<&Tok>) -> Option<&'static str> { COMP.iter().find(|&&c| Some(c) == sym(t))?.strip_suffix('=') }
fn one_val(v: Vals, w: &str) -> Res<Ev> { if v.len() == 1 { Ok(Ev::N(v[0].0, v[0].1)) } else { Err(perr(w)) } }
fn scalar_elem(name: &str) -> Option<(Et, Vt, Domain)> {
  let &(_, e, v, _, _, _) = SCALARS.iter().find(|s| s.0 == name)?;
  Some((e, v, if v == Vt::Usize { Domain::range(0, i64::MAX) }
    else if v == Vt::Bool { Domain::range(0, 1) } else { e.natural_domain() })) }
fn lex_rust(src: &str, strict: bool) -> Res<Vec<Tok>> {
  let cl: String = if strict { "".into() } else { src.chars().map(|c| if c.is_ascii() { c } else { ' ' }).collect() };
  let text = if strict { src } else { cl.as_str() };
  if !text.is_ascii() { return Err(perr("non-ASCII Rust source")) }
  let (b, n, mut i, mut out) = (text.as_bytes(), text.len(), 0usize, Vec::new());
  while i < n {
    i = skip_w(b, i, |c| c.is_ascii_whitespace());
    if i >= n { break }
    let (tok, nx) = if b[i..].starts_with(b"//") {
      let e = skip_w(b, i + 2, |c| c != b'\n');
      let d = text[i..e].strip_prefix("///").map(str::trim)
        .and_then(|t| t.strip_prefix("@domain").or_else(|| t.strip_prefix("@zero_domain")));
      (d.map(|s| Tok::Doc(s.trim().into())), e)
    } else if b[i..].starts_with(b"/*") {
      let (mut j, mut d) = (i + 2, 1isize);
      while j + 1 < n && d > 0 {
        let (dd, st) = if &b[j..j + 2] == b"/*" { (1, 2) } else if &b[j..j + 2] == b"*/" { (-1, 2) } else { (0, 1) };
        (d, j) = (d + dd, j + st); }
      (None, if d > 0 { if strict { return Err(perr("unterminated comment")) } else { n } } else { j })
    } else if !strict && (b[i] == b'"' || (b[i] == b'\'' && i + 2 < n && (b[i + 1] == b'\\' || b[i + 2] == b'\''))) {
      let mut e = if b[i] == b'"' { i + 1 }
        else if b[i + 1] == b'\\' { skip_w(b, (i + 3).min(n), |c| c != b'\'') } else { i + 2 };
      if b[i] == b'"' { while e < n && b[e] != b'"' { e = (e + if b[e] == b'\\' { 2 } else { 1 }).min(n) } }
      (Some(Tok::Other), (e + 1).min(n))
    } else if b[i].is_ascii_alphabetic() || b[i] == b'_' {
      let e = skip_w(b, i, |c| c.is_ascii_alphanumeric() || c == b'_');
      (Some(Tok::Id(text[i..e].into())), e)
    } else if b[i].is_ascii_digit() {
      let r = if b[i] == b'0' { [16u32, 2, 8].iter().zip(b"xbo")
        .find(|(_, k)| b.get(i + 1).is_some_and(|x| x.eq_ignore_ascii_case(k))).map_or(10, |(&r, _)| r) } else { 10 };
      let st = i + usize::from(r != 10) * 2;
      let end = skip_w(b, st, |c| c == b'_' || (c as char).is_digit(r));
      let is_dot_f = r == 10 && b.get(end) == Some(&b'.') && b.get(end + 1).is_some_and(|&c| c.is_ascii_digit());
      let is_exp_f = r == 10 && b.get(end).is_some_and(|&c| c == b'e' || c == b'E');
      let is_f32_suf = r == 10 && text.get(end..).is_some_and(|t| t.trim_start_matches('_').starts_with("f32"));
      if is_dot_f || is_exp_f || is_f32_suf {
        let mut fe = end;
        if is_dot_f { fe = skip_w(b, end + 1, |c| c == b'_' || c.is_ascii_digit()); }
        if b.get(fe).is_some_and(|&c| c == b'e' || c == b'E') {
          fe += 1;
          if b.get(fe).is_some_and(|&c| c == b'+' || c == b'-') { fe += 1; }
          fe = skip_w(b, fe, |c| c == b'_' || c.is_ascii_digit());
        }
        let nx = skip_w(b, fe, |c| c.is_ascii_alphanumeric() || c == b'_');
        let suf = text[fe..nx].trim_start_matches('_');
        let val = (suf.is_empty() || suf == "f32").then(|| text[i..fe].replace('_', "").parse::<f32>().ok()).flatten();
        (Some(if let Some(fv) = val { Tok::Float(fv.to_bits()) }
          else if strict { return Err(perr("bad float")) } else { Tok::Other }), nx)
      } else {
        let nx = skip_w(b, end, |c| c.is_ascii_alphanumeric() || c == b'_');
        let suf = text[end..nx].trim_start_matches('_');
        let sty = if suf.is_empty() { Some(None) }
          else { SCALARS.iter().find(|s| s.0 == suf && s.2 != Vt::Bool && s.2 != Vt::F32).map(|s| Some(s.2)) };
        let val = u64::from_str_radix(&text[st..end].replace('_', ""), r).ok().map(|v| v as i64);
        (Some(if let (Some(st), Some(v)) = (sty, val) { Tok::Int(v, st) }
          else if strict { return Err(perr("bad int")) } else { Tok::Other }), nx)
      }
    } else {
      let step = MULTI_SYMS.iter().find(|s| b[i..].starts_with(s.as_bytes())).map_or(1, |s| s.len());
      (Some(Tok::Sy(text[i..i + step].into())), i + step) };
    out.extend(tok);
    i = nx; }
  Ok(out) }
fn parse_signed(toks: &[Tok], p: &mut usize, skip: usize) -> Res<i64> {
  let neg = sym(toks.get(*p + skip)) == Some("-");
  let Some(&Tok::Int(v, _)) = toks.get(*p + skip + usize::from(neg)) else { return Err(perr("expected int")) };
  *p += skip + 1 + usize::from(neg);
  if neg { v.checked_neg().ok_or_else(|| perr("int ovf")) } else { Ok(v) } }
fn parse_domain_list(ts: &[Tok], out: &mut DMap) -> Res<()> {
  let sl = if sym(ts.first()) == Some("(") && sym(ts.last()) == Some(")") { &ts[1..ts.len() - 1] } else { ts };
  for part in sl.split(|t| *t == Tok::Sy(",".into())).filter(|p| !p.is_empty()) {
    let Some(Tok::Id(name)) = part.first().filter(|_| matches!(sym(part.get(1)), Some(":" | "=")))
      else { return Err(perr("bad domain")) };
    let mut p = 2usize;
    let lo = parse_signed(part, &mut p, 0)?;
    let hi = if sym(part.get(p)) == Some("..=") { parse_signed(part, &mut p, 1)? } else { return Err(perr("..=")) };
    let dom = if id_s(part.get(p)) == Some("mod") {
      let m = parse_signed(part, &mut p, 1)?;
      if m <= 0 || sym(part.get(p)) != Some("==") { return Err(perr("bad mod")) }
      Domain::congruent(lo, hi, m, parse_signed(part, &mut p, 1)?)
    } else { Domain::range(lo, hi) };
    if lo > hi || p != part.len() { return Err(perr("bad domain")) }
    out.insert(name.clone(), dom); }
  Ok(()) }
pub fn parse_domain_spec(spec: &str) -> Res<DMap> {
  let mut out = BTreeMap::new();
  if !spec.trim().is_empty() { parse_domain_list(&lex_rust(spec, true)?, &mut out)? }
  Ok(out) }
fn disjoint_or_pow2(da: Domain, db: Domain) -> bool {
  if da.is_empty() || db.is_empty() || !db.is_nonneg() || db.hi == i64::MAX { return false }
  let m = ((db.hi as u64) + 1).next_power_of_two();
  db.hi == 0 || ((2..=(1u64 << 48)).contains(&m) && da.rem_mod(m as i64) == Some(0)) }
fn refine_clamp(k: &Kernel, doms: &[Domain], op: Op<Nid>, mut d: Domain) -> Domain {
  let Op::Select { cond, t_val, f_val } = op else { return d };
  let Some(&Op::Cmp(ck, a, b)) = k.nodes.get(cond) else { return d };
  let Some(c) = doms.get(b).copied().and_then(Domain::is_const) else { return d };
  let flip = if t_val == b && f_val == a { false } else if t_val == a && f_val == b { true } else { return d };
  let up = if matches!(ck, Ck::Gt | Ck::Ge) { !flip } else if matches!(ck, Ck::Lt | Ck::Le) { flip } else { return d };
  if up && d.hi > c { d.hi = c } else if !up && d.lo < c { d.lo = c } else { return d }
  Domain { modulus: 1, rem: 0, ..d } }
struct Lowerer {
  toks: Vec<Tok>, pos: usize, k: Kernel, doms: Vec<Domain>, mem: Nid, bufs: BTreeMap<String, u8>,
  sbufs: BTreeMap<u8, Option<u8>>, loaded: BTreeSet<u8>, vars: BTreeMap<String, (Nid, Vt)>, consts: CMap,
  outs: BTreeMap<(u8, u8), Nid>, helpers: HMap, bmasks: BTreeMap<Nid, (Nid, u64)>, inl_d: usize, depth: usize }
impl Lowerer {
  fn new(toks: Vec<Tok>, helpers: HMap, consts: CMap) -> Self {
    Self { toks, pos: 0, k: Kernel::new(""), doms: Vec::new(), mem: 0, bufs: BTreeMap::new(), sbufs: BTreeMap::new(),
      loaded: BTreeSet::new(), vars: BTreeMap::new(), consts, outs: BTreeMap::new(), helpers, bmasks: BTreeMap::new(),
      inl_d: 0, depth: 0 } }
  fn with_d<T>(&mut self, f: impl FnOnce(&mut Self) -> Res<T>) -> Res<T> {
    if self.depth >= 128 { return Err(perr("depth")) }
    self.depth += 1;
    let r = f(self);
    self.depth -= 1;
    r }
  fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
  fn pid(&self) -> Option<&str> { id_s(self.peek()) }
  fn skip(&mut self, n: usize) -> &mut Self { if n > 0 { self.pos += n } self }
  fn adv<T>(&mut self, n: usize, v: T) -> T { if n > 0 { self.pos += n } v }
  fn is_s(&self, o: usize, s: &str) -> bool { sym(self.toks.get(self.pos + o)) == Some(s) }
  fn eat_s(&mut self, s: &str) -> bool { if self.is_s(0, s) { self.adv(1, true) } else { false } }
  fn eat_i(&mut self, s: &str) -> bool { if self.pid() == Some(s) { self.adv(1, true) } else { false } }
  fn opt_i(&mut self, s: &str) -> &mut Self { if self.pid() == Some(s) { self.pos += 1 } self }
  fn rs(&mut self, s: &str) -> Res<&mut Self> { if self.eat_s(s) { Ok(self) } else { Err(perr(s)) } }
  fn rk(&mut self, s: &str) -> Res<&mut Self> { if self.eat_i(s) { Ok(self) } else { Err(perr(s)) } }
  fn id(&mut self) -> Res<String> { self.pid().map(str::to_owned).map(|s| self.adv(1, s)).ok_or_else(|| perr("id")) }
  fn skip_pub(&mut self) -> Res<&mut Self> {
    if self.eat_i("pub") && self.is_s(0, "(") { self.bal("(", ")").map(|_| self) } else { Ok(self) } }
  fn dom(&self, id: Nid) -> Domain { self.doms.get(id).copied().unwrap_or_else(Domain::top) }
  fn touch(&mut self, b: u8, l: u8) { if let Some(s) = self.sbufs.get_mut(&b) { *s = Some(s.map_or(l, |c| c.max(l))) } }
  fn p(&mut self, op: Op<Nid>) -> Nid {
    let raw = op.eval_domain(|i| self.dom(i), |b| self.k.bufs.get(b as usize).map_or_else(Domain::top, |d| d.domain),
      |i| matches!(self.k.nodes.get(i), Some(Op::MemInit)));
    let (d, id) = (refine_clamp(&self.k, &self.doms, op, raw), self.k.push(op));
    self.doms.push(d);
    if let Op::Sub(a, b) = op {
      if self.dom(a).is_const() == Some(0) && is01(self.dom(b)) { self.bmasks.extend([(id, (b, u64::MAX))]) } }
    id }
  fn c(&mut self, v: i64) -> Nid { self.p(Op::Const(v)) }
  fn pc(&mut self, f: impl FnOnce(Nid) -> Op<Nid>, v: i64) -> Nid { self.po(Op::Const(v), f, None) }
  fn po(&mut self, a: Op<Nid>, f: impl FnOnce(Nid) -> Op<Nid>, ty: Option<Vt>) -> Nid {
    let op = f(self.p(a));
    if let Some(t) = ty { self.pm(op, t) } else { self.p(op) } }
  fn pm(&mut self, op: Op<Nid>, ty: Vt) -> Nid {
    let v = self.p(op);
    self.mty(v, ty) }
  fn ushr(&self, v: Nid, s: u8) -> Op<Nid> { if !self.dom(v).is_nonneg() { Op::Ushr(v, s) } else { Op::Shr(v, s) } }
  fn mty(&mut self, n: Nid, ty: Vt) -> Nid {
    if ty == Vt::F32 { return n }
    let (d, (b, sgn, _)) = (self.dom(n), ty.inf());
    if b == 64 { return n }
    if !sgn {
      let lim = (1i64 << b) - 1;
      if !d.is_empty() && d.is_nonneg() && d.hi <= lim { return n }
      if let Some(c) = d.is_const() { return self.c(c & lim) }
      let (bm, r) = (self.bmasks.get(&n).copied(), self.pc(|m| Op::BitAnd(n, m), lim));
      self.bmasks.extend(bm.map(|(c, mb)| (r, (c, mb & (lim as u64)))));
      return r; }
    let (sh, lo) = (64 - b, i64::MIN >> (64 - b));
    if !d.is_empty() && d.lo >= lo && d.hi <= !lo { n } else if let Some(c) = d.is_const() { self.c((c << sh) >> sh) }
    else { self.po(Op::Shl(n, sh), |s| Op::Shr(s, sh), None) } }
  fn msh(&mut self, amt: Nid, ty: Vt) -> Nid {
    let (m, d) = (i64::from(ty.inf().0) - 1, self.dom(amt));
    if !d.is_empty() && d.is_nonneg() && d.hi <= m { amt }
    else if let Some(c) = d.is_const() { self.c(c & m) } else { self.pc(|mc| Op::BitAnd(amt, mc), m) } }
  fn zext(&mut self, n: Nid, t: Vt) -> Nid { if matches!(t, Vt::I32 | Vt::I16) { self.mty(n, t.inf().2) } else { n } }
  fn as_b(&mut self, id: Nid) -> Nid { if is01(self.dom(id)) { id } else { self.pc(|z| Op::Cmp(Ck::Ne, id, z), 0) } }
  fn sel(&mut self, c: Nid, t: Nid, f: Nid) -> Nid {
    if t == f { t } else { self.p(Op::Select { cond: c, t_val: t, f_val: f }) } }
  fn store(&mut self, buf: u8, lane: u8, val: Nid) {
    self.mem = self.p(Op::Store { mem: self.mem, buf, lane, val });
    self.outs.insert((buf, lane), val); }
  fn real(&mut self, ev: Ev) -> (Nid, Vt) {
    let z = if matches!(ev, Ev::Mxr(..)) { self.c(0) } else { 0 };
    match ev { Ev::N(n, t) => (n, t),
      Ev::Xor(a, b, t) => (self.p(Op::BitXor(a, b)), t),
      Ev::Mxr(a, b, c, t) => (self.po(Op::BitXor(a, b), |x| Op::Select { cond: c, t_val: x, f_val: z }, None), t) } }
  fn node(&mut self, min_p: u8) -> Res<(Nid, Vt)> { self.prec(min_p).map(|e| self.real(e)) }
  fn pnode(&mut self) -> Res<(Nid, Vt)> { self.prim().map(|e| self.real(e)) }
  fn shift(&mut self, base: Nid, ty: Vt, amt: Nid, shr: bool, wrap: bool) -> Res<Nid> {
    let (sgn, a) = (ty.inf().1, if wrap || ty.inf().0 < 64 { self.msh(amt, ty) } else { amt });
    if let Some(s) = self.dom(a).is_const() {
      if !(0..64).contains(&s) { return Err(CompileError::InvalidShift(s as u8)) }
      return Ok(if !shr { self.pm(Op::Shl(base, s as u8), ty) }
        else { self.p(if !sgn { self.ushr(base, s as u8) } else { Op::Shr(base, s as u8) }) }); }
    if !shr { Ok(self.pm(Op::ShlVar(base, a), ty)) }
    else if !sgn { Ok(self.p(Op::ShrVarU(base, a))) } else { Err(perr("signed var shr")) } }
  fn div(&mut self, lhs: Nid, rhs: Nid) -> Nid {
    let (dx, d) = (self.dom(lhs), self.dom(rhs).is_const().unwrap_or(0));
    let ma = dx.lo.unsigned_abs().max(dx.hi.unsigned_abs());
    let ok = d >= 3 && !(d as u64).is_power_of_two() && !dx.is_empty() && dx.lo < 0 && ma <= i64::MAX as u64;
    if let Some((m, s)) = ok.then(|| barrett_params(d, ma as i64)).flatten() {
      let (p, sg) = (self.pc(|mc| Op::Mul(lhs, mc), m), self.p(Op::Shr(lhs, 63)));
      return self.po(Op::Shr(p, s), |q| Op::Sub(q, sg), None); }
    self.p(Op::Div(lhs, rhs)) }
  fn ovf(&mut self, lhs: Nid, rhs: Nid, ty: Vt, sub: bool) -> (Nid, Nid, Option<Nid>) {
    if sub { return (self.p(Op::Sub(lhs, rhs)), self.p(Op::Cmp(Ck::Ult, lhs, rhs)), None) }
    let (raw, (b, sgn, _)) = (self.p(Op::Add(lhs, rhs)), ty.inf());
    let lc = (b < 64 && !sgn).then(|| self.c((1i64 << b) - 1));
    (raw, self.p(if let Some(l) = lc { Op::Cmp(Ck::Gt, raw, l) } else { Op::Cmp(Ck::Ult, raw, lhs) }), lc) }
  fn ovf_call(&mut self, lhs: Nid, ty: Vt, sub: bool) -> Res<(Nid, Nid)> {
    let rhs = self.delim("(", ")", |s| s.node(1))?.0;
    let (raw, ovf, _) = self.ovf(lhs, rhs, ty, sub);
    Ok((self.mty(raw, ty), ovf)) }
  fn bal(&mut self, o: &str, c: &str) -> Res<usize> {
    let st = self.rs(o)?.pos;
    self.pos = bal_toks(&self.toks, st - 1, o, c).ok_or_else(|| perr("unclosed"))? + 1;
    Ok(st) }
  fn delim<T>(&mut self, o: &str, c: &str, f: impl FnOnce(&mut Self) -> Res<T>) -> Res<T> {
    f(self.rs(o)?).and_then(|v| self.rs(c).map(|_| v)) }
  fn list<T>(&mut self, mut f: impl FnMut(&mut Self) -> Res<T>) -> Res<Vec<T>> {
    let mut out = Vec::new();
    while !self.eat_s(")") {
      out.push(f(self)?);
      if !self.eat_s(",") { return self.rs(")").map(|_| out) } }
    Ok(out) }
  fn sty(&mut self) -> Res<(Et, Vt, Domain)> { scalar_elem(&self.id()?).ok_or_else(|| perr("bad scalar ty")) }
  fn ret_tys(&mut self) -> Res<Vec<(Et, Domain)>> {
    if self.rs("->")?.eat_s("(") { self.list(|s| s.sty().map(|(e, _, d)| (e, d))) }
    else { self.sty().map(|(e, _, d)| vec![(e, d)]) } }
  fn try_const(&self, p: usize, end: usize) -> Option<i64> {
    let get = |q: usize| if let Some(&Tok::Int(v, _)) = self.toks.get(q) { Some(v) }
      else { id_s(self.toks.get(q)).and_then(|s| self.consts.get(s).copied()) };
    let mut val = get(p)?;
    for q in (p + 1..end).step_by(2) {
      let (op, r) = (sym(self.toks.get(q))?, (q + 1 < end).then(|| get(q + 1)).flatten()?);
      val = if op == "+" { val.checked_add(r)? } else if op == "-" { val.checked_sub(r)? }
        else if op == "*" { val.checked_mul(r)? } else { return None }; }
    Some(val) }
  fn s_int(&mut self) -> Res<i64> { parse_signed(&self.toks, &mut self.pos, 0) }
  fn parse_assert(&mut self) -> Res<(Option<String>, Nid)> { self.rs("!")?.delim("(", ")", |s| {
    let (vn, idx) = (s.pid().filter(|_| s.is_s(1, ">=")).map(str::to_owned), s.node(4)?.0);
    if s.rs(">=")?.s_int()? != 0 { return Err(perr("assert lb")) }
    let lim = s.rs("&&")?.node(4).and_then(|_| s.rs("<")?.s_int())?;
    if lim <= 0 { Err(perr("assert lim")) } else { Ok((vn, s.p(Op::BoundsCheck { idx, limit: lim }))) } }) }
  fn merge_v(&mut self, c: Nid, tv: Vals, fv: Vals, inv: bool) -> Res<Vals> {
    if tv.len() != fv.len() { return Err(perr("arity")) }
    Ok(tv.into_iter().zip(fv).map(|((t, ty), (f, _))|
      (if inv { self.sel(c, f, t) } else { self.sel(c, t, f) }, ty)).collect()) }
  fn with_n<T>(&mut self, cond: Nid, is_t: bool, f: impl FnOnce(&mut Self) -> Res<T>) -> Res<T> {
    let saved = (|| {
      let Op::Cmp(k, a, b) = *self.k.nodes.get(cond)? else { return None };
      let (c, old) = (self.dom(b).is_const()?, self.dom(a));
      let (lo, hi) = if matches!((k, is_t), (Ck::Lt | Ck::Le, true) | (Ck::Gt, false)) {
        (old.lo, old.hi.min(c.saturating_sub(i64::from(k == Ck::Lt))))
      } else if matches!((k, is_t), (Ck::Gt | Ck::Ge, true) | (Ck::Lt, false)) {
        (old.lo.max(c.saturating_add(i64::from(k == Ck::Gt))), old.hi)
      } else { return None };
      if lo > hi || (lo, hi) == (old.lo, old.hi) { return None }
      *self.doms.get_mut(a)? = Domain::range(lo, hi);
      Some((a, old)) })();
    let res = f(self);
    if let Some((id, d)) = saved.filter(|(i, _)| *i < self.doms.len()) { self.doms[id] = d }
    res }
  fn if_stmt(&mut self) -> Res<()> { self.with_d(|s| {
    let (cond, sv, sm, so) = (s.node(1)?.0, s.vars.clone(), s.mem, s.outs.clone());
    s.with_n(cond, true, Self::blk)?;
    let (tv, to, _) = (replace(&mut s.vars, sv.clone()), replace(&mut s.outs, so.clone()), replace(&mut s.mem, sm));
    s.rk("else")?.with_n(cond, false, |s| if s.eat_i("if") { s.if_stmt() } else { s.blk() })?;
    let (fv, fo, _) = (replace(&mut s.vars, sv), replace(&mut s.outs, so), replace(&mut s.mem, sm));
    for (name, &(tid, ty)) in &tv {
      if let Some(&(fid, _)) = fv.get(name).filter(|_| s.vars.contains_key(name)) {
        let v = s.sel(cond, tid, fid);
        s.vars.extend([(name.clone(), (v, ty))]) } }
    for (b, l) in to.keys().chain(fo.keys()).copied().collect::<BTreeSet<_>>() {
      let t = to.get(&(b, l)).copied().unwrap_or_else(|| s.p(Op::Load { mem: sm, buf: b, lane: l }));
      let f = fo.get(&(b, l)).copied().unwrap_or_else(|| s.p(Op::Load { mem: sm, buf: b, lane: l }));
      if t != f || Some(t) != s.outs.get(&(b, l)).copied() {
        let m = s.sel(cond, t, f);
        s.store(b, l, m) } }
    Ok(()) }) }
  fn else_v(&mut self) -> Res<Vals> { if self.eat_i("if") { self.if_vals() } else { self.eblk() } }
  fn if_vals(&mut self) -> Res<Vals> { self.with_d(|s| {
    if s.eat_i("let") {
      let (bind, (lhs, lty), m) = (s.rk("Some")?.delim("(", ")", Self::id)?, s.rs("=")?.pnode()?, s.rs(".")?.id()?);
      if !matches!(m.as_str(), "checked_add" | "checked_sub") { return Err(perr("bad if-let")) }
      let ((val, ovf), saved) = (s.ovf_call(lhs, lty, m == "checked_sub")?, s.vars.clone());
      s.vars.insert(bind, (val, lty));
      let (tv, _) = (s.eblk()?, replace(&mut s.vars, saved));
      return s.rk("else")?.else_v().and_then(|fv| s.merge_v(ovf, tv, fv, true)); }
    let cond = s.node(1)?.0;
    let tv = s.with_n(cond, true, Self::eblk).and_then(|v| if v.is_empty() { Err(perr("empty if")) } else { Ok(v) })?;
    s.rk("else")?.with_n(cond, false, Self::else_v).and_then(|fv| s.merge_v(cond, tv, fv, false)) }) }
  fn trhs(&mut self) -> Res<Vals> {
    if self.eat_i("if") { return self.if_vals() }
    if let Some(h) = self.pid().filter(|_| self.is_s(1, "(")).and_then(|n| self.helpers.get(n).cloned()) {
      return self.skip(2).list(|s| s.node(1)).and_then(|a| self.inline_fn(h, a)); }
    let saved = self.pos;
    if self.eat_s("(") {
      let first = self.node(1)?;
      if self.eat_s(",") { return Ok([vec![first], self.list(|s| s.node(1))?].concat()) }
      self.pos = saved; }
    if let Some(ev) = self.prim().ok()
      .filter(|_| self.eat_s(".") && matches!(self.pid(), Some("overflowing_add" | "overflowing_sub"))) {
      let (sub, (lhs, ty)) = (self.id()? == "overflowing_sub", self.real(ev));
      return self.ovf_call(lhs, ty, sub).map(|(v, o)| vec![(v, ty), (o, Vt::Bool)]); }
    self.pos = saved;
    Ok(vec![self.node(1)?]) }
  fn let_stmt(&mut self) -> Res<()> {
    if self.eat_s("(") {
      let names = self.list(|s| s.opt_i("mut").id())?;
      if self.eat_s(":") { self.bal("(", ")")?; }
      let vals = self.rs("=")?.trhs()?;
      if self.rs(";").is_err() || vals.len() != names.len() { return Err(perr("tuple let")) }
      self.vars.extend(names.into_iter().zip(vals).filter(|(n, _)| n != "_"))
    } else {
      let name = self.opt_i("mut").id()?;
      let ann = if self.eat_s(":") { scalar_elem(&self.id()?).map(|x| x.1) } else { None };
      let (vn, vt) = self.rs("=")?.node(1).and_then(|v| self.rs(";").map(|_| v))?;
      let pair = ann.map_or((vn, vt), |t| (self.mty(vn, t), t));
      self.vars.extend([(name, pair)]) }
    Ok(()) }
  fn bound(&mut self) -> Res<i64> {
    if let Some(&v) = self.pid().and_then(|c| self.consts.get(c)) { Ok(self.adv(1, v)) } else { self.s_int() } }
  fn for_stmt(&mut self) -> Res<()> {
    let (var, lo, incl) = (self.id()?, self.rk("in")?.bound()?, self.eat_s("..="));
    if !incl { self.rs("..")?; }
    let hi = self.bound()?.checked_add(i64::from(incl)).ok_or_else(|| perr("ovf"))?;
    if lo < 0 || hi < lo || hi - lo > 16 { return Err(perr("for bounds 0..=16")) }
    if lo == hi { return self.bal("{", "}").map(|_| ()) }
    let (bpos, mut epos, sc) = (self.pos, self.pos, self.consts.clone());
    let outer: BTreeSet<String> = self.vars.keys().cloned().collect();
    for v in lo..hi {
      (self.consts, self.pos) = (sc.clone(), bpos);
      self.consts.insert(var.clone(), v);
      self.blk()?;
      (epos, _) = (self.pos, self.vars.retain(|k, _| outer.contains(k))); }
    (self.consts, self.pos) = (sc, epos);
    Ok(()) }
  fn assign_rhs(&mut self, bop: Option<&str>, old: impl FnOnce(&mut Self) -> (Nid, Vt)) -> Res<Nid> {
    let val = if let Some(op) = bop {
      let ((o, ty), rhs) = (old(self), self.prec(1)?);
      self.binop(op, Ev::N(o, ty), rhs).map(|ev| self.real(ev).0)?
    } else { self.node(1)?.0 };
    self.rs(";").map(|_| val) }
  fn stmt(&mut self, allow_if: bool) -> Res<bool> {
    if self.eat_i("for") { return self.for_stmt().map(|()| true) }
    if self.eat_i("let") { return self.let_stmt().map(|()| true) }
    if allow_if && self.eat_i("if") { return self.if_stmt().map(|()| true) }
    if self.eat_i("assert") {
      let (vn, chk) = self.parse_assert()?;
      self.vars.extend(vn.map(|n| (n, (chk, Vt::I64))));
      return self.rs(";").map(|_| true); }
    let Some(Tok::Id(name)) = self.peek().cloned() else { return Ok(false) };
    if let Some(&buf) = self.bufs.get(&name).filter(|_| self.is_s(1, "[")) {
      let e = bal_toks(&self.toks, self.pos + 1, "[", "]").ok_or_else(|| perr("unclosed ["))?;
      let bop = comp_op(self.toks.get(e + 1));
      if sym(self.toks.get(e + 1)) != Some("=") && bop.is_none() { return Ok(false) }
      let (be, bl) = self.k.bufs.get(buf as usize).map_or((Et::I64, LANES), |b| (b.elem_ty, usize::from(b.lanes)));
      let lane = chk_lane(self.try_const(self.pos + 2, e).ok_or_else(|| perr("non-const lane"))?, bl)?;
      self.pos = e + 2;
      let val = self.assign_rhs(bop, |s| (s.p(Op::Load { mem: s.mem, buf, lane }), Vt::from_elem(be)))?;
      self.touch(buf, lane);
      self.store(buf, lane, val);
      return Ok(true); }
    let bop = comp_op(self.toks.get(self.pos + 1));
    if let Some(&(oid, oty)) = self.vars.get(&name).filter(|_| self.is_s(1, "=") || bop.is_some()) {
      let m = self.skip(2).assign_rhs(bop, |_| (oid, oty)).map(|v| self.mty(v, oty))?;
      self.vars.insert(name, (m, oty));
      return Ok(true); }
    Ok(false) }
  fn blk(&mut self) -> Res<()> { self.with_d(|s| {
    s.rs("{")?;
    while !s.eat_s("}") { if s.pos >= s.toks.len() || !s.stmt(true)? { return Err(perr("bad stmt")) } }
    Ok(()) }) }
  fn is_early_ret(&self) -> bool {
    let mut d = 0usize;
    for (p, t) in self.toks.iter().enumerate().skip(self.pos + 1) {
      if matches!(sym(Some(t)), Some("(" | "[")) { d += 1 }
      else if matches!(sym(Some(t)), Some(")" | "]")) { d = d.saturating_sub(1) }
      else if sym(Some(t)) == Some("{") && d == 0 { return id_s(self.toks.get(p + 1)) == Some("return") } }
    false }
  fn tail_close(&mut self) -> Res<Vals> {
    self.trhs().and_then(|o| self.skip(usize::from(self.is_s(0, ";"))).rs("}").map(|_| o)) }
  fn etail(&mut self) -> Res<Vals> { self.with_d(|s| loop {
    if s.pos >= s.toks.len() { return Err(perr("unclosed expr")) }
    if s.stmt(false)? { continue }
    if s.eat_i("return") { return s.tail_close() }
    if s.pid() == Some("if") && s.is_early_ret() {
      let cond = s.skip(1).node(1)?.0;
      let tv = s.with_n(cond, true, |s| s.rs("{")?.rk("return")?.tail_close())?;
      return s.with_n(cond, false, Self::etail).and_then(|fv| s.merge_v(cond, tv, fv, false)); }
    return s.tail_close() }) }
  fn eblk(&mut self) -> Res<Vals> { self.rs("{")?.etail() }
  fn prec(&mut self, min_p: u8) -> Res<Ev> { self.with_d(|s| {
    let mut lhs = s.cast()?;
    while let Some(&(op, p)) = BINOPS.iter().find(|b| Some(b.0) == sym(s.peek())) {
      if p < min_p { break }
      lhs = s.skip(1).prec(p + 1).and_then(|rhs| s.binop(op, lhs, rhs))?; }
    Ok(lhs) }) }
  fn binop(&mut self, op: &str, lhs: Ev, rhs: Ev) -> Res<Ev> {
    if op == "^" {
      if let (Ev::N(o, ty), Ev::Mxr(a, b, c, _)) | (Ev::Mxr(a, b, c, _), Ev::N(o, ty)) = (lhs, rhs) {
        if o == a || o == b { return Ok(Ev::N(self.sel(c, if o == b { a } else { b }, o), ty)) } }
      let ((ln, ty), (rn, _)) = (self.real(lhs), self.real(rhs));
      return Ok(if let (Some(a), Some(b)) = (self.dom(ln).is_const(), self.dom(rn).is_const()) {
        Ev::N(self.c(a ^ b), ty) } else { Ev::Xor(ln, rn, ty) }); }
    if op == "&" {
      if let (Ev::Xor(a, b, ty), Ev::N(m, _)) | (Ev::N(m, _), Ev::Xor(a, b, ty)) = (lhs, rhs) {
        if let Some(&(c, mb)) = self.bmasks.get(&m) {
          if mb == u64::MAX || [self.dom(a), self.dom(b)].iter()
            .all(|d| !d.is_empty() && d.is_nonneg() && (d.hi as u64) <= mb) { return Ok(Ev::Mxr(a, b, c, ty)) } } } }
    let ((ln, ty), (rn, rty)) = (self.real(lhs), self.real(rhs));
    if ty == Vt::F32 || rty == Vt::F32 {
      if let Some(k) = cmp_k(op, false) { return Ok(Ev::N(self.p(Op::FCmp(k, ln, rn)), Vt::Bool)) }
      let id = match op {
        "+" => self.p(Op::FAdd(ln, rn)),
        "-" => self.p(Op::FSub(ln, rn)),
        "*" => self.p(Op::FMul(ln, rn)),
        "/" => self.p(Op::FDiv(ln, rn)),
        _ => return Err(perr(format!("bad f32 binop '{op}'"))),
      };
      return Ok(Ev::N(id, Vt::F32));
    }
    if let Some(k) = cmp_k(op, ty.u64_cmp() || rty.u64_cmp()) { return Ok(Ev::N(self.p(Op::Cmp(k, ln, rn)), Vt::Bool)) }
    if op == "||" || op == "&&" {
      let (ba, bb) = (self.as_b(ln), self.as_b(rn));
      let s = self.p(if op == "&&" { Op::BitAnd(ba, bb) } else { Op::Add(ba, bb) });
      return Ok(Ev::N(if op == "&&" { s } else { self.as_b(s) }, Vt::Bool)); }
    let id = match op {
      "|" => if let (Some(a), Some(b)) = (self.dom(ln).is_const(), self.dom(rn).is_const()) { self.c(a | b) }
        else if disjoint_or_pow2(self.dom(ln), self.dom(rn)) || disjoint_or_pow2(self.dom(rn), self.dom(ln)) {
          self.p(Op::Add(ln, rn)) } else { self.p(Op::BitOr(ln, rn)) },
      "&" => self.p(Op::BitAnd(ln, rn)),
      "<<" | ">>" => self.shift(ln, ty, rn, op == ">>", false)?,
      "+" | "-" | "*" => self.pm(if op == "+" { Op::Add(ln, rn) }
        else if op == "-" { Op::Sub(ln, rn) } else { Op::Mul(ln, rn) }, ty),
      "/" => self.div(ln, rn),
      _ => self.p(Op::Rem(ln, rn)) };
    Ok(Ev::N(id, ty)) }
  fn cast(&mut self) -> Res<Ev> {
    let mut expr = self.prim()?;
    while self.eat_s(".") {
      let ((n, ty), m) = (self.real(expr), self.id()?);
      if ty == Vt::F32 {
        let args = self.rs("(").and_then(|s| s.list(|s| s.node(1).map(|x| x.0)))?;
        let res = match (m.as_str(), args.as_slice()) {
          ("sqrt", []) => self.p(Op::FSqrt(n)),
          ("abs", []) => self.p(Op::FAbs(n)),
          ("max", &[a]) => self.p(Op::FMax(n, a)),
          ("min", &[a]) => self.p(Op::FMin(n, a)),
          ("mul_add", &[b, c]) => self.p(Op::FFma(n, b, c)),
          _ => return Err(perr(format!("bad f32 method '.{m}'"))),
        };
        expr = Ev::N(res, Vt::F32);
        continue;
      }
      let arg = self.delim("(", ")", |s| if s.is_s(0, ")") { Ok(None) } else { Ok(Some(s.node(1)?.0)) })?;
      let (b, sgn, uty) = ty.inf();
      let zv = if matches!(m.as_str(), "count_ones" | "leading_zeros" | "swap_bytes" | "rotate_left" | "rotate_right") {
        self.zext(n, ty) } else { n };
      let (res, out_ty) = match (m.as_str(), arg) {
        ("wrapping_neg", None) => (self.po(Op::Const(0), |z| Op::Sub(z, n), Some(ty)), ty),
        ("abs", None) => { let z = self.c(0);
          let (lt, neg) = (self.p(Op::Cmp(Ck::Lt, n, z)), self.p(Op::Sub(z, n)));
          (self.pm(Op::Select { cond: lt, t_val: neg, f_val: n }, ty), ty) }
        ("count_ones", None) => (self.p(Op::Popcnt(zv)), Vt::U32),
        ("trailing_zeros", None) => { let v = if b < 64 { self.pc(|h| Op::BitOr(n, h), 1i64 << b) } else { n };
          (self.p(Op::Tzcnt(v)), Vt::U32) }
        ("leading_zeros", None) => { let mut v = zv;
          for s in [1u8, 2, 4, 8, 16, 32] { if s < b { v = self.po(self.ushr(v, s), |sh| Op::BitOr(v, sh), None) } }
          let pc = self.p(Op::Popcnt(v));
          (self.pc(|wc| Op::Sub(wc, pc), i64::from(b)), Vt::U32) }
        ("swap_bytes", None) => { let mut v = zv;
          for (sh, mask) in [(8, 0x00FF_00FF_00FF_00FFu64), (16, 0x0000_FFFF_0000_FFFF), (32, 0x0000_0000_FFFF_FFFF)] {
            if sh >= b { break }
            let mc = self.c(mask as i64);
            let lsh = self.po(Op::BitAnd(v, mc), |lo| Op::Shl(lo, sh), None);
            let hi = self.po(self.ushr(v, sh), |r| Op::BitAnd(r, mc), None);
            v = self.p(Op::BitOr(lsh, hi)); }
          (self.mty(v, ty), ty) }
        ("wrapping_add" | "wrapping_sub" | "wrapping_mul", Some(a)) => (self.pm(if m.ends_with('d') { Op::Add(n, a) }
          else if m.ends_with('b') { Op::Sub(n, a) } else { Op::Mul(n, a) }, ty), ty),
        ("saturating_sub" | "saturating_add", Some(a)) if !sgn => {
          let (sub, (raw, ovf, lc)) = (m == "saturating_sub", self.ovf(n, a, ty, m == "saturating_sub"));
          let sat = if sub { self.c(0) } else { lc.unwrap_or_else(|| self.c(-1)) };
          (self.sel(ovf, sat, raw), ty) }
        ("abs_diff", Some(a)) => { let k = if ty.u64_cmp() { Ck::Uge } else { Ck::Ge };
          let (ge, d1, d2) = (self.p(Op::Cmp(k, n, a)), self.p(Op::Sub(n, a)), self.p(Op::Sub(a, n)));
          (self.pm(Op::Select { cond: ge, t_val: d1, f_val: d2 }, uty), uty) }
        ("wrapping_shl" | "wrapping_shr", Some(a)) => (self.shift(n, ty, a, m == "wrapping_shr", true)?, ty),
        ("rotate_left" | "rotate_right", Some(a)) => { let s = self.msh(a, ty);
          let diff = self.pc(|wc| Op::Sub(wc, s), i64::from(b));
          let (ls, rs) = if m == "rotate_left" { (s, self.msh(diff, ty)) } else { (self.msh(diff, ty), s) };
          let (x, y) = (self.shift(zv, uty, ls, false, true)?, self.shift(zv, uty, rs, true, true)?);
          (self.pm(Op::BitOr(x, y), ty), ty) }
        ("min" | "max", Some(a)) => { let k = cmp_k(if m == "min" { "<" } else { ">" }, ty.u64_cmp()).unwrap_or(Ck::Lt);
          (self.po(Op::Cmp(k, n, a), |c| Op::Select { cond: c, t_val: n, f_val: a }, None), ty) }
        ("div_ceil", Some(a)) => {
          let d = self.dom(a).is_const().filter(|&d| d > 0 && self.dom(n).is_nonneg()).ok_or_else(|| perr("div_ceil"))?;
          let sum = self.pc(|k| Op::Add(n, k), d - 1);
          (self.div(sum, a), ty) }
        _ => return Err(perr(format!("bad method '.{m}'"))) };
      expr = Ev::N(res, out_ty); }
    while self.eat_i("as") {
      let (n, (_, ty, _)) = (self.real(expr).0, self.sty()?);
      expr = Ev::N(self.mty(n, ty), ty); }
    Ok(expr) }
  fn inline_fn(&mut self, helper: HelperFn, args: Vals) -> Res<Vals> {
    if self.inl_d >= 6 || args.len() != helper.params.len() { return Err(perr("helper")) }
    let (st, sp, sv) = (replace(&mut self.toks, helper.body), replace(&mut self.pos, 0), take(&mut self.vars));
    self.vars.extend(helper.params.into_iter().zip(args).map(|((pn, pt), (aid, _))| (pn, (aid, Vt::from_elem(pt)))));
    self.inl_d += 1;
    let res = self.eblk();
    (self.inl_d, self.toks, self.pos, self.vars) = (self.inl_d - 1, st, sp, sv);
    res.and_then(|v| if v.is_empty() { Err(perr("empty helper")) } else { Ok(v) }) }
  fn prim(&mut self) -> Res<Ev> { self.with_d(|s| {
    let neg = s.eat_s("-");
    if let Some(&Tok::Float(bits)) = s.peek() {
      let b = if neg { (-f32::from_bits(bits)).to_bits() } else { bits };
      return Ok(Ev::N(s.skip(1).p(Op::FConst(b)), Vt::F32)); }
    if let Some(&Tok::Int(v, sty)) = s.peek() {
      let v = if neg { v.checked_neg().ok_or_else(|| perr("int ovf"))? } else { v };
      let (ty, n) = (sty.unwrap_or(Vt::I64), s.skip(1).c(v));
      return Ok(Ev::N(if sty.is_some() { s.mty(n, ty) } else { n }, ty)); }
    if neg { return s.pnode().map(|(e, ty)| if ty == Vt::F32 { Ev::N(s.p(Op::FNeg(e)), Vt::F32) }
      else { Ev::N(s.po(Op::Const(0), |z| Op::Sub(z, e), Some(ty)), ty) }) }
    if s.eat_s("!") {
      let (e, ty) = s.pnode()?;
      return Ok(Ev::N(if ty == Vt::Bool { s.pc(|z| Op::Cmp(Ck::Eq, e, z), 0) }
        else { s.po(Op::Const(-1), |m| Op::BitXor(e, m), Some(ty)) }, ty)); }
    if s.is_s(0, "(") { return s.delim("(", ")", |s| s.prec(1)) }
    if s.eat_i("if") { return s.if_vals().and_then(|v| one_val(v, "if")) }
    if s.is_s(0, "{") { return s.delim("{", "}", |s| {
      let chk = s.rk("assert")?.parse_assert()?.1;
      s.rs(";")?.node(1).map(|_| Ev::N(chk, Vt::I64)) }) }
    let name = s.id()?;
    if name == "true" || name == "false" { return Ok(Ev::N(s.c(i64::from(name == "true")), Vt::Bool)) }
    if s.eat_s("::") {
      let m = s.id()?;
      if m == "from" {
        let (a, ty) = (s.delim("(", ")", |s| s.node(1))?.0, scalar_elem(&name).map_or(Vt::I64, |x| x.1));
        return Ok(Ev::N(s.mty(a, ty), ty)); }
      if m == "MAX" || m == "MIN" {
        let c = SCALARS.iter().find(|x| x.0 == name && x.2 != Vt::Bool).ok_or_else(|| perr("bad const"))?;
        let mn = if c.4 { i64::MIN >> (64 - c.3) } else { 0 };
        return Ok(Ev::N(s.c(if m == "MIN" { mn } else if c.4 { !mn } else { (u64::MAX >> (64 - c.3)) as i64 }), c.2)); }
      return Err(perr("bad path")); }
    if s.eat_s("(") {
      let h = s.helpers.get(&name).cloned().ok_or_else(|| perr("unknown helper"))?;
      return s.list(|s| s.node(1)).and_then(|a| s.inline_fn(h, a)).and_then(|v| one_val(v, "helper")); }
    if let Some(&buf) = s.bufs.get(&name) {
      let e = s.is_s(0, "[").then(|| bal_toks(&s.toks, s.pos, "[", "]")).flatten().ok_or_else(|| perr("need ["))?;
      let (be, bl, ba) = s.k.bufs.get(buf as usize)
        .map_or((Et::I64, LANES, Ba::ReadOnly), |b| (b.elem_ty, usize::from(b.lanes), b.access));
      let ty = Vt::from_elem(be);
      if !matches!(ba, Ba::LookupTable(_)) {
        if let Some(lv) = s.try_const(s.pos + 1, e) {
          let lane = chk_lane(lv, bl)?;
          (s.pos, _) = (e + 1, s.loaded.insert(buf));
          s.touch(buf, lane);
          return Ok(Ev::N(s.p(Op::Load { mem: if ba == Ba::ReadOnly { 0 } else { s.mem }, buf, lane }), ty)); }
        if s.sbufs.contains_key(&buf) { return Err(perr("dyn slice idx")) }
        if ba == Ba::ReadOnly && !s.loaded.contains(&buf) {
          if let Some(slot) = s.k.bufs.get_mut(buf as usize) { slot.access = Ba::LookupTable(bl) } } }
      return s.delim("[", "]", |s| s.node(1)).map(|(idx, _)| Ev::N(s.p(Op::LoadLut { buf, idx }), ty)); }
    if let Some(&(n, t)) = s.vars.get(&name) { Ok(Ev::N(n, t)) }
    else if let Some(&c) = s.consts.get(&name) { Ok(Ev::N(s.c(c), Vt::I64)) } else { Err(perr(name)) } }) }
  fn fn_hdr(&mut self, domains: &mut DMap) -> Res<String> {
    while let Some(tok) = self.peek().cloned() {
      if let Tok::Doc(s) = tok { parse_domain_list(&lex_rust(&self.adv(1, s), true)?, domains)? }
      else if matches!(sym(Some(&tok)), Some("#!" | "#")) {
        let st = self.skip(1).bal("[", "]")?;
        let a = &self.toks[st..self.pos - 1];
        if id_s(a.first()) == Some("zero_domain") && a.len() > 2 { parse_domain_list(&a[1..], domains)? }
      } else { break } }
    self.skip_pub()?;
    while self.eat_i("const") || self.eat_i("unsafe") {}
    self.rk("fn")?.id().and_then(|name| self.rs("(").map(|_| name)) } }
fn parse_kernel_internal(toks: Vec<Tok>, ov: &DMap, helpers: &HMap, consts: &CMap, allow_sc: bool) -> Res<Kernel> {
  let (mut domains, mut low) = (DMap::new(), Lowerer::new(toks, helpers.clone(), consts.clone()));
  low.k.name = low.fn_hdr(&mut domains)?;
  domains.extend(ov.iter().map(|(k, &v)| (k.clone(), v)));
  let mut sparams = Vec::new();
  let tight = |n: &str, e: Et, b: Domain|
    domains.get(n).copied().map_or(b, |d| d.try_meet(e.natural_domain()).unwrap_or_else(|| e.natural_domain()));
  low.list(|low| {
    let pn = low.opt_i("mut").id()?;
    if low.rs(":")?.eat_s("&") {
      let is_mut = low.eat_i("mut");
      let (tn, len, is_sl) = low.delim("[", "]", |low| {
        let tn = low.id()?;
        if !low.eat_s(";") { return Ok((tn, LANES, true)) }
        low.bound().and_then(|n| if n <= 0 { Err(perr("bad len")) } else { Ok((tn, n as usize, false)) }) })?;
      let et = Et::from_rust_name(&tn).ok_or_else(|| perr("bad elem"))?;
      let ml = et.max_lanes();
      let (acc, lanes) = if is_mut && (1..=ml).contains(&len) { (Ba::MutNoAlias, len as u8) }
        else if is_mut { return Err(perr("&mut lanes 1..=max")) }
        else if len <= ml { (Ba::ReadOnly, len as u8) } else { (Ba::LookupTable(len), ml as u8) };
      let id = low.k.add_typed_buf(pn.clone(), acc, et, lanes, tight(&pn, et, et.natural_domain()));
      low.bufs.insert(pn, id);
      if is_sl { low.sbufs.insert(id, None); }
    } else if !allow_sc || !low.bufs.is_empty() { return Err(perr("need buffer param")) } else {
      let (e, vty, nat) = low.sty()?;
      sparams.push((pn.clone(), e, vty, tight(&pn, e, nat))) }
    Ok(()) })?;
  low.mem = low.p(Op::MemInit);
  if !low.bufs.is_empty() { if !sparams.is_empty() { return Err(perr("mixed params")) } else { low.blk()? } } else {
    if !allow_sc || !(1..=4).contains(&sparams.len()) { return Err(perr("need 1..=4 scalar params")) }
    let rt = low.ret_tys()?;
    if !(1..=4).contains(&rt.len()) || sparams.len() + rt.len() > 5 { return Err(perr("bad scalar lift")) }
    let lift_lanes = if sparams.iter().any(|(_, e, _, _)| e.is_float()) || rt.iter().any(|(e, _)| e.is_float()) {
      crate::ir::AVX512_F32_LANES as u8
    } else { LANES as u8 };
    let in_b: Vec<_> = sparams.into_iter()
      .map(|(p, e, v, d)| (p.clone(), low.k.add_typed_buf(p, Ba::ReadOnly, e, lift_lanes, d), v)).collect();
    let out_b: Vec<_> = rt.iter().enumerate().map(|(i, &(e, d))| low.k.add_typed_buf(
      if rt.len() == 1 { "out".into() } else { format!("out{i}") }, Ba::MutNoAlias, e, lift_lanes, d)).collect();
    let bpos = low.pos;
    for lane in 0..lift_lanes {
      (low.vars, low.pos) = (BTreeMap::new(), bpos);
      for (p, bid, vty) in &in_b {
        let ld = low.p(Op::Load { mem: 0, buf: *bid, lane });
        low.vars.extend([(p.clone(), (ld, *vty))]) }
      let res = low.eblk()?;
      if res.len() != out_b.len() { return Err(perr("ret arity")) }
      for (&ob, (val, _)) in out_b.iter().zip(res) { low.store(ob, lane, val) } } }
  for (id, ml) in low.sbufs { if let Some(b) = low.k.bufs.get_mut(id as usize) { b.lanes = ml.map_or(1, |m| m + 1) } }
  for ((buf, lane), val) in low.outs { low.k.add_output(buf, lane, val) }
  low.k.validate().map(|()| low.k) }
fn bal_toks(toks: &[Tok], open_pos: usize, o: &str, c: &str) -> Option<usize> {
  let mut d = 1usize;
  for (i, t) in toks.iter().enumerate().skip(open_pos + 1) {
    d = (d + usize::from(sym(Some(t)) == Some(o))).saturating_sub(usize::from(sym(Some(t)) == Some(c)));
    if d == 0 { return Some(i) } }
  None }
fn try_const_item(toks: &[Tok], consts: &CMap) -> Option<(String, i64)> {
  let mut low = Lowerer::new(toks.to_vec(), HMap::new(), consts.clone());
  let (name, ty, id, _) = (low.skip_pub().ok()?.rk("const").ok()?.id().ok()?,
    low.rs(":").ok()?.sty().ok()?.1, low.rs("=").ok()?.node(1).ok()?.0, low.rs(";").ok()?);
  let m = low.mty(id, ty);
  low.dom(m).is_const().map(|v| (name, v)) }
fn extract_items(toks: &[Tok]) -> (Vec<(String, Vec<Tok>)>, CMap) {
  let (mut out, mut consts, mut pfx, mut skip, mut i) = (Vec::new(), CMap::new(), Vec::new(), false, 0usize);
  while i < toks.len() {
    if matches!(toks[i], Tok::Doc(_)) {
      pfx.push(toks[i].clone());
      i += 1;
    } else if matches!(sym(toks.get(i)), Some("#" | "#!")) && sym(toks.get(i + 1)) == Some("[") {
      let Some(e) = bal_toks(toks, i + 1, "[", "]") else { break };
      let a = &toks[i + 2..e];
      if a.windows(4).any(|w| matches!(w, [Tok::Id(c), Tok::Sy(l), Tok::Id(t), Tok::Sy(r)]
        if c == "cfg" && l == "(" && t == "test" && r == ")")) { skip = true }
      else if id_s(a.first()) == Some("zero_domain") { pfx.extend_from_slice(&toks[i..=e]) }
      i = e + 1;
    } else {
      let (st, mut s, mut is_c, mut fn_n) = (i, i, false, None);
      while let Some(Tok::Id(w)) = toks.get(s) {
        s += 1;
        if w == "pub" && sym(toks.get(s)) == Some("(") { s = bal_toks(toks, s, "(", ")").map_or(s, |e| e + 1) }
        else if w == "const" { is_c = true }
        else if w != "pub" && w != "unsafe" {
          if w == "fn" { (is_c, fn_n) = (false, id_s(toks.get(s)).map(str::to_owned)) }
          break } }
      let mut j = st;
      let end = loop { match sym(toks.get(j)) {
        Some(o @ ("(" | "[")) => if let Some(e) = bal_toks(toks, j, o, if o == "(" { ")" } else { "]" }) { j = e + 1 }
          else { break toks.len() },
        Some(";") => { if is_c && !skip { consts.extend(try_const_item(&toks[st..=j], &consts)) } break j + 1 }
        Some("{") => break bal_toks(toks, j, "{", "}").map_or(toks.len(), |e| {
          if let Some(n) = fn_n.filter(|_| !skip) { out.push((n, [pfx.as_slice(), &toks[st..=e]].concat())) }
          e + 1 }),
        _ if j < toks.len() => j += 1,
        _ => break toks.len() } };
      pfx.clear();
      (skip, i) = (false, end.max(i + 1)) } }
  (out, consts) }
fn try_helper(toks: &[Tok], consts: &CMap) -> Option<(String, HelperFn)> {
  let mut low = Lowerer::new(toks.to_vec(), HMap::new(), consts.clone());
  let (name, params) = (low.fn_hdr(&mut DMap::new()).ok()?,
    low.list(|s| Ok((s.opt_i("mut").id()?, s.rs(":")?.sty()?.0))).ok()?);
  let rets = (1..=8).contains(&params.len()).then(|| low.ret_tys().ok()).flatten()?;
  if !(1..=4).contains(&rets.len()) || !low.is_s(0, "{") { return None }
  Some((name, HelperFn { params, body: low.toks[low.pos..].to_vec() })) }
fn build_helpers(fns: &[(String, Vec<Tok>)], c: &CMap, ex: Option<&str>) -> HMap {
  fns.iter().filter(|(n, _)| ex != Some(n.as_str())).filter_map(|(_, ts)| try_helper(ts, c)).collect() }
fn parse_src(src: &str, ov: &DMap, entry: Option<&str>, lift: bool) -> Res<Kernel> {
  let toks = lex_rust(src, true)?;
  let (fns, consts) = extract_items(&toks);
  let h = if entry.is_some() || (!lift && fns.len() > 1) { build_helpers(&fns, &consts, entry) } else { HMap::new() };
  if let Some(e) = entry {
    let (_, ts) = fns.iter().find(|(n, _)| n == e).ok_or_else(|| perr(format!("entry '{e}' not found")))?;
    return parse_kernel_internal(ts.clone(), ov, &h, &consts, true); }
  if let Some((_, ts)) = (!h.is_empty()).then(|| fns.iter().find(|(n, _)| !h.contains_key(n))).flatten() {
    return parse_kernel_internal(ts.clone(), ov, &h, &consts, true) }
  let one = fns.len() == 1 && !consts.is_empty();
  parse_kernel_internal(if one { fns[0].1.clone() } else { toks }, ov, &HMap::new(), &consts, lift || one) }
pub fn parse_rust_kernel(src: &str) -> Res<Kernel> { parse_src(src, &DMap::new(), None, false) }
pub fn parse_rust_pipeline(src: &str, entry: &str) -> Res<Kernel> { parse_src(src, &DMap::new(), Some(entry), true) }
pub fn parse_rust_kernel_with_domains(src: &str, ov: &DMap) -> Res<Kernel> { parse_src(src, ov, None, true) }
pub fn compile_rust_source(s: &str, b: RegBudget) -> Res<CompiledKernel> { compile_kernel(&parse_rust_kernel(s)?, b) }
fn parse_file_cnt(src: &str, ovs: &DOvr) -> (usize, Vec<Kernel>) {
  let Ok(toks) = lex_rust(src, false) else { return (0, Vec::new()) };
  let (fns, consts) = extract_items(&toks);
  let (seen, h, g) = (fns.len(), build_helpers(&fns, &consts, None), ovs.get("*").cloned().unwrap_or_default());
  (seen, fns.into_iter().filter_map(|(name, ts)| {
    let mut ov = g.clone();
    ov.extend(ovs.get(&name).into_iter().flat_map(|sp| sp.iter().map(|(k, &v)| (k.clone(), v))));
    parse_kernel_internal(ts, &ov, &h, &consts, true).ok() }).collect()) }
pub fn parse_rust_file_kernels(src: &str, ov: &DOvr) -> Vec<Kernel> { parse_file_cnt(src, ov).1 }
fn collect_rs(path: &Path, out: &mut Vec<PathBuf>) -> Res<()> {
  if path.is_file() {
    if path.extension() == Some("rs".as_ref()) { out.push(path.to_path_buf()) }
    return Ok(()); }
  let mut es: Vec<_> = fs::read_dir(path).map_err(|e| perr(format!("rd: {e}")))?
    .map(|r| r.map(|e| e.path()).map_err(|e| perr(format!("dir: {e}")))).collect::<Res<_>>()?;
  es.sort();
  for p in es {
    if p.is_dir() && p.file_name().and_then(|n| n.to_str())
      .is_some_and(|n| matches!(n, "target" | "tests" | "benches" | "examples" | ".git")) { continue }
    if p.is_dir() || p.is_file() { collect_rs(&p, out)? } }
  Ok(()) }
pub fn extract_crate_kernels(path: &Path, ov: &DOvr) -> Res<(usize, usize, Vec<ExtractedKernel>)> {
  let (mut files, mut scanned, mut seen_tot, mut extracted) = (Vec::new(), 0usize, 0usize, Vec::new());
  collect_rs(path, &mut files)?;
  for file in files {
    let Some(src) = fs::metadata(&file).ok().filter(|m| m.len() <= 2 * 1024 * 1024)
      .and_then(|_| fs::read_to_string(&file).ok()) else { continue };
    let ((sn, ks), lb) = (parse_file_cnt(&src, ov), file.strip_prefix(path).unwrap_or(&file).display().to_string());
    (scanned, seen_tot) = (scanned + 1, seen_tot + sn);
    extracted.extend(ks.into_iter().map(|k| ExtractedKernel { file: lb.clone(), name: k.name.clone(), kernel: k })) }
  Ok((scanned, seen_tot, extracted)) }
pub fn kernel_reg_budget(kernel: &Kernel, base: RegBudget) -> RegBudget {
  if kernel.bufs.iter().all(|b| b.elem_ty == Et::I64 && usize::from(b.lanes) == LANES) { base }
  else { RegBudget { max_gprs: base.max_gprs, max_vecs: base.max_vecs.max(32) } } }
pub fn extract_and_compile_crate(path: &Path, isa: TargetIsa, budget: RegBudget, ov: &DOvr) -> Res<CrateCompileReport> {
  let (files_scanned, functions_seen, extracted) = extract_crate_kernels(path, ov)?;
  let kernels_extracted = extracted.len();
  let compiled: Vec<_> = extracted.into_iter().filter_map(|ek| {
    let (original_cost, kb) = (ek.kernel.cost(), kernel_reg_budget(&ek.kernel, budget));
    let optimized_cost = crate::egraph::optimize_kernel(&ek.kernel).ok()?.cost();
    let (o, e) = (compile_kernel_direct(&ek.kernel, kb, isa).ok()?,
      compile_standalone_elf_exec(&ek.kernel, kb, isa, &[]).ok()?);
    verify_elf64_object(&o.elf_bytes).ok().and(verify_elf64_executable(&e.elf_bytes).ok())?;
    let max_lanes = ek.kernel.bufs.iter().map(|b| b.lanes).max().unwrap_or(0);
    Some(CompiledCrateKernel { file: ek.file, name: ek.name, original_cost, optimized_cost,
      vector_mode: o.vector_mode, max_lanes, direct_insns: o.direct_insns, direct_share_pct: o.direct_share_pct,
      exec_share_pct: e.direct_share_pct, zero_spills: o.zero_spills && e.zero_spills, obj_bytes: o.elf_bytes.len(),
      exec_bytes: e.elf_bytes.len() })
  }).collect();
  Ok(CrateCompileReport { root: path.display().to_string(), files_scanned, functions_seen, kernels_extracted,
    skipped: kernels_extracted - compiled.len(), compiled }) }
