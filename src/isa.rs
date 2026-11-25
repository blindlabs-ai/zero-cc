use crate::codegen::{remat_readonly_loads, schedule_and_verify, RegBudget, ScheduleReport};
use crate::elf64::{elf64_exec_data_vaddr, emit_elf64_executable, emit_elf64_object, verify_elf64_executable,
  verify_elf64_object, ElfReloc, ElfSymbol, TargetIsa, ELF_EXEC_TEXT_VADDR, R_AARCH64_CALL26, R_X86_64_PLT32};
use crate::ir::{BufAccess, BufDecl, CmpKind, CompileError, Domain, ElemTy, Kernel, NodeId, Op, LANES};
use crate::{optimize_kernel, rust_fe::parse_rust_kernel};
const NO: usize = usize::MAX;
const X86_ARGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
const CMP_CC: [(u8, u32); 10] = [
  (0x94, 1), (0x95, 0), (0x9C, 10), (0x9E, 12), (0x9F, 13), (0x9D, 11), (0x92, 2), (0x96, 8), (0x97, 9), (0x93, 3)];
#[derive(Clone, Debug)] pub struct RegAlloc {
  pub node_reg: Vec<Option<u8>>, pub buf_reg: Vec<u8>, pub callee_saved: Vec<u8>, pub spills: usize }
#[derive(Clone, Debug)] pub struct CompiledObject {
  pub isa: TargetIsa, pub kernel_sym: String, pub batch_sym: String, pub code_bytes: Vec<u8>, pub elf_bytes: Vec<u8>,
  pub direct_insns: usize, pub shim_glue_insns: usize, pub direct_share_pct: u8,
  pub zero_spills: bool, pub rust_ffi_shim: String }
#[derive(Clone, Debug)] pub struct StandaloneExec {
  pub isa: TargetIsa, pub kernel_sym: String, pub batch_sym: String, pub entry_sym: String,
  pub entry_vaddr: u64, pub data_vaddr: u64, pub code_bytes: Vec<u8>, pub elf_bytes: Vec<u8>,
  pub direct_insns: usize, pub shim_glue_insns: usize, pub direct_share_pct: u8,
  pub zero_spills: bool, pub blocks: usize, pub output_bytes: usize }
type Res<T> = Result<T, CompileError>;
type ModuleCode = (Vec<u8>, Vec<ElfSymbol>, Vec<ElfReloc>, usize);
type Lanes<const N: usize> = [NodeId; N];
fn berr(m: &str) -> CompileError { CompileError::BackendError(m.into()) }
fn is_inl(op: Op<NodeId>) -> bool { matches!(op, Op::MemInit | Op::Const(_) | Op::Store { .. }) }
fn is_64(t: ElemTy) -> bool { matches!(t, ElemTy::I64 | ElemTy::U64) }
fn breg(ra: &RegAlloc, b: u8) -> Res<u8> { ra.buf_reg.get(b as usize).copied().ok_or(CompileError::InvalidBuffer(b)) }
fn bdecl(k: &Kernel, b: u8) -> Res<&BufDecl> { k.bufs.get(b as usize).ok_or(CompileError::InvalidBuffer(b)) }
fn nop(k: &Kernel, id: NodeId) -> Res<Op<NodeId>> { k.nodes.get(id).copied().ok_or(CompileError::InvalidNode(id)) }
fn reg_of(rs: &[Option<u8>], i: NodeId) -> Res<u8> { rs.get(i).and_then(|&r| r).ok_or(CompileError::InvalidNode(i)) }
fn clit(k: &Kernel, i: NodeId) -> Option<i64> { if let Some(&Op::Const(c)) = k.nodes.get(i) { Some(c) } else { None } }
fn cof(k: &Kernel, i: NodeId) -> Option<i64> { if k.nodes.get(i) == Some(&Op::MemInit) { Some(0) } else { clit(k, i) } }
fn dom(d: &[Domain], i: NodeId) -> Domain { d.get(i).copied().unwrap_or_else(Domain::top) }
fn all_d(d: &[Domain], xs: &[NodeId], f: fn(Domain) -> bool) -> bool { xs.iter().all(|&x| f(dom(d, x))) }
fn cmp_eq(k: CmpKind) -> bool { matches!(k, CmpKind::Eq | CmpKind::Ne) }
fn cmp_inv(k: CmpKind) -> bool { matches!(k, CmpKind::Ne | CmpKind::Le | CmpKind::Ge | CmpKind::Ule | CmpKind::Uge) }
fn bin_op(op: Op<NodeId>) -> Option<(u8, u8, u8, u32)> { match op {
  Op::Add(..) => Some((0, 0x01, 0xD4, 0x8B00_0000)),
  Op::Sub(..) => Some((5, 0x29, 0xFB, 0xCB00_0000)),
  Op::BitAnd(..) => Some((4, 0x21, 0xDB, 0x8A00_0000)),
  Op::BitOr(..) => Some((1, 0x09, 0xEB, 0xAA00_0000)),
  Op::BitXor(..) => Some((6, 0x31, 0xEF, 0xCA00_0000)),
  _ => None } }
fn vex_load_enc(ty: ElemTy, w64: bool) -> Res<[u8; 6]> {
  let (sh, b) = (ty.scale_shift(), if matches!(ty, ElemTy::I32 | ElemTy::I16) { 0x20 } else { 0x30 });
  if sh == 3 && !w64 { return Err(berr("64b load in w32")) }
  Ok(if sh == 3 || (sh == 2 && !w64) { [1, 1, 2, 0, 0, 0x6F] }
    else { [2, 1, 1, 0, 0, b + if sh == 0 { 1 } else { sh + 2 } + u8::from(w64)] }) }
fn remat_try<T>(base: &Kernel, ns: &[u8], mut f: impl FnMut(&Kernel) -> Option<T>) -> Option<(Kernel, T)> {
  ns.iter().find_map(|&n| {
    let c = if n == 0 { base.clone() } else { remat_readonly_loads(base, n) };
    f(&c).map(|v| (c, v)) }) }
/// @domain reg: 0..=15, idx: 0..=15, base: 0..=15
pub const fn x86_rex_byte(w: bool, reg: u8, idx: u8, base: u8) -> u8 {
  0x40 | (if w { 8 } else { 0 }) | (((reg >> 3) & 1) << 2) | (((idx >> 3) & 1) << 1) | ((base >> 3) & 1)
}
/// @domain reg: 0..=15, rm: 0..=15
pub const fn x86_rex_w(reg: u8, rm: u8) -> u8 { 0x48 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1) }
/// @domain reg: 0..=15, idx: 0..=15, base: 0..=15
pub const fn x86_rex_sib(reg: u8, idx: u8, base: u8) -> u8 { x86_rex_byte(true, reg, idx, base) }
/// @domain md: 0..=3, reg: 0..=15, rm: 0..=15
pub const fn x86_modrm(md: u8, reg: u8, rm: u8) -> u8 { ((md & 3) << 6) | ((reg & 7) << 3) | (rm & 7) }
/// @domain scale: 0..=3, idx: 0..=15, base: 0..=15
pub const fn x86_sib(scale: u8, idx: u8, base: u8) -> u8 { ((scale & 3) << 6) | ((idx & 7) << 3) | (base & 7) }
/// @domain r: 0..=1, b: 0..=1, map: 1..=3, w: 0..=1, vvvv: 0..=15, l: 0..=1, pp: 0..=3
pub const fn x86_vex3_word(r: u8, b: u8, map: u8, w: u8, vvvv: u8, l: u8, pp: u8) -> u16 {
  let b1 = ((!r & 1) << 7) | 0x40 | ((!b & 1) << 5) | (map & 0x1F);
  let b2 = ((w & 1) << 7) | ((!vvvv & 15) << 3) | ((l & 1) << 2) | (pp & 3);
  ((b2 as u16) << 8) | (b1 as u16)
}
/// @domain rd: 0..=31, rn: 0..=31, rm: 0..=31
pub const fn arm64_rrr(op_hi: u32, rm: u8, mid: u32, rn: u8, rd: u8) -> u32 {
  op_hi | ((rm as u32 & 31) << 16) | mid | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
/// @domain rd: 0..=31, rn: 0..=31, imm12: 0..=4095
pub const fn arm64_imm12(op_hi: u32, imm12: u16, rn: u8, rd: u8) -> u32 {
  op_hi | ((imm12 as u32 & 0xFFF) << 10) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
fn linear_scan(k: &Kernel, s: &ScheduleReport, pool: &[u8], hold: &[NodeId], ru: bool) -> Res<(Vec<Option<u8>>, u32)> {
  let mut last = vec![0usize; k.nodes.len()];
  for (st, &id) in s.order.iter().enumerate() {
    let (kids, n) = k.nodes[id].children();
    for &c in &kids[..n] { if let Some(l) = last.get_mut(c) { *l = (*l).max(st) } } }
  if hold.is_empty() { for &(os, oi) in &s.output_order {
    if let Some(l) = k.outputs.get(oi).and_then(|o| last.get_mut(o.val)) { *l = (*l).max(os + usize::from(!ru)) }
  } } else { for &o in hold { if let Some(l) = last.get_mut(o) { *l = s.order.len().saturating_add(1) } } }
  let (mut free, mut regs) = (pool.iter().rev().copied().collect::<Vec<_>>(), vec![None; k.nodes.len()]);
  let (mut act, mut used) = (Vec::<(NodeId, u8)>::new(), 0u32);
  let ret = |a: &mut Vec<(NodeId, u8)>, f: &mut Vec<u8>, lim: usize| a.retain(|&(id, r)| {
    let keep = last.get(id).copied().unwrap_or(0) > lim;
    if !keep { f.push(r) }
    keep });
  for (st, &id) in s.order.iter().enumerate() {
    if ru { ret(&mut act, &mut free, st) }
    if !is_inl(nop(k, id)?) {
      let r = free.pop().ok_or(CompileError::RegisterSpill {
        peak_gprs: act.len() + 1, max_gprs: 0, peak_vecs: s.peak_vecs, max_vecs: 0 })?;
      used |= 1u32 << r;
      if let Some(sl) = regs.get_mut(id) { *sl = Some(r) }
      if last.get(id).copied().unwrap_or(0) > st { act.push((id, r)) } else { free.push(r) } }
    if !ru { ret(&mut act, &mut free, st) }
    if hold.is_empty() && s.output_order.iter().any(|&(os, _)| os == st + 1) { ret(&mut act, &mut free, st + 1) } }
  Ok((regs, used)) }
pub fn allocate_registers(k: &Kernel, s: &ScheduleReport, isa: TargetIsa, b: RegBudget) -> Res<RegAlloc> {
  let (nb, x86) = (k.bufs.len(), isa == TargetIsa::X86_64);
  if (x86 && nb > 5) || (!x86 && nb > 7) { return Err(berr("kernel buffer count exceeds ABI args")) }
  let div = s.order.iter().any(|&id| matches!(k.nodes.get(id), Some(Op::Div(..) | Op::Rem(..))));
  let mut buf_reg: Vec<u8> = if x86 { X86_ARGS[..nb].to_vec() } else { (0..nb as u8).collect() };
  if x86 && div && nb > 2 { buf_reg[2] = if nb < 5 { 9 } else { 15 } }
  let cnt = X86_ARGS.get(nb).copied().unwrap_or(255);
  let pool: Vec<u8> = if x86 { [10, 1, 2, 6, 7, 8, 9, 3, 5, 12, 13, 14, 15].into_iter()
    .filter(|&r| !buf_reg.contains(&r) && r != cnt && (!div || r != 2)).collect()
  } else { vec![9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25] };
  let (node_reg, used) = linear_scan(k, s, &pool, &[], x86).map_err(|e| match e {
    CompileError::RegisterSpill { peak_gprs: p, .. } => CompileError::RegisterSpill {
      peak_gprs: nb + p, max_gprs: b.max_gprs, peak_vecs: s.peak_vecs, max_vecs: b.max_vecs },
    o => o })?;
  let cs: &[u8] = if x86 { &[3, 5, 12, 13, 14, 15] } else { &[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] };
  let callee_saved = cs.iter().copied().filter(|&r| (used & (1u32 << r)) != 0).collect();
  Ok(RegAlloc { node_reg, buf_reg, callee_saved, spills: 0 }) }
#[derive(Default)] pub(crate) struct X86Emit {
  pub(crate) code: Vec<u8>, pub(crate) insns: usize, const_pool: Vec<i64>, const_patches: Vec<(usize, usize)> }
impl X86Emit {
  fn end(&mut self) {}
  fn raw(&mut self, b: &[u8]) -> &mut Self {
    self.code.extend_from_slice(b);
    self }
  fn emit(&mut self, b: &[u8]) -> &mut Self {
    self.insns += 1;
    self.raw(b) }
  fn e2(&mut self, a: &[u8], b: &[u8]) -> &mut Self { self.raw(a).emit(b) }
  fn seq(&mut self, s: &[&[u8]]) -> &mut Self { s.iter().fold(self, |e, b| e.emit(b)) }
  fn ei32(&mut self, b: &[u8], i: i32) -> &mut Self { self.e2(b, &i.to_le_bytes()) }
  fn prr(&mut self, o: u8, r: u8, m: u8) -> &mut Self { self.emit(&[x86_rex_w(r, m), o, x86_modrm(3, r, m)]) }
  fn mr(&mut self, d: u8, s: u8) -> &mut Self { if d != s { self.prr(0x89, s, d) } else { self } }
  fn m64(&mut self, d: u8, v: u64) -> &mut Self { self.e2(&[x86_rex_w(0, d), 0xB8 + (d & 7)], &v.to_le_bytes()) }
  fn mi(&mut self, d: u8, v: i64) -> &mut Self {
    if v == 0 { self.e2(if d >= 8 { &[0x45] } else { &[] }, &[0x31, x86_modrm(3, d, d)]) }
    else if let Ok(v32) = i32::try_from(v) { self.ei32(&[x86_rex_w(0, d), 0xC7, x86_modrm(3, 0, d)], v32) }
    else { self.m64(d, v as u64) } }
  fn md(&mut self, r: u8, b: u8, disp: i32) -> &mut Self {
    let (b7, d) = (b & 7, disp.to_le_bytes());
    let (md, n) = if disp == 0 && b7 != 5 { (0, 0) } else if i8::try_from(disp).is_ok() { (1, 1) } else { (2, 4) };
    self.raw(&[x86_modrm(md, r, b)]).raw(if b7 == 4 { &[0x24] } else { &[] }).emit(&d[..n]) }
  fn lp(&mut self, ty: ElemTy, d: u8, idx: u8, b: u8) -> &mut Self {
    let (sh, sg) = (ty.scale_shift(), matches!(ty, ElemTy::I32 | ElemTy::I16));
    if sh == 3 || sg || d >= 8 || idx >= 8 || b >= 8 { self.code.push(x86_rex_byte(sh == 3 || sg, d, idx, b)) }
    let lo = [0x0F, 0xB6 | sh | (u8::from(sg) << 3)];
    self.raw(if sh < 2 { &lo } else if sg { &[0x63] } else { &[0x8B] }) }
  fn store_lane(&mut self, ty: ElemTy, s: u8, b: u8, ln: u8) {
    let w = is_64(ty);
    if matches!(ty, ElemTy::I16 | ElemTy::U16) { self.code.push(0x66) }
    if w || s >= 8 || b >= 8 || (ty == ElemTy::U8 && s >= 4) { self.code.push(x86_rex_byte(w, s, 0, b)) }
    self.raw(&[if ty == ElemTy::U8 { 0x88 } else { 0x89 }]).md(s, b, i32::from(ln) * i32::from(ty.bytes())); }
  fn i8(&mut self, su: u8, d: u8, i: u8) -> &mut Self { self.emit(&[x86_rex_w(0, d), 0x83, x86_modrm(3, su, d), i]) }
  fn i8s(&mut self, s: &[[u8; 3]]) -> &mut Self { s.iter().fold(self, |e, &[o, d, i]| e.i8(o, d, i)) }
  fn ai(&mut self, su: u8, d: u8, i: i32) -> &mut Self {
    if let Ok(c) = i8::try_from(i) { self.i8(su, d, c as u8) }
    else { self.ei32(&[x86_rex_w(0, d), 0x81, x86_modrm(3, su, d)], i) } }
  fn vh(&mut self, [m, w, l, pp, o]: [u8; 5], r: u8, v: u8, rm: u8) -> &mut Self {
    let vw = x86_vex3_word((r >> 3) & 1, (rm >> 3) & 1, m, w, v, l, pp).to_le_bytes();
    self.raw(&[0xC4, vw[0], vw[1], o]) }
  fn vr(&mut self, c: [u8; 5], d: u8, v: u8, r: u8, i: Option<u8>) -> &mut Self {
    let e = self.vh(c, d, v, r).emit(&[x86_modrm(3, d, r)]);
    if let Some(m) = i { e.raw(&[m]) } else { e } }
  fn vop(&mut self, m: u8, o: u8, d: u8, v: u8, r: u8) -> &mut Self { self.vr([m, 0, 1, 1, o], d, v, r, None) }
  fn vi(&mut self, m: u8, o: u8, d: u8, v: u8, r: u8, i: u8) -> &mut Self { self.vr([m, 0, 1, 1, o], d, v, r, Some(i)) }
  fn vsh(&mut self, o: u8, su: u8, d: u8, s: u8, i: u8) -> &mut Self { self.vr([1, 0, 1, 1, o], su, d, s, Some(i)) }
  fn vmem(&mut self, e: [u8; 6], r: u8, b: u8, d: i32) -> &mut Self {
    self.vh([e[0], e[3], e[1], e[2], e[5]], r, e[4], b).md(r, b, d) }
  fn vmim(&mut self, e: [u8; 6], r: u8, b: u8, d: i32, i: u8) -> &mut Self { self.vmem(e, r, b, d).raw(&[i]) }
  fn vops(&mut self, s: &[[u8; 5]]) -> &mut Self { s.iter().fold(self, |e, &t| e.vop(t[0], t[1], t[2], t[3], t[4])) }
  pub(crate) fn vex_ymm_op_w(&mut self, m: u8, w: u8, o: u8, d: u8, v: u8, r: u8) {
    self.vr([m, w, 1, 1, o], d, v, r, None); }
  pub(crate) fn vex_ymm_op(&mut self, m: u8, o: u8, d: u8, v: u8, r: u8) { self.vop(m, o, d, v, r); }
  pub(crate) fn vex_ymm_imm(&mut self, m: u8, o: u8, d: u8, v: u8, r: u8, i: u8) { self.vi(m, o, d, v, r, i); }
  pub(crate) fn vex_vshufps(&mut self, d: u8, v: u8, r: u8, i: u8) { self.vr([1, 0, 1, 0, 0xC6], d, v, r, Some(i)); }
  pub(crate) fn vex_vpermq(&mut self, d: u8, s: u8, i: u8) { self.vr([3, 1, 1, 1, 0], d, 0, s, Some(i)); }
  pub(crate) fn vex_vptest(&mut self, a: u8, b: u8) { self.vr([2, 0, 1, 1, 0x17], a, 0, b, None); }
  pub(crate) fn vex_shift_imm(&mut self, o: u8, su: u8, d: u8, s: u8, i: u8) { self.vsh(o, su, d, s, i); }
  pub(crate) fn vex_mem_op(&mut self, e: [u8; 6], r: u8, b: u8, d: i32) { self.vmem(e, r, b, d); }
  fn vcmp(&mut self, w64: bool, k: CmpKind, d: u8, va: u8, vb: u8, one: impl FnOnce(&mut Self) -> Option<u8>) {
    let (mp, eq, gt, so, sh) = if w64 { (2, 0x29, 0x37, 0x73, 63) } else { (1, 0x76, 0x66, 0x72, 31) };
    let fl = matches!(k, CmpKind::Lt | CmpKind::Ge | CmpKind::Ult | CmpKind::Uge);
    let (o, x, y) = if cmp_eq(k) { (eq, va, vb) } else if fl { (gt, vb, va) } else { (gt, va, vb) };
    self.vop(mp, o, d, x, y).vsh(so, 2, d, d, sh);
    if let Some(or) = one(self) { self.vop(1, 0xEF, d, d, or).end() } }
  pub(crate) fn jcc_placeholder(&mut self, cc: u8) -> usize { self.jph(&[0x0F, cc]) }
  fn jph(&mut self, op: &[u8]) -> usize {
    let p = self.code.len();
    self.ei32(op, 0);
    p }
  fn jcc(&mut self, c: u8, t: usize) -> &mut Self {
    self.ei32(&[0x0F, c], ((t as isize) - ((self.code.len() + 6) as isize)) as i32) }
  fn patch(&mut self, p: usize, ol: usize) -> &mut Self {
    let rel = ((self.code.len() as isize) - ((p + ol + 4) as isize)) as i32;
    if let Some(s) = self.code.get_mut(p + ol..p + ol + 4) { s.copy_from_slice(&rel.to_le_bytes()) }
    self }
  fn bcast(&mut self, d: u8, v: i64, opc: u8) {
    if v == 0 { return self.vop(1, 0xEF, d, d, d).end() }
    if opc == 0x58 && v == i64::from(u32::MAX) { return self.vop(1, 0x76, d, d, d).end() }
    let i = self.const_pool.iter().position(|&c| c == v).unwrap_or_else(|| {
      self.const_pool.push(v);
      self.const_pool.len() - 1 });
    let off = self.vh([2, 0, 1, 1, opc], d, 0, 0).raw(&[x86_modrm(0, d, 5)]).code.len();
    self.const_patches.push((off, i));
    self.emit(&[0, 0, 0, 0]); }
  pub(crate) fn vex_broadcast_rip(&mut self, d: u8, v: i64) { self.bcast(d, v, 0x59) }
  pub(crate) fn vex_broadcast32_rip(&mut self, d: u8, v: u32) { self.bcast(d, i64::from(v), 0x58) }
  fn finalize_pool(&mut self) {
    if self.const_pool.is_empty() { return }
    while self.code.len() % 8 != 0 { self.code.push(0x90) }
    let base = self.code.len();
    for &c in &self.const_pool { self.code.extend_from_slice(&c.to_le_bytes()) }
    for &(off, i) in &self.const_patches {
      let d = (base.saturating_add(i.saturating_mul(8)) as isize) - ((off + 4) as isize);
      if let Some(s) = self.code.get_mut(off..off + 4) { s.copy_from_slice(&(d as i32).to_le_bytes()) } } }
  fn tge(&mut self, r: u8, l: i64) -> &mut Self {
    self.mi(11, l).seq(&[&[x86_rex_w(11, r), 0x39, x86_modrm(3, 11, r)], &[0x72, 2], &[0x0F, 0x0B]]) }
  fn vzu(&mut self) -> &mut Self { self.emit(&[0xC5, 0xF8, 0x77]) } }
fn rreg(k: &Kernel, ra: &RegAlloc, i: NodeId, sc: u8, mut imm: impl FnMut(u8, i64)) -> Res<u8> {
  if let Some(c) = cof(k, i) {
    imm(sc, c);
    Ok(sc)
  } else { reg_of(&ra.node_reg, i) } }
fn drain_outs<E>(e: &mut E, k: &Kernel, s: &ScheduleReport, ra: &RegAlloc, oc: &mut usize, step: Option<usize>,
  mut st: impl FnMut(&mut E, &BufDecl, u8, u8, NodeId) -> Res<()>) -> Res<()> {
  while let Some(&(os, oi)) = s.output_order.get(*oc) {
    if step.is_some_and(|t| os != t) { break }
    let o = k.outputs.get(oi).copied().ok_or(CompileError::InvalidNode(oi))?;
    st(e, bdecl(k, o.buf)?, breg(ra, o.buf)?, o.lane, o.val)?;
    *oc += 1; }
  Ok(()) }
fn walk_body<E>(e: &mut E, k: &Kernel, s: &ScheduleReport, ra: &RegAlloc,
  mut on_st: impl FnMut(&mut E, &BufDecl, u8, u8, NodeId) -> Res<()>,
  mut on_op: impl FnMut(&mut E, Op<NodeId>, u8) -> Res<()>) -> Res<()> {
  let mut oc = 0usize;
  for (st, &id) in s.order.iter().enumerate() {
    drain_outs(e, k, s, ra, &mut oc, Some(st), &mut on_st)?;
    let op = nop(k, id)?;
    if let Op::Store { buf, lane, val, .. } = op { on_st(e, bdecl(k, buf)?, breg(ra, buf)?, lane, val)? }
    else if !matches!(op, Op::MemInit | Op::Const(_)) { on_op(e, op, reg_of(&ra.node_reg, id)?)? } }
  drain_outs(e, k, s, ra, &mut oc, None, &mut on_st) }
fn lut_trap<'a>(k: &'a Kernel, ds: &[Domain], b: u8, idx: NodeId) -> Res<(&'a BufDecl, Option<i64>)> {
  let bd = bdecl(k, b)?;
  let BufAccess::LookupTable(len) = bd.access else { return Err(CompileError::InvalidBuffer(b)) };
  let d = dom(ds, idx);
  Ok((bd, (d.is_empty() || d.lo < 0 || d.hi >= len as i64).then_some(len as i64))) }
fn xv(e: &mut X86Emit, k: &Kernel, ra: &RegAlloc, i: NodeId, s: u8) -> Res<u8> {
  rreg(k, ra, i, s, |r, c| e.mi(r, c).end()) }
fn match_same_op<const N: usize>(k: &Kernel, ids: Lanes<N>) -> Option<(Op<NodeId>, [Lanes<N>; 3])> {
  let (o0, mut args) = (*k.nodes.get(ids[0])?, [[0; N], [0; N], [0; N]]);
  for (i, &id) in ids.iter().enumerate() {
    let op = *k.nodes.get(id)?;
    if std::mem::discriminant(&op) != std::mem::discriminant(&o0) || !match (o0, op) {
      (Op::Shl(_, a), Op::Shl(_, b)) | (Op::Shr(_, a), Op::Shr(_, b)) => a == b,
      (Op::Cmp(a, ..), Op::Cmp(b, ..)) => a == b,
      (Op::Const(a), Op::Const(b)) => a == b,
      _ => true } { return None }
    let (kids, n) = op.children();
    for c in 0..n { args[c][i] = kids[c] } }
  Some((o0, args)) }
#[derive(Clone, Copy)]
enum HOp { Load(u8, ElemTy, i8), Store(u8, Option<bool>, i8), Bin(u8, u8), Shift(u8, u8), Cmp(CmpKind), Sel }
#[derive(Clone, Copy)] struct HStep(HOp, usize, [usize; 4]);
struct HVecPlan { is_64: bool, consts: Vec<(u8, i64)>, steps: Vec<HStep> }
struct HCtx<'a, const N: usize> {
  k: &'a Kernel, ds: &'a [Domain], nf: Vec<bool>, memo: Vec<(Lanes<N>, usize)>,
  consts: Vec<(usize, i64)>, steps: Vec<HStep>, next_v: usize }
impl<'a, const N: usize> HCtx<'a, N> {
  fn push(&mut self, op: HOp, s: &[usize]) -> usize {
    let (d, mut src) = (self.next_v, [NO; 4]);
    self.next_v += 1;
    for (p, &v) in src.iter_mut().zip(s) { *p = v }
    self.steps.push(HStep(op, d, src));
    d }
  fn st(&mut self, op: HOp, r: usize) { self.steps.push(HStep(op, NO, [r, NO, NO, NO])) }
  fn ad(&self, a: &[usize], b: &[usize], f: fn(Domain) -> bool) -> bool { [a, b].iter().all(|s| all_d(self.ds, s, f)) }
  fn get_c(&mut self, val: i64) -> usize {
    if let Some(&(v, _)) = self.consts.iter().find(|&&(_, c)| c == val) { return v }
    let v = self.next_v;
    self.next_v += 1;
    self.consts.push((v, val));
    v }
  fn match_vec(&mut self, ids: Lanes<N>) -> Option<usize> {
    if let Some(&(_, v)) = self.memo.iter().find(|&&(m, _)| m == ids) { return Some(v) }
    if N == 8 && ids.iter().any(|&i| self.nf[i] && !all_d(self.ds, &[i], |d| d.fits_i32() || d.fits_u32())) {
      return None }
    let o0 = *self.k.nodes.get(ids[0])?;
    let r = match o0 {
      Op::Const(c) if ids[1..].iter().all(|&i| clit(self.k, i) == Some(c)) =>
        self.get_c(if N == 4 { c } else { i64::from(c as i32) }),
      Op::Load { mem, buf, lane }
        if (lane == 0 || (N == 4 && lane == 4)) && self.k.nodes.get(mem) == Some(&Op::MemInit) => {
        let (k, ty) = (self.k, self.k.bufs.get(buf as usize)?.elem_ty);
        if (N == 4 && ty != ElemTy::I64) || (N == 8 && is_64(ty))
          || !(1..N).all(|i| k.nodes.get(ids[i]) == Some(&Op::Load { mem, buf, lane: lane + i as u8 })) { return None }
        self.push(HOp::Load(buf, ty, if lane == 4 { 32 } else { 0 }), &[]) }
      Op::Add(..) | Op::Sub(..) | Op::BitAnd(..) | Op::Mul(..) => {
        let (_, [la, lb, _]) = match_same_op(self.k, ids)?;
        if let Some((o, _)) = (N == 8 && matches!(o0, Op::BitAnd(..))).then(|| [(la, lb), (lb, la)].into_iter().find(
          |(_, m)| m.iter().all(|&x| clit(self.k, x) == Some(0xFFFF_FFFF)))).flatten() { self.match_vec(o)? } else {
          if N == 4 && matches!(o0, Op::Mul(..)) && !self.ad(&la, &lb, Domain::fits_u32) { return None }
          let (va, vb) = (self.match_vec(la)?, self.match_vec(lb)?);
          let (mp, opc) = if matches!(o0, Op::Add(..)) { (1, if N == 4 { 0xD4 } else { 0xFE }) }
            else if matches!(o0, Op::Sub(..)) { (1, if N == 4 { 0xFB } else { 0xFA }) }
            else if matches!(o0, Op::BitAnd(..)) { (1, 0xDB) } else if N == 4 { (1, 0xF4) } else { (2, 0x40) };
          self.push(HOp::Bin(mp, opc), &[va, vb]) } }
      Op::Shl(_, s) | Op::Shr(_, s) if N == 4 || s < 32 => {
        let ((_, [la, ..]), shr) = (match_same_op(self.k, ids)?, matches!(o0, Op::Shr(..)));
        let u = shr && all_d(self.ds, &la, |d| d.is_nonneg() && (N == 4 || d.fits_u32()));
        if shr && !u && (N == 4 || !all_d(self.ds, &la, Domain::fits_i32)) { return None }
        let va = self.match_vec(la)?;
        self.push(HOp::Shift(if !shr { 6 } else if u { 2 } else { 4 }, s), &[va]) }
      Op::Cmp(ck, ..) if N == 8 => {
        let (_, [la, lb, _]) = match_same_op(self.k, ids)?;
        let (ai, au) = (self.ad(&la, &lb, Domain::fits_i32), self.ad(&la, &lb, Domain::fits_u32));
        if (!ai || ck.is_unsigned()) && !au { return None }
        let (va, vb) = (self.match_vec(la)?, self.match_vec(lb)?);
        let bi = if (ck.is_unsigned() || !ai) && !cmp_eq(ck) { self.get_c(i64::from(i32::MIN)) } else { NO };
        let one = if cmp_inv(ck) { self.get_c(1) } else { NO };
        self.push(HOp::Cmp(ck), &[va, vb, bi, one]) }
      Op::Select { .. } if N == 8 => {
        let (_, [lc, lt, lf]) = match_same_op(self.k, ids)?;
        let (vc, vt, vf, z) = (self.match_vec(lc)?, self.match_vec(lt)?, self.match_vec(lf)?, self.get_c(0));
        self.push(HOp::Sel, &[vc, vt, vf, z]) }
      _ => return None };
    self.memo.push((ids, r));
    Some(r) }
  fn finalize(self, is_64: bool, mv: usize) -> Option<HVecPlan> {
    let (lim, nc) = (mv.min(if is_64 { 16 } else { 14 }), self.consts.len());
    if nc >= lim { return None }
    let (mut ph, mut last) = (vec![255u8; self.next_v], vec![0usize; self.next_v]);
    for (i, &(v, _)) in self.consts.iter().enumerate() { ph[v] = i as u8 }
    let consts = self.consts.iter().enumerate().map(|(i, &(_, c))| (i as u8, c)).collect();
    for (i, st) in self.steps.iter().enumerate() { for s in st.2 { if s != NO { last[s] = last[s].max(i) } } }
    let (mut free, mut steps) = ((nc as u8..lim as u8).rev().collect::<Vec<_>>(), Vec::with_capacity(self.steps.len()));
    for (i, &st) in self.steps.iter().enumerate() {
      for (j, s) in st.2.into_iter().enumerate() {
        if s != NO && ph[s] >= nc as u8 && last[s] == i && st.2.iter().take(j).all(|&x| x != s) { free.push(ph[s]) } }
      if st.1 != NO {
        let pr = free.pop()?;
        ph[st.1] = pr;
        if last[st.1] <= i { free.push(pr) } }
      let r = |v: usize| if v == NO { NO } else { usize::from(ph[v]) };
      steps.push(HStep(st.0, r(st.1), st.2.map(r))); }
    Some(HVecPlan { is_64, consts, steps }) } }
fn try_plan_hvec(k: &Kernel, s: &ScheduleReport, ds: &[Domain], mv: usize) -> Option<HVecPlan> {
  if k.outputs.is_empty() || s.order.iter().any(|&i| matches!(k.nodes.get(i), Some(Op::Store { .. }))
    || matches!(k.nodes.get(i), Some(&Op::Load { mem, .. }) if k.nodes.get(mem) != Some(&Op::MemInit))) { return None }
  let mut og: Vec<(u8, ElemTy, [NodeId; LANES])> = k.bufs.iter()
    .filter(|b| b.access == BufAccess::MutNoAlias).map(|b| (b.id, b.elem_ty, [NO; LANES])).collect();
  for o in &k.outputs {
    let g = og.iter_mut().find(|g| g.0 == o.buf)?;
    if usize::from(o.lane) >= LANES || g.2[usize::from(o.lane)] != NO { return None }
    g.2[usize::from(o.lane)] = o.val; }
  if og.is_empty() || og.iter().any(|g| g.2.contains(&NO)) { return None }
  if og.len() == 1 && k.bufs.iter().all(|b| b.elem_ty == ElemTy::I64
    && (matches!(b.access, BufAccess::LookupTable(_)) || usize::from(b.lanes) == LANES)) {
    let ((ob, _, l), ro) = (og[0], k.nodes.iter().any(|&op| matches!(op, Op::Load { buf, .. } if buf == og[0].0)));
    let mut c = HCtx::<4> { k, ds, nf: vec![], memo: vec![], consts: vec![], steps: vec![], next_v: 0 };
    if let Some(lo) = c.match_vec([l[0], l[1], l[2], l[3]]) {
      if !ro { c.st(HOp::Store(ob, None, 0), lo) }
      if let Some(hi) = c.match_vec([l[4], l[5], l[6], l[7]]) {
        if ro { c.st(HOp::Store(ob, None, 0), lo) }
        c.st(HOp::Store(ob, None, 32), hi);
        if let Some(p) = c.finalize(true, mv) { return Some(p) } } } }
  if k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || usize::from(b.lanes) != LANES)
    || og.iter().any(|g| !matches!(g.1, ElemTy::I32 | ElemTy::U32 | ElemTy::I64))
    || s.order.iter().any(|&i| matches!(k.nodes.get(i),
      Some(Op::LoadLut { .. } | Op::BoundsCheck { .. } | Op::Div(..) | Op::Rem(..)))) { return None }
  let mut nf = vec![false; k.nodes.len()];
  for g in &og { if g.1 == ElemTy::I64 { for &v in &g.2 { nf[v] = true } } }
  for &id in s.order.iter().rev() { match k.nodes[id] {
    Op::Cmp(_, a, b) => for x in [a, b] { nf[x] = true },
    Op::Shr(a, sh) if sh > 0 => nf[a] = true,
    Op::Select { cond, t_val, f_val } => for (x, v) in [(cond, true), (t_val, nf[id]), (f_val, nf[id])] { nf[x] |= v },
    _ => {} } }
  let mut c = HCtx::<8> { k, ds, nf, memo: vec![], consts: vec![], steps: vec![], next_v: 0 };
  let outs: Vec<_> = og.iter().map(|&(b, t, l)| c.match_vec(l).map(|v| (b, t, l, v))).collect::<Option<_>>()?;
  for (buf, ty, lanes, vr) in outs {
    let sg = if ty == ElemTy::I64 {
      let (si, ui) = (all_d(ds, &lanes, Domain::fits_i32), all_d(ds, &lanes, Domain::fits_u32));
      if !si && !ui { return None }
      Some(si)
    } else { None };
    c.st(HOp::Store(buf, sg, 0), vr); }
  c.finalize(false, mv) }
fn emit_hvec_consts(e: &mut X86Emit, p: &HVecPlan) {
  let (w, o) = if p.is_64 { (1, 0x59) } else { (0, 0x58) };
  for &(r, v) in &p.consts {
    if v == 0 { e.vop(1, 0xEF, r, r, r).end() } else { e.mi(0, if p.is_64 { v } else { i64::from(v as i32) })
      .vr([1, w, 0, 1, 0x6E], r, 0, 0, None).vop(2, o, r, 0, r).end() } } }
fn emit_hvec_body(e: &mut X86Emit, k: &Kernel, p: &HVecPlan, ra: &RegAlloc, blk: i32) -> Res<()> {
  let (sh, bo) = (if p.is_64 { 0x73 } else { 0x72 }, |b| Ok(blk * (bdecl(k, b)?.block_bytes() as i32)));
  for &HStep(op, dst, s) in &p.steps {
    let (d, a, b) = (dst as u8, s[0] as u8, s[1] as u8);
    match op {
      HOp::Load(bf, ty, dp) => e.vmem(vex_load_enc(ty, p.is_64)?, d, breg(ra, bf)?, bo(bf)? + i32::from(dp)).end(),
      HOp::Store(bf, None, dp) => e.vmem([1, 1, 2, 0, 0, 0x7F], a, breg(ra, bf)?, bo(bf)? + i32::from(dp)).end(),
      HOp::Store(bf, Some(sg), _) => {
        let (br, off, ext) = (breg(ra, bf)?, bo(bf)?, if sg { 0x25 } else { 0x35 });
        e.vop(2, ext, 15, 0, a).vmem([1, 1, 2, 0, 0, 0x7F], 15, br, off)
          .vi(3, 0x39, a, 0, 14, 1).vop(2, ext, 15, 0, 14).vmem([1, 1, 2, 0, 0, 0x7F], 15, br, off + 32).end() }
      HOp::Bin(mp, opc) => e.vop(mp, opc, d, a, b).end(),
      HOp::Shift(su, shf) => e.vsh(sh, su, d, a, shf).end(),
      HOp::Cmp(ck) => {
        let (va, vb) = if s[2] != NO { (14, 15) } else { (a, b) };
        if s[2] != NO { e.vops(&[[1, 0xEF, 14, a, s[2] as u8], [1, 0xEF, 15, b, s[2] as u8]]); }
        e.vcmp(false, ck, d, va, vb, |_| (s[3] != NO).then_some(s[3] as u8)); }
      HOp::Sel => e.vop(1, 0x76, 15, a, s[3] as u8).vi(3, 0x4C, d, b, s[2] as u8, 15 << 4).end() } }
  Ok(()) }
#[derive(Clone, Copy)] struct TransposeOut { buf: u8, n: usize, lanes: [NodeId; 4] }
struct Batch4Plan { node_ymm: Vec<Option<u8>>, hoisted: Vec<(u8, i64)>, tr: Option<TransposeOut> }
impl Batch4Plan {
  fn cst(&self, e: &mut X86Emit, c: i64, sc: u8) -> u8 {
    if let Some(&(r, _)) = self.hoisted.iter().find(|&&(_, v)| v == c) { return r }
    e.vex_broadcast_rip(sc, c);
    sc }
  fn val(&self, e: &mut X86Emit, k: &Kernel, id: NodeId, sc: u8) -> Res<u8> {
    if let Some(c) = cof(k, id) { Ok(self.cst(e, c, sc)) } else { reg_of(&self.node_ymm, id) } }
  fn biased(&self, e: &mut X86Emit, k: &Kernel, id: NodeId, sc: u8, nb: bool) -> Res<u8> {
    if !nb { return self.val(e, k, id, sc) }
    if let Some(c) = cof(k, id) { return Ok(self.cst(e, c ^ i64::MIN, sc)) }
    let (r, br) = (reg_of(&self.node_ymm, id)?, self.cst(e, i64::MIN, sc));
    e.vop(1, 0xEF, sc, r, br);
    Ok(sc) } }
fn try_plan_avx2_batch4(k: &Kernel, s: &ScheduleReport, ds: &[Domain], mv: usize) -> Option<Batch4Plan> {
  if k.outputs.is_empty() || k.bufs.iter().any(|b| matches!(b.access, BufAccess::LookupTable(_)) || b.lanes == 0
    || !((b.access == BufAccess::ReadOnly && b.lanes == 1)
      || matches!(b.elem_ty, ElemTy::I64 | ElemTy::U64 | ElemTy::I32 | ElemTy::U32))) { return None }
  if !s.order.iter().all(|&id| match k.nodes.get(id) {
    Some(Op::MemInit | Op::Const(_) | Op::Add(..) | Op::Sub(..) | Op::BitAnd(..) | Op::BitOr(..)
      | Op::BitXor(..) | Op::ShlVar(..) | Op::ShrVarU(..) | Op::Cmp(..) | Op::Select { .. }) => true,
    Some(Op::Load { mem, .. }) => k.nodes.get(*mem) == Some(&Op::MemInit),
    Some(Op::Mul(a, b)) => (ds[*a].fits_i32() && ds[*b].fits_i32()) || ds[*a].fits_u32() || ds[*b].fits_u32(),
    Some(Op::Shl(_, sh) | Op::Shr(_, sh) | Op::Ushr(_, sh)) => *sh < 64,
    _ => false }) { return None }
  let cand = matches!(k.outputs.len(), 2 | 4).then(|| {
    let (nl, ob, mut lanes) = (k.outputs.len(), k.outputs[0].buf, [NO; 4]);
    let bd = k.bufs.get(ob as usize)?;
    if !is_64(bd.elem_ty) || usize::from(bd.lanes) != nl
      || k.bufs.iter().filter(|b| b.access == BufAccess::MutNoAlias).count() != 1 { return None }
    for o in &k.outputs {
      let ln = usize::from(o.lane);
      if o.buf != ob || ln >= nl || lanes[ln] != NO
        || matches!(k.nodes.get(o.val), Some(Op::Const(_) | Op::MemInit)) { return None }
      lanes[ln] = o.val; }
    (!lanes[..nl].contains(&NO)).then_some(TransposeOut { buf: ob, n: nl, lanes }) }).flatten();
  let (pool, lim14) = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13u8], mv.min(14));
  let (node_ymm, used, plim, tr) = cand.and_then(|t| {
    let lim = mv.min(if t.n == 2 { 14 } else { 13 });
    linear_scan(k, s, &pool[..lim], &t.lanes[..t.n], false).ok().map(|(ny, u)| (ny, u, lim, Some(t)))
  }).or_else(|| linear_scan(k, s, &pool[..lim14], &[], false).ok().map(|(ny, u)| (ny, u, lim14, None)))?;
  let mut cc: Vec<(i64, usize)> = Vec::new();
  let mut bump = |c, w| if let Some(e) = cc.iter_mut().find(|e| e.0 == c) { e.1 += w } else { cc.push((c, w)) };
  for &op in s.order.iter().filter_map(|&id| k.nodes.get(id)) { match op {
      Op::Cmp(ck, a, b) => {
        let nb = ck.is_unsigned() && !(ds[a].is_nonneg() && ds[b].is_nonneg());
        for x in [a, b] {
          if let Some(c) = cof(k, x) { bump(if nb { c ^ i64::MIN } else { c }, 2 + usize::from(nb)) }
          else if nb { bump(i64::MIN, 6) } }
        if cmp_inv(ck) { bump(1, 6) } }
      Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
        if let Some(c) = clit(k, a) { bump(c, 2) }
        if let Some(c) = clit(k, b) { bump(c & 63, 2) } else if !(ds[b].is_nonneg() && ds[b].hi < 64) { bump(63, 4) } }
      _ => {
        if matches!(op, Op::Select { .. }) { bump(0, 6) }
        let (kids, n) = op.children();
        for &kid in &kids[..n] { if let Some(c) = clit(k, kid) { bump(c, 2) } }
        if let Op::Shr(a, sh) = op { if sh > 0 && !dom(ds, a).is_nonneg() {
          if sh == 63 { bump(0, 3) }
          else { for (c, w) in [(i64::MIN, 3), (((1u64 << 63) >> sh) as i64, 2)] { bump(c, w) } } } } } } }
  for o in &k.outputs { if let Some(c) = clit(k, o.val) { bump(c, 2) } }
  cc.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
  let hoisted = (0..plim as u8).filter(|&r| (used & (1u32 << r)) == 0).zip(cc).map(|(r, (c, _))| (r, c)).collect();
  Some(Batch4Plan { node_ymm, hoisted, tr }) }
fn b4_sc(e: &mut X86Emit, bd: &BufDecl, vr: u8, br: u8, blk: i32, lane: u8) {
  let (ty, st) = (bd.elem_ty, bd.block_bytes() as i32);
  if bd.lanes == 1 && is_64(ty) { return e.vmem([1, 1, 2, 0, 0, 0x7F], vr, br, blk * st).end() }
  let (o, (op, w, i)) = (
    blk * st + i32::from(lane) * i32::from(ty.bytes()), if is_64(ty) { (0xD6, 1, 1) } else { (0x7E, 0, 2) });
  e.vmem([1, 0, 1, 0, 0, op], vr, br, o).vmim([3, 0, 1, w, 0, 0x16], vr, br, st + o, i).vi(3, 0x39, vr, 0, 15, 1)
    .vmem([1, 0, 1, 0, 0, op], 15, br, 2 * st + o).vmim([3, 0, 1, w, 0, 0x16], 15, br, 3 * st + o, i); }
fn emit_x86_batch4_body(e: &mut X86Emit, k: &Kernel, s: &ScheduleReport, ra: &RegAlloc,
  ds: &[Domain], p: &Batch4Plan, blk: i32) -> Res<()> {
  let (mut oc, sc) = (0usize, p.tr.is_none());
  let ab = |em: &mut X86Emit, a, b| -> Res<(u8, u8)> { Ok((p.val(em, k, a, 14)?, p.val(em, k, b, 15)?)) };
  let do_sc = |em: &mut X86Emit, bd: &BufDecl, br, ln, v| p.val(em, k, v, 14).map(|r| b4_sc(em, bd, r, br, blk, ln));
  for (step, &id) in s.order.iter().enumerate() {
    if sc { drain_outs(e, k, s, ra, &mut oc, Some(step), do_sc)? }
    let op = nop(k, id)?;
    if is_inl(op) { continue }
    let d = reg_of(&p.node_ymm, id)?;
    match op {
      Op::Load { buf, lane, .. } => {
        let (bd, br) = (bdecl(k, buf)?, breg(ra, buf)?);
        let (st, bo) = (bd.block_bytes() as i32, blk * (bd.block_bytes() as i32));
        if bd.lanes == 1 { e.vmem(vex_load_enc(bd.elem_ty, true)?, d, br, bo).end() } else {
          let o = bo + i32::from(lane) * i32::from(bd.elem_ty.bytes());
          if is_64(bd.elem_ty) {
            e.vmem([1, 0, 2, 0, 0, 0x7E], d, br, o).vmim([3, 0, 1, 1, d, 0x22], d, br, st + o, 1)
              .vmem([1, 0, 2, 0, 0, 0x7E], 15, br, 2 * st + o).vmim([3, 0, 1, 1, 15, 0x22], 15, br, 3 * st + o, 1)
              .vi(3, 0x38, d, d, 15, 1).end()
          } else { e.vmem([1, 0, 1, 0, 0, 0x6E], d, br, o).vmim([3, 0, 1, 0, d, 0x22], d, br, st + o, 1)
            .vmim([3, 0, 1, 0, d, 0x22], d, br, 2 * st + o, 2).vmim([3, 0, 1, 0, d, 0x22], d, br, 3 * st + o, 3)
            .vop(2, if bd.elem_ty == ElemTy::I32 { 0x25 } else { 0x35 }, d, 0, d).end() } } }
      Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) =>
        ab(e, a, b).map(|(x, y)| e.vop(1, bin_op(op).map_or(0xEF, |t| t.2), d, x, y).end())?,
      Op::Mul(a, b) => if (ds[a].fits_i32() && ds[b].fits_i32()) || (ds[a].fits_u32() && ds[b].fits_u32()) {
        let (mp, oc) = if ds[a].fits_i32() && ds[b].fits_i32() { (2, 0x28) } else { (1, 0xF4) };
        ab(e, a, b).map(|(x, y)| e.vop(mp, oc, d, x, y).end())?;
      } else {
        let (x, y) = if ds[b].fits_u32() { (a, b) } else { (b, a) };
        let (vx, vy) = ab(e, x, y)?;
        let t = if d == vx { 14 } else if d == vy { 15 } else { d };
        let (lo, add) = if t == d { (14, 14) } else { (d, t) };
        e.vsh(0x73, 2, t, vx, 32).vop(1, 0xF4, t, t, vy).vsh(0x73, 6, t, t, 32)
          .vops(&[[1, 0xF4, lo, vx, vy], [1, 0xD4, d, d, add]]); }
      Op::Shl(a, sh) | Op::Ushr(a, sh) | Op::Shr(a, sh) => {
        let va = p.val(e, k, a, d)?;
        if sh == 0 { if d != va { e.vop(1, 0xEB, d, va, va).end() } }
        else if !matches!(op, Op::Shr(..)) || ds[a].is_nonneg() {
          e.vsh(0x73, if matches!(op, Op::Shl(..)) { 6 } else { 2 }, d, va, sh).end()
        } else if sh == 63 {
          let zr = p.cst(e, 0, 15);
          e.vop(2, 0x37, d, zr, va).end()
        } else {
          let (sb, sbs) = (p.cst(e, i64::MIN, 14), p.cst(e, ((1u64 << 63) >> sh) as i64, 15));
          e.vop(1, 0xEF, d, va, sb).vsh(0x73, 2, d, d, sh).vop(1, 0xFB, d, d, sbs).end() } }
      Op::ShlVar(a, b) | Op::ShrVarU(a, b) => {
        let va = p.val(e, k, a, 14)?;
        let vb = if let Some(c) = cof(k, b) { p.cst(e, c & 63, 15) } else {
          let r = p.val(e, k, b, 15)?;
          if ds[b].is_nonneg() && ds[b].hi < 64 { r } else {
            let m = p.cst(e, 63, 15);
            e.vop(1, 0xDB, 15, r, m);
            15 } };
        e.vex_ymm_op_w(2, 1, if matches!(op, Op::ShlVar(..)) { 0x47 } else { 0x45 }, d, va, vb); }
      Op::Cmp(ck, a, b) => {
        let nb = !cmp_eq(ck) && ck.is_unsigned() && !(ds[a].is_nonneg() && ds[b].is_nonneg());
        let (va, vb) = (p.biased(e, k, a, 14, nb)?, p.biased(e, k, b, 15, nb)?);
        e.vcmp(true, ck, d, va, vb, |em| cmp_inv(ck).then(|| p.cst(em, 1, 15))); }
      Op::Select { cond, t_val, f_val } => {
        let (vc, zr) = (p.val(e, k, cond, 14)?, p.cst(e, 0, 15));
        e.vop(2, 0x29, 15, vc, zr);
        let (zf, zt) = (cof(k, f_val) == Some(0), cof(k, t_val) == Some(0));
        if zf || zt {
          let v = p.val(e, k, if zf { t_val } else { f_val }, 14)?;
          e.vop(1, if zf { 0xDF } else { 0xDB }, d, 15, v).end()
        } else {
          let vt = p.val(e, k, t_val, 14)?;
          if vt == d { e.vop(1, 0xEB, 14, vt, vt); }
          let vf = p.val(e, k, f_val, d)?;
          e.vi(3, 0x4C, d, if vt == d { 14 } else { vt }, vf, 15 << 4).end() } }
      _ => return Err(berr("unsupported op in batch4")) } }
  if let Some(TransposeOut { buf, n, lanes: l }) = p.tr {
    let (br, r0, r1) = (breg(ra, buf)?, reg_of(&p.node_ymm, l[0])?, reg_of(&p.node_ymm, l[1])?);
    let ps = |e: &mut X86Emit, d, a, b, o0, o1| e.vi(3, 0x46, d, a, b, 0x20)
      .vmem([1, 1, 2, 0, 0, 0x7F], d, br, o0).vi(3, 0x46, d, a, b, 0x31).vmem([1, 1, 2, 0, 0, 0x7F], d, br, o1).end();
    if n == 2 {
      e.vops(&[[1, 0x6C, 14, r0, r1], [1, 0x6D, 15, r0, r1]]);
      ps(e, r0, 14, 15, blk * 16, blk * 16 + 32);
    } else {
      let (r2, r3) = (reg_of(&p.node_ymm, l[2])?, reg_of(&p.node_ymm, l[3])?);
      for (op, o0, o1) in [(0x6Cu8, 0, 64), (0x6Du8, 32, 96)] {
        e.vops(&[[1, op, 13, r0, r1], [1, op, 14, r2, r3]]);
        ps(e, 15, 13, 14, blk * 32 + o0, blk * 32 + o1) } }
  } else { drain_outs(e, k, s, ra, &mut oc, None, do_sc)? }
  Ok(()) }
fn x86_div(e: &mut X86Emit, dst: u8, ra: u8, rb: u8, rem: bool) {
  e.mr(11, rb).seq(&[&[0x4D, 0x85, 0xDB], &[0x75, 2], &[0x0F, 0x0B]]).mr(0, ra).seq(&[
    &[0x48, 0xBA, 0, 0, 0, 0, 0, 0, 0, 0x80], &[0x48, 0x39, 0xD0], &[0x75, 8], &[0x49, 0x83, 0xFB, 0xFF],
    &[0x75, 2], &[0x0F, 0x0B], &[0x48, 0x99], &[0x49, 0xF7, 0xFB]]).mr(dst, if rem { 2 } else { 0 }); }
fn emit_x86_body(e: &mut X86Emit, k: &Kernel, s: &ScheduleReport, ra: &RegAlloc, ds: &[Domain]) -> Res<()> {
  let xp = |em: &mut X86Emit, a, b| -> Res<(u8, u8)> { Ok((xv(em, k, ra, a, 0)?, xv(em, k, ra, b, 11)?)) };
  let ci = |i| clit(k, i).and_then(|c| i32::try_from(c).ok());
  let ip = |cm: bool, a, b| ci(b).map(|v| (a, v)).or_else(|| cm.then(|| ci(a).map(|v| (b, v))).flatten());
  let st = |em: &mut X86Emit, b: &BufDecl, br, l, v| xv(em, k, ra, v, 0).map(|r| em.store_lane(b.elem_ty, r, br, l));
  walk_body(e, k, s, ra, st, |em, op, d| { match op {
      Op::Load { buf, lane, .. } => {
        let (t, br) = (bdecl(k, buf)?.elem_ty, breg(ra, buf)?);
        em.lp(t, d, 0, br).md(d, br, i32::from(lane) * i32::from(t.bytes())); }
      Op::LoadLut { buf, idx } => {
        let ((b, tr), br, ir) = (lut_trap(k, ds, buf, idx)?, breg(ra, buf)?, xv(em, k, ra, idx, 0)?);
        if let Some(lim) = tr { em.tge(ir, lim); }
        let md = u8::from((br & 7) == 5);
        em.lp(b.elem_ty, d, ir, br).e2(&[x86_modrm(md, d, 4), x86_sib(b.elem_ty.scale_shift(), ir, br)],
          if md != 0 { &[0] } else { &[] }); }
      Op::BoundsCheck { idx, limit } => xv(em, k, ra, idx, 0).map(|r| em.tge(r, limit).mr(d, r).end())?,
      Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b) => {
        let (su, oc, _, _) = bin_op(op).unwrap_or((6, 0x31, 0, 0));
        if let Some((x, c)) = ip(!matches!(op, Op::Sub(..)), a, b) {
          xv(em, k, ra, x, 0).map(|v| em.mr(d, v).ai(su, d, c).end())?;
        } else {
          let (va, vb) = xp(em, a, b)?;
          let cl = d == vb && d != va;
          if cl { em.mr(11, vb); }
          em.mr(d, va).prr(oc, if cl { 11 } else { vb }, d).end() } }
      Op::Mul(a, b) => if let Some((x, c)) = ip(true, a, b) {
        xv(em, k, ra, x, 0).map(|v| em.ei32(&[x86_rex_w(d, v), 0x69, x86_modrm(3, d, v)], c).end())?;
      } else {
        let (va, vb) = xp(em, a, b)?;
        let (r1, r2) = if d == vb && d != va { (vb, va) } else { (va, vb) };
        em.mr(d, r1).emit(&[x86_rex_w(d, r2), 0x0F, 0xAF, x86_modrm(3, d, r2)]); }
      Op::Div(a, b) | Op::Rem(a, b) => xp(em, a, b).map(|(x, y)| x86_div(em, d, x, y, matches!(op, Op::Rem(..))))?,
      Op::Shl(a, sh) | Op::Ushr(a, sh) | Op::Shr(a, sh) => {
        let su = if matches!(op, Op::Shl(..)) { 4 }
          else if matches!(op, Op::Shr(..)) && !dom(ds, a).is_nonneg() { 7 } else { 5 };
        xv(em, k, ra, a, 0).map(|v| em.mr(d, v).end())?;
        if sh != 0 { em.emit(&[x86_rex_w(0, d), 0xC1, x86_modrm(3, su, d), sh]).end() } }
      Op::ShlVar(a, b) | Op::ShrVarU(a, b) => xp(em, a, b)
        .map(|(x, y)| em.vr([2, 1, 0, if matches!(op, Op::ShlVar(..)) { 1 } else { 3 }, 0xF7], d, y, x, None).end())?,
      Op::Popcnt(a) | Op::Tzcnt(a) => {
        let (v, o) = (xv(em, k, ra, a, 0)?, if matches!(op, Op::Popcnt(_)) { 0xB8 } else { 0xBC });
        em.emit(&[0xF3, x86_rex_w(d, v), 0x0F, o, x86_modrm(3, d, v)]); }
      Op::Cmp(ck, a, b) => {
        let (x, y) = xp(em, a, b)?;
        em.seq(&[&[x86_rex_w(y, x), 0x39, x86_modrm(3, y, x)], &[0x0F, CMP_CC[ck as usize].0, 0xC0],
          &[x86_rex_w(d, 0), 0x0F, 0xB6, x86_modrm(3, d, 0)]]); }
      Op::Select { cond, t_val, f_val } => {
        xv(em, k, ra, cond, 0).map(|v| em.mr(0, v).end())?;
        let th = xv(em, k, ra, t_val, 11)?;
        let cl = d == th && t_val != f_val;
        if cl { em.mr(11, th); }
        let (vf, sr) = (if f_val == cond { 0 } else { xv(em, k, ra, f_val, d)? }, if cl { 11 } else { th });
        em.mr(d, vf).seq(&[&[0x48, 0x85, 0xC0], &[x86_rex_w(d, sr), 0x0F, 0x45, x86_modrm(3, d, sr)]]); }
      Op::MemInit | Op::Const(_) | Op::Store { .. } => {} }
    Ok(()) }) }
fn bump_x86<'a>(e: &'a mut X86Emit, k: &Kernel, rs: &[u8], m: i32) -> &'a mut X86Emit {
  rs.iter().zip(&k.bufs).filter(|(_, b)| !matches!(b.access, BufAccess::LookupTable(_)) && b.block_bytes() != 0)
    .fold(e, |em, (&r, b)| em.ai(0, r, (b.block_bytes() as i32) * m)) }
fn step_x86<'a>(e: &'a mut X86Emit, k: &Kernel, rs: &[u8], c: u8, s: u8) -> &'a mut X86Emit {
  bump_x86(e, k, rs, i32::from(s)).i8s(&[[5, c, s], [7, c, s]]) }
fn vec_loop_x86(e: &mut X86Emit, k: &Kernel, ra: &RegAlloc, cnt: u8, st: u8, pf: i32,
  mut body: impl FnMut(&mut X86Emit, i32) -> Res<()>) -> Res<()> {
  let (jb2, top) = (e.i8(7, cnt, st * 2).jcc_placeholder(0x82), e.code.len());
  for (&br, _) in ra.buf_reg.iter().zip(&k.bufs).filter(|(_, b)| b.access == BufAccess::ReadOnly) {
    (if br >= 8 { e.raw(&[x86_rex_byte(false, 0, 0, br)]) } else { &mut *e }).raw(&[0x0F, 0x18]).md(1, br, pf); }
  body(e, 0)?;
  body(e, i32::from(st))?;
  let jb1 = step_x86(e, k, &ra.buf_reg, cnt, st * 2).jcc(0x83, top).patch(jb2, 2).i8(7, cnt, st).jcc_placeholder(0x82);
  body(e, 0)?;
  if st > 1 { bump_x86(e, k, &ra.buf_reg, i32::from(st)).i8(5, cnt, st); }
  e.patch(jb1, 2);
  Ok(()) }
fn mod_syms(name: &str, ksz: u64, boff: u64, bsz: u64) -> Vec<ElfSymbol> {
  vec![ElfSymbol { name: format!("zero_cc_{name}"), offset: 0, size: ksz },
    ElfSymbol { name: format!("zero_cc_{name}_batch"), offset: boff, size: bsz }] }
fn x86_frame<'a>(e: &'a mut X86Emit, sv: &[u8], b2: Option<u8>, epi: bool) -> &'a mut X86Emit {
  let e = (0..sv.len()).fold(e, |em, i| {
    let r = sv[if epi { sv.len() - 1 - i } else { i }];
    em.e2(if r >= 8 { &[0x41] } else { &[] }, &[(if epi { 0x58 } else { 0x50 }) + (r & 7)]) });
  if epi { e.emit(&[0xC3]) } else if let Some(r) = b2.filter(|&r| r != 2) { e.mr(r, 2) } else { e } }
fn emit_x86_module(k: &Kernel, base: &Kernel, s: &ScheduleReport, ra: &RegAlloc,
  ds: &[Domain], mv: usize, cb: bool) -> Res<ModuleCode> {
  let (mut e, hvec) = (X86Emit::default(), try_plan_hvec(k, s, ds, mv));
  let b8_plan = hvec.is_none().then(|| (base.nodes.len() < k.nodes.len()).then(|| remat_try(base, &[0, 1, 3, 6, 12],
    |c| (c.nodes.len() < k.nodes.len()).then(|| schedule_and_verify(c, RegBudget { max_gprs: 64, max_vecs: mv }).ok()
      .and_then(|bs| crate::batch8::try_plan_avx2_batch8(c, &bs, &c.infer_domains(), mv))).flatten())
    .map(|(_, p)| p)).flatten().or_else(|| crate::batch8::try_plan_avx2_batch8(k, s, ds, mv))).flatten();
  let b4_plan = (hvec.is_none() && b8_plan.is_none()).then(|| try_plan_avx2_batch4(k, s, ds, mv)).flatten();
  let (mut saved, b2) = (if hvec.is_some() { Vec::new() } else { ra.callee_saved.clone() }, ra.buf_reg.get(2).copied());
  if b2 == Some(15) && !saved.contains(&15) { saved.push(15) }
  x86_frame(&mut e, &saved, b2, false);
  if let Some(ref hv) = hvec {
    emit_hvec_consts(&mut e, hv);
    emit_hvec_body(&mut e, k, hv, ra, 0)?;
    e.vzu();
  } else { emit_x86_body(&mut e, k, s, ra, ds)? }
  let k_size = x86_frame(&mut e, &saved, None, true).code.len() as u64;
  while e.code.len() % 16 != 0 { e.code.push(0x90) }
  let (b_off, nb, mut relocs) = (e.code.len() as u64, k.bufs.len(), Vec::new());
  let cnt = *X86_ARGS.get(nb).ok_or_else(|| berr("x86_64 batch requires <= 5 kernel buffers"))?;
  if cb {
    let bsv = [12u8, 13, 14, 15, 3, 5].get(..nb + 1).ok_or_else(|| berr("bsaved overflow"))?;
    let (pad, ch) = ((bsv.len() & 1) == 0, bsv[nb]);
    x86_frame(&mut e, bsv, None, false);
    (if pad { e.i8(5, 4, 8) } else { &mut e }).mr(ch, cnt);
    for (&bs, &arg) in bsv.iter().zip(&X86_ARGS).take(nb) { e.mr(bs, arg); }
    let (jz, top) = (e.prr(0x85, ch, ch).jcc_placeholder(0x84), e.code.len());
    for (&bs, &arg) in bsv.iter().zip(&X86_ARGS).take(nb) { e.mr(arg, bs); }
    relocs.push(ElfReloc { offset: (e.code.len() + 1) as u64, sym_idx: 0, r_type: R_X86_64_PLT32, addend: -4 });
    bump_x86(e.ei32(&[0xE8], 0), k, bsv, 1).i8(0, ch, 0xFF).jcc(0x85, top).patch(jz, 2);
    x86_frame(if pad { e.i8(0, 4, 8) } else { &mut e }, bsv, None, true);
  } else {
    x86_frame(&mut e, &saved, b2, false);
    if let Some(ref b8) = b8_plan {
      let (jb, mut jfp, mut jms) = (e.i8(7, cnt, 8).jcc_placeholder(0x82), None, None);
      if let Some(fp) = b8.fastpath.as_ref().filter(|_| !ra.buf_reg.contains(&11) && cnt != 11) {
        let fp_hoist = e.mi(11, 0).code.len();
        crate::batch8::emit_x86_batch8_fp_consts(&mut e, fp);
        let fp_loop = e.code.len();
        let jnz = crate::batch8::emit_x86_batch8_fp_body(&mut e, ra, fp)?;
        jfp = Some(step_x86(&mut e, k, &ra.buf_reg, cnt, 8).jcc(0x83, fp_loop).jph(&[0xE9]));
        e.patch(jnz, 2);
        for &(r, c) in &b8.hoisted { e.vex_broadcast32_rip(r, c) }
        crate::batch8::emit_x86_batch8_body(&mut e, k, ra, b8, 0)?;
        jms = Some(step_x86(&mut e, k, &ra.buf_reg, cnt, 8).jcc_placeholder(0x82));
        e.i8s(&[[0, 11, 1], [7, 11, 2]]).jcc(0x82, fp_hoist);
      } else { for &(r, c) in &b8.hoisted { e.vex_broadcast32_rip(r, c) } }
      vec_loop_x86(&mut e, k, ra, cnt, 8, 512, |em, o| crate::batch8::emit_x86_batch8_body(em, k, ra, b8, o))?;
      for (p, l) in [(jfp, 1), (jms, 2)] { if let Some(p) = p { e.patch(p, l).end() } }
      e.vzu().patch(jb, 2);
    } else if let Some(ref b4) = b4_plan {
      let jb = e.i8(7, cnt, 4).jcc_placeholder(0x82);
      for &(r, c) in &b4.hoisted { e.vex_broadcast_rip(r, c) }
      vec_loop_x86(&mut e, k, ra, cnt, 4, 256, |em, o| emit_x86_batch4_body(em, k, s, ra, ds, b4, o))?;
      e.vzu().patch(jb, 2); }
    let jz = e.prr(0x85, cnt, cnt).jcc_placeholder(0x84);
    if let Some(ref hv) = hvec {
      emit_hvec_consts(&mut e, hv);
      vec_loop_x86(&mut e, k, ra, cnt, 1, 256, |em, o| emit_hvec_body(em, k, hv, ra, o))?;
      e.vzu();
    } else {
      let top = e.code.len();
      emit_x86_body(&mut e, k, s, ra, ds)?;
      bump_x86(&mut e, k, &ra.buf_reg, 1).i8(0, cnt, 0xFF).jcc(0x85, top); }
    x86_frame(e.patch(jz, 2), &saved, None, true); }
  let b_size = (e.code.len() as u64) - b_off;
  e.finalize_pool();
  Ok((e.code, mod_syms(&k.name, k_size, b_off, b_size), relocs, e.insns)) }
#[derive(Default)] struct ArmEmit { code: Vec<u8>, insns: usize }
impl ArmEmit {
  fn end(&mut self) {}
  fn put(&mut self, w: u32) -> &mut Self {
    self.code.extend_from_slice(&w.to_le_bytes());
    self.insns += 1;
    self }
  fn puts(&mut self, ws: &[u32]) -> &mut Self { ws.iter().fold(self, |e, &w| e.put(w)) }
  fn mv(&mut self, d: u8, s: u8) -> &mut Self {
    if d != s { self.put(arm64_rrr(0xAA00_0000, s, 0, 31, d)) } else { self } }
  fn m64(&mut self, d: u8, u: u64, fx: bool) -> &mut Self {
    let r = u32::from(d & 31);
    (1..4u32).fold(self.put(0xD280_0000 | (((u & 0xFFFF) as u32) << 5) | r), |e, hw| {
      let p = (u >> (hw * 16)) & 0xFFFF;
      if fx || p != 0 { e.put(0xF280_0000 | (hw << 21) | ((p as u32) << 5) | r) } else { e } }) }
  fn sz(ty: ElemTy) -> u32 { 0x3800_0000 | (u32::from(ty.scale_shift()) << 30) }
  fn ldst(&mut self, ty: ElemTy, r: u8, b: u8, ln: u8, st: bool) -> &mut Self {
    let m = if st { 0 } else if matches!(ty, ElemTy::I32 | ElemTy::I16) { 0x0080_0000 } else { 0x0040_0000 };
    self.put(arm64_imm12(Self::sz(ty) | 0x0100_0000 | m, u16::from(ln), b, r)) }
  fn tge(&mut self, r: u8, l: i64) -> &mut Self {
    self.m64(17, l as u64, false).puts(&[arm64_rrr(0xEB00_0000, 17, 0, r, 31), 0x5400_0043, 0xD420_0000]) } }
fn av(e: &mut ArmEmit, k: &Kernel, ra: &RegAlloc, i: NodeId, s: u8) -> Res<u8> {
  rreg(k, ra, i, s, |r, c| e.m64(r, c as u64, false).end()) }
fn emit_arm_body(e: &mut ArmEmit, k: &Kernel, s: &ScheduleReport, ra: &RegAlloc, ds: &[Domain]) -> Res<()> {
  let ap = |em: &mut ArmEmit, a, b| -> Res<(u8, u8)> { Ok((av(em, k, ra, a, 16)?, av(em, k, ra, b, 17)?)) };
  let st = |e: &mut ArmEmit, b: &BufDecl, p, l, v| av(e, k, ra, v, 16).map(|r| e.ldst(b.elem_ty, r, p, l, true).end());
  walk_body(e, k, s, ra, st, |em, op, d| { match op {
      Op::Load { buf, lane, .. } => em.ldst(bdecl(k, buf)?.elem_ty, d, breg(ra, buf)?, lane, false).end(),
      Op::LoadLut { buf, idx } => {
        let ((b, tr), br, ir) = (lut_trap(k, ds, buf, idx)?, breg(ra, buf)?, av(em, k, ra, idx, 16)?);
        if let Some(lim) = tr { em.tge(ir, lim); }
        let (t, m) = (b.elem_ty, if matches!(b.elem_ty, ElemTy::I32 | ElemTy::I16) { 0xA0_0000 } else { 0x60_0000 });
        em.put(arm64_rrr(ArmEmit::sz(t) | m, ir, if t == ElemTy::U8 { 0x6800 } else { 0x7800 }, br, d)); }
      Op::BoundsCheck { idx, limit } => av(em, k, ra, idx, 16).map(|r| em.tge(r, limit).mv(d, r).end())?,
      Op::Add(a, b) | Op::Sub(a, b) | Op::BitAnd(a, b) | Op::BitOr(a, b) | Op::BitXor(a, b)
      | Op::ShlVar(a, b) | Op::ShrVarU(a, b) | Op::Mul(a, b) => {
        let (hi, mid) = if let Some((_, _, _, h)) = bin_op(op) { (h, 0) }
          else if matches!(op, Op::ShlVar(..)) { (0x9AC0_0000, 0x2000) }
          else if matches!(op, Op::ShrVarU(..)) { (0x9AC0_0000, 0x2400) } else { (0x9B00_0000, 0x7C00) };
        ap(em, a, b).map(|(va, vb)| em.put(arm64_rrr(hi, vb, mid, va, d)).end())?; }
      Op::Popcnt(a) => av(em, k, ra, a, 16).map(|v| em.puts(&[0x9E67_0000 | ((v as u32 & 31) << 5),
        0x0E20_5800, 0x0E31_B800, 0x0E01_3C00 | (d as u32 & 31)]).end())?,
      Op::Tzcnt(a) => av(em, k, ra, a, 16).map(|v| em.puts(&[
        arm64_rrr(0xDAC0_0000, 0, 0, v, d), arm64_rrr(0xDAC0_1000, 0, 0, d, d)]).end())?,
      Op::Div(a, b) | Op::Rem(a, b) => {
        let ((va, vb), (da, db)) = (ap(em, a, b)?, (dom(ds, a), dom(ds, b)));
        if db.is_empty() || db.contains(0) { em.puts(&[0xB500_0040 | (vb as u32 & 31), 0xD420_0000]); }
        if (da.is_empty() || da.contains(i64::MIN)) && (db.is_empty() || db.contains(-1)) {
          em.puts(&[arm64_imm12(0xB100_0000, 1, vb, 31), if vb == 17 { 0x5400_00C1 } else { 0x5400_00A1 },
            0xD2E0_0000 | (0x8000 << 5) | 17, arm64_rrr(0xEB00_0000, 17, 0, va, 31), 0x5400_0041, 0xD420_0000]);
          if vb == 17 { em.put(0x9280_0011).end() } }
        if matches!(op, Op::Div(..)) { em.put(arm64_rrr(0x9AC0_0000, vb, 0x0C00, va, d)).end() } else {
          let q = if va == 16 && vb != 17 { 17 } else if va != 16 { 16 } else { d };
          if q == d && (d == va || d == vb) { return Err(berr("arm64 rem scratch conflict")) }
          em.puts(&[arm64_rrr(0x9AC0_0000, vb, 0x0C00, va, q),
            arm64_rrr(0x9B00_0000, vb, 0x8000 | ((va as u32 & 31) << 10), q, d)]).end() } }
      Op::Shl(a, sh) | Op::Ushr(a, sh) | Op::Shr(a, sh) => {
        let (va, s) = (av(em, k, ra, a, 16)?, u32::from(sh));
        if sh == 0 { em.mv(d, va).end() } else {
          let w = if matches!(op, Op::Shl(..)) { 0xD340_0000 | (((64 - s) & 63) << 16) | ((63 - s) << 10) }
            else if matches!(op, Op::Shr(..)) && !dom(ds, a).is_nonneg() { 0x9340_FC00 | ((s & 63) << 16) }
            else { 0xD340_FC00 | ((s & 63) << 16) };
          em.put(w | ((va as u32 & 31) << 5) | (d as u32 & 31)).end() } }
      Op::Cmp(ck, a, b) => ap(em, a, b).map(|(x, y)| em.puts(&[arm64_rrr(0xEB00_0000, y, 0, x, 31),
        0x9A9F_07E0 | (CMP_CC[ck as usize].1 << 12) | (d as u32 & 31)]).end())?,
      Op::Select { cond, t_val, f_val } => {
        av(em, k, ra, cond, 16).map(|c| em.put(arm64_rrr(0xEB00_0000, 31, 0, c, 31)).end())?;
        ap(em, t_val, f_val).map(|(t, f)| em.put(arm64_rrr(0x9A80_0000, f, 0x1000, t, d)).end())?; }
      Op::MemInit | Op::Const(_) | Op::Store { .. } => {} }
    Ok(()) }) }
fn arm_frame<'a>(e: &'a mut ArmEmit, sv: &[u8], epi: bool) -> &'a mut ArmEmit {
  if !sv.is_empty() {
    let (ps, opc) = (sv.len().div_ceil(2), if epi { 0xA940_0000 } else { 0xA900_0000 });
    if !epi { e.put(arm64_imm12(0xD100_0000, (ps as u16) * 16, 31, 31)); }
    for p in 0..ps {
      let (r1, r2) = (sv[p * 2], sv.get(p * 2 + 1).copied().unwrap_or(31));
      e.put(opc | ((((p as u32) * 2) & 0x7F) << 15) | ((r2 as u32 & 31) << 10) | (31 << 5) | (r1 as u32 & 31)); }
    if epi { e.put(arm64_imm12(0x9100_0000, (ps as u16) * 16, 31, 31)).end() } }
  if epi { e.put(0xD65F_03C0) } else { e } }
fn emit_arm_module(k: &Kernel, s: &ScheduleReport, ra: &RegAlloc, ds: &[Domain], cb: bool) -> Res<ModuleCode> {
  let mut e = ArmEmit::default();
  arm_frame(&mut e, &ra.callee_saved, false);
  emit_arm_body(&mut e, k, s, ra, ds)?;
  let k_size = arm_frame(&mut e, &ra.callee_saved, true).code.len() as u64;
  while e.code.len() % 16 != 0 { e.put(0xD503_201F); }
  let (b_off, nb, mut relocs) = (e.code.len() as u64, k.bufs.len(), Vec::new());
  let (sv, pb, cr) = if cb { (std::iter::once(30).chain((0..=nb).map(|i| 19 + i as u8)).collect(), 19, 19 + nb as u8)
  } else { (ra.callee_saved.clone(), 0, nb as u8) };
  arm_frame(&mut e, &sv, false);
  if cb { (0..nb).fold(e.mv(cr, nb as u8), |e, i| e.mv(19 + i as u8, i as u8)); }
  let (cbz, top) = (e.code.len(), e.put(0xB400_0000 | (cr as u32 & 31)).code.len());
  if cb {
    (0..nb).fold(&mut e, |e, i| e.mv(i as u8, 19 + i as u8));
    relocs.push(ElfReloc { offset: e.code.len() as u64, sym_idx: 0, r_type: R_AARCH64_CALL26, addend: 0 });
    e.put(0x9400_0000);
  } else { emit_arm_body(&mut e, k, s, ra, ds)? }
  for (i, b) in k.bufs.iter().enumerate().filter(|(_, b)| !matches!(b.access, BufAccess::LookupTable(_))) {
    e.put(arm64_imm12(0x9100_0000, b.block_bytes() as u16, pb + i as u8, pb + i as u8)); }
  let bk = ((top as isize) - (e.put(arm64_imm12(0xF100_0000, 1, cr, cr)).code.len() as isize)) / 4;
  let end = e.put(0x5400_0001 | (((bk as u32) & 0x7_FFFF) << 5)).code.len();
  let w = 0xB400_0000 | (((((end as isize) - (cbz as isize)) / 4) as u32 & 0x7_FFFF) << 5) | (cr as u32 & 31);
  if let Some(sl) = e.code.get_mut(cbz..cbz + 4) { sl.copy_from_slice(&w.to_le_bytes()) }
  let b_size = (arm_frame(&mut e, &sv, true).code.len() as u64) - b_off;
  Ok((e.code, mod_syms(&k.name, k_size, b_off, b_size), relocs, e.insns)) }
fn emit_ffi_shim(k: &Kernel, ks: &str, bs: &str) -> (String, usize) {
  let (mut c, mut lb): ([Vec<String>; 5], Vec<&str>) = (Default::default(), Vec::new());
  for b in &k.bufs {
    let (nm, ty, n) = (b.name.as_str(), b.elem_ty.rust_name(), b.lanes);
    let row = if let BufAccess::LookupTable(l) = b.access {
      [format!("{nm}: *const {ty}"), format!("{nm}: &[{ty}; {l}]"), format!("{nm}.as_ptr()"),
        format!("{nm}: &[{ty}; {l}]"), format!("{nm}.as_ptr()")]
    } else {
      let (p, r, a) = if b.access == BufAccess::MutNoAlias { ("mut", "&mut ", "as_mut_ptr") }
        else { ("const", "&", "as_ptr") };
      lb.push(nm);
      [format!("{nm}: *{p} [{ty}; {n}]"), format!("{nm}: {r}[{ty}; {n}]"), format!("{nm}.{a}().cast()"),
        format!("{nm}: {r}[[{ty}; {n}]]"), format!("{nm}.{a}()")] };
    for (v, s) in c.iter_mut().zip(row) { v.push(s) } }
  let ba = c[0].iter().cloned().chain(["blocks: usize".into()]).collect::<Vec<_>>().join(", ");
  let a4 = c[4].iter().cloned().chain(["blocks".into()]).collect::<Vec<_>>().join(", ");
  let le = lb.first().map_or_else(|| "0".into(),
    |f| lb[1..].iter().fold(format!("{f}.len()"), |a, &x| format!("{a}.min({x}.len())")));
  (format!(concat!("extern \"C\" {{\n    fn {ks}({a0});\n    fn {bs}({ba});\n}}\n",
    "#[inline(never)]\npub fn {nm}({a1}) {{\n    unsafe {{ {ks}({a2}); }}\n}}\n",
    "#[inline(never)]\npub fn {nm}_batch({a3}) {{\n    let blocks = {le};\n    unsafe {{ {bs}({a4}); }}\n}}\n"),
    ks = ks, bs = bs, nm = k.name, le = le, a0 = c[0].join(", "), ba = ba,
    a1 = c[1].join(", "), a2 = c[2].join(", "), a3 = c[3].join(", "), a4 = a4), 4 + lb.len().saturating_sub(1) * 2) }
type PackedExecData = (Vec<u8>, Vec<u64>, usize, Option<(u64, usize)>);
type StartOut = Res<(Vec<u8>, usize)>;
type LowerOut = Res<(Kernel, RegAlloc, ModuleCode)>;
fn pack_exec_data(k: &Kernel, inputs: &[&[i64]]) -> Res<PackedExecData> {
  let blocks = if inputs.is_empty() { 1 } else {
    if inputs.len() != k.bufs.len() { return Err(CompileError::InvalidBuffer(inputs.len() as u8)) }
    let mut bo = None;
    for (b, sl) in k.bufs.iter().zip(inputs) {
      if let BufAccess::LookupTable(l) = b.access {
        if sl.len() < l { return Err(CompileError::OutOfBounds { idx: sl.len() as i64, limit: l as i64 }) }
      } else {
        let ln = usize::from(b.lanes);
        if ln == 0 || sl.len() % ln != 0 { return Err(berr("standalone input length not multiple of lanes")) }
        if bo.is_some_and(|p| p != sl.len() / ln) { return Err(berr("mismatched standalone block counts")) }
        bo = Some(sl.len() / ln) } }
    bo.unwrap_or(1) };
  let (mut data, mut offs, mut po) = (Vec::new(), Vec::with_capacity(k.bufs.len()), None);
  for (i, b) in k.bufs.iter().enumerate() {
    while data.len() % 64 != 0 { data.push(0) }
    let off = data.len() as u64;
    offs.push(off);
    let cnt = if let BufAccess::LookupTable(l) = b.access { l } else { blocks.saturating_mul(usize::from(b.lanes)) };
    let sl = inputs.get(i).copied().unwrap_or(&[]);
    for j in 0..cnt { b.elem_ty.push_le_elem(&mut data, sl.get(j).copied().unwrap_or(0)) }
    if b.access == BufAccess::MutNoAlias && po.is_none() { po = Some((off, blocks.saturating_mul(b.block_bytes()))) } }
  Ok((data, offs, blocks, po)) }
fn emit_x86_start(so: usize, bo: u64, dv: u64, bs: &[u64], bl: usize, po: Option<(u64, usize)>) -> StartOut {
  let mut e = X86Emit::default();
  e.emit(&[0x48, 0x83, 0xE4, 0xF0]);
  for (i, &o) in bs.iter().enumerate() {
    e.m64(*X86_ARGS.get(i).ok_or_else(|| berr("x86_64 _start arg overflow"))?, dv.saturating_add(o)); }
  e.m64(*X86_ARGS.get(bs.len()).ok_or_else(|| berr("x86_64 _start count overflow"))?, bl as u64);
  e.ei32(&[0xE8], ((bo as isize) - ((so + e.code.len() + 5) as isize)) as i32);
  if let Some((oo, ob)) = po.filter(|&(_, n)| n > 0) {
    let top = e.m64(6, dv.saturating_add(oo)).m64(2, ob as u64)
      .seq(&[&[0xB8, 1, 0, 0, 0], &[0xBF, 1, 0, 0, 0], &[0x0F, 5], &[0x31, 0xFF], &[0x31, 0xC9]]).code.len();
    e.seq(&[&[0x0F, 0xB6, 4, 0x0E], &[1, 0xC7], &[0x48, 0xFF, 0xC1], &[0x48, 0x39, 0xD1]]);
    e.seq(&[&[0x72, ((top as isize) - ((e.code.len() + 2) as isize)) as u8], &[0x83, 0xE7, 0x7F]])
  } else { e.emit(&[0x31, 0xFF]) };
  e.seq(&[&[0xB8, 0x3C, 0, 0, 0], &[0x0F, 5], &[0x0F, 0x0B]]);
  Ok((e.code, e.insns)) }
fn emit_arm_start(so: usize, bo: u64, dv: u64, bs: &[u64], bl: usize, po: Option<(u64, usize)>) -> StartOut {
  if bs.len() > 7 { return Err(berr("aarch64 _start arg overflow")) }
  let mut e = ArmEmit::default();
  for (i, &o) in bs.iter().enumerate() { e.m64(i as u8, dv.saturating_add(o), true); }
  e.m64(bs.len() as u8, bl as u64, true);
  e.put(0x9400_0000 | (((((bo as isize) - ((so + e.code.len()) as isize)) / 4) as u32) & 0x03FF_FFFF));
  if let Some((oo, ob)) = po.filter(|&(_, n)| n > 0) {
    let top = e.m64(19, dv.saturating_add(oo), true).m64(20, ob as u64, true).puts(&[
      0xD280_0020, arm64_rrr(0xAA00_0000, 19, 0, 31, 1), arm64_rrr(0xAA00_0000, 20, 0, 31, 2),
      0xD280_0808, 0xD400_0001, 0xD280_0000, 0xD280_0001]).code.len();
    e.puts(&[arm64_rrr(0x3860_0000, 1, 0x6800, 19, 2), arm64_rrr(0x8B00_0000, 2, 0, 0, 0),
      arm64_imm12(0x9100_0000, 1, 1, 1), arm64_rrr(0xEB00_0000, 20, 0, 1, 31)]);
    e.puts(&[0x5400_0003 | ((((((top as isize) - (e.code.len() as isize)) / 4) as u32) & 0x7_FFFF) << 5), 0x9240_1800])
  } else { e.put(0xD280_0000) };
  e.puts(&[0xD280_0BA8, 0xD400_0001, 0xD420_0000]);
  Ok((e.code, e.insns)) }
fn lower_kernel_code(k: &Kernel, b: RegBudget, isa: TargetIsa, opt: bool, cb: bool) -> LowerOut {
  k.validate()?;
  let base = if opt { optimize_kernel(k)? } else { k.clone() };
  let ts = |kn: &Kernel| schedule_and_verify(kn, b).and_then(|s| allocate_registers(kn, &s, isa, b).map(|r| (s, r)));
  let (low, (sched, ra)) = match ts(&base) {
    Ok(p) => (base.clone(), p),
    Err(e @ CompileError::RegisterSpill { .. }) => remat_try(&base, &[1, 3, 6, 12, 24, 48], |c| ts(c).ok()).ok_or(e)?,
    Err(e) => return Err(e) };
  let ds = low.infer_domains();
  let m = if isa == TargetIsa::X86_64 { emit_x86_module(&low, &base, &sched, &ra, &ds, b.max_vecs, cb)? }
    else { emit_arm_module(&low, &sched, &ra, &ds, cb)? };
  Ok((low, ra, m)) }
pub fn emit_kernel_elf(k: &Kernel, b: RegBudget, isa: TargetIsa, opt: bool, cb: bool) -> Res<CompiledObject> {
  let (low, ra, (code_bytes, syms, relocs, direct_insns)) = lower_kernel_code(k, b, isa, opt, cb)?;
  let elf_bytes = emit_elf64_object(isa, &code_bytes, &syms, &relocs)?;
  let sum = verify_elf64_object(&elf_bytes)?;
  if sum.machine != isa.e_machine() || !sum.has_nonexec_gnu_stack || sum.symbols.len() != 2 {
    return Err(berr("elf64 airlock verification mismatch")) }
  let (kernel_sym, batch_sym) = (syms[0].name.clone(), syms[1].name.clone());
  let (rust_ffi_shim, shim_glue_insns) = emit_ffi_shim(&low, &kernel_sym, &batch_sym);
  let tot = direct_insns + shim_glue_insns;
  Ok(CompiledObject { isa, kernel_sym, batch_sym, code_bytes, elf_bytes, direct_insns, shim_glue_insns,
    direct_share_pct: if tot == 0 { 100 } else { ((direct_insns * 100) / tot) as u8 },
    zero_spills: ra.spills == 0, rust_ffi_shim }) }
pub fn compile_kernel_direct(k: &Kernel, b: RegBudget, isa: TargetIsa) -> Res<CompiledObject> {
  emit_kernel_elf(k, b, isa, true, false) }
pub fn compile_rust_to_elf(src: &str, b: RegBudget, isa: TargetIsa) -> Res<CompiledObject> {
  compile_kernel_direct(&parse_rust_kernel(src)?, b, isa) }
pub fn compile_standalone_elf_exec(k: &Kernel, b: RegBudget, isa: TargetIsa, inp: &[&[i64]]) -> Res<StandaloneExec> {
  let (low, ra, (mut code_bytes, mut syms, _, mut direct_insns)) = lower_kernel_code(k, b, isa, true, false)?;
  let (data, boffs, blocks, pout) = pack_exec_data(&low, inp)?;
  let pad: &[u8] = if isa == TargetIsa::X86_64 { &[0x90] } else { &0xD503_201Fu32.to_le_bytes() };
  while code_bytes.len() % 16 != 0 { code_bytes.extend_from_slice(pad) }
  let (boff, soff) = (syms.get(1).ok_or_else(|| berr("missing batch symbol"))?.offset, code_bytes.len());
  let emit_st = |va: u64| match isa {
    TargetIsa::X86_64 => emit_x86_start(soff, boff, va, &boffs, blocks, pout),
    TargetIsa::Aarch64 => emit_arm_start(soff, boff, va, &boffs, blocks, pout) };
  let (p1, _) = emit_st(0)?;
  let data_vaddr = elf64_exec_data_vaddr(soff + p1.len());
  let (sb, si) = emit_st(data_vaddr)?;
  if sb.len() != p1.len() { return Err(berr("standalone _start length invariance violated")) }
  code_bytes.extend_from_slice(&sb);
  direct_insns += si;
  syms.push(ElfSymbol { name: String::from("_start"), offset: soff as u64, size: sb.len() as u64 });
  let elf_bytes = emit_elf64_executable(isa, &code_bytes, soff as u64, &data, 0, &syms)?;
  let (sum, entry_vaddr) = (verify_elf64_executable(&elf_bytes)?, ELF_EXEC_TEXT_VADDR + soff as u64);
  if sum.machine != isa.e_machine() || !sum.has_nonexec_gnu_stack || sum.symbols.len() != 3
    || sum.entry_vaddr != entry_vaddr || sum.data_vaddr != data_vaddr {
    return Err(berr("elf64 exec airlock verification mismatch")) }
  Ok(StandaloneExec { isa, kernel_sym: syms[0].name.clone(), batch_sym: syms[1].name.clone(),
    entry_sym: "_start".into(), entry_vaddr, data_vaddr, code_bytes, elf_bytes, direct_insns, shim_glue_insns: 0,
    direct_share_pct: 100, zero_spills: ra.spills == 0, blocks, output_bytes: pout.map_or(0, |(_, n)| n) }) }
pub fn compile_rust_to_elf_exec(src: &str, b: RegBudget, isa: TargetIsa, inp: &[&[i64]]) -> Res<StandaloneExec> {
  compile_standalone_elf_exec(&parse_rust_kernel(src)?, b, isa, inp) }
