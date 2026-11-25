use std::collections::BTreeSet;
use crate::CompiledKernel;
const SVO_BAN: &[&str] = &["the", "is", "are", "was", "were", "this", "that", "should", "must", "we", "our"];
define_errors! { Ntf1Error,
    NonAscii(usize) | Self::NonAscii(l) => "ntf1: non-ASCII line {l}";
    MissingHeader | Self::MissingHeader => "ntf1: missing header";
    SvoTax { line: usize, word: String } | Self::SvoTax { line, word } => "ntf1: SVO '{word}' line {line}";
    Syntax { line: usize, detail: String } | Self::Syntax { line, detail } => "ntf1: syntax line {line}: {detail}";
    UncoveredFault(String) | Self::UncoveredFault(a) => "ntf1: uncovered fault {a}";
    UngroundedAirlock(String) | Self::UngroundedAirlock(a) => "ntf1: ungrounded airlock {a}";
    AirlockMismatch | Self::AirlockMismatch => "ntf1: airlock mismatch";
}
#[derive(Clone, Debug, PartialEq, Eq)] pub enum Ntf1Stmt {
    Forward { lhs: Vec<String>, rhs: Vec<String> }, Trap { fault: Vec<String>, guard: Vec<String> },
    Airlock { left: Vec<String>, right: Vec<String> }, Pipe(Vec<String>),
}
impl Ntf1Stmt { fn parts(&self) -> (bool, bool, &'static str, &'static str, &[String], &[String]) { match self {
    Self::Forward { lhs, rhs } => (true, false, "+t ", " -> ", lhs, rhs),
    Self::Trap { fault, guard } => (true, true, "-t ", " <- ", fault, guard),
    Self::Airlock { left, right } => (false, false, "0t ", " >< ", left, right),
    Self::Pipe(c) => (true, false, "", "", c, &[]),
}}}
#[derive(Clone, Debug, PartialEq, Eq)] pub struct Ntf1Spec { pub module: Option<String>, pub stmts: Vec<Ntf1Stmt> }
fn syn(line: usize, detail: impl Into<String>) -> Ntf1Error { Ntf1Error::Syntax { line, detail: detail.into() } }
fn check_svo_line(line: usize, raw: &str) -> Result<(), Ntf1Error> {
    let (mut in_q, mut w) = (false, String::new());
    for ch in raw.chars().chain([' ']) {
        if ch == '"' { (in_q, w) = (!in_q, String::new()); }
        else if !in_q && (ch.is_ascii_alphanumeric() || ch == '_') { w.push(ch.to_ascii_lowercase()); }
        else if !in_q && SVO_BAN.contains(&w.as_str()) { return Err(Ntf1Error::SvoTax { line, word: w }); }
        else if !in_q { w.clear(); }
    }
    Ok(())
}
fn split_and_tokenize(raw: &str, line: usize) -> Result<Vec<Vec<String>>, Ntf1Error> {
    let (mut in_q, mut cur, mut toks, mut stmts) = (false, String::new(), Vec::new(), Vec::new());
    for ch in raw.chars() {
        if ch == '"' {
            cur.push(ch);
            in_q = !in_q;
            if !in_q { toks.push(std::mem::take(&mut cur)); }
        } else if in_q { cur.push(ch); }
        else if ch == '#' { break; }
        else if ch == ';' || ch.is_ascii_whitespace() {
            if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); }
            if ch == ';' && !toks.is_empty() { stmts.push(std::mem::take(&mut toks)); }
        } else { cur.push(ch); }
    }
    if in_q { return Err(syn(line, "unterminated quote")); }
    if !cur.is_empty() { toks.push(cur); }
    if !toks.is_empty() { stmts.push(toks); }
    Ok(stmts)
}
fn is_atom(t: &str) -> bool {
    let ok = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'/' | b'@' | b'=' | b'-');
    (t.len() >= 2 && t.starts_with('"') && t.ends_with('"'))
        || (!t.is_empty() && !matches!(t, "+t" | "-t" | "0t" | "->" | "<-" | "><" | "|>") && t.bytes().all(ok))
}
fn parse_chain(toks: &[String], line: usize) -> Result<Vec<String>, Ntf1Error> {
    if toks.is_empty() || toks.len() % 2 == 0 { return Err(syn(line, "invalid pipeline chain")); }
    toks.iter().enumerate().filter_map(|(i, t)| match (i % 2 == 0, t.as_str()) {
        (true, _) if is_atom(t) => Some(Ok(t.clone())),
        (true, _) => Some(Err(syn(line, format!("bad atom '{t}'")))),
        (false, "|>") => None,
        (false, _) => Some(Err(syn(line, format!("expected '|>', got '{t}'")))),
    }).collect()
}
fn parse_pair(toks: &[String], op: &str, line: usize) -> Result<(Vec<String>, Vec<String>), Ntf1Error> {
    let mut it = toks.iter().enumerate().filter(|(_, t)| *t == op).map(|(i, _)| i);
    let idx = it.next().ok_or_else(|| syn(line, format!("missing '{op}'")))?;
    if it.next().is_some() { return Err(syn(line, format!("duplicate '{op}'"))); }
    Ok((parse_chain(&toks[..idx], line)?, parse_chain(&toks[idx + 1..], line)?))
}
fn parse_stmt(toks: &[String], line: usize) -> Result<Ntf1Stmt, Ntf1Error> { match toks[0].as_str() {
    "+t" => parse_pair(&toks[1..], "->", line).map(|(lhs, rhs)| Ntf1Stmt::Forward { lhs, rhs }),
    "-t" => parse_pair(&toks[1..], "<-", line).map(|(fault, guard)| Ntf1Stmt::Trap { fault, guard }),
    "0t" if toks[1..].iter().any(|t| t == "->" || t == "<-") => Err(syn(line, "0t forbids -> or <-")),
    "0t" => parse_pair(&toks[1..], "><", line).map(|(left, right)| Ntf1Stmt::Airlock { left, right }),
    _ => parse_chain(toks, line).and_then(|c| (c.len() >= 2).then_some(Ntf1Stmt::Pipe(c))
        .ok_or_else(|| syn(line, "expected combinator"))),
}}
pub fn parse_ntf1(src: &str) -> Result<Ntf1Spec, Ntf1Error> {
    let (mut header_seen, mut module, mut stmts) = (false, None, Vec::new());
    for (line, raw) in (1..).zip(src.lines()) {
        if !raw.is_ascii() { return Err(Ntf1Error::NonAscii(line)); }
        check_svo_line(line, raw)?;
        for toks in split_and_tokenize(raw, line)? {
            if !header_seen {
                if toks[0] != "NTF1" || toks.len() > 2 { return Err(syn(line, "expected NTF1 header")); }
                if let Some(m) = toks.get(1) {
                    let ok = m.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
                    if !ok { return Err(syn(line, format!("bad module '{m}'"))); }
                    module = Some(m.clone());
                }
                header_seen = true;
            } else { stmts.push(parse_stmt(&toks, line)?); }
        }
    }
    if !header_seen { return Err(Ntf1Error::MissingHeader); }
    let (mut trapped, mut grounded, mut all) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for (gr, is_trap, _, _, l, r) in stmts.iter().map(Ntf1Stmt::parts) {
        if is_trap { trapped.extend(l.iter().map(String::as_str)); }
        all.extend(l.iter().chain(r).map(String::as_str));
        if gr { grounded.extend(l.iter().chain(r).map(String::as_str)); }
    }
    if let Some(&a) = all.iter().find(|&&a| a.starts_with("Fault.") && !trapped.contains(a)) {
        return Err(Ntf1Error::UncoveredFault(a.to_string()));
    }
    if let Some(a) = stmts.iter().find_map(|s| match s {
        Ntf1Stmt::Airlock { left: l, right: r } if l != r => l.iter().chain(r).find(|x| !grounded.contains(x.as_str())),
        _ => None,
    }) { return Err(Ntf1Error::UngroundedAirlock(a.clone())); }
    Ok(Ntf1Spec { module, stmts })
}
pub fn emit_ntf1(spec: &Ntf1Spec) -> String {
    let mut out = spec.module.as_deref().map_or_else(|| "NTF1\n".into(), |m| format!("NTF1 {m}\n"));
    for (_, _, pre, sep, l, r) in spec.stmts.iter().map(Ntf1Stmt::parts) {
        out.push_str(&format!("{pre}{}{sep}{}\n", l.join(" |> "), r.join(" |> ")));
    }
    out
}
pub fn verify_ntf1_airlock(src: &str) -> Result<String, Ntf1Error> {
    let spec = parse_ntf1(src)?;
    let canon = emit_ntf1(&spec);
    let rep = parse_ntf1(&canon)?;
    (rep == spec && emit_ntf1(&rep) == canon).then_some(canon).ok_or(Ntf1Error::AirlockMismatch)
}
pub fn export_kernel_ntf1(c: &CompiledKernel) -> String {
    let (k, ol, sc, oc) = (&c.optimized.name, c.original_cost.total_latency, &c.schedule, &c.optimized_cost);
    let (g, v, dr, bc) = (sc.peak_gprs, sc.peak_vecs, oc.divs_rems, oc.bounds_checks);
    format!("NTF1 kernel.{k}\n+t Kernel.{k} -> IR |> EGraph |> SethiUllman |> RustO3\n\
        +t Cost.orig/lat={ol} -> Cost.opt/lat={}\n+t Regs/gprs={g} -> Regs/vecs={v}\n\
        -t Fault.RegSpill <- Guard.RegBudget_gprs={g}_vecs={v}\n-t Fault.Div0 <- Guard.DivRem={dr}\n\
        -t Fault.OOB <- Guard.BoundsChecks={bc}\n0t Kernel.{k} >< RustO3\n", oc.total_latency)
}
