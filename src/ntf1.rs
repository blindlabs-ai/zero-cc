use std::collections::BTreeSet;
use std::fmt;

use crate::CompiledKernel;

const SVO_BAN: &[&str] = &["the", "is", "are", "was", "were", "this", "that", "should", "must", "we", "our"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ntf1Error {
    NonAscii(usize), MissingHeader, SvoTax { line: usize, word: String },
    Syntax { line: usize, detail: String }, UncoveredFault(String), UngroundedAirlock(String), AirlockMismatch,
}

impl fmt::Display for Ntf1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ntf1: {self:?}") }
}

impl std::error::Error for Ntf1Error {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ntf1Stmt {
    Forward { lhs: Vec<String>, rhs: Vec<String> }, Trap { fault: Vec<String>, guard: Vec<String> },
    Airlock { left: Vec<String>, right: Vec<String> }, Pipe(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ntf1Spec { pub module: Option<String>, pub stmts: Vec<Ntf1Stmt> }

fn syn(line: usize, detail: impl Into<String>) -> Ntf1Error { Ntf1Error::Syntax { line, detail: detail.into() } }

fn check_svo_line(line: usize, raw: &str) -> Result<(), Ntf1Error> {
    let (mut in_q, mut word) = (false, String::new());
    for ch in raw.chars().chain(std::iter::once(' ')) {
        if ch == '"' { in_q = !in_q; word.clear(); }
        else if !in_q && (ch.is_ascii_alphanumeric() || ch == '_') { word.push(ch.to_ascii_lowercase()); }
        else if !in_q && !word.is_empty() {
            if SVO_BAN.contains(&word.as_str()) { return Err(Ntf1Error::SvoTax { line, word }); }
            word.clear();
        }
    }
    Ok(())
}

fn split_and_tokenize(raw: &str, line: usize) -> Result<Vec<Vec<String>>, Ntf1Error> {
    let (mut in_q, mut cur, mut toks, mut stmts) = (false, String::new(), Vec::new(), Vec::new());
    for ch in raw.chars() {
        if ch == '"' { cur.push(ch); in_q = !in_q; if !in_q { toks.push(std::mem::take(&mut cur)); } }
        else if in_q { cur.push(ch); }
        else if ch == '#' { break; }
        else if ch == ';' {
            if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); }
            if !toks.is_empty() { stmts.push(std::mem::take(&mut toks)); }
        } else if ch.is_ascii_whitespace() { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
        else { cur.push(ch); }
    }
    if in_q { return Err(syn(line, "unterminated quote")); }
    if !cur.is_empty() { toks.push(cur); }
    if !toks.is_empty() { stmts.push(toks); }
    Ok(stmts)
}

fn is_atom(tok: &str) -> bool {
    (tok.len() >= 2 && tok.starts_with('"') && tok.ends_with('"'))
        || (!tok.is_empty() && !matches!(tok, "+t" | "-t" | "0t" | "->" | "<-" | "><" | "|>")
            && tok.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'/' | b'@' | b'=' | b'-')))
}

fn parse_chain(toks: &[String], line: usize) -> Result<Vec<String>, Ntf1Error> {
    if toks.is_empty() || toks.len() % 2 == 0 { return Err(syn(line, "invalid pipeline chain")); }
    let mut out = Vec::with_capacity(toks.len().div_ceil(2));
    for (i, t) in toks.iter().enumerate() {
        if i % 2 == 0 {
            if !is_atom(t) { return Err(syn(line, format!("bad atom '{t}'"))); }
            out.push(t.clone());
        } else if t != "|>" { return Err(syn(line, format!("expected '|>', got '{t}'"))); }
    }
    Ok(out)
}

fn split_op<'a>(toks: &'a [String], op: &str, line: usize) -> Result<(&'a [String], &'a [String]), Ntf1Error> {
    let mut pos = None;
    for (i, t) in toks.iter().enumerate() {
        if t == op { if pos.is_some() { return Err(syn(line, format!("duplicate '{op}'"))); } pos = Some(i); }
    }
    let idx = pos.ok_or_else(|| syn(line, format!("missing '{op}'")))?;
    Ok((&toks[..idx], &toks[idx + 1..]))
}

pub fn parse_ntf1(src: &str) -> Result<Ntf1Spec, Ntf1Error> {
    let (mut header_seen, mut module, mut stmts) = (false, None, Vec::new());
    for (idx, raw) in src.lines().enumerate() {
        let line = idx + 1;
        if !raw.is_ascii() { return Err(Ntf1Error::NonAscii(line)); }
        check_svo_line(line, raw)?;
        for toks in split_and_tokenize(raw, line)? {
            if !header_seen {
                if toks[0] != "NTF1" || toks.len() > 2 { return Err(syn(line, "expected NTF1 header")); }
                if let Some(m) = toks.get(1) {
                    if !m.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-')) {
                        return Err(syn(line, format!("bad module '{m}'")));
                    }
                    module = Some(m.clone());
                }
                header_seen = true;
                continue;
            }
            let stmt = match toks[0].as_str() {
                "+t" => { let (l, r) = split_op(&toks[1..], "->", line)?; Ntf1Stmt::Forward { lhs: parse_chain(l, line)?, rhs: parse_chain(r, line)? } }
                "-t" => { let (l, r) = split_op(&toks[1..], "<-", line)?; Ntf1Stmt::Trap { fault: parse_chain(l, line)?, guard: parse_chain(r, line)? } }
                "0t" => {
                    if toks[1..].iter().any(|t| t == "->" || t == "<-") { return Err(syn(line, "0t forbids -> or <-")); }
                    let (l, r) = split_op(&toks[1..], "><", line)?;
                    Ntf1Stmt::Airlock { left: parse_chain(l, line)?, right: parse_chain(r, line)? }
                }
                _ => {
                    let c = parse_chain(&toks, line)?;
                    if c.len() < 2 { return Err(syn(line, "expected combinator")); }
                    Ntf1Stmt::Pipe(c)
                }
            };
            stmts.push(stmt);
        }
    }
    if !header_seen { return Err(Ntf1Error::MissingHeader); }
    let (mut trapped, mut grounded, mut all) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for s in &stmts {
        match s {
            Ntf1Stmt::Forward { lhs, rhs } => for a in lhs.iter().chain(rhs) { grounded.insert(a.as_str()); all.insert(a.as_str()); },
            Ntf1Stmt::Trap { fault, guard } => {
                for a in fault { trapped.insert(a.as_str()); grounded.insert(a.as_str()); all.insert(a.as_str()); }
                for a in guard { grounded.insert(a.as_str()); all.insert(a.as_str()); }
            }
            Ntf1Stmt::Pipe(c) => for a in c { grounded.insert(a.as_str()); all.insert(a.as_str()); },
            Ntf1Stmt::Airlock { left, right } => for a in left.iter().chain(right) { all.insert(a.as_str()); },
        }
    }
    for a in all { if a.starts_with("Fault.") && !trapped.contains(a) { return Err(Ntf1Error::UncoveredFault(a.to_string())); } }
    for s in &stmts {
        if let Ntf1Stmt::Airlock { left, right } = s {
            if left != right {
                for a in left.iter().chain(right) { if !grounded.contains(a.as_str()) { return Err(Ntf1Error::UngroundedAirlock(a.clone())); } }
            }
        }
    }
    Ok(Ntf1Spec { module, stmts })
}

pub fn emit_ntf1(spec: &Ntf1Spec) -> String {
    let mut out = match &spec.module { Some(m) => format!("NTF1 {m}\n"), None => String::from("NTF1\n") };
    for s in &spec.stmts {
        match s {
            Ntf1Stmt::Forward { lhs, rhs } => out.push_str(&format!("+t {} -> {}\n", lhs.join(" |> "), rhs.join(" |> "))),
            Ntf1Stmt::Trap { fault, guard } => out.push_str(&format!("-t {} <- {}\n", fault.join(" |> "), guard.join(" |> "))),
            Ntf1Stmt::Airlock { left, right } => out.push_str(&format!("0t {} >< {}\n", left.join(" |> "), right.join(" |> "))),
            Ntf1Stmt::Pipe(c) => out.push_str(&format!("{}\n", c.join(" |> "))),
        }
    }
    out
}

pub fn verify_ntf1_airlock(src: &str) -> Result<String, Ntf1Error> {
    let spec = parse_ntf1(src)?;
    let canon = emit_ntf1(&spec);
    let reparsed = parse_ntf1(&canon)?;
    if reparsed != spec || emit_ntf1(&reparsed) != canon { return Err(Ntf1Error::AirlockMismatch); }
    Ok(canon)
}

pub fn export_kernel_ntf1(c: &CompiledKernel) -> String {
    let k = &c.optimized.name;
    format!(
        "NTF1 kernel.{k}\n+t Kernel.{k} -> IR |> EGraph |> SethiUllman |> RustO3\n\
         +t Cost.orig/lat={} -> Cost.opt/lat={}\n+t Regs/gprs={} -> Regs/vecs={}\n\
         -t Fault.RegSpill <- Guard.RegBudget_gprs={}_vecs={}\n\
         -t Fault.Div0 <- Guard.DivRem={}\n-t Fault.OOB <- Guard.BoundsChecks={}\n\
         0t Kernel.{k} >< RustO3\n",
        c.original_cost.total_latency, c.optimized_cost.total_latency,
        c.schedule.peak_gprs, c.schedule.peak_vecs, c.schedule.peak_gprs, c.schedule.peak_vecs,
        c.optimized_cost.divs_rems, c.optimized_cost.bounds_checks,
    )
}
