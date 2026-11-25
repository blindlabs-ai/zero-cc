use crate::ir::CompileError;
pub const ET_REL: u16 = 1;
pub const ET_EXEC: u16 = 2;
pub const EM_X86_64: u16 = 62;
pub const EM_AARCH64: u16 = 183;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_AARCH64_CALL26: u32 = 283;
pub const ELF_EXEC_BASE: u64 = 0x0040_0000;
pub const ELF_EXEC_TEXT_OFF: u64 = 0xF0;
pub const ELF_EXEC_TEXT_VADDR: u64 = ELF_EXEC_BASE + ELF_EXEC_TEXT_OFF;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PG: u64 = 0x1000;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum TargetIsa { X86_64, Aarch64 }
impl TargetIsa {
    pub const fn e_machine(self) -> u16 { if matches!(self, Self::X86_64) { EM_X86_64 } else { EM_AARCH64 } }}
#[derive(Clone, Debug, PartialEq, Eq)] pub struct ElfSymbol { pub name: String, pub offset: u64, pub size: u64 }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElfReloc { pub offset: u64, pub sym_idx: u32, pub r_type: u32, pub addend: i64 }
#[derive(Clone, Debug, PartialEq, Eq)] pub struct ElfSummary {
    pub machine: u16, pub text_size: u64, pub rela_count: usize,
    pub has_nonexec_gnu_stack: bool, pub symbols: Vec<ElfSymbol> }
#[derive(Clone, Debug, PartialEq, Eq)] pub struct ElfExecSummary {
    pub machine: u16, pub entry_vaddr: u64, pub text_vaddr: u64, pub text_size: u64, pub data_vaddr: u64,
    pub data_filesz: u64, pub data_memsz: u64, pub has_nonexec_gnu_stack: bool, pub symbols: Vec<ElfSymbol> }
/// @domain text_len: 1..=16777216
pub const fn elf64_exec_data_vaddr(text_len: usize) -> u64 {
    ELF_EXEC_BASE + ((ELF_EXEC_TEXT_OFF + (text_len as u64) + (PG - 1)) & !(PG - 1)) }
fn berr(msg: &str) -> CompileError { CompileError::BackendError(msg.into()) }
fn push_le(out: &mut Vec<u8>, w: usize, xs: &[u64]) { for &x in xs { out.extend_from_slice(&x.to_le_bytes()[..w]); }}
fn pad_align(out: &mut Vec<u8>, a: usize) { out.resize(out.len().div_ceil(a).saturating_mul(a), 0); }
fn add_str(tab: &mut Vec<u8>, s: &str) -> u32 {
    let off = tab.len() as u32;
    tab.extend(s.bytes().chain([0]));
    off }
fn add_strs<const N: usize>(tab: &mut Vec<u8>, ss: [&str; N]) -> [u32; N] { ss.map(|s| add_str(tab, s)) }
fn emit_sec(out: &mut Vec<u8>, align: usize, data: &[u8]) -> u64 {
    pad_align(out, align);
    let off = out.len() as u64;
    out.extend_from_slice(data);
    off }
fn push_shdr(out: &mut Vec<u8>, u32s: [u32; 4], u64s: [u64; 6]) {
    push_le(out, 4, &[u64::from(u32s[0]), u64::from(u32s[1])]);
    push_le(out, 8, &u64s[..4]);
    push_le(out, 4, &[u64::from(u32s[2]), u64::from(u32s[3])]);
    push_le(out, 8, &u64s[4..]);
}
fn push_phdr(out: &mut Vec<u8>, hdr: [u32; 2], f: [u64; 5]) {
    push_le(out, 4, &[u64::from(hdr[0]), u64::from(hdr[1])]);
    push_le(out, 8, &[f[0], f[1], f[1], f[2], f[3], f[4]]);
}
fn push_ehdr(out: &mut Vec<u8>, e_type: u16, isa: TargetIsa, entry: u64, ph: [u64; 3], sh: [u64; 2]) {
    out.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    push_le(out, 2, &[u64::from(e_type), u64::from(isa.e_machine())]);
    push_le(out, 4, &[1]);
    push_le(out, 8, &[entry, ph[0], 0]);
    push_le(out, 4, &[0]);
    push_le(out, 2, &[64, ph[1], ph[2], 64, sh[0], sh[1]]);
}
fn build_symtab(text_len: usize, syms: &[ElfSymbol], base: u64) -> Result<(Vec<u8>, Vec<u8>), CompileError> {
    let (mut strtab, mut symtab) = (vec![0u8], vec![0u8; 24]);
    let mut push_sym = |nm: u32, info: u8, val: u64, sz: u64| {
        push_le(&mut symtab, 4, &[u64::from(nm)]);
        symtab.extend_from_slice(&[info, 0, 1, 0]);
        push_le(&mut symtab, 8, &[val, sz]);
    };
    push_sym(add_str(&mut strtab, ""), 0x03, base, 0);
    for sym in syms {
        if sym.offset.saturating_add(sym.size) > text_len as u64 { return Err(berr("symbol exceeds .text")); }
        push_sym(add_str(&mut strtab, &sym.name), 0x12, base.saturating_add(sym.offset), sym.size);
    }
    Ok((symtab, strtab)) }
fn finish_tables(out: &mut Vec<u8>, symtab: &[u8], strtab: &[u8], shstr: &[u8]) -> ([u64; 3], [u64; 3]) {
    pad_align(out, 8);
    let offs = [emit_sec(out, 1, symtab), emit_sec(out, 1, strtab), emit_sec(out, 1, shstr)];
    pad_align(out, 8);
    let shoff = out.len() as u64;
    if let Some(slot) = out.get_mut(40..48) { slot.copy_from_slice(&shoff.to_le_bytes()); }
    (offs, [symtab.len() as u64, strtab.len() as u64, shstr.len() as u64]) }
fn push_tail_shdrs(out: &mut Vec<u8>, names: [u32; 3], offs: [u64; 3], lens: [u64; 3], str_link: u32) {
    push_shdr(out, [names[0], 2, str_link, 2], [0, 0, offs[0], lens[0], 8, 24]);
    push_shdr(out, [names[1], 3, 0, 0], [0, 0, offs[1], lens[1], 1, 0]);
    push_shdr(out, [names[2], 3, 0, 0], [0, 0, offs[2], lens[2], 1, 0]);
}
pub fn emit_elf64_object(isa: TargetIsa, text: &[u8], syms: &[ElfSymbol],
    relocs: &[ElfReloc]) -> Result<Vec<u8>, CompileError> {
    if text.is_empty() || syms.is_empty() { return Err(berr("elf64 requires non-empty .text and symbols")); }
    let mut shstr = vec![0u8];
    let (n_text, n_rela) = (add_str(&mut shstr, ".text"),
        if relocs.is_empty() { 0 } else { add_str(&mut shstr, ".rela.text") });
    let [n_note, t0, t1, t2] = add_strs(&mut shstr, [".note.GNU-stack", ".symtab", ".strtab", ".shstrtab"]);
    let ((symtab, strtab), mut rela) = (build_symtab(text.len(), syms, 0)?, Vec::with_capacity(relocs.len() * 24));
    for rel in relocs {
        let elf_sym = u64::from(rel.sym_idx) + 2;
        if (elf_sym as usize) * 24 >= symtab.len() { return Err(berr("reloc sym out of bounds")); }
        push_le(&mut rela, 8, &[rel.offset, (elf_sym << 32) | u64::from(rel.r_type), rel.addend as u64]);
    }
    let (has_rela, shnum) = (!rela.is_empty(), if rela.is_empty() { 6u64 } else { 7u64 });
    let mut out = Vec::with_capacity(512 + text.len() + symtab.len() + rela.len());
    push_ehdr(&mut out, ET_REL, isa, 0, [0, 0, 0], [shnum, shnum - 1]);
    let (text_off, rela_off) = (emit_sec(&mut out, 16, text), emit_sec(&mut out, 8, &rela));
    let (note_off, (offs, lens)) = (out.len() as u64, finish_tables(&mut out, &symtab, &strtab, &shstr));
    let sym_idx = if has_rela { 4u32 } else { 3u32 };
    push_shdr(&mut out, [0; 4], [0; 6]);
    push_shdr(&mut out, [n_text, 1, 0, 0], [6, 0, text_off, text.len() as u64, 16, 0]);
    if has_rela { push_shdr(&mut out, [n_rela, 4, sym_idx, 1], [0x40, 0, rela_off, rela.len() as u64, 8, 24]); }
    push_shdr(&mut out, [n_note, 1, 0, 0], [0, 0, note_off, 0, 1, 0]);
    push_tail_shdrs(&mut out, [t0, t1, t2], offs, lens, sym_idx + 1);
    Ok(out) }
pub fn emit_elf64_executable(isa: TargetIsa, text: &[u8], entry_off: u64, data: &[u8],
    bss_size: usize, syms: &[ElfSymbol]) -> Result<Vec<u8>, CompileError> {
    if text.is_empty() || text.len() > 16_777_216 || syms.is_empty() {
        return Err(berr("elf64 exec requires valid non-empty .text and symbols")); }
    if entry_off >= text.len() as u64 { return Err(berr("elf64 exec entry exceeds .text")); }
    let data_vaddr = elf64_exec_data_vaddr(text.len());
    let (data_off, text_end) = (data_vaddr - ELF_EXEC_BASE, ELF_EXEC_TEXT_OFF + text.len() as u64);
    let (rw_fsz, rw_msz) = (data.len() as u64, data.len().saturating_add(bss_size).max(64) as u64);
    let mut shstr = vec![0u8];
    let [n_text, n_data, n_bss, t0, t1, t2] = add_strs(&mut shstr,
        [".text", ".data", ".bss", ".symtab", ".strtab", ".shstrtab"]);
    let (symtab, strtab) = build_symtab(text.len(), syms, ELF_EXEC_TEXT_VADDR)?;
    let mut out = Vec::with_capacity((data_off as usize) + data.len() + symtab.len() + strtab.len() + 512);
    push_ehdr(&mut out, ET_EXEC, isa, ELF_EXEC_TEXT_VADDR + entry_off, [64, 56, 3], [7, 6]);
    push_phdr(&mut out, [1, 5], [0, ELF_EXEC_BASE, text_end, text_end, PG]);
    push_phdr(&mut out, [1, 6], [data_off, data_vaddr, rw_fsz, rw_msz, PG]);
    push_phdr(&mut out, [PT_GNU_STACK, 6], [0, 0, 0, 0, 16]);
    pad_align(&mut out, 16);
    if out.len() as u64 != ELF_EXEC_TEXT_OFF { return Err(berr("elf64 exec text offset mismatch")); }
    out.extend_from_slice(text);
    if (out.len() as u64) > data_off { return Err(berr("elf64 exec text overlaps data page")); }
    out.resize(data_off as usize, 0);
    out.extend_from_slice(data);
    let (offs, lens) = finish_tables(&mut out, &symtab, &strtab, &shstr);
    push_shdr(&mut out, [0; 4], [0; 6]);
    push_shdr(&mut out, [n_text, 1, 0, 0], [6, ELF_EXEC_TEXT_VADDR, ELF_EXEC_TEXT_OFF, text.len() as u64, 16, 0]);
    push_shdr(&mut out, [n_data, 1, 0, 0], [3, data_vaddr, data_off, rw_fsz, 64, 0]);
    push_shdr(&mut out, [n_bss, 8, 0, 0], [3, data_vaddr + rw_fsz, data_off + rw_fsz, rw_msz - rw_fsz, 64, 0]);
    push_tail_shdrs(&mut out, [t0, t1, t2], offs, lens, 5);
    Ok(out) }
fn rd(b: &[u8], off: usize, w: usize) -> Result<u64, CompileError> {
    let s = b.get(off..off + w).ok_or_else(|| berr("elf64 unexpected eof"))?;
    Ok(s.iter().enumerate().fold(0u64, |acc, (i, &byte)| acc | (u64::from(byte) << (i * 8)))) }
fn rd_cstr(tab: &[u8], off: u32) -> Result<String, CompileError> {
    let tail = tab.get(off as usize..).ok_or_else(|| berr("elf64 bad str offset"))?;
    let end = tail.iter().position(|&c| c == 0).ok_or_else(|| berr("elf64 unterminated str"))?;
    std::str::from_utf8(&tail[..end]).map(String::from).map_err(|_| berr("elf64 non-utf8 str")) }
#[derive(Default)] struct ScannedShdrs<'a> {
    text_vaddr: u64, text_size: u64, rela_count: usize,
    has_nonexec_gnu_stack: bool, sym_slice: &'a [u8], str_slice: &'a [u8] }
fn scan_shdrs(bytes: &[u8], check_all: bool) -> Result<ScannedShdrs<'_>, CompileError> {
    let (shoff, entsz) = (rd(bytes, 40, 8)? as usize, rd(bytes, 58, 2)? as usize);
    let (shnum, shstrndx) = (rd(bytes, 60, 2)? as usize, rd(bytes, 62, 2)? as usize);
    if entsz != 64 || shnum < 5 || shstrndx >= shnum { return Err(berr("elf64 bad section header table")); }
    let sh_tab = bytes.get(shoff..shoff.saturating_add(shnum.saturating_mul(64)))
        .ok_or_else(|| berr("elf64 truncated shdrs"))?;
    let h = shoff + shstrndx * 64;
    let (so, ss) = (rd(bytes, h + 24, 8)? as usize, rd(bytes, h + 32, 8)? as usize);
    let shstr = bytes.get(so..so.saturating_add(ss)).ok_or_else(|| berr("elf64 truncated shstrtab"))?;
    let mut out = ScannedShdrs::default();
    for i in 0..shnum {
        let b = i * 64;
        let (nm, ty, fl) = (rd(sh_tab, b, 4)? as u32, rd(sh_tab, b + 4, 4)? as u32, rd(sh_tab, b + 8, 8)?);
        let (ad, off, sz) = (rd(sh_tab, b + 16, 8)?, rd(sh_tab, b + 24, 8)? as usize, rd(sh_tab, b + 32, 8)? as usize);
        let sec = bytes.get(off..off.saturating_add(sz));
        if check_all && sec.is_none() { return Err(berr("elf64 section out of bounds")); }
        match rd_cstr(shstr, nm)?.as_str() {
            ".text" if ty == 1 && (fl & 4) != 0 => {
                if sec.is_none() { return Err(berr("elf64 .text out of bounds")); }
                (out.text_vaddr, out.text_size) = (ad, sz as u64);
            }
            ".rela.text" if ty == 4 && sz % 24 == 0 => out.rela_count = sz / 24,
            ".note.GNU-stack" if ty == 1 && (fl & 4) == 0 => out.has_nonexec_gnu_stack = true,
            ".symtab" if ty == 2 && sz % 24 == 0 => out.sym_slice = sec.ok_or_else(|| berr("bad .symtab"))?,
            ".strtab" if ty == 3 => out.str_slice = sec.ok_or_else(|| berr("bad .strtab"))?,
            _ => {}
        }
    }
    Ok(out) }
fn parse_func_syms(sym: &[u8], str_tab: &[u8], base: u64, text_sz: u64) -> Result<Vec<ElfSymbol>, CompileError> {
    let mut syms = Vec::new();
    for i in 1..(sym.len() / 24) {
        let b = i * 24;
        if sym.get(b + 4) != Some(&0x12) { continue; }
        let (nm, val, sz) = (rd(sym, b, 4)? as u32, rd(sym, b + 8, 8)?, rd(sym, b + 16, 8)?);
        let off = val.checked_sub(base).ok_or_else(|| berr("elf64 global func precedes .text"))?;
        if off.saturating_add(sz) > text_sz { return Err(berr("elf64 global func exceeds .text")); }
        syms.push(ElfSymbol { name: rd_cstr(str_tab, nm)?, offset: off, size: sz });
    }
    Ok(syms) }
fn check_ident(bytes: &[u8], want: u16, msg: &str) -> Result<u16, CompileError> {
    if bytes.get(0..7) != Some(b"\x7fELF\x02\x01\x01") { return Err(berr("bad elf64 magic/ident")); }
    if rd(bytes, 16, 2)? != u64::from(want) { return Err(berr(msg)); }
    Ok(rd(bytes, 18, 2)? as u16) }
pub fn verify_elf64_object(bytes: &[u8]) -> Result<ElfSummary, CompileError> {
    let (machine, sc) = (check_ident(bytes, ET_REL, "elf64 not ET_REL")?, scan_shdrs(bytes, true)?);
    if sc.text_size == 0 || !sc.has_nonexec_gnu_stack || sc.sym_slice.is_empty() {
        return Err(berr("elf64 missing mandatory section")); }
    let symbols = parse_func_syms(sc.sym_slice, sc.str_slice, 0, sc.text_size)?;
    Ok(ElfSummary { machine, text_size: sc.text_size, rela_count: sc.rela_count,
        has_nonexec_gnu_stack: true, symbols }) }
pub fn verify_elf64_executable(bytes: &[u8]) -> Result<ElfExecSummary, CompileError> {
    let machine = check_ident(bytes, ET_EXEC, "elf64 not ET_EXEC")?;
    let (entry, phoff) = (rd(bytes, 24, 8)?, rd(bytes, 32, 8)? as usize);
    let (phent, phnum) = (rd(bytes, 54, 2)? as usize, rd(bytes, 56, 2)? as usize);
    if phent != 56 || phnum < 2 { return Err(berr("elf64 bad program header table")); }
    let ph_tab = bytes.get(phoff..phoff.saturating_add(phnum.saturating_mul(56)))
        .ok_or_else(|| berr("elf64 truncated phdrs"))?;
    let (mut rx_seg, mut rw_seg, mut has_nx) = (None, None, false);
    for i in 0..phnum {
        let b = i * 56;
        let (pty, pfl, poff, pv) = (rd(ph_tab, b, 4)? as u32, rd(ph_tab, b + 4, 4)? as u32,
            rd(ph_tab, b + 8, 8)?, rd(ph_tab, b + 16, 8)?);
        let (pfsz, pmsz, pal) = (rd(ph_tab, b + 32, 8)?, rd(ph_tab, b + 40, 8)?, rd(ph_tab, b + 48, 8)?);
        if pty == 1 {
            if (pfl & 3) == 3 { return Err(berr("elf64 W^X violation on PT_LOAD")); }
            if pfsz > pmsz || poff.saturating_add(pfsz) > bytes.len() as u64 { return Err(berr("PT_LOAD oob")); }
            if pal < PG || (poff % PG) != (pv % PG) { return Err(berr("elf64 PT_LOAD not page-congruent")); }
            if (pfl & 1) != 0 {
                if (pfl & 4) == 0 { return Err(berr("elf64 RX segment missing PF_R")); }
                rx_seg = Some((pv, pmsz));
            } else if (pfl & 2) != 0 { rw_seg = Some((pv, pfsz, pmsz)); }
        } else if pty == PT_GNU_STACK {
            if (pfl & 1) != 0 { return Err(berr("elf64 executable PT_GNU_STACK")); }
            has_nx = true;
        }
    }
    let (rx_v, rx_m) = rx_seg.ok_or_else(|| berr("elf64 missing RX PT_LOAD"))?;
    let (dv, dfs, dms) = rw_seg.ok_or_else(|| berr("elf64 missing RW PT_LOAD"))?;
    if !has_nx { return Err(berr("elf64 missing non-exec PT_GNU_STACK")); }
    let pg_end = |v: u64, sz: u64| v.saturating_add(sz).div_ceil(PG).saturating_mul(PG);
    if pg_end(rx_v, rx_m) > (dv & !(PG - 1)) && pg_end(dv, dms) > (rx_v & !(PG - 1)) {
        return Err(berr("elf64 RX and RW PT_LOAD overlap page")); }
    if entry < rx_v || entry >= rx_v.saturating_add(rx_m) { return Err(berr("entry_vaddr outside RX PT_LOAD")); }
    let sc = scan_shdrs(bytes, false)?;
    if sc.text_size == 0 || sc.sym_slice.is_empty() || entry < sc.text_vaddr
        || entry >= sc.text_vaddr.saturating_add(sc.text_size) {
        return Err(berr("elf64 exec missing valid .text/.symtab")); }
    let symbols = parse_func_syms(sc.sym_slice, sc.str_slice, sc.text_vaddr, sc.text_size)?;
    Ok(ElfExecSummary { machine, entry_vaddr: entry, text_vaddr: sc.text_vaddr, text_size: sc.text_size,
        data_vaddr: dv, data_filesz: dfs, data_memsz: dms, has_nonexec_gnu_stack: has_nx, symbols }) }
