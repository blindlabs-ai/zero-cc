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

const PT_LOAD: u32 = 1;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;
const ELF_PAGE_SIZE: u64 = 0x1000;
const MAX_EXEC_TEXT_BYTES: usize = 16_777_216;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const SHF_INFO_LINK: u64 = 0x40;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetIsa { X86_64, Aarch64 }
impl TargetIsa {
    pub const fn e_machine(self) -> u16 { match self { Self::X86_64 => EM_X86_64, Self::Aarch64 => EM_AARCH64 } }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElfSymbol { pub name: String, pub offset: u64, pub size: u64 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElfReloc { pub offset: u64, pub sym_idx: u32, pub r_type: u32, pub addend: i64 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElfSummary {
    pub machine: u16, pub text_size: u64, pub rela_count: usize,
    pub has_nonexec_gnu_stack: bool, pub symbols: Vec<ElfSymbol>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElfExecSummary {
    pub machine: u16, pub entry_vaddr: u64, pub text_vaddr: u64, pub text_size: u64,
    pub data_vaddr: u64, pub data_filesz: u64, pub data_memsz: u64,
    pub has_nonexec_gnu_stack: bool, pub symbols: Vec<ElfSymbol>,
}

/// Page-aligned RW `.data` virtual address for a static `ET_EXEC` with `text_len` bytes of `.text`.
/// @domain text_len: 1..=16777216
pub const fn elf64_exec_data_vaddr(text_len: usize) -> u64 {
    let end = ELF_EXEC_TEXT_OFF + (text_len as u64);
    let data_off = (end + (ELF_PAGE_SIZE - 1)) & !(ELF_PAGE_SIZE - 1);
    ELF_EXEC_BASE + data_off
}

fn berr(msg: impl Into<String>) -> CompileError { CompileError::BackendError(msg.into()) }
fn push_le<const N: usize>(out: &mut Vec<u8>, bytes: [u8; N]) { out.extend_from_slice(&bytes); }
fn pad_align(out: &mut Vec<u8>, align: usize) { while out.len() % align != 0 { out.push(0); } }

fn add_str(tab: &mut Vec<u8>, s: &str) -> u32 {
    let off = tab.len() as u32;
    tab.extend_from_slice(s.as_bytes());
    tab.push(0);
    off
}

#[derive(Clone, Copy)]
struct Shdr { name: u32, ty: u32, flags: u64, addr: u64, off: u64, size: u64, link: u32, info: u32, align: u64, entsize: u64 }

fn push_shdr(out: &mut Vec<u8>, s: Shdr) {
    push_le(out, s.name.to_le_bytes()); push_le(out, s.ty.to_le_bytes()); push_le(out, s.flags.to_le_bytes());
    push_le(out, s.addr.to_le_bytes()); push_le(out, s.off.to_le_bytes()); push_le(out, s.size.to_le_bytes());
    push_le(out, s.link.to_le_bytes()); push_le(out, s.info.to_le_bytes());
    push_le(out, s.align.to_le_bytes()); push_le(out, s.entsize.to_le_bytes());
}

#[derive(Clone, Copy)]
struct Phdr { ty: u32, flags: u32, off: u64, vaddr: u64, filesz: u64, memsz: u64, align: u64 }

fn push_phdr(out: &mut Vec<u8>, p: Phdr) {
    for w in [p.ty, p.flags] { push_le(out, w.to_le_bytes()); }
    for q in [p.off, p.vaddr, p.vaddr, p.filesz, p.memsz, p.align] { push_le(out, q.to_le_bytes()); }
}

fn push_ehdr(out: &mut Vec<u8>, e_type: u16, isa: TargetIsa, entry: u64, ph: (u64, u16, u16), sh: (u16, u16)) {
    out.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    push_le(out, e_type.to_le_bytes()); push_le(out, isa.e_machine().to_le_bytes()); push_le(out, 1u32.to_le_bytes());
    push_le(out, entry.to_le_bytes()); push_le(out, ph.0.to_le_bytes()); push_le(out, 0u64.to_le_bytes());
    push_le(out, 0u32.to_le_bytes()); push_le(out, 64u16.to_le_bytes());
    push_le(out, ph.1.to_le_bytes()); push_le(out, ph.2.to_le_bytes());
    push_le(out, 64u16.to_le_bytes()); push_le(out, sh.0.to_le_bytes()); push_le(out, sh.1.to_le_bytes());
}

fn build_symtab(text_len: usize, syms: &[ElfSymbol], vaddr_base: u64) -> Result<(Vec<u8>, Vec<u8>), CompileError> {
    let (mut strtab, mut symtab) = (vec![0u8], vec![0u8; 24]);
    push_le(&mut symtab, add_str(&mut strtab, "").to_le_bytes());
    symtab.extend_from_slice(&[0x03, 0]);
    push_le(&mut symtab, 1u16.to_le_bytes());
    push_le(&mut symtab, vaddr_base.to_le_bytes());
    push_le(&mut symtab, 0u64.to_le_bytes());
    for sym in syms {
        if sym.offset.saturating_add(sym.size) > text_len as u64 { return Err(berr("symbol exceeds .text")); }
        push_le(&mut symtab, add_str(&mut strtab, &sym.name).to_le_bytes());
        symtab.extend_from_slice(&[0x12, 0]);
        push_le(&mut symtab, 1u16.to_le_bytes());
        push_le(&mut symtab, vaddr_base.saturating_add(sym.offset).to_le_bytes());
        push_le(&mut symtab, sym.size.to_le_bytes());
    }
    Ok((symtab, strtab))
}

/// Emit a standalone relocatable `ELF64` `.o` (`ET_REL`, `.text`, optional `.rela.text`, `.note.GNU-stack`, `.symtab`, `.strtab`, `.shstrtab`).
pub fn emit_elf64_object(isa: TargetIsa, text: &[u8], syms: &[ElfSymbol], relocs: &[ElfReloc]) -> Result<Vec<u8>, CompileError> {
    if text.is_empty() || syms.is_empty() { return Err(berr("elf64 requires non-empty .text and symbols")); }
    let mut shstr = vec![0u8];
    let n_text = add_str(&mut shstr, ".text");
    let n_rela = if relocs.is_empty() { 0 } else { add_str(&mut shstr, ".rela.text") };
    let n_note = add_str(&mut shstr, ".note.GNU-stack");
    let n_sym = add_str(&mut shstr, ".symtab");
    let n_str = add_str(&mut shstr, ".strtab");
    let n_shstr = add_str(&mut shstr, ".shstrtab");

    let (symtab, strtab) = build_symtab(text.len(), syms, 0)?;
    let mut rela = Vec::with_capacity(relocs.len() * 24);
    for rel in relocs {
        let elf_sym = u64::from(rel.sym_idx) + 2;
        if (elf_sym as usize) * 24 >= symtab.len() { return Err(berr("reloc sym out of bounds")); }
        push_le(&mut rela, rel.offset.to_le_bytes());
        push_le(&mut rela, ((elf_sym << 32) | u64::from(rel.r_type)).to_le_bytes());
        push_le(&mut rela, (rel.addend as u64).to_le_bytes());
    }

    let has_rela = !rela.is_empty();
    let shnum: u16 = if has_rela { 7 } else { 6 };
    let mut out = Vec::with_capacity(512 + text.len() + symtab.len() + rela.len());
    push_ehdr(&mut out, ET_REL, isa, 0, (0, 0, 0), (shnum, shnum - 1));

    pad_align(&mut out, 16);
    let text_off = out.len() as u64; out.extend_from_slice(text);
    pad_align(&mut out, 8);
    let rela_off = out.len() as u64; out.extend_from_slice(&rela);
    let note_off = out.len() as u64;
    pad_align(&mut out, 8);
    let sym_off = out.len() as u64; out.extend_from_slice(&symtab);
    let str_off = out.len() as u64; out.extend_from_slice(&strtab);
    let shstr_off = out.len() as u64; out.extend_from_slice(&shstr);
    pad_align(&mut out, 8);
    let shoff = out.len() as u64;
    if let Some(slot) = out.get_mut(40..48) { slot.copy_from_slice(&shoff.to_le_bytes()); }

    let sym_idx: u32 = if has_rela { 4 } else { 3 };
    push_shdr(&mut out, Shdr { name: 0, ty: SHT_NULL, flags: 0, addr: 0, off: 0, size: 0, link: 0, info: 0, align: 0, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_text, ty: SHT_PROGBITS, flags: SHF_ALLOC | SHF_EXECINSTR, addr: 0, off: text_off, size: text.len() as u64, link: 0, info: 0, align: 16, entsize: 0 });
    if has_rela {
        push_shdr(&mut out, Shdr { name: n_rela, ty: SHT_RELA, flags: SHF_INFO_LINK, addr: 0, off: rela_off, size: rela.len() as u64, link: sym_idx, info: 1, align: 8, entsize: 24 });
    }
    push_shdr(&mut out, Shdr { name: n_note, ty: SHT_PROGBITS, flags: 0, addr: 0, off: note_off, size: 0, link: 0, info: 0, align: 1, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_sym, ty: SHT_SYMTAB, flags: 0, addr: 0, off: sym_off, size: symtab.len() as u64, link: sym_idx + 1, info: 2, align: 8, entsize: 24 });
    push_shdr(&mut out, Shdr { name: n_str, ty: SHT_STRTAB, flags: 0, addr: 0, off: str_off, size: strtab.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_shstr, ty: SHT_STRTAB, flags: 0, addr: 0, off: shstr_off, size: shstr.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
    Ok(out)
}

/// Emit a static, directly Linux-executable `ELF64` `ET_EXEC` binary with strict W^X `PT_LOAD` segments,
/// non-executable `PT_GNU_STACK`, and trailing `.symtab` for `readelf`/`objdump` inspection.
pub fn emit_elf64_executable(
    isa: TargetIsa, text: &[u8], entry_off: u64, data: &[u8], bss_size: usize, syms: &[ElfSymbol],
) -> Result<Vec<u8>, CompileError> {
    if text.is_empty() || text.len() > MAX_EXEC_TEXT_BYTES || syms.is_empty() {
        return Err(berr("elf64 exec requires valid non-empty .text and symbols"));
    }
    if entry_off >= text.len() as u64 { return Err(berr("elf64 exec entry exceeds .text")); }
    let data_vaddr = elf64_exec_data_vaddr(text.len());
    let data_off = data_vaddr - ELF_EXEC_BASE;
    let text_end = ELF_EXEC_TEXT_OFF + text.len() as u64;
    let rw_filesz = data.len() as u64;
    let rw_memsz = data.len().saturating_add(bss_size).max(64) as u64;

    let mut shstr = vec![0u8];
    let n_text = add_str(&mut shstr, ".text");
    let n_data = add_str(&mut shstr, ".data");
    let n_bss = add_str(&mut shstr, ".bss");
    let n_sym = add_str(&mut shstr, ".symtab");
    let n_str = add_str(&mut shstr, ".strtab");
    let n_shstr = add_str(&mut shstr, ".shstrtab");
    let (symtab, strtab) = build_symtab(text.len(), syms, ELF_EXEC_TEXT_VADDR)?;

    let mut out = Vec::with_capacity((data_off as usize) + data.len() + symtab.len() + strtab.len() + 512);
    push_ehdr(&mut out, ET_EXEC, isa, ELF_EXEC_TEXT_VADDR + entry_off, (64, 56, 3), (7, 6));
    push_phdr(&mut out, Phdr { ty: PT_LOAD, flags: PF_R | PF_X, off: 0, vaddr: ELF_EXEC_BASE, filesz: text_end, memsz: text_end, align: ELF_PAGE_SIZE });
    push_phdr(&mut out, Phdr { ty: PT_LOAD, flags: PF_R | PF_W, off: data_off, vaddr: data_vaddr, filesz: rw_filesz, memsz: rw_memsz, align: ELF_PAGE_SIZE });
    push_phdr(&mut out, Phdr { ty: PT_GNU_STACK, flags: PF_R | PF_W, off: 0, vaddr: 0, filesz: 0, memsz: 0, align: 16 });
    pad_align(&mut out, 16);
    if out.len() as u64 != ELF_EXEC_TEXT_OFF { return Err(berr("elf64 exec text offset mismatch")); }
    out.extend_from_slice(text);
    if (out.len() as u64) > data_off { return Err(berr("elf64 exec text overlaps data page")); }
    out.resize(data_off as usize, 0);
    out.extend_from_slice(data);

    pad_align(&mut out, 8);
    let sym_off = out.len() as u64; out.extend_from_slice(&symtab);
    let str_off = out.len() as u64; out.extend_from_slice(&strtab);
    let shstr_off = out.len() as u64; out.extend_from_slice(&shstr);
    pad_align(&mut out, 8);
    let shoff = out.len() as u64;
    if let Some(slot) = out.get_mut(40..48) { slot.copy_from_slice(&shoff.to_le_bytes()); }

    push_shdr(&mut out, Shdr { name: 0, ty: SHT_NULL, flags: 0, addr: 0, off: 0, size: 0, link: 0, info: 0, align: 0, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_text, ty: SHT_PROGBITS, flags: SHF_ALLOC | SHF_EXECINSTR, addr: ELF_EXEC_TEXT_VADDR, off: ELF_EXEC_TEXT_OFF, size: text.len() as u64, link: 0, info: 0, align: 16, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_data, ty: SHT_PROGBITS, flags: SHF_ALLOC | SHF_WRITE, addr: data_vaddr, off: data_off, size: rw_filesz, link: 0, info: 0, align: 64, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_bss, ty: SHT_NOBITS, flags: SHF_ALLOC | SHF_WRITE, addr: data_vaddr + rw_filesz, off: data_off + rw_filesz, size: rw_memsz.saturating_sub(rw_filesz), link: 0, info: 0, align: 64, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_sym, ty: SHT_SYMTAB, flags: 0, addr: 0, off: sym_off, size: symtab.len() as u64, link: 5, info: 2, align: 8, entsize: 24 });
    push_shdr(&mut out, Shdr { name: n_str, ty: SHT_STRTAB, flags: 0, addr: 0, off: str_off, size: strtab.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_shstr, ty: SHT_STRTAB, flags: 0, addr: 0, off: shstr_off, size: shstr.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
    Ok(out)
}

fn rd_le(b: &[u8], off: usize, width: usize) -> Result<u64, CompileError> {
    let s = b.get(off..off + width).ok_or_else(|| berr("elf64 unexpected eof"))?;
    let mut acc = 0u64;
    for (i, &byte) in s.iter().enumerate() { acc |= u64::from(byte) << (i * 8); }
    Ok(acc)
}

fn rd_cstr(tab: &[u8], off: u32) -> Result<String, CompileError> {
    let tail = tab.get(off as usize..).ok_or_else(|| berr("elf64 bad str offset"))?;
    let end = tail.iter().position(|&c| c == 0).ok_or_else(|| berr("elf64 unterminated str"))?;
    std::str::from_utf8(&tail[..end]).map(String::from).map_err(|_| berr("elf64 non-utf8 str"))
}

fn read_shdrs_and_shstr(bytes: &[u8]) -> Result<(&[u8], &[u8], usize), CompileError> {
    let (shoff, shentsize, shnum, shstrndx) = (
        rd_le(bytes, 40, 8)? as usize, rd_le(bytes, 58, 2)? as usize,
        rd_le(bytes, 60, 2)? as usize, rd_le(bytes, 62, 2)? as usize,
    );
    if shentsize != 64 || shnum < 5 || shstrndx >= shnum { return Err(berr("elf64 bad section header table")); }
    let sh_table = bytes.get(shoff..shoff.saturating_add(shnum.saturating_mul(64))).ok_or_else(|| berr("elf64 truncated shdrs"))?;
    let shstr_hdr = shoff + shstrndx * 64;
    let (shstr_off, shstr_sz) = (rd_le(bytes, shstr_hdr + 24, 8)? as usize, rd_le(bytes, shstr_hdr + 32, 8)? as usize);
    let shstr = bytes.get(shstr_off..shstr_off.saturating_add(shstr_sz)).ok_or_else(|| berr("elf64 truncated shstrtab"))?;
    Ok((sh_table, shstr, shnum))
}

fn parse_func_syms(sym_slice: &[u8], str_slice: &[u8], base_vaddr: u64, text_size: u64) -> Result<Vec<ElfSymbol>, CompileError> {
    let mut symbols = Vec::new();
    for i in 1..(sym_slice.len() / 24) {
        let b = i * 24;
        let (st_name, st_info, st_value, st_size) = (
            rd_le(sym_slice, b, 4)? as u32, sym_slice.get(b + 4).copied().unwrap_or(0),
            rd_le(sym_slice, b + 8, 8)?, rd_le(sym_slice, b + 16, 8)?,
        );
        if st_info == 0x12 {
            let rel_off = st_value.checked_sub(base_vaddr).ok_or_else(|| berr("elf64 global func precedes .text"))?;
            if rel_off.saturating_add(st_size) > text_size { return Err(berr("elf64 global func exceeds .text")); }
            symbols.push(ElfSymbol { name: rd_cstr(str_slice, st_name)?, offset: rel_off, size: st_size });
        }
    }
    Ok(symbols)
}

/// Self-verifying `0t` airlock parser for relocatable `ELF64` objects emitted by `zero-cc`.
pub fn verify_elf64_object(bytes: &[u8]) -> Result<ElfSummary, CompileError> {
    if bytes.get(0..7) != Some(b"\x7fELF\x02\x01\x01") { return Err(berr("bad elf64 magic/ident")); }
    if rd_le(bytes, 16, 2)? != u64::from(ET_REL) { return Err(berr("elf64 not ET_REL")); }
    let machine = rd_le(bytes, 18, 2)? as u16;
    let (sh_table, shstr, shnum) = read_shdrs_and_shstr(bytes)?;

    let (mut text_size, mut rela_count, mut has_nonexec_gnu_stack) = (0u64, 0usize, false);
    let (mut sym_slice, mut str_slice): (&[u8], &[u8]) = (&[], &[]);
    for i in 0..shnum {
        let base = i * 64;
        let (nm_off, ty, flags) = (rd_le(sh_table, base, 4)? as u32, rd_le(sh_table, base + 4, 4)? as u32, rd_le(sh_table, base + 8, 8)?);
        let (off, sz) = (rd_le(sh_table, base + 24, 8)? as usize, rd_le(sh_table, base + 32, 8)? as usize);
        let sec = bytes.get(off..off.saturating_add(sz)).ok_or_else(|| berr("elf64 section out of bounds"))?;
        match rd_cstr(shstr, nm_off)?.as_str() {
            ".text" if ty == SHT_PROGBITS && (flags & SHF_EXECINSTR) != 0 => text_size = sz as u64,
            ".rela.text" if ty == SHT_RELA && sz % 24 == 0 => rela_count = sz / 24,
            ".note.GNU-stack" if ty == SHT_PROGBITS && (flags & SHF_EXECINSTR) == 0 => has_nonexec_gnu_stack = true,
            ".symtab" if ty == SHT_SYMTAB && sz % 24 == 0 => sym_slice = sec,
            ".strtab" if ty == SHT_STRTAB => str_slice = sec,
            _ => {}
        }
    }
    if text_size == 0 || !has_nonexec_gnu_stack || sym_slice.is_empty() { return Err(berr("elf64 missing mandatory section")); }
    let symbols = parse_func_syms(sym_slice, str_slice, 0, text_size)?;
    Ok(ElfSummary { machine, text_size, rela_count, has_nonexec_gnu_stack, symbols })
}

/// Self-verifying `0t` airlock parser for static Linux `ELF64` `ET_EXEC` binaries emitted by `zero-cc`.
pub fn verify_elf64_executable(bytes: &[u8]) -> Result<ElfExecSummary, CompileError> {
    if bytes.get(0..7) != Some(b"\x7fELF\x02\x01\x01") { return Err(berr("bad elf64 magic/ident")); }
    if rd_le(bytes, 16, 2)? != u64::from(ET_EXEC) { return Err(berr("elf64 not ET_EXEC")); }
    let (machine, entry_vaddr, phoff, phentsize, phnum) = (
        rd_le(bytes, 18, 2)? as u16, rd_le(bytes, 24, 8)?, rd_le(bytes, 32, 8)? as usize,
        rd_le(bytes, 54, 2)? as usize, rd_le(bytes, 56, 2)? as usize,
    );
    if phentsize != 56 || phnum < 2 { return Err(berr("elf64 bad program header table")); }
    let ph_table = bytes.get(phoff..phoff.saturating_add(phnum.saturating_mul(56))).ok_or_else(|| berr("elf64 truncated phdrs"))?;

    let (mut rx_seg, mut rw_seg, mut has_nonexec_gnu_stack) = (None, None, false);
    for i in 0..phnum {
        let base = i * 56;
        let (p_type, p_flags) = (rd_le(ph_table, base, 4)? as u32, rd_le(ph_table, base + 4, 4)? as u32);
        let (p_off, p_vaddr) = (rd_le(ph_table, base + 8, 8)?, rd_le(ph_table, base + 16, 8)?);
        let (p_filesz, p_memsz, p_align) = (rd_le(ph_table, base + 32, 8)?, rd_le(ph_table, base + 40, 8)?, rd_le(ph_table, base + 48, 8)?);
        if p_type == PT_LOAD {
            if (p_flags & (PF_W | PF_X)) == (PF_W | PF_X) { return Err(berr("elf64 W^X violation on PT_LOAD")); }
            if p_filesz > p_memsz || p_off.saturating_add(p_filesz) > bytes.len() as u64 { return Err(berr("elf64 PT_LOAD out of bounds")); }
            if p_align < ELF_PAGE_SIZE || (p_off % ELF_PAGE_SIZE) != (p_vaddr % ELF_PAGE_SIZE) {
                return Err(berr("elf64 PT_LOAD not page-congruent"));
            }
            if (p_flags & PF_X) != 0 {
                if (p_flags & PF_R) == 0 { return Err(berr("elf64 RX segment missing PF_R")); }
                rx_seg = Some((p_vaddr, p_memsz));
            } else if (p_flags & PF_W) != 0 {
                rw_seg = Some((p_vaddr, p_filesz, p_memsz));
            }
        } else if p_type == PT_GNU_STACK {
            if (p_flags & PF_X) != 0 { return Err(berr("elf64 executable PT_GNU_STACK")); }
            has_nonexec_gnu_stack = true;
        }
    }
    let Some((rx_vaddr, rx_memsz)) = rx_seg else { return Err(berr("elf64 missing RX PT_LOAD")); };
    let Some((data_vaddr, data_filesz, data_memsz)) = rw_seg else { return Err(berr("elf64 missing RW PT_LOAD")); };
    if !has_nonexec_gnu_stack { return Err(berr("elf64 missing non-exec PT_GNU_STACK")); }
    let rx_page_end = rx_vaddr.saturating_add(rx_memsz).div_ceil(ELF_PAGE_SIZE).saturating_mul(ELF_PAGE_SIZE);
    let rw_page_start = data_vaddr & !(ELF_PAGE_SIZE - 1);
    let rw_page_end = data_vaddr.saturating_add(data_memsz).div_ceil(ELF_PAGE_SIZE).saturating_mul(ELF_PAGE_SIZE);
    let rx_page_start = rx_vaddr & !(ELF_PAGE_SIZE - 1);
    if rx_page_end > rw_page_start && rw_page_end > rx_page_start { return Err(berr("elf64 RX and RW PT_LOAD overlap page")); }
    if entry_vaddr < rx_vaddr || entry_vaddr >= rx_vaddr.saturating_add(rx_memsz) {
        return Err(berr("elf64 entry_vaddr outside RX PT_LOAD"));
    }

    let (sh_table, shstr, shnum) = read_shdrs_and_shstr(bytes)?;
    let (mut text_vaddr, mut text_size) = (0u64, 0u64);
    let (mut sym_slice, mut str_slice): (&[u8], &[u8]) = (&[], &[]);
    for i in 0..shnum {
        let base = i * 64;
        let (nm_off, ty, flags, addr) = (
            rd_le(sh_table, base, 4)? as u32, rd_le(sh_table, base + 4, 4)? as u32,
            rd_le(sh_table, base + 8, 8)?, rd_le(sh_table, base + 16, 8)?,
        );
        let (off, sz) = (rd_le(sh_table, base + 24, 8)? as usize, rd_le(sh_table, base + 32, 8)? as usize);
        match rd_cstr(shstr, nm_off)?.as_str() {
            ".text" if ty == SHT_PROGBITS && (flags & SHF_EXECINSTR) != 0 => {
                if off.saturating_add(sz) > bytes.len() { return Err(berr("elf64 .text out of bounds")); }
                text_vaddr = addr;
                text_size = sz as u64;
            }
            ".symtab" if ty == SHT_SYMTAB && sz % 24 == 0 => {
                sym_slice = bytes.get(off..off.saturating_add(sz)).ok_or_else(|| berr("elf64 .symtab out of bounds"))?;
            }
            ".strtab" if ty == SHT_STRTAB => {
                str_slice = bytes.get(off..off.saturating_add(sz)).ok_or_else(|| berr("elf64 .strtab out of bounds"))?;
            }
            _ => {}
        }
    }
    if text_size == 0 || sym_slice.is_empty() || entry_vaddr < text_vaddr || entry_vaddr >= text_vaddr.saturating_add(text_size) {
        return Err(berr("elf64 exec missing valid .text/.symtab"));
    }
    let symbols = parse_func_syms(sym_slice, str_slice, text_vaddr, text_size)?;
    Ok(ElfExecSummary { machine, entry_vaddr, text_vaddr, text_size, data_vaddr, data_filesz, data_memsz, has_nonexec_gnu_stack, symbols })
}
