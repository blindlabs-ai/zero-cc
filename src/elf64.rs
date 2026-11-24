use crate::ir::CompileError;

pub const EM_X86_64: u16 = 62;
pub const EM_AARCH64: u16 = 183;
pub const R_X86_64_PLT32: u32 = 4;
pub const R_AARCH64_CALL26: u32 = 283;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
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
struct Shdr { name: u32, ty: u32, flags: u64, off: u64, size: u64, link: u32, info: u32, align: u64, entsize: u64 }

fn push_shdr(out: &mut Vec<u8>, s: Shdr) {
    push_le(out, s.name.to_le_bytes()); push_le(out, s.ty.to_le_bytes()); push_le(out, s.flags.to_le_bytes());
    push_le(out, 0u64.to_le_bytes()); push_le(out, s.off.to_le_bytes()); push_le(out, s.size.to_le_bytes());
    push_le(out, s.link.to_le_bytes()); push_le(out, s.info.to_le_bytes());
    push_le(out, s.align.to_le_bytes()); push_le(out, s.entsize.to_le_bytes());
}

/// Emit a standalone relocatable `ELF64` `.o` (`ET_REL`, `.text`, optional `.rela.text`, `.note.GNU-stack`, `.symtab`, `.strtab`, `.shstrtab`).
pub fn emit_elf64_object(isa: TargetIsa, text: &[u8], syms: &[ElfSymbol], relocs: &[ElfReloc]) -> Result<Vec<u8>, CompileError> {
    if text.is_empty() || syms.is_empty() { return Err(berr("elf64 requires non-empty .text and symbols")); }
    let (mut shstr, mut strtab, mut symtab, mut rela) = (vec![0u8], vec![0u8], vec![0u8; 24], Vec::new());
    let n_text = add_str(&mut shstr, ".text");
    let n_rela = if relocs.is_empty() { 0 } else { add_str(&mut shstr, ".rela.text") };
    let n_note = add_str(&mut shstr, ".note.GNU-stack");
    let n_sym = add_str(&mut shstr, ".symtab");
    let n_str = add_str(&mut shstr, ".strtab");
    let n_shstr = add_str(&mut shstr, ".shstrtab");

    push_le(&mut symtab, add_str(&mut strtab, "").to_le_bytes());
    symtab.extend_from_slice(&[0x03, 0]);
    push_le(&mut symtab, 1u16.to_le_bytes()); push_le(&mut symtab, 0u64.to_le_bytes()); push_le(&mut symtab, 0u64.to_le_bytes());

    for sym in syms {
        if sym.offset.saturating_add(sym.size) > text.len() as u64 { return Err(berr("symbol exceeds .text")); }
        push_le(&mut symtab, add_str(&mut strtab, &sym.name).to_le_bytes());
        symtab.extend_from_slice(&[0x12, 0]);
        push_le(&mut symtab, 1u16.to_le_bytes());
        push_le(&mut symtab, sym.offset.to_le_bytes()); push_le(&mut symtab, sym.size.to_le_bytes());
    }
    for rel in relocs {
        let elf_sym = u64::from(rel.sym_idx) + 2;
        if (elf_sym as usize) * 24 >= symtab.len() { return Err(berr("reloc sym out of bounds")); }
        push_le(&mut rela, rel.offset.to_le_bytes());
        push_le(&mut rela, ((elf_sym << 32) | u64::from(rel.r_type)).to_le_bytes());
        push_le(&mut rela, (rel.addend as u64).to_le_bytes());
    }

    let mut out = Vec::with_capacity(512 + text.len() + symtab.len() + rela.len());
    out.extend_from_slice(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    push_le(&mut out, 1u16.to_le_bytes()); push_le(&mut out, isa.e_machine().to_le_bytes()); push_le(&mut out, 1u32.to_le_bytes());
    push_le(&mut out, [0u8; 24]); push_le(&mut out, 0u32.to_le_bytes());
    push_le(&mut out, 64u16.to_le_bytes()); push_le(&mut out, [0u8; 4]); push_le(&mut out, 64u16.to_le_bytes());
    let has_rela = !rela.is_empty();
    let shnum: u16 = if has_rela { 7 } else { 6 };
    push_le(&mut out, shnum.to_le_bytes()); push_le(&mut out, (shnum - 1).to_le_bytes());

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
    push_shdr(&mut out, Shdr { name: 0, ty: SHT_NULL, flags: 0, off: 0, size: 0, link: 0, info: 0, align: 0, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_text, ty: SHT_PROGBITS, flags: SHF_ALLOC | SHF_EXECINSTR, off: text_off, size: text.len() as u64, link: 0, info: 0, align: 16, entsize: 0 });
    if has_rela {
        push_shdr(&mut out, Shdr { name: n_rela, ty: SHT_RELA, flags: SHF_INFO_LINK, off: rela_off, size: rela.len() as u64, link: sym_idx, info: 1, align: 8, entsize: 24 });
    }
    push_shdr(&mut out, Shdr { name: n_note, ty: SHT_PROGBITS, flags: 0, off: note_off, size: 0, link: 0, info: 0, align: 1, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_sym, ty: SHT_SYMTAB, flags: 0, off: sym_off, size: symtab.len() as u64, link: sym_idx + 1, info: 2, align: 8, entsize: 24 });
    push_shdr(&mut out, Shdr { name: n_str, ty: SHT_STRTAB, flags: 0, off: str_off, size: strtab.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
    push_shdr(&mut out, Shdr { name: n_shstr, ty: SHT_STRTAB, flags: 0, off: shstr_off, size: shstr.len() as u64, link: 0, info: 0, align: 1, entsize: 0 });
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

/// Self-verifying `0t` airlock parser for relocatable `ELF64` objects emitted by `zero-cc`.
pub fn verify_elf64_object(bytes: &[u8]) -> Result<ElfSummary, CompileError> {
    if bytes.get(0..7) != Some(b"\x7fELF\x02\x01\x01") { return Err(berr("bad elf64 magic/ident")); }
    if rd_le(bytes, 16, 2)? != 1 { return Err(berr("elf64 not ET_REL")); }
    let (machine, shoff, shentsize, shnum, shstrndx) = (
        rd_le(bytes, 18, 2)? as u16, rd_le(bytes, 40, 8)? as usize,
        rd_le(bytes, 58, 2)? as usize, rd_le(bytes, 60, 2)? as usize, rd_le(bytes, 62, 2)? as usize,
    );
    if shentsize != 64 || shnum < 5 || shstrndx >= shnum { return Err(berr("elf64 bad section header table")); }
    let sh_table = bytes.get(shoff..shoff.saturating_add(shnum.saturating_mul(64))).ok_or_else(|| berr("elf64 truncated shdrs"))?;
    let shstr_hdr = shoff + shstrndx * 64;
    let (shstr_off, shstr_sz) = (rd_le(bytes, shstr_hdr + 24, 8)? as usize, rd_le(bytes, shstr_hdr + 32, 8)? as usize);
    let shstr = bytes.get(shstr_off..shstr_off.saturating_add(shstr_sz)).ok_or_else(|| berr("elf64 truncated shstrtab"))?;

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
    let mut symbols = Vec::new();
    for i in 1..(sym_slice.len() / 24) {
        let b = i * 24;
        let (st_name, st_info, st_value, st_size) = (
            rd_le(sym_slice, b, 4)? as u32, sym_slice.get(b + 4).copied().unwrap_or(0),
            rd_le(sym_slice, b + 8, 8)?, rd_le(sym_slice, b + 16, 8)?,
        );
        if st_info == 0x12 {
            if st_value.saturating_add(st_size) > text_size { return Err(berr("elf64 global func exceeds .text")); }
            symbols.push(ElfSymbol { name: rd_cstr(str_slice, st_name)?, offset: st_value, size: st_size });
        }
    }
    Ok(ElfSummary { machine, text_size, rela_count, has_nonexec_gnu_stack, symbols })
}
