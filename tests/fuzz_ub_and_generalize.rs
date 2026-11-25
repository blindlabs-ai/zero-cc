use std::collections::BTreeMap;
use std::fs;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::process::Command;
use zero_cc::rust_fe::{kernel_reg_budget, parse_rust_file_kernels};
use zero_cc::{
    compile_kernel_direct, compile_kernel_wasm, compile_rust_to_elf, compile_rust_to_wasm,
    parse_rust_kernel, verify_elf64_object, BufAccess, Domain, Kernel, Op, RegBudget, TargetIsa,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn tmp_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("zcc_fuzz_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("mkdir");
    p
}

#[cfg(target_arch = "x86_64")]
mod native {
    use super::*;
    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIV_ANON: i32 = 0x22;
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: isize) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    pub struct ExecMem { base: *mut u8, len: usize, batch_off: usize }
    impl ExecMem {
        pub fn from_kernel(k: &Kernel, budget: RegBudget) -> (Self, zero_cc::CompiledObject) {
            let obj = compile_kernel_direct(k, budget, TargetIsa::X86_64).expect("compile x86");
            assert!(obj.zero_spills, "zero spills required on {}", k.name);
            let sum = verify_elf64_object(&obj.elf_bytes).expect("elf64");
            assert_eq!(sum.rela_count, 0, "direct code must have 0 relocs");
            let batch_off = sum.symbols.iter().find(|s| s.name == obj.batch_sym).expect("batch sym").offset as usize;
            let len = obj.code_bytes.len().max(1).div_ceil(4096) * 4096;
            let base = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIV_ANON, -1, 0) };
            assert_ne!(base as isize, -1, "mmap exec");
            unsafe {
                std::ptr::copy_nonoverlapping(obj.code_bytes.as_ptr(), base, obj.code_bytes.len());
                assert_eq!(mprotect(base, len, PROT_READ | PROT_EXEC), 0, "mprotect rx");
            }
            (Self { base, len, batch_off }, obj)
        }
        pub fn new(src: &str, budget: RegBudget) -> (Self, zero_cc::CompiledObject, Kernel) {
            let k = parse_rust_kernel(src).expect("parse kernel");
            let (ex, obj) = Self::from_kernel(&k, budget);
            (ex, obj, k)
        }
        pub fn ptr(&self) -> *const u8 { unsafe { self.base.add(self.batch_off) } }
    }
    impl Drop for ExecMem { fn drop(&mut self) { unsafe { let _ = munmap(self.base, self.len); } } }

    pub struct GuardPage(*mut u8);
    impl GuardPage {
        pub fn new() -> Self {
            let base = unsafe { mmap(std::ptr::null_mut(), 8192, PROT_READ | PROT_WRITE, MAP_PRIV_ANON, -1, 0) };
            assert_ne!(base as isize, -1, "mmap guard");
            unsafe { assert_eq!(mprotect(base.add(4096), 4096, PROT_NONE), 0, "mprotect guard"); }
            Self(base)
        }
        pub fn tail(&self, bytes: usize) -> *mut u8 { assert!(bytes <= 4096); unsafe { self.0.add(4096 - bytes) } }
    }
    impl Drop for GuardPage { fn drop(&mut self) { unsafe { let _ = munmap(self.0, 8192); } } }

    pub unsafe fn call_abi6(f: *const u8, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
        let err: u64;
        core::arch::asm!(
            "push rbx", "push rbp", "push r12", "push r13", "push r14", "push r15",
            "mov rax, rsp", "and rsp, -16", "sub rsp, 16", "mov [rsp], rax",
            "movabs rax, 0xCAFEBABEDEADBEEF", "mov [rsp + 8], rax",
            "movabs rbx, 0x1122334466778899", "movabs rbp, 0x778899AABBCCEEFF",
            "movabs r12, 0x22334466778899AA", "movabs r13, 0x334466778899AABB",
            "movabs r14, 0x4466778899AABBCC", "movabs r15, 0x66778899AABBCCEE",
            "call r11",
            "xor r10, r10", "mov rax, rsp", "and rax, 15", "or r10, rax",
            "movabs rax, 0xCAFEBABEDEADBEEF", "xor rax, [rsp + 8]", "or r10, rax",
            "movabs rax, 0x1122334466778899", "xor rax, rbx", "or r10, rax",
            "movabs rax, 0x778899AABBCCEEFF", "xor rax, rbp", "or r10, rax",
            "movabs rax, 0x22334466778899AA", "xor rax, r12", "or r10, rax",
            "movabs rax, 0x334466778899AABB", "xor rax, r13", "or r10, rax",
            "movabs rax, 0x4466778899AABBCC", "xor rax, r14", "or r10, rax",
            "movabs rax, 0x66778899AABBCCEE", "xor rax, r15", "or r10, rax",
            "mov rsp, [rsp]", "pop r15", "pop r14", "pop r13", "pop r12", "pop rbp", "pop rbx",
            "mov rax, r10",
            inout("r11") f => _, inout("rdi") a0 => _, inout("rsi") a1 => _, inout("rdx") a2 => _,
            inout("rcx") a3 => _, inout("r8") a4 => _, inout("r9") a5 => _,
            lateout("rax") err, clobber_abi("C"),
        );
        err
    }
}

fn run_node_js(dir: &Path, name: &str, js: &str) -> std::process::Output {
    let p = dir.join(format!("{name}.mjs"));
    fs::write(&p, js).expect("write js");
    Command::new("node").arg(&p).output().expect("run node")
}

#[test]
fn layer_a_hostile_parser_10k_fuzz() {
    let mut seeds: Vec<String> = vec![
        "/*\u{1F600}".into(), "/*x".into(), "/* /* /*\u{1F600}".into(),
        "pub fn f(x: &[u32;1], o: &mut [u32;1]) { /*\u{1F600}".into(),
        format!("pub fn f(x: &[u32;1], o: &mut [u32;1]) {{ o[0] = {}x[0]{}; }}", "(".repeat(200), ")".repeat(200)),
        format!("pub fn f(x: &[u32;1], o: &mut [u32;1]) {{ o[0] = {}x[0]{}; }}", "{ ".repeat(200), " }".repeat(200)),
        format!("pub fn f(x: &[u32;1], o: &mut [u32;1]) {{ o[0] = {}0{}; }}", "if x[0] > 0 { ".repeat(160), " } else { 1 }".repeat(160)),
    ];
    let toks = [
        "pub ", "fn ", "const ", "let ", "mut ", "if ", "else ", "for ", "in ", "0..8", "return ",
        "u32", "i16", "u64", "bool", "&[u32]", "&mut [u32]", "(", ")", "{", "}", "[", "]", ";", ",",
        "=", "+=", "^=", "<<=", "->", "/*", "*/", "//", "!", "&", "|", "^", "<<", ">>",
        "wrapping_add", "0xFFFF", "\u{1F600}", "\u{4E2D}", "\0", "\n", "x", "a", "1", "999999999999999999999",
    ];
    let mut rng = Rng(0xA1B2_C3D4_E5F6_0718);
    for _ in 0..10_000 {
        let len = (rng.next() % 28 + 1) as usize;
        let mut s = String::new();
        for _ in 0..len { s.push_str(toks[(rng.next() as usize) % toks.len()]); }
        seeds.push(s);
    }
    let ov = BTreeMap::new();
    for s in &seeds {
        assert!(catch_unwind(|| { let _ = parse_rust_kernel(s); }).is_ok(), "panic on parse_rust_kernel");
        assert!(catch_unwind(|| { let _ = parse_rust_file_kernels(s, &ov); }).is_ok(), "panic on parse_rust_file_kernels");
    }
    assert!(parse_rust_kernel(&seeds[0]).is_err() && parse_rust_kernel(&seeds[4]).is_err());
}

#[test]
fn layer_b_boundary_100k_prot_none_abi_ubsan_wasm() {
    let dir = tmp_dir("layer_b");
    let src32 = "pub fn mix32(a: &[u32; 1], b: &[u32; 1], s: &[u32; 1], o: &mut [u32; 1]) { let t = a[0].wrapping_add(b[0]).wrapping_mul(b[0] ^ 0x9E37_79B1); o[0] = (t.wrapping_shl(s[0]) ^ t.wrapping_shr(s[0])).wrapping_sub(!a[0]); }";
    let src16 = "pub fn mix16(a: &[i16; 1], b: &[i16; 1], o: &mut [i32; 1]) { let x = a[0] as i32; let y = b[0] as i32; o[0] = if x < y { (x - y) >> 1 } else { (x + y) & 0x7FFF }; }";
    let src64 = "pub fn mix64(a: &[u64; 1], b: &[u64; 1], o: &mut [u64; 1]) { let x = a[0].wrapping_add(b[0]).rotate_left(13); o[0] = if x > a[0] { x ^ b[0] } else { x.wrapping_sub(b[0]) }; }";
    let ref32 = |a: u32, b: u32, s: u32| -> u32 { let t = a.wrapping_add(b).wrapping_mul(b ^ 0x9E37_79B1); (t.wrapping_shl(s) ^ t.wrapping_shr(s)).wrapping_sub(!a) };
    let ref16 = |a: i16, b: i16| -> i32 { let (x, y) = (i32::from(a), i32::from(b)); if x < y { (x - y) >> 1 } else { (x + y) & 0x7FFF } };
    let ref64 = |a: u64, b: u64| -> u64 { let x = a.wrapping_add(b).rotate_left(13); if x > a { x ^ b } else { x.wrapping_sub(b) } };
    let e32: [u32; 13] = [0, 1, 7, 15, 16, 31, 32, 63, 64, 255, i32::MAX as u32, i32::MIN as u32, u32::MAX];
    let e64: [u64; 12] = [0, 1, 31, 32, 63, 64, 255, u32::MAX as u64, i32::MIN as i64 as u64, i64::MAX as u64, i64::MIN as u64, u64::MAX];
    let mut rng = Rng(0x0102_0304_0607_0809);
    let n_big = 100_032usize;
    let (mut va, mut vb, mut vs) = (Vec::with_capacity(n_big), Vec::with_capacity(n_big), Vec::with_capacity(n_big));
    for &x in &e32 { for &y in &e32 { for &s in &[0u32, 1, 31, 32, 63, 64, 255] { va.push(x); vb.push(y); vs.push(s); } } }
    while va.len() < n_big { va.push(rng.next() as u32); vb.push(rng.next() as u32); vs.push((rng.next() & 255) as u32); }

    #[cfg(target_arch = "x86_64")]
    {
        let (ex32, obj32, k32) = native::ExecMem::new(src32, RegBudget::default());
        let (ex16, obj16, _) = native::ExecMem::new(src16, RegBudget::default());
        let (ex64, obj64, _) = native::ExecMem::new(src64, RegBudget::default());
        let mut out = vec![0u32; n_big];
        assert_eq!(unsafe { native::call_abi6(ex32.ptr(), va.as_ptr() as u64, vb.as_ptr() as u64, vs.as_ptr() as u64, out.as_mut_ptr() as u64, n_big as u64, 0) }, 0);
        for i in 0..n_big { assert_eq!(out[i], ref32(va[i], vb[i], vs[i]), "mix32 mismatch {i}"); }
        for i in 0..256 { assert_eq!(k32.eval(&[&[i64::from(va[i])], &[i64::from(vb[i])], &[i64::from(vs[i])], &[0]]).unwrap()[3][0] as u32, ref32(va[i], vb[i], vs[i])); }
        let (ga, gb, gc, go) = (native::GuardPage::new(), native::GuardPage::new(), native::GuardPage::new(), native::GuardPage::new());
        for n in 0..=33usize {
            unsafe {
                let (pa, pb, ps, po) = (ga.tail(n * 4) as *mut u32, gb.tail(n * 4) as *mut u32, gc.tail(n * 4) as *mut u32, go.tail(n * 4) as *mut u32);
                for i in 0..n { *pa.add(i) = va[i]; *pb.add(i) = vb[i]; *ps.add(i) = vs[i]; }
                assert_eq!(native::call_abi6(ex32.ptr(), pa as u64, pb as u64, ps as u64, po as u64, n as u64, 0), 0);
                for i in 0..n { assert_eq!(*po.add(i), ref32(va[i], vb[i], vs[i]), "PROT_NONE u32 n={n}"); }
                let (p16a, p16b, p32o) = (ga.tail(n * 2) as *mut i16, gb.tail(n * 2) as *mut i16, go.tail(n * 4) as *mut i32);
                for i in 0..n { *p16a.add(i) = va[i] as i16; *p16b.add(i) = vb[i] as i16; }
                assert_eq!(native::call_abi6(ex16.ptr(), p16a as u64, p16b as u64, p32o as u64, n as u64, 0, 0), 0);
                for i in 0..n { assert_eq!(*p32o.add(i), ref16(va[i] as i16, vb[i] as i16), "PROT_NONE i16 n={n}"); }
                let (p64a, p64b, p64o) = (ga.tail(n * 8) as *mut u64, gb.tail(n * 8) as *mut u64, go.tail(n * 8) as *mut u64);
                for i in 0..n { *p64a.add(i) = e64[i % e64.len()]; *p64b.add(i) = e64[(i * 5 + 3) % e64.len()]; }
                assert_eq!(native::call_abi6(ex64.ptr(), p64a as u64, p64b as u64, p64o as u64, n as u64, 0, 0), 0);
                for i in 0..n { assert_eq!(*p64o.add(i), ref64(*p64a.add(i), *p64b.add(i)), "PROT_NONE u64 n={n}"); }
            }
        }
        let (o32, o16, o64) = (dir.join("m32.o"), dir.join("m16.o"), dir.join("m64.o"));
        fs::write(&o32, &obj32.elf_bytes).unwrap(); fs::write(&o16, &obj16.elf_bytes).unwrap(); fs::write(&o64, &obj64.elf_bytes).unwrap();
        let c_src = format!(r#"#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
extern void {s32}(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint64_t);
extern void {s16}(const int16_t*, const int16_t*, int32_t*, uint64_t);
extern void {s64}(const uint64_t*, const uint64_t*, uint64_t*, uint64_t);
int main(void) {{
  uint32_t e[13]={{0,1,7,15,16,31,32,63,64,255,0x7FFFFFFFu,0x80000000u,0xFFFFFFFFu}};
  for (uint64_t n=0; n<=33; n++) {{
    uint32_t *a=malloc(n?n*4:4),*b=malloc(n?n*4:4),*s=malloc(n?n*4:4),*o=malloc(n?n*4:4);
    int16_t *i1=malloc(n?n*2:2),*i2=malloc(n?n*2:2); int32_t *io=malloc(n?n*4:4);
    uint64_t *u1=malloc(n?n*8:8),*u2=malloc(n?n*8:8),*uo=malloc(n?n*8:8);
    for (uint64_t i=0;i<n;i++){{ a[i]=e[i%13]; b[i]=e[(i*3+1)%13]; s[i]=e[(i*7)%13]; i1[i]=(int16_t)a[i]; i2[i]=(int16_t)b[i]; u1[i]=((uint64_t)a[i]<<32)|b[i]; u2[i]=~u1[i]; }}
    {s32}(a,b,s,o,n); {s16}(i1,i2,io,n); {s64}(u1,u2,uo,n);
    for (uint64_t i=0;i<n;i++){{ uint32_t sh=s[i]&31u,t=(a[i]+b[i])*(b[i]^0x9E3779B1u); assert(o[i]==(((sh?(t<<sh):t)^(sh?(t>>sh):t))-(~a[i]))); int32_t x=i1[i],y=i2[i]; assert(io[i]==(x<y?((x-y)>>1):((x+y)&0x7FFF))); }}
    free(a);free(b);free(s);free(o);free(i1);free(i2);free(io);free(u1);free(u2);free(uo);
  }} return 0;
}}"#, s32 = obj32.batch_sym, s16 = obj16.batch_sym, s64 = obj64.batch_sym);
        let (cp, bin) = (dir.join("h.c"), dir.join("h_san"));
        fs::write(&cp, c_src).unwrap();
        let cc = Command::new("clang").args(["-O2", "-no-pie", "-fsanitize=undefined,address", "-fno-omit-frame-pointer"]).arg(&cp).arg(&o32).arg(&o16).arg(&o64).arg("-o").arg(&bin).output().expect("clang");
        assert!(cc.status.success(), "clang san: {}", String::from_utf8_lossy(&cc.stderr));
        let run = Command::new(&bin).output().expect("run h_san");
        assert!(run.status.success(), "h_san: {}", String::from_utf8_lossy(&run.stderr));
    }
    let cw = compile_rust_to_wasm(src32, RegBudget::default()).expect("wasm mix32");
    let wp = dir.join("mix32.wasm");
    fs::write(&wp, &cw.wasm_bytes).unwrap();
    let js = format!(r#"import fs from 'fs';
const {{ instance }} = await WebAssembly.instantiate(fs.readFileSync({wp:?}));
const m = new Uint32Array(instance.exports.memory.buffer);
const e = [0,1,7,15,16,31,32,63,64,255,0x7FFFFFFF,0x80000000,0xFFFFFFFF]; const n = 256;
for (let i=0;i<n;i++){{ m[64+i]=e[i%13]; m[64+n+i]=e[(i*3+1)%13]; m[64+2*n+i]=e[(i*5+2)%13]; }}
instance.exports[{bsym:?}](64*4, (64+n)*4, (64+2*n)*4, (64+3*n)*4, n);
for (let i=0;i<n;i++){{ const a=m[64+i],b=m[64+n+i],s=m[64+2*n+i]&31; const t=Math.imul((a+b)>>>0,(b^0x9E3779B1)>>>0)>>>0; if (m[64+3*n+i] !== ((((t<<s)^(t>>>s))>>>0)-(~a>>>0))>>>0) throw new Error('wasm '+i); }}"#, bsym = cw.batch_sym);
    let out = run_node_js(&dir, "mix32", &js);
    assert!(out.status.success(), "node wasm: {}", String::from_utf8_lossy(&out.stderr));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn layer_c_random_2k_dags_and_10_knockouts() {
    let mut rng = Rng(0x7172_7374_7677_7879);
    for idx in 0..2_000usize {
        let is32 = (idx & 1) == 0;
        let ty = if is32 { "u32" } else { "u64" };
        let op1 = ["wrapping_add", "wrapping_sub", "wrapping_mul"][(rng.next() as usize) % 3];
        let op2 = ["&", "|", "^"][(rng.next() as usize) % 3];
        let sh = (rng.next() % if is32 { 31 } else { 63 }) as u32;
        let cmp = ["<", ">", "==", "!="][(rng.next() as usize) % 4];
        let unroll = (idx % 5) == 0;
        let body = if unroll {
            format!("let mut acc = a[0].{op1}(b[0]); for i in 0..4 {{ acc = acc.wrapping_add((b[0] {op2} (i as {ty})).wrapping_shl({sh})); }} o[0] = if acc {cmp} a[0] {{ !acc }} else {{ acc ^ b[0] }};")
        } else {
            format!("let t = a[0].{op1}(b[0]); let u = (t {op2} !a[0]).wrapping_shl({sh}); o[0] = if u {cmp} b[0] {{ u.wrapping_sub(a[0]) }} else {{ u ^ t }};")
        };
        let src = format!("pub fn k_{idx}(a: &[{ty}; 1], b: &[{ty}; 1], o: &mut [{ty}; 1]) {{ {body} }}");
        let eval_ref = |a: u64, b: u64| -> u64 {
            let m = |v: u64| if is32 { v & 0xFFFF_FFFF } else { v };
            let f1 = |x: u64, y: u64| match op1 { "wrapping_add" => m(x.wrapping_add(y)), "wrapping_sub" => m(x.wrapping_sub(y)), _ => m(x.wrapping_mul(y)) };
            let f2 = |x: u64, y: u64| match op2 { "&" => x & y, "|" => x | y, _ => x ^ y };
            let fc = |x: u64, y: u64| match cmp { "<" => x < y, ">" => x > y, "==" => x == y, _ => x != y };
            let (a, b) = (m(a), m(b));
            if unroll {
                let mut acc = f1(a, b);
                for i in 0..4u64 { acc = m(acc.wrapping_add(m(f2(b, i).wrapping_shl(sh)))); }
                if fc(acc, a) { m(!acc) } else { acc ^ b }
            } else {
                let t = f1(a, b); let u = m(f2(t, m(!a)).wrapping_shl(sh));
                if fc(u, b) { m(u.wrapping_sub(a)) } else { u ^ t }
            }
        };
        let k = parse_rust_kernel(&src).unwrap_or_else(|e| panic!("parse k_{idx}: {e:?}"));
        let pairs = [(0u64, 0u64), (1, 0xFFFF_FFFF), (0xFFFF_FFFF, 1), (u64::MAX, 7), (rng.next(), rng.next()), (rng.next(), rng.next()), (rng.next(), rng.next()), (rng.next(), rng.next())];
        for &(a, b) in &pairs[..4] {
            let ev = k.eval(&[&[a as i64], &[b as i64], &[0]]).expect("eval");
            let got = if is32 { (ev[2][0] as u64) & 0xFFFF_FFFF } else { ev[2][0] as u64 };
            assert_eq!(got, eval_ref(a, b), "IR k_{idx}");
        }
        #[cfg(target_arch = "x86_64")]
        if idx < 200 || (idx % 10) == 0 {
            let (ex, _) = native::ExecMem::from_kernel(&k, RegBudget::default());
            if is32 {
                let (va, vb): (Vec<u32>, Vec<u32>) = pairs.iter().map(|&(a, b)| (a as u32, b as u32)).unzip();
                let mut vo = vec![0u32; pairs.len()];
                assert_eq!(unsafe { native::call_abi6(ex.ptr(), va.as_ptr() as u64, vb.as_ptr() as u64, vo.as_mut_ptr() as u64, pairs.len() as u64, 0, 0) }, 0);
                for (i, &(a, b)) in pairs.iter().enumerate() { assert_eq!(u64::from(vo[i]), eval_ref(a, b), "x86 u32 k_{idx}"); }
            } else {
                let (va, vb): (Vec<u64>, Vec<u64>) = pairs.iter().copied().unzip();
                let mut vo = vec![0u64; pairs.len()];
                assert_eq!(unsafe { native::call_abi6(ex.ptr(), va.as_ptr() as u64, vb.as_ptr() as u64, vo.as_mut_ptr() as u64, pairs.len() as u64, 0, 0) }, 0);
                for (i, &(a, b)) in pairs.iter().enumerate() { assert_eq!(vo[i], eval_ref(a, b), "x86 u64 k_{idx}"); }
            }
        }
    }
    let eval1 = |src: &str, ins: &[i64]| -> i64 {
        let k = parse_rust_kernel(src).expect("parse ko");
        let mut args: Vec<Vec<i64>> = ins.iter().map(|&v| vec![v]).collect();
        args.push(vec![0]);
        let refs: Vec<&[i64]> = args.iter().map(Vec::as_slice).collect();
        k.eval(&refs).expect("eval ko").last().unwrap()[0]
    };
    let k1 = eval1("pub fn f(a: &[u32;1], b: &[u32;1], o: &mut [u64;1]) { o[0] = a[0].wrapping_add(b[0]) as u64; }", &[u32::MAX as i64, 1]);
    assert_eq!(k1, 0); assert_ne!(k1, 1i64 << 32);
    let k2 = eval1("pub fn f(a: &[u32;1], b: &[u32;1], o: &mut [i64;1]) { o[0] = a[0].wrapping_sub(b[0]) as i64; }", &[0, 1]);
    assert_eq!(k2, u32::MAX as i64); assert_ne!(k2, -1);
    let k3 = eval1("pub fn f(a: &[u32;1], b: &[u32;1], o: &mut [u32;1]) { o[0] = a[0].wrapping_mul(b[0]).wrapping_neg().wrapping_shl(32); }", &[1, u32::MAX as i64]);
    assert_eq!(k3, 1); assert_ne!(k3, 0);
    let k4 = eval1("pub fn f(x: &[u32;1], o: &mut [u32;1]) { let mut acc = x[0]; for i in 0..4 { acc += i as u32; } o[0] = acc; }", &[0]);
    assert_eq!(k4, 6); assert_ne!(k4, 0);
    let k5a = eval1("pub fn f(x: &[u32;1], o: &mut [u32;1]) { o[0] = !x[0]; }", &[1]);
    let k5b = eval1("pub fn f(b: &[u32;1], o: &mut [i64;1]) { let f = b[0] != 0; o[0] = if !f { 1 } else { 0 }; }", &[1]);
    assert_eq!(k5a as u32, !1u32); assert_ne!(k5a, 0); assert_eq!(k5b, 0); assert_ne!(k5b, -2);
    #[cfg(target_arch = "x86_64")]
    {
        let (ex6, _, _) = native::ExecMem::new("pub fn f(a: &[u64;1], b: &[u64;1], s: &[u64;1], o: &mut [u32;1]) { o[0] = ((a[0] + b[0]) >> (s[0] & 31)) as u32; }", RegBudget::default());
        let (a6, b6, s6) = ([u32::MAX as u64; 8], [1u64; 8], [1u64; 8]); let mut o6 = [0u32; 8];
        assert_eq!(unsafe { native::call_abi6(ex6.ptr(), a6.as_ptr() as u64, b6.as_ptr() as u64, s6.as_ptr() as u64, o6.as_mut_ptr() as u64, 8, 0) }, 0);
        assert_eq!(o6[0], 0x8000_0000); assert_ne!(o6[0], 0);
        let (ex7, _, _) = native::ExecMem::new("pub fn f(a: &[i32;1], o: &mut [i64;1]) { o[0] = (a[0] as i64) >> 1; }", RegBudget::default());
        let a7 = [-5i32; 8]; let mut o7 = [0i64; 8];
        assert_eq!(unsafe { native::call_abi6(ex7.ptr(), a7.as_ptr() as u64, o7.as_mut_ptr() as u64, 8, 0, 0, 0) }, 0);
        assert_eq!(o7[0], -3); assert_ne!(o7[0], 0x7FFF_FFFD);
        let (ex9, _, _) = native::ExecMem::new("pub fn f(a: &[u64;1], b: &[u64;1], o: &mut [u64;2]) { o[0] = if a[0] > b[0] { 1u64 } else { 0u64 }; o[1] = a[0] ^ b[0]; }", RegBudget::default());
        let (a9, b9) = ([u64::MAX, 0, 9, 4], [0u64, u64::MAX, 2, 4]); let mut o9 = [[0u64; 2]; 4];
        assert_eq!(unsafe { native::call_abi6(ex9.ptr(), a9.as_ptr() as u64, b9.as_ptr() as u64, o9.as_mut_ptr() as u64, 4, 0, 0) }, 0);
        assert_eq!(o9[0], [1, u64::MAX]); assert_ne!(o9[0][0], 0); assert_eq!(o9[1], [0, u64::MAX]);
    }
    let arm = compile_rust_to_elf("pub fn f(a: &[i64;1], b: &[i64;1], o: &mut [i64;1]) { o[0] = a[0] / b[0]; }", RegBudget::default(), TargetIsa::Aarch64).expect("arm64");
    let words: Vec<u32> = arm.code_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(words.iter().any(|&w| (w & !0x1F) == 0xB500_0040) && !words.iter().any(|&w| (w & !0x1F) == 0xB400_0040));
    let dir = tmp_dir("ko10");
    let cw_rem = compile_rust_to_wasm("pub fn r(a: &[i64;1], b: &[i64;1], o: &mut [i64;1]) { o[0] = a[0] % b[0]; }", RegBudget::default()).expect("wasm rem");
    let mut k_lut = Kernel::new("lut_oob");
    let bi = k_lut.add_buf("idx", BufAccess::ReadOnly, Domain::range(-16, 16));
    let bl = k_lut.add_buf("lut", BufAccess::LookupTable(8), Domain::range(0, 100));
    let bo = k_lut.add_buf("out", BufAccess::MutNoAlias, Domain::range(0, 100));
    let m0 = k_lut.push(Op::MemInit);
    for l in 0..4u8 { let ix = k_lut.push(Op::Load { mem: m0, buf: bi, lane: l }); let v = k_lut.push(Op::LoadLut { buf: bl, idx: ix }); k_lut.add_output(bo, l, v); }
    let cw_lut = compile_kernel_wasm(&k_lut, RegBudget::default()).expect("wasm lut");
    let (pr, pl) = (dir.join("rem.wasm"), dir.join("lut.wasm"));
    fs::write(&pr, &cw_rem.wasm_bytes).unwrap(); fs::write(&pl, &cw_lut.wasm_bytes).unwrap();
    let js = format!(r#"import fs from 'fs';
const r = (await WebAssembly.instantiate(fs.readFileSync({pr:?}))).instance;
const mr = new BigInt64Array(r.exports.memory.buffer); mr[8] = -(1n<<63n); mr[9] = -1n;
let rt = false; try {{ r.exports[{rsym:?}](64, 72, 80); }} catch {{ rt = true; }} if (!rt) throw new Error('rem no trap');
const l = (await WebAssembly.instantiate(fs.readFileSync({pl:?}))).instance;
new BigInt64Array(l.exports.memory.buffer)[8] = -1n;
let lt = false; try {{ l.exports[{lsym:?}](64, 128, 256); }} catch {{ lt = true; }} if (!lt) throw new Error('lut no trap');"#, rsym = cw_rem.kernel_sym, lsym = cw_lut.kernel_sym);
    let out = run_node_js(&dir, "traps", &js);
    assert!(out.status.success(), "WASM trap parity: {}", String::from_utf8_lossy(&out.stderr));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unmodified_4_domain_generalization() {
    let dir = tmp_dir("gen4");
    let domains: &[(&str, &str)] = &[
        ("morton4d", r#"
pub fn spread4_16(v: u16) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 24)) & 0x0000_00FF_0000_00FF; x = (x | (x << 12)) & 0x000F_000F_000F_000F;
    x = (x | (x << 6)) & 0x0303_0303_0303_0303; (x | (x << 3)) & 0x1111_1111_1111_1111
}
pub fn compact4_16(code: u64) -> u16 {
    let mut x = code & 0x1111_1111_1111_1111;
    x = (x | (x >> 3)) & 0x0303_0303_0303_0303; x = (x | (x >> 6)) & 0x000F_000F_000F_000F;
    x = (x | (x >> 12)) & 0x0000_00FF_0000_00FF; ((x | (x >> 24)) & 0xFFFF) as u16
}
pub fn decode_morton4d_slots(code: u64) -> (u16, u16, u16, u16) { (compact4_16(code), compact4_16(code >> 1), compact4_16(code >> 2), compact4_16(code >> 3)) }"#),
        ("static_arith", r#"
pub fn guard_metrics(a: u32, b: u32) -> u32 {
    if a == 0 { return 32u32; }
    a.saturating_add(b).saturating_sub(a.abs_diff(b)) ^ a.leading_zeros()
}"#),
        ("cidr_ring", r#"
const RING_MASK: u32 = 1023;
pub fn ring_admit(src: &[u32], out: &mut [u32]) {
    let mut h = src[0]; h ^= src[1] << 3; h += src[2] & RING_MASK;
    out[0] = h & RING_MASK; out[1] = if (src[0] & 0xFFFF_0000) == (src[3] & 0xFFFF_0000) { 1u32 } else { 0u32 };
    out[2] = src[2].wrapping_add(1) & RING_MASK; out[3] = h ^ src[3];
}"#),
        ("hash_checksum", r#"
pub fn mix_hash32(v0: u32, v1: u32) -> u32 {
    let mut h = v0.wrapping_mul(0x0100_0193) ^ v1.swap_bytes();
    for i in 0..4 { h = h.rotate_left(7).wrapping_add(v1.rotate_right(3)) ^ (i as u32); }
    h
}"#),
    ];
    let ov = BTreeMap::new();
    let mut all_ks = Vec::new();
    for &(dom_name, src) in domains {
        let ks = parse_rust_file_kernels(src, &ov);
        assert!(!ks.is_empty(), "domain {dom_name} extracted 0 kernels");
        for k in ks {
            let kb = kernel_reg_budget(&k, RegBudget::default());
            let obj = compile_kernel_direct(&k, kb, TargetIsa::X86_64).unwrap_or_else(|e| panic!("x86 {dom_name}::{}: {e:?}", k.name));
            assert!(obj.zero_spills && verify_elf64_object(&obj.elf_bytes).is_ok());
            assert!(!compile_kernel_wasm(&k, kb).unwrap_or_else(|e| panic!("wasm {dom_name}::{}: {e:?}", k.name)).wasm_bytes.is_empty());
            all_ks.push(k);
        }
    }
    assert!(all_ks.len() >= 6);
    let k_dec = all_ks.iter().find(|k| k.name == "decode_morton4d_slots").unwrap();
    let spread = |v: u16| -> u64 { let mut x = u64::from(v); x=(x|(x<<24))&0x0000_00FF_0000_00FF; x=(x|(x<<12))&0x000F_000F_000F_000F; x=(x|(x<<6))&0x0303_0303_0303_0303; (x|(x<<3))&0x1111_1111_1111_1111 };
    let coords: [(u16, u16, u16, u16); 8] = [
        (0, 1, 2, 3), (0xFFFF, 0x1234, 0x4321, 0xABCD), (7, 42, 999, 65000), (0x8000, 0x7FFF, 0, 1),
        (13, 37, 255, 256), (0x5555, 0xAAAA, 0x0F0F, 0xF0F0), (1000, 2000, 3000, 4000), (65534, 0, 1, 32768),
    ];
    let codes: [u64; 8] = std::array::from_fn(|i| { let (a, b, c, d) = coords[i]; spread(a) | (spread(b) << 1) | (spread(c) << 2) | (spread(d) << 3) });
    let codes_i64: [i64; 8] = codes.map(|c| c as i64);
    let ev = k_dec.eval(&[&codes_i64, &[0; 8], &[0; 8], &[0; 8], &[0; 8]]).expect("ir morton");
    for i in 0..8 { assert_eq!((ev[1][i] as u16, ev[2][i] as u16, ev[3][i] as u16, ev[4][i] as u16), coords[i], "ir morton4d"); }
    #[cfg(target_arch = "x86_64")]
    {
        let (ex, _) = native::ExecMem::from_kernel(k_dec, kernel_reg_budget(k_dec, RegBudget::default()));
        let (mut o0, mut o1, mut o2, mut o3) = ([0u16; 8], [0u16; 8], [0u16; 8], [0u16; 8]);
        assert_eq!(unsafe { native::call_abi6(ex.ptr(), codes.as_ptr() as u64, o0.as_mut_ptr() as u64, o1.as_mut_ptr() as u64, o2.as_mut_ptr() as u64, o3.as_mut_ptr() as u64, 1) }, 0);
        for i in 0..8 { assert_eq!((o0[i], o1[i], o2[i], o3[i]), coords[i], "x86 morton4d"); }
        let k_h = all_ks.iter().find(|k| k.name == "mix_hash32").unwrap();
        let (exh, _) = native::ExecMem::from_kernel(k_h, kernel_reg_budget(k_h, RegBudget::default()));
        let (v0, v1) = ([0u32, 1, 0xDEAD_BEEF, u32::MAX, 7, 13, 0x1337_C0DE, 0x8000_0000], [42u32, 0x1234_5678, 0, 99, 0xFFFF_0000, 1, 0xCAFE_BABE, u32::MAX]);
        let mut oh = [0u32; 8];
        assert_eq!(unsafe { native::call_abi6(exh.ptr(), v0.as_ptr() as u64, v1.as_ptr() as u64, oh.as_mut_ptr() as u64, 1, 0, 0) }, 0);
        for i in 0..8 { let mut h = v0[i].wrapping_mul(0x0100_0193) ^ v1[i].swap_bytes(); for j in 0..4u32 { h = h.rotate_left(7).wrapping_add(v1[i].rotate_right(3)) ^ j; } assert_eq!(oh[i], h, "x86 hash"); }
    }
    let cw_m = compile_kernel_wasm(k_dec, kernel_reg_budget(k_dec, RegBudget::default())).expect("wasm morton");
    let wp = dir.join("morton.wasm"); fs::write(&wp, &cw_m.wasm_bytes).unwrap();
    let c_js = codes.iter().map(|c| format!("{c}n")).collect::<Vec<_>>().join(", ");
    let w_js: [String; 4] = [
        coords.iter().map(|c| c.0.to_string()).collect::<Vec<_>>().join(","),
        coords.iter().map(|c| c.1.to_string()).collect::<Vec<_>>().join(","),
        coords.iter().map(|c| c.2.to_string()).collect::<Vec<_>>().join(","),
        coords.iter().map(|c| c.3.to_string()).collect::<Vec<_>>().join(","),
    ];
    let js = format!(r#"import fs from 'fs';
const inst = (await WebAssembly.instantiate(fs.readFileSync({wp:?}))).instance;
const u64 = new BigUint64Array(inst.exports.memory.buffer); const u16 = new Uint16Array(inst.exports.memory.buffer);
const codes = [{c_js}]; for (let i=0;i<8;i++) u64[8+i] = codes[i];
inst.exports[{bsym:?}](64, 128, 144, 160, 176, 1);
const w = [[{w0}],[{w1}],[{w2}],[{w3}]];
for (let o=0;o<4;o++) for (let i=0;i<8;i++) if (u16[64+o*8+i]!==w[o][i]) throw new Error('wasm morton '+o+' '+i);"#,
        w0 = w_js[0], w1 = w_js[1], w2 = w_js[2], w3 = w_js[3], bsym = cw_m.batch_sym);
    let out = run_node_js(&dir, "morton", &js);
    assert!(out.status.success(), "wasm morton: {}", String::from_utf8_lossy(&out.stderr));
    let _ = fs::remove_dir_all(&dir);
}
