# zero-cc

Zero-dependency, zero-spill Rust specialized kernel compiler & sovereign Direct `ELF64` (`x86_64` + `AArch64`) machine-code backend in **2,309 LOC** across `src/`.

## Architecture

`zero-cc` (`zero_cc`) solves typed buffer ownership/noalias, exact value range and CRT/quadratic modular congruence domains, and cycle-free DAG algebraic super-optimization, then lowers **either** directly to standalone relocatable `ELF64` `.o` machine code (`>= 80%` direct `zero-cc` instructions, `0` LLVM `undef`/`poison` drift, `0` stack spills) **or** to branchless safe Rust (`&[i64; 8]` / `&mut [i64; 8]`):

- `src/lib.rs` (54 LOC): `#![forbid(unsafe_code)]` compile-time declarative macros (`define_errors!`, `define_ops!` with unified `children`/`map_ids` arity-3 traversal) and pipeline entrypoints `compile_kernel`, `compile_rust_source`, `compile_kernel_direct`, and `compile_rust_to_elf`.
- `src/rust_fe.rs` (311 LOC): Zero-dependency raw safe-Rust `.rs` kernel frontend (`parse_rust_kernel` + `compile_rust_source`) supporting `/// @domain` doc comments and `#[zero_domain(...)]` attributes, constant `for` loop unrolling (`<= 16` trips), exact Rust operator precedence including `==`/`!=`/`<`/`<=`/`>`/`>=`, value-form and statement-form `if`/`else` lowering to branchless `Op::Select`, fail-closed cast/ownership/bounds checks, and fixed-point round-trip compilation `compile_rust_source(&compiled.rust_source)`.
- `src/ir.rs` (328 LOC): Typed ownership (`BufAccess::{ReadOnly, MutNoAlias, LookupTable(usize)}`), exact interval + modular congruence lattice `Domain { lo, hi, modulus, rem }` with explicit bottom `Domain::empty()`, `tighten()`, `i128` tail-recursive `const fn` `gcd_u128`/`ext_gcd_i128` CRT/LCM `try_meet`/`meet`, exact congruence-preserving `join()`, `eval_cmp()`, constant-preserving `eval_add_sub`/`eval_mul`, exact quadratic residue lifting `eval_square` `(m*k + r)^2 == r^2 (mod m * gcd(m + 2r, 2m))`, unified generic `Op<Id>` (`Cmp`, `Select`), `const fn` `u128`-checked non-negative `barrett_params(d, max_x)`, DAG validation, cost model, and bit-exact interpreter `eval()`.
- `src/egraph.rs` (264 LOC): Deterministic Union-Find + `BTreeMap` hash-consed E-Graph with domain intersection, upward domain refinement, `noalias` store-forwarding and load-hoisting, dead bounds-check elimination, domain-proven dead-branch `Select` folding and `Cmp` bool-domain round-trip normalization, distributive/butterfly factoring, shift-add/sub strength reduction, shift fusion, modular congruence zeroing, domain-exact Barrett reduction, cycle-free DAG-cost extraction (`dag_cost`), and preserved `proven_domains` across extraction.
- `src/codegen.rs` (187 LOC): Generalized arity-3 Sethi-Ullman (`calc_su`) minimum-live-register DAG scheduler with memory-chain anti-dependency ordering, early output emission, pending-read hazard tracking, and fail-closed zero-spill register cliff guard (`RegBudget { max_gprs: 14, max_vecs: 16 }`), plus warning-free (`-D warnings`) Rust emitter with `/// @domain` doc-comment propagation, branchless `Cmp`/`Select` (`cmov`/`vblend`) emission, narrow `u32 * u32 -> u64` multiplies (`vpmuludq`), and logical unsigned right shifts (`vpsrlq`).
- `src/isa.rs` (829 LOC) & `src/elf64.rs` (178 LOC): Linear-scan physical GPR allocator (`allocate_registers`) over `ScheduleReport` proving `spills == 0`, System V AMD64 `x86_64` scalar (`cmovne`, bounds-checked SIB `LoadLut`, REX.W/ModRM) + 256-bit AVX2 `ymm0..ymm15` (`vpaddq`, `vpsubq`, `vpand`, `vpmuludq`, `vpsllq`, `vpsrlq`, `vmovdqu`) and AAPCS64 `AArch64` (`csel`, `cset`, `ubfm`/`sbfm`, bounds-checked scaled `ldr`/`str`, `i64::MIN / -1` `cmn`+`brk` guard, `cbz`/`subs`/`b.ne`) encoders, 16-byte-aligned counted `<name>_batch` multi-block loops + `R_X86_64_PLT32` / `R_AARCH64_CALL26` `.rela.text` relocations, standalone `ELF64` `.o` emitter (`emit_elf64_object` with non-exec `.note.GNU-stack`), and `0t` self-verifier (`verify_elf64_object`).
- `src/ntf1.rs` (158 LOC) & `specs/zero_cc.ntf1`: Zero-SVO `NTF1` (Native Token-Morphism Format v1) parser, canonical emitter, `0t` airlock fixed-point verifier (`+t ->`, `-t <-`, `0t ><`, `|>`), `0t Elf64.Emit >< Elf64.ParseHeader`, `-t Fault.LlvmPoisonDrift <- Guard.DirectElf64Backend`, `0t RustSrc.Parse >< RustO3.Emit`, and `export_kernel_ntf1`.

## Verified Live Benchmarks (AMD Ryzen 9 3900X, `rustc 1.91.0`, 32,768 lanes)

All 3 compiled kernels are verified bit-exact against naive slice Rust across all 32,768 lanes before timing and compile with `rustc -D warnings` (0 warnings).

| Kernel | `rustc -C opt-level=3` (SSE2) | `rustc -C opt-level=3 -C target-cpu=native` (AVX2) | `objdump -d` (`zero-cc` AVX2) | `objdump -d` (`naive` AVX2) |
| :--- | :---: | :---: | :--- | :--- |
| `modular_hash_lut` | `105,678 ns -> 43,775 ns` (**`2.41x`**) | `112,517 ns -> 27,796 ns` (**`4.05x`**) | `81` insns, `56` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `101` insns, `0` vec, `5` panic calls |
| `butterfly_8lane` | `18,000 ns -> 19,548 ns` (`0.92x` scalar; `57` vs `145` insns, `0` vs `12` calls) | `19,490 ns -> 7,095 ns` (**`2.75x`**) | `34` insns, `11` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `145` insns, `0` vec, `12` panic calls |
| `affine_filter_8lane` | `72,924 ns -> 44,522 ns` (**`1.64x`**) | `59,187 ns -> 18,652 ns` (**`3.17x`**) | `305` insns, `266` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `58` insns, `0` vec, `3` panic calls |

Across all compiled kernels (including raw-`.rs` `quad_crt_8lane`, `virtio_ring_filter`, and `branchless_clamp_and_mux`) under both profiles, `objdump -d` confirms: `0` stack frame allocations (`sub $..., %rsp`), `0` stack spills, `0` hot-loop `(%rsp)` accesses, `0` hardware `idiv`, and `0` panic/bounds-check calls.

## Raw `.rs` Round-Trip Airlock (`zero-cc >< zero-check`) & Compile-Time Recursion

- **Direct `.rs -> E-Graph -> Zero-Spill .rs` Fixed-Point (`0t RustSrc.Parse >< RustO3.Emit`):** `compile_rust_source` ingests raw safe-Rust `.rs` kernels (`virtio_ring_filter`, `branchless_clamp_and_mux`) with `/// @domain` doc comments, lowers `if / else` and comparisons into branchless `Op::Cmp` + `Op::Select` (`cmov`/`vblend`), eliminates divisions/moduli/bounds checks via CRT + quadratic congruence E-Graph rewrites, hits exact node-count idempotence on `compile_rust_source(&compiled.rust_source)`, and passes `zero-check --prove-fns --kernel-strict` in `< 2 ms`.
- **Compile-Time Recursive Compression (`1,272` LOC across `src/`):** `define_ops!` (`[Id; 3]` arity traversal delegating `children` to `map_ids`), `define_errors!`, and tail-recursive `const fn` number-theory solvers (`gcd_u128`, `ext_gcd_i128`, `barrett_params`) keep the entire compiler, E-Graph, register-cliff scheduler, `NTF1` airlock, and `.rs` frontend under `1,300` LOC with `0` external crates and `0` `ripwire` clones/findings.

## Verification & Hygiene Gates

```bash
env -u RUSTFLAGS cargo test --release --offline -- --nocapture
env -u RUSTFLAGS cargo clippy --all-targets --offline -- -D warnings
env -u RUSTFLAGS cargo +nightly miri test --offline
env -u RUSTFLAGS cargo kani --tests
ripwire --clones src
ripwire --lint --lint-ignore=naming src
```

Hygiene invariants:
- `src/` LOC: `2,309` (`8` modules)
- `Cargo.lock` external dependencies: `0` (`0` `"source ="` lines)
- `#![forbid(unsafe_code)]`, `0` `.clamp(` / `::clamp(`, `0` `#[allow(` / `#[expect(`, `100%` 7-bit ASCII

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) and [NOTICE](NOTICE)).

Copyright 2026 Blind Labs (https://blindlabs.ai).
