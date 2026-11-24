# zero-cc

Zero-dependency, zero-spill Rust specialized kernel compiler & sovereign **100.0% Direct `ELF64`** (`ET_EXEC` + `ET_REL`, `x86_64` + `AArch64`) machine-code backend in **4,103 LOC** across `src/`.

![zero-cc vs rustc -O3 on zero-rans](assets/zero_rans_speedup.svg)

## Architecture

`zero-cc` (`zero_cc`) extracts pure numeric kernels directly from unmodified `.rs` files or crate trees, solves typed buffer ownership/noalias, narrow element widths (`i64`/`i32`/`i16`/`u32`/`u16`/`u8`) and variable lane widths (`1..=8`), exact value-range + CRT/quadratic modular congruence domains, and cycle-free DAG algebraic super-optimization (including signed Barrett division `((x * m) >> s) - (x >> 63)`), then lowers directly to:
1. **Standalone `100.0%` Direct Linux `ELF64` `ET_EXEC` binaries** (zero `rustc`, zero `cc`, zero `ld`, zero LLVM, strict W^X page-congruent `RX`/`RW` `PT_LOAD` + non-executable `PT_GNU_STACK`, self-verified by `verify_elf64_executable`);
2. **Relocatable `ELF64` `ET_REL` `.o` object files** with typed `extern "C"` Rust FFI shims (`0` LLVM `undef`/`poison` drift, `0` stack spills); or
3. **Branchless safe Rust** (`&[T; N]` / `&mut [T; N]`).

Module breakdown (`4,103` total LOC across `9` files, `0` external crates):

- `src/lib.rs` (64 LOC): `#![forbid(unsafe_code)]` compile-time declarative macros (`define_errors!`, `define_ops!` with unified `children`/`map_ids` arity-3 traversal) and pipeline entrypoints (`compile_kernel`, `compile_rust_source`, `compile_kernel_direct`, `compile_rust_to_elf`, `compile_standalone_elf_exec`, `compile_rust_to_elf_exec`).
- `src/rust_fe.rs` (1,352 LOC): Zero-dependency multi-file Rust crate & kernel frontend (`parse_rust_kernel`, `parse_rust_file_kernels`, `extract_crate_kernels`, `extract_and_compile_crate`, `parse_domain_spec`, `kernel_reg_budget`) supporting narrow `ElemTy` buffers, scalar/tuple auto-batching, bounded helper inlining (`<= 6` depth), constant `for` loop unrolling (`<= 16` trips), `/// @domain` doc comments, `#[zero_domain(...)]` attributes, CLI `--domain` provenance overrides, value/statement `if`/`else` -> branchless `Op::Select`, and exact signed Barrett lowering (`((x * m) >> s) - (x >> 63)`).
- `src/ir.rs` (416 LOC): Typed ownership (`BufAccess::{ReadOnly, MutNoAlias, LookupTable(usize)}`), `ElemTy::{I64, I32, I16, U32, U16, U8}` (`natural_domain`, `cast_i64`, `push_le_elem`, `scale_shift`), variable `lanes` (`BufDecl::{elem_ty, lanes, block_bytes}`), exact interval + modular congruence lattice `Domain { lo, hi, modulus, rem }` with `i128` tail-recursive `const fn` `gcd_u128`/`ext_gcd_i128` CRT/LCM `try_meet`/`meet`, quadratic residue lifting `eval_square`, `const fn` `u128`-checked `barrett_params(d, max_x)`, DAG validation, cost model, and bit-exact interpreter `eval()`.
- `src/egraph.rs` (264 LOC): Deterministic Union-Find + `BTreeMap` hash-consed E-Graph with domain intersection, upward domain refinement, `noalias` store-forwarding and load-hoisting, dead bounds-check and dead saturation-clamp elimination, domain-proven dead-branch `Select` folding, distributive/butterfly factoring, shift-add/sub strength reduction, modular congruence zeroing, domain-exact Barrett reduction, and cycle-free DAG-cost extraction (`dag_cost`).
- `src/codegen.rs` (187 LOC): Generalized arity-3 Sethi-Ullman (`calc_su`) minimum-live-register DAG scheduler with memory-chain anti-dependency ordering, early output emission, pending-read hazard tracking, and fail-closed zero-spill register cliff guard (`RegBudget { max_gprs: 14, max_vecs: 16 }`), plus warning-free (`-D warnings`) Rust emitter.
- `src/isa.rs` (1,094 LOC) & `src/elf64.rs` (374 LOC): Linear-scan physical GPR allocator (`allocate_registers`) proving `spills == 0`, System V AMD64 `x86_64` scalar (exact width-matched loads/stores across all 6 `ElemTy` widths with mandatory REX prefix on `u8` stores so `sil/dil/bpl/r8b..r15b` never collapse into `ah/ch/dh/bh`) + 256-bit AVX2 `ymm0..ymm15` (`i64x8`) and AAPCS64 `AArch64` encoders, 16-byte-aligned counted `<name>_batch` loops, two-pass fixed-width `_start` synthesis invoking Linux `sys_write`/`sys_exit`, standalone `ET_EXEC` (`emit_elf64_executable` + `verify_elf64_executable`) and `ET_REL` (`emit_elf64_object` + `verify_elf64_object`) emitters with strict W^X airlock.
- `src/ntf1.rs` (158 LOC) & `specs/zero_cc.ntf1`: Zero-SVO `NTF1` (Native Token-Morphism Format v1) parser, canonical emitter, and `0t` airlock fixed-point verifier.
- `src/bin/zero-cc.rs` (194 LOC): Sovereign CLI (`zero-cc <file_or_dir> [--kernel <name>] [--domain <spec>] [--emit-elf <out.o>] [--emit-exec <out_bin>] [--isa x86_64|aarch64] [--bench]`).

## Live Benchmarks: Real `zero-rans` Kernels (`rustc -O3` vs `zero-cc` Direct `ELF64`)

Measured live on **AMD Ryzen 9 3900X** (`rustc 1.91.0 -C opt-level=3`, `512` blocks x `256` iters = `131,072` calls/trial, median of `9` interleaved trials, `std::hint::black_box`). All `6` kernels are extracted straight from unmodified `../zero-rans/src/{residual.rs,rans8.rs}`, verified bit-exact across `4,096` blocks (including signed Barrett corners `-262144, -289, -29, -1, 0, +1, +29, +289, +262144`) on linked `ET_REL` `.o` and across `512` blocks (`16,384` raw `stdout` bytes + exit code) via direct Linux execution of standalone `100.0%` `ET_EXEC` binaries (zero `rustc`/`cc`/`ld`). `objdump -d -M intel` confirms `0` `idiv`, `0` `div`, and `0` stack spills across every `ET_REL` and `ET_EXEC` binary.

| Kernel (`zero-rans`) | IR Ops (`orig -> opt`) | `idiv` / Spills | `ET_EXEC` Direct Share | `rustc -O3` (`ns/call`) | `zero-cc` `ELF64` (`ns/call`) | Speedup (`rustc / zcc`) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `idct8_1d_combine` | `165 -> 17` | `0` / `0` | `100.0%` (`91` insns) | `8.81` | `2.03` | **`4.34x`** | Domain proves `[-70B, 70B]` guard & clamps dead |
| `dct8_1d_even` | `69 -> 31` | `0` / `0` | `100.0%` (`151` insns) | `4.35` | `3.07` | **`1.42x`** | Domain proves all `saturate_i32_from_i64` clamps dead |
| `dct8_1d_odd` | `93 -> 53` | `0` / `0` | `100.0%` (`253` insns) | `6.12` | `5.25` | **`1.17x`** | Domain proves all `saturate_i32_from_i64` clamps dead |
| `idct8_1d_odd` | `93 -> 71` | `0` / `0` | `100.0%` (`317` insns) | `6.38` | `7.06` | `0.90x` | Signed Barrett `/ 289` (`0` `idiv`); scalar GPR + batch `call` |
| `idct8_1d_even` | `49 -> 37` | `0` / `0` | `100.0%` (`177` insns) | `3.29` | `3.72` | `0.89x` | Signed Barrett `/ 29` (`0` `idiv`); scalar GPR + batch `call` |
| `rans_renorm_branchless` | `137 -> 68` | `0` / `0` | `100.0%` (`397` insns) | `2.86` | `10.86` | `0.26x` | Scalar GPR (`374` insns) vs LLVM SSE2 auto-vec on `u32`/`u16` |

Full-crate scan (`./target/release/zero-cc ../zero-rans/src --domain "raw: -262144..=262144, x: -32768..=32767, even: -1000000000..=1000000000, odd: -1000000000..=1000000000"`): `7` files, `152` functions, `26` kernels extracted, **`25/26` compiled to `100.0%` direct `ET_EXEC` + `ET_REL` with `0` spills in `22.80 ms`**.

## Live Benchmarks: `i64x8` AVX2 Kernels (32,768 lanes, AMD Ryzen 9 3900X)

All `3` compiled `i64x8` kernels are verified bit-exact against naive slice Rust across all 32,768 lanes before timing and compile with `rustc -D warnings` (0 warnings).

| Kernel (`i64x8`) | `rustc -C opt-level=3` (SSE2) | `rustc -C opt-level=3 -C target-cpu=native` (AVX2) | `objdump -d` (`zero-cc` AVX2) | `objdump -d` (`naive` AVX2) |
| :--- | :---: | :---: | :--- | :--- |
| `modular_hash_lut` | `105,678 ns -> 43,775 ns` (**`2.41x`**) | `112,517 ns -> 27,796 ns` (**`4.05x`**) | `81` insns, `56` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `101` insns, `0` vec, `5` panic calls |
| `butterfly_8lane` | `18,000 ns -> 19,548 ns` (`0.92x` scalar) | `19,490 ns -> 7,095 ns` (**`2.75x`**) | `34` insns, `11` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `145` insns, `0` vec, `12` panic calls |
| `affine_filter_8lane` | `72,924 ns -> 44,522 ns` (**`1.64x`**) | `59,187 ns -> 18,652 ns` (**`3.17x`**) | `305` insns, `266` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `58` insns, `0` vec, `3` panic calls |

## CLI & Verification Commands

```bash
# Build release CLI (never export RUSTFLAGS)
env -u RUSTFLAGS cargo build --release --offline

# Scan an entire Rust crate and emit report for every pure numeric kernel
./target/release/zero-cc ../zero-rans/src \
  --domain "raw: -262144..=262144, x: -32768..=32767, even: -1000000000..=1000000000, odd: -1000000000..=1000000000"

# Compile a single kernel directly to ET_REL (.o) and standalone 100% Linux ET_EXEC
./target/release/zero-cc ../zero-rans/src/residual.rs \
  --kernel idct8_1d_odd --domain "raw: -262144..=262144" \
  --emit-elf /tmp/idct8_1d_odd.o --emit-exec /tmp/idct8_1d_odd --bench

# Full test suite (13/13: direct ELF64 + equiv/bench + live zero-rans 6-kernel bench)
env -u RUSTFLAGS cargo test --release --offline -- --nocapture

# Clippy, Ripwire clone/lint audit, and zero-check proof gauntlet
env -u RUSTFLAGS cargo clippy --all-targets --offline -- -D warnings
ripwire --clones src
ripwire --lint --lint-ignore=naming src
../zero-check/target/release/zero-check --no-ledger --prove-fns --kernel-strict --ntf1 specs/zero_cc.ntf1 src
```

Hygiene invariants:
- `src/` LOC: `4,103` (`9` files)
- `Cargo.lock` external dependencies: `0` (`0` `"source ="` lines)
- `#![forbid(unsafe_code)]`, `0` `.clamp(` / `::clamp(`, `0` `#[allow(` / `#[expect(`, `100%` 7-bit ASCII

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) and [NOTICE](NOTICE)).

Copyright 2026 Blind Labs (https://blindlabs.ai).
