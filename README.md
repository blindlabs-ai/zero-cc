# zero-cc

Zero-dependency, zero-spill Rust specialized kernel compiler in **1,250 LOC** across `src/`.

## Architecture

`zero-cc` (`zero_cc`) solves the three things general-purpose LLVM IR cannot prove on its own -- typed buffer ownership/noalias, exact value range and CRT/quadratic modular congruence domains, and cycle-free DAG algebraic super-optimization -- then lowers to branchless, fixed-size, zero-spill Rust (`&[i64; 8]` / `&mut [i64; 8]`) so `rustc -C opt-level=3` handles instruction selection, AVX2 SLP vectorization, and register allocation cleanly:

- `src/lib.rs`: `#![forbid(unsafe_code)]` pipeline entrypoint `compile_kernel(kernel, RegBudget)` -> validate -> E-Graph -> zero-spill schedule -> Rust codegen.
- `src/ir.rs`: Typed ownership (`BufAccess::{ReadOnly, MutNoAlias, LookupTable(usize)}`), exact interval + modular congruence lattice `Domain { lo, hi, modulus, rem }` with explicit bottom `Domain::empty()`, `tighten()`, `i128` extended-GCD CRT/LCM `try_meet`/`meet`, constant-preserving `eval_add_sub`/`eval_mul`, exact quadratic residue lifting `eval_square` `(m*k + r)^2 == r^2 (mod m * gcd(m + 2r, 2m))`, unified generic `Op<Id>`, `u128`-checked non-negative `barrett_params(d, max_x)`, DAG validation, cost model, and bit-exact interpreter `eval()`.
- `src/egraph.rs`: Deterministic Union-Find + `BTreeMap` hash-consed E-Graph with domain intersection, upward domain refinement, `noalias` store-forwarding and load-hoisting, dead bounds-check elimination, distributive/butterfly factoring, shift-add/sub strength reduction, shift fusion, modular congruence zeroing, domain-exact Barrett reduction, cycle-free DAG-cost extraction (`dag_cost`), and preserved `proven_domains` across extraction.
- `src/codegen.rs`: Sethi-Ullman minimum-live-register DAG scheduler with early output emission, pending-read hazard tracking, and fail-closed zero-spill register cliff guard (`RegBudget { max_gprs: 14, max_vecs: 16 }`), plus warning-free (`-D warnings`) Rust emitter with narrow `u32 * u32 -> u64` multiplies (`vpmuludq`) and logical unsigned right shifts (`vpsrlq`).
- `src/ntf1.rs` & `specs/zero_cc.ntf1`: Zero-SVO `NTF1` (Native Token-Morphism Format v1) parser, canonical emitter, `0t` airlock fixed-point verifier (`+t ->`, `-t <-`, `0t ><`, `|>`), and `export_kernel_ntf1` causal-graph exporter for recursive `zero-cc >< zero-check` co-verification.

## Verified Live Benchmarks (AMD Ryzen 9 3900X, `rustc 1.91.0`, 32,768 lanes)

All 3 compiled kernels are verified bit-exact against naive slice Rust across all 32,768 lanes before timing and compile with `rustc -D warnings` (0 warnings).

| Kernel | `rustc -C opt-level=3` (SSE2) | `rustc -C opt-level=3 -C target-cpu=native` (AVX2) | `objdump -d` (`zero-cc` AVX2) | `objdump -d` (`naive` AVX2) |
| :--- | :---: | :---: | :--- | :--- |
| `modular_hash_lut` | `105,678 ns -> 43,775 ns` (**`2.41x`**) | `112,517 ns -> 27,796 ns` (**`4.05x`**) | `81` insns, `56` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `101` insns, `0` vec, `5` panic calls |
| `butterfly_8lane` | `18,000 ns -> 19,548 ns` (`0.92x` scalar; `57` vs `145` insns, `0` vs `12` calls) | `19,490 ns -> 7,095 ns` (**`2.75x`**) | `34` insns, `11` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `145` insns, `0` vec, `12` panic calls |
| `affine_filter_8lane` | `72,924 ns -> 44,522 ns` (**`1.64x`**) | `59,187 ns -> 18,652 ns` (**`3.17x`**) | `305` insns, `266` AVX2 vec, `0` spills, `0` `idiv`, `0` calls | `58` insns, `0` vec, `3` panic calls |

Across all 3 kernels under both profiles, `objdump -d` confirms: `0` stack frame allocations (`sub $..., %rsp`), `0` stack spills, `0` hot-loop `(%rsp)` accesses, `0` hardware `idiv`, and `0` panic/bounds-check calls.

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
- `src/` LOC: `<= 1,250` ceiling
- `Cargo.lock` external dependencies: `0` (`0` `"source ="` lines)
- `#![forbid(unsafe_code)]`, `0` `.clamp(` / `::clamp(`, `0` `#[allow(` / `#[expect(`, `100%` 7-bit ASCII

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) and [NOTICE](NOTICE)).

Copyright 2026 Blind Labs (https://blindlabs.ai).
