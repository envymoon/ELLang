# Native Runtime

This directory contains the native runtime path for ELLang.

## Purpose

The Python layer is currently used for:

- natural-language ideation
- syntax and `ProgramSpec` handling
- compiler prototyping
- typed IR and bytecode generation

The native runtime exists to move execution hot paths out of Python and toward a stable VM boundary.

## Current Goals

- execute stable bytecode in a native VM
- expand native intrinsics for reusable algorithm families
- support structured-program execution beyond Python-only semantics
- provide a path toward richer AOT/JIT and FFI support

## Layout

```text
native/runtime-rs/
  Cargo.toml
  src/
    main.rs
    bytecode.rs
    capabilities.rs
    ffi.rs
    jit.rs
    quotas.rs
    vm.rs
```

## Build

```bash
cd native/runtime-rs
cargo build --release
```

## Current State

Today the native runtime can execute:

- the main bytecode pipeline
- algorithm-family intrinsics
- the current structured `MinStack` path

The Python runtime automatically prefers the native executable when it is present at the expected build output path.

## Long-Term Direction

- broaden generic structured-program native execution
- replace more reference-VM-only semantics
- expand intrinsic registries
- strengthen FFI boundaries
- deepen JIT/AOT specialization for hot paths
