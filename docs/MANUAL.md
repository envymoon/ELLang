# User Manual

## Overview

ELLang is an experimental AI-native programming language runtime. It is not designed as a prompt wrapper around a traditional language. Instead, it treats planning, typed IR generation, bytecode lowering, verification, and runtime execution as language-native concerns.

The most important design principle is:

- the model helps produce executable program structure
- the VM executes the program
- the final result should come from execution, not from chat text

## Core Pipeline

The execution pipeline is:

1. natural-language idea or `.ell`
2. `ProgramSpec`
3. typed IR
4. verified typed IR
5. stable bytecode
6. VM execution
7. replay, trace, diagnostics, and debug preparation

## Two User Entry Points

### 1. Natural-language programming

Use this when you want the system to generate the intermediate program for you.

```bash
python -m ellang.ideate "select the top 2 students by score" examples/students.json --show-ell
```

### 2. Direct `.ell` authoring

Use this when you want to inspect or directly edit the mid-level representation.

```bash
python -m ellang.cli examples/select_top_students.ell examples/students.json
```

## What `.ell` Is

`.ell` is a human-readable mid-level language. It is not the final machine representation. It sits between natural-language intent and typed IR.

In public usage, `.ell` should be thought of as:

- an editable compiler-facing program description
- a debugging and inspection layer
- a stable place for humans to refine AI-generated programs

## `.ell` Syntax

### Top-level sections

An `.ell` program can contain:

- `program`
- `intent`
- `input`
- `output`
- `constraint`
- `object`
- `module`
- `project`
- `flow`

### Minimal example

```text
program "sort_students"
intent "Sort students by score descending and return the top 2"

input students: dataset
output result: dataset

constraint deterministic = true

flow:
  use "sort students by score descending and return the top 2"
```

### Objects

```text
object Student:
  field name: string
  field score: number
```

### Modules

```text
module rank_students:
  intent "Rank students by score"
  use "sort students by score descending"
```

Modules with parameters:

```text
module push(value):
  append value to stack
```

### Flow and statements

Supported statement families include:

- `use "..."`  
  Freeform action/intention step
- `use module Name`
- `emit expr`
- `append expr to target`
- `remove last(target)`
- `return expr`
- `initialize ObjectName`
- `call module(arg1, arg2)`
- `set name = expr`
- `if condition:`
- `else:`
- `loop item in source:`
- `while condition:`
- `break`
- `continue`

### Structured example: MinStack

```text
program "MinStackProgram"
intent "Implement a stack that supports push, pop, top, and getMin in O(1) time"

input operations: list
input values: list
output result: list

constraint deterministic = true

object MinStack:
  field stack: list
  field min_stack: list

module push(value):
  append value to stack
  if min_stack is empty OR value <= current minimum:
    append value to min_stack

module pop:
  if top of stack == top of min_stack:
    remove last(min_stack)
  remove last(stack)

module top:
  return last(stack)

module getMin:
  return last(min_stack)

flow:
  initialize MinStack
  loop op in operations:
    if op == "MinStack":
      append null to results
    else:
      if op == "push":
        call push(values[next])
        append null to results
      else:
        if op == "pop":
          call pop()
          append null to results
        else:
          if op == "top":
            append call top() to results
          else:
            if op == "getMin":
              append call getMin() to results
  emit results
```

## Input Data

Bindings are usually provided as JSON.

Example:

```json
{
  "students": [
    {"name": "Ada", "score": 91},
    {"name": "Linus", "score": 88},
    {"name": "Grace", "score": 99}
  ],
  "top_only": true
}
```

## CLI Commands

### Run `.ell`

```bash
python -m ellang.cli path/to/program.ell path/to/bindings.json
```

### Inspect model profiles

```bash
python -m ellang.cli --models consumer
```

### Start from natural language

```bash
python -m ellang.ideate "your idea here" path/to/bindings.json --show-ell
```

### Write generated `.ell` to disk

```bash
python -m ellang.ideate "your idea here" path/to/bindings.json --write-ell generated.ell --show-ell
```

## Output Structure

The CLI returns JSON that typically includes:

- `program`
- `intent`
- `diagnostics`
- `result`
- `suggested_tests`
- `debug_report`
- `typed_ir_nodes`
- `bytecode_instructions`
- `vm_backend`
- `replay`
- `flowchart`
- `tracechart`

## Model Integration

The model integration is intentionally configurable and public-repo-safe.

### Main file

Model selection and planning behavior live in:

- [src/ellang/models/backend.py](../src/ellang/models/backend.py)

### Main extension points

- `QwenBackendConfig`
- `MODEL_PROFILES`
- `QwenLocalBackend`
- `discover_local_model_paths`
- `describe_model_profile`

### Environment-based configuration

Do not hard-code machine-specific paths in source files. Use environment variables:

- `ELLANG_MODEL_PROFILE`
- `ELLANG_MODEL_PATH`
- `ELLANG_MODEL_ID`
- `ELLANG_MODEL_QUANTIZATION`
- `ELLANG_MODEL_DIR`
- `HF_HOME`

See [LOCAL_MODELS.md](LOCAL_MODELS.md) for concrete setup examples.

## Runtime Architecture

### Python side

Important files:

- [src/ellang/ideation.py](../src/ellang/ideation.py)
- [src/ellang/syntax.py](../src/ellang/syntax.py)
- [src/ellang/compiler.py](../src/ellang/compiler.py)
- [src/ellang/typed_ir.py](../src/ellang/typed_ir.py)
- [src/ellang/bytecode.py](../src/ellang/bytecode.py)
- [src/ellang/runtime.py](../src/ellang/runtime.py)
- [src/ellang/vm.py](../src/ellang/vm.py)

### Rust side

Important files:

- [native/runtime-rs/src/main.rs](../native/runtime-rs/src/main.rs)
- [native/runtime-rs/src/bytecode.rs](../native/runtime-rs/src/bytecode.rs)
- [native/runtime-rs/src/vm.rs](../native/runtime-rs/src/vm.rs)
- [native/runtime-rs/src/jit.rs](../native/runtime-rs/src/jit.rs)
- [native/runtime-rs/src/ffi.rs](../native/runtime-rs/src/ffi.rs)

## Native vs Reference VM

ELLang prefers the native Rust VM when available. The Python reference VM is still valuable for:

- semantic prototyping
- fallback execution
- debugging new operators
- serving as an oracle while native support expands

The active backend is reported in the CLI output:

- `vm_backend: "native"`
- `vm_backend: "reference"`

## Caching

Typed IR caching is handled by:

- [src/ellang/cache.py](../src/ellang/cache.py)

The cache currently supports:

- canonicalized keys
- hit/miss/store statistics
- TTL eviction
- max-entry pruning

Default cache location:

- `.ellang-cache/` in the current working directory

This directory is safe to ignore in Git.

## Capabilities and Safety

ELLang includes architectural constraints intended to support safer AI-native execution:

- capability-aware typed IR
- runtime quotas
- deterministic lowering when possible
- replay metadata
- local-first debug flow

Current capability-sensitive areas include:

- model inference
- git/project access
- debug escalation
- FFI calls

## Debugging Model

The current debug path is:

1. local diagnostics
2. trace and replay capture
3. debug preparation
4. optional future external escalation

The goal is that external API use remains a fallback, not the core executor.

## Public Repository Guidance

If you are publishing or forking ELLang:

- keep user-specific paths out of the source tree
- keep caches and build outputs out of version control
- configure models through environment variables
- document local-model assumptions in `docs/LOCAL_MODELS.md`
- treat `.ell` as a compiler-visible language layer, not just a prompt log

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Current Scope

ELLang is a serious prototype, not a finished language product. It already demonstrates:

- natural-language to program lowering
- typed IR generation
- stable bytecode lowering
- VM execution
- native runtime integration

What remains is broader generalization, deeper native semantics, and a larger reusable primitive set.
