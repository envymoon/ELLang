# ELLang

ELLang is an experimental AI-native programming language runtime.

The project is built around one core idea: the model should not behave like an external coding assistant that simply returns text. Instead, natural-language intent is compiled into an executable program representation:

`natural idea -> ProblemSpec -> ProgramSpec -> typed IR -> stable bytecode -> VM`

The execution result comes from the runtime executing that compiled representation, not from the model directly writing the final answer as chat output.

## Why This Project Exists

Most "LLM coding" systems are wrappers around a general-purpose language plus API calls. ELLang explores a different direction:

- the language itself is AI-aware
- intent understanding is part of compilation
- deterministic execution is separated from model planning
- IR, bytecode, tracing, and replay are first-class runtime concepts
- local models are preferred, with minimal external token dependence

The long-term goal is an executable model language for software construction, not a prompt orchestration toolkit.

## Current Architecture

ELLang is organized as a layered system.

1. User layer
   - The primary entrypoint is natural language.
   - Users can also inspect or edit `.ell` files as a mid-level representation.

2. Program layer
   - Natural language is first lowered into a `ProblemSpec`.
   - `ProblemSpec` captures problem type, I/O contracts, constraints, and correctness conditions.
   - `ProblemSpec` is lowered into a `ProgramSpec`.
   - `ProgramSpec` can be rendered as `.ell`.

3. Compiler layer
   - `ProgramSpec` is compiled into typed IR.
   - Typed IR is verified for types, capabilities, and resource budgets.
   - Typed IR is lowered into stable bytecode.

4. Runtime layer
   - Bytecode is executed by a VM.
   - A Rust native VM is preferred when available.
   - A Python reference VM remains available as a fallback and development oracle.

## What Works Today

- Natural-language entrypoint: `python -m ellang.ideate "..."`
- Editable `.ell` source as a mid-level view
- Intent classification and operator selection with lazy AI planning
- Structured planner JSON schema instead of free-form planning text
- Automatic validation loop after `.ell` generation:
  - parse
  - compile / type check
  - sample execution
  - mismatch diagnosis
  - constrained regeneration
- Typed IR generation and verification
- Stable bytecode generation
- Native Rust VM execution for the main bytecode path
- Native intrinsic execution for:
  - algorithm-family tasks
  - structured `MinStack` execution
- Capability, quota, replay, trace, and debug preparation layers
- Local-first Hugging Face model discovery for Qwen-family models
- Git/project snapshot metadata inside runtime execution

## Supported Algorithm Families

The project currently includes family-level intrinsic support for representative tasks in:

- array manipulation
- arrays and two pointers
- hash maps and counting
- stack / queue / heap
- linked lists
- trees and graphs
- dynamic programming and backtracking

Representative built-in tasks now include:

- `array_manipulation`
  - `rotate_array`
  - `binary_search`
  - `max_subarray`
  - `product_except_self`
- `array_two_pointers`
  - `three_sum`
- `hashmap_counting`
  - `group_anagrams`
  - `most_frequent_element`
- `stack_queue_heap`
  - `valid_parentheses`
  - `top_k_frequent`
- `linked_list`
  - `reverse_list`
- `tree_graph`
  - `binary_tree_level_order`
  - `binary_tree_right_side_view`
  - `num_islands`
- `dp_backtracking`
  - `coin_change`
  - `subsets`
  - `longest_increasing_subsequence`

This is not yet the same as "all of LeetCode is solved," but the architecture is organized around reusable algorithm families and executable primitives rather than one-off prompt templates.

## Repository Layout

```text
src/ellang/
  algorithm_families.py
  bytecode.py
  cache.py
  cli.py
  compiler.py
  ideate.py
  ideation.py
  runtime.py
  syntax.py
  typed_ir.py
  verifier.py
  vm.py
  models/
    backend.py

native/runtime-rs/
  src/
    bytecode.rs
    ffi.rs
    jit.rs
    main.rs
    vm.rs

examples/
tests/
docs/
```

## Installation

### Python runtime

```bash
python -m pip install -e .
```

Optional local-model dependencies:

```bash
python -m pip install -e .[hf]
```

### Native Rust VM

Build the native runtime from the repository root:

```bash
cd native/runtime-rs
cargo build --release
```

The Python runtime automatically uses the native VM when the built executable is present at the default location.

## Quick Start

### Run an existing `.ell` program

```bash
python -m ellang.cli examples/select_top_students.ell examples/students.json
```

### Program from natural language

```bash
python -m ellang.ideate "select the top 2 students by score" examples/students.json --show-ell
```

### Inspect local model discovery

```bash
python -m ellang.cli --models consumer
```

See:

- [Quickstart](docs/QUICKSTART.md)
- [User Manual](docs/MANUAL.md)
- [Local Models Guide](docs/LOCAL_MODELS.md)
- [Native Runtime Notes](native/README.md)

## Model Configuration

ELLang is designed to avoid hard-coded machine-specific paths.

Model selection can be changed in two ways:

1. Change defaults in [backend.py](src/ellang/models/backend.py)
   - `QwenBackendConfig`
   - `MODEL_PROFILES`

2. Override behavior with environment variables
   - `ELLANG_MODEL_PROFILE`
   - `ELLANG_MODEL_PATH`
   - `ELLANG_MODEL_ID`
   - `ELLANG_MODEL_QUANTIZATION`
   - `ELLANG_MODEL_DIR`
   - `HF_HOME`

The runtime auto-discovers local Hugging Face cache directories when possible. No personal filesystem path is required in the codebase.

## Public Project Notes

This repository is prepared to be shared publicly:

- documentation is written for external users
- tests use repository-relative paths
- local cache and build artifacts are git-ignored
- model configuration is environment-driven
- examples are included for both `.ell` and natural-language entrypoints

## Current Limitations

- The language is still experimental.
- The natural-language front end is intentionally heuristic and compiler-oriented, not a general chat agent.
- Planner output is checked against supported algorithm families so unsupported family/task pairs do not silently pass into runtime execution.
- The structured-program native runtime is still expanding toward more generic semantics.
- The native runtime does not yet replace every Python reference path.

## Verification

The core test suite can be run with:

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- broaden generic structured native execution beyond current specializations
- expand intrinsic families into richer reusable primitives
- move more hot execution paths out of Python
- strengthen type inference and FFI boundaries
- support additional runtime targets beyond the current native-first path
