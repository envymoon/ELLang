# Local Models

ELLang is designed to be local-first. The planner layer can use a local Hugging Face model when available, and the rest of the pipeline remains compiler- and runtime-driven.

## Default Profiles

The default model profiles live in [src/ellang/models/backend.py](../src/ellang/models/backend.py).

Current defaults:

- `consumer`
  - `Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4`
  - fallback: `Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4`
- `midrange`
  - `Qwen/Qwen2.5-Coder-3B-Instruct`
- `enhanced`
  - `Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4`

`7B+` is treated as an enhancement tier, not the default requirement.

## Where To Change the Model API

If you want to change the open-model backend, the main integration points are:

- [src/ellang/models/backend.py](../src/ellang/models/backend.py)
  - `QwenBackendConfig`
  - `MODEL_PROFILES`
  - `discover_local_model_paths`
  - `QwenLocalBackend._resolve_model_choice`
  - `QwenLocalBackend._get_model_bundle`

This is the file to edit if you want to:

- switch the default model family
- add a new hardware profile
- replace Hugging Face Transformers loading behavior
- wire in a different local model runtime
- change quantization defaults

## Environment Variables

ELLang avoids hard-coded personal paths. Use environment variables instead.

- `ELLANG_MODEL_PROFILE`
  - Selects `consumer`, `midrange`, or `enhanced`
- `ELLANG_MODEL_PATH`
  - Explicit path to a local model directory
- `ELLANG_MODEL_ID`
  - Explicit Hugging Face model id
- `ELLANG_MODEL_QUANTIZATION`
  - `int4`, `int8`, or `none`
- `ELLANG_MODEL_DIR`
  - Additional model root to search
- `HF_HOME`
  - Hugging Face home directory; ELLang will search `HF_HOME/hub`
- `ELLANG_DISABLE_MODEL_PLANNER`
  - Set to `1` to force deterministic fallback planning

## Auto-Discovery

ELLang searches for local models in this order:

1. `ELLANG_MODEL_PATH`
2. `ELLANG_MODEL_ID`
3. `ELLANG_MODEL_DIR`
4. `HF_HOME/hub`
5. `~/.cache/huggingface/hub`
6. `~/.huggingface/hub`

## Download Options

You can download models with the Hugging Face CLI.

### Download into the default Hugging Face cache

```bash
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4
```

### Download into an explicit directory

```bash
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4 --local-dir ./models/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4
```

Then point ELLang at that directory:

```bash
export ELLANG_MODEL_PATH=./models/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4
```

On PowerShell:

```powershell
$env:ELLANG_MODEL_PATH = ".\\models\\Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4"
```

## Inspecting Model Resolution

Use the CLI to inspect what ELLang can see:

```bash
python -m ellang.cli --models consumer
```

The output includes:

- requested profile
- configured candidate models
- discovered local model directories

## Recommended Consumer Setup

For modest local hardware:

- start with `1.5B int4`
- use `0.5B int4` as the lower-cost fallback
- move to `3B` only when you have more VRAM and want stronger planning
- treat `7B+` as opt-in

## Important Design Note

In ELLang, the local model is used for planning-oriented language tasks such as:

- intent classification
- parameter extraction
- operator selection
- constrained planning

It is not intended to be the runtime executor. Execution should come from typed IR, bytecode, and the VM.
