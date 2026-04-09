# Quickstart

This quickstart shows the two main ways to use ELLang:

- run an existing `.ell` program
- start from a natural-language idea

## 1. Install the Project

From the repository root:

```bash
python -m pip install -e .
```

Optional local-model dependencies:

```bash
python -m pip install -e .[hf]
```

## 2. Run an Existing `.ell` Program

Example program:

- [examples/select_top_students.ell](../examples/select_top_students.ell)
- [examples/students.json](../examples/students.json)

Run:

```bash
python -m ellang.cli examples/select_top_students.ell examples/students.json
```

This will:

1. parse the `.ell` source
2. compile it into typed IR
3. lower it into bytecode
4. execute it in the VM
5. print diagnostics, result, replay metadata, and visualization output

## 3. Start From Natural Language

Run:

```bash
python -m ellang.ideate "select the top 2 students by score" examples/students.json --show-ell
```

This will:

1. lower the idea into `ProgramSpec`
2. generate `.ell`
3. compile the generated program
4. execute it
5. print the result and the generated `.ell`

## 4. Inspect Local Model Discovery

```bash
python -m ellang.cli --models consumer
```

This shows:

- the selected hardware profile
- candidate model ids
- auto-discovered local model directories

## 5. Build the Native Runtime

If you want the Rust VM:

```bash
cd native/runtime-rs
cargo build --release
```

After the native runtime is built, ELLang will automatically prefer it when the executable is found in the expected output directory.

## 6. Run the Test Suite

```bash
python -m unittest discover -s tests -v
```

## 7. Good First Examples

- [examples/select_top_students.ell](../examples/select_top_students.ell)
- [examples/minStack.ell](../examples/minStack.ell)
- [examples/Substring.ell](../examples/Substring.ell)
- [examples/three_sum_input.json](../examples/three_sum_input.json)

## Next

For detailed syntax, architecture, runtime behavior, and model configuration, see:

- [User Manual](MANUAL.md)
- [Local Models Guide](LOCAL_MODELS.md)
