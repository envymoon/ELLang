from __future__ import annotations

from .bytecode import BytecodeProgram


def emit_backend_prototypes(program: BytecodeProgram) -> dict[str, str]:
    return {
        "wasm": _emit_wasm_module(program),
        "jvm": _emit_jvm_stub(program),
    }


def _emit_wasm_module(program: BytecodeProgram) -> str:
    lines = [
        f";; ELLang WASM prototype for {program.program_name}",
        "(module",
        '  (memory (export "memory") 1)',
        '  (func (export "run") (result i32)',
    ]
    for index, instruction in enumerate(program.instructions):
        lines.append(f"    ;; {index}: {instruction.opcode.value} -> {instruction.operand.get('node_id', 'node')}")
    lines.extend(
        [
            "    i32.const 0",
            "  )",
            ")",
        ]
    )
    return "\n".join(lines)


def _emit_jvm_stub(program: BytecodeProgram) -> str:
    class_name = "".join(part.capitalize() for part in program.program_name.replace("-", "_").split("_") if part) or "ELLangProgram"
    lines = [
        f"; ELLang JVM prototype for {program.program_name}",
        f".class public {class_name}",
        ".super java/lang/Object",
        "",
        ".method public static run()Ljava/lang/Object;",
        "  .limit stack 4",
        "  .limit locals 4",
    ]
    for index, instruction in enumerate(program.instructions):
        lines.append(f"  ; {index}: {instruction.opcode.value} -> {instruction.operand.get('node_id', 'node')}")
    lines.extend(
        [
            "  aconst_null",
            "  areturn",
            ".end method",
        ]
    )
    return "\n".join(lines)
