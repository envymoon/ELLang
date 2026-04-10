from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .compiler import Compiler
from .ideation import IdeationEngine
from .models import describe_model_profile
from .output import is_error_result, print_error, print_result
from .runtime import ExecutionEngine
from .syntax import parse_program
from .visualization import mermaid_flowchart, mermaid_trace


EXAMPLE_DATA = {
    "students": [
        {"name": "Ada", "score": 91},
        {"name": "Linus", "score": 88},
        {"name": "Grace", "score": 99},
        {"name": "Margaret", "score": 94},
    ],
    "top_only": True,
}


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--models":
        profile = sys.argv[2] if len(sys.argv) >= 3 else "consumer"
        print(json.dumps(describe_model_profile(profile), ensure_ascii=False, indent=2))
        return 0

    if len(sys.argv) < 2:
        print('Usage: ellang <program.ell> [bindings.json] [--extend]')
        print('       ellang "<natural idea>" [bindings.json] [--bind key=value ...] [--extend]')
        print("       ellang --models [consumer|midrange|enhanced]")
        return 1
    if _looks_like_program_path(sys.argv[1]):
        return _run_program_cli(sys.argv[1:])
    return _run_idea_cli(sys.argv[1:])


def _run_program_cli(args: list[str]) -> int:
    program_path = Path(args[0])
    bindings = EXAMPLE_DATA
    extend = False
    index = 1
    while index < len(args):
        item = args[index]
        if item == "--extend":
            extend = True
            index += 1
            continue
        bindings = json.loads(Path(item).read_text(encoding="utf-8"))
        index += 1
    spec = parse_program(program_path.read_text(encoding="utf-8"))
    plan = Compiler().compile(spec)
    result = ExecutionEngine().execute(plan, bindings)

    if is_error_result(result.value) and not extend:
        print_error(result.value)
        return 1
    if extend:
        payload = {
            "program": spec.name,
            "intent": spec.intent,
            "diagnostics": result.diagnostics,
            "result": result.value,
            "suggested_tests": plan.suggested_tests,
            "debug_report": asdict(result.debug_report) if result.debug_report else None,
            "typed_ir_nodes": len(plan.typed_program.nodes) if plan.typed_program else 0,
            "bytecode_instructions": len(plan.bytecode.instructions) if plan.bytecode else 0,
            "vm_backend": result.vm_backend,
            "replay": result.replay,
            "backend_prototypes": plan.backend_prototypes,
            "flowchart": mermaid_flowchart(plan),
            "tracechart": mermaid_trace(result.trace) if result.trace else "",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_result(result.value)
    return 0


def _run_idea_cli(args: list[str]) -> int:
    idea = args[0]
    bindings: dict[str, object] = {}
    extend = False
    index = 1
    while index < len(args):
        item = args[index]
        if item == "--extend":
            extend = True
            index += 1
            continue
        if item == "--bind":
            if index + 1 >= len(args):
                raise SystemExit("--bind requires key=value.")
            key, value = _parse_inline_binding(args[index + 1])
            bindings[key] = value
            index += 2
            continue
        bindings = json.loads(Path(item).read_text(encoding="utf-8"))
        index += 1
    drafted = IdeationEngine().ideate(idea, bindings)
    plan = Compiler().compile(drafted.spec)
    result = ExecutionEngine().execute(plan, bindings)
    if is_error_result(result.value) and not extend:
        print_error(result.value)
        return 1
    if extend:
        payload = {
            "idea": idea,
            "program": drafted.spec.name,
            "problem_spec": asdict(drafted.problem_spec),
            "ell_source": drafted.source,
            "ideation_diagnostics": drafted.diagnostics,
            "diagnostics": result.diagnostics,
            "result": result.value,
            "suggested_tests": plan.suggested_tests,
            "debug_report": asdict(result.debug_report) if result.debug_report else None,
            "typed_ir_nodes": len(plan.typed_program.nodes) if plan.typed_program else 0,
            "bytecode_instructions": len(plan.bytecode.instructions) if plan.bytecode else 0,
            "vm_backend": result.vm_backend,
            "replay": result.replay,
            "backend_prototypes": plan.backend_prototypes,
            "flowchart": mermaid_flowchart(plan),
            "tracechart": mermaid_trace(result.trace) if result.trace else "",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_result(result.value)
    return 0


def _looks_like_program_path(value: str) -> bool:
    path = Path(value)
    return path.suffix.lower() == ".ell" or path.exists()


def _parse_inline_binding(payload: str) -> tuple[str, object]:
    if "=" not in payload:
        raise SystemExit(f"Inline binding must look like key=value, got: {payload}")
    key, raw = payload.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        elif lowered == "null":
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
    return key, value


if __name__ == "__main__":
    raise SystemExit(main())
