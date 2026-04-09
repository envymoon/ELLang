from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .compiler import Compiler
from .models import describe_model_profile
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
        print("Usage: ellang <program.ell> [bindings.json]")
        print("       ellang --models [consumer|midrange|enhanced]")
        return 1

    program_path = Path(sys.argv[1])
    bindings = EXAMPLE_DATA
    if len(sys.argv) >= 3:
        bindings = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    spec = parse_program(program_path.read_text(encoding="utf-8"))
    plan = Compiler().compile(spec)
    result = ExecutionEngine().execute(plan, bindings)

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
        "flowchart": mermaid_flowchart(plan),
        "tracechart": mermaid_trace(result.trace) if result.trace else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
