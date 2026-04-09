from __future__ import annotations

from dataclasses import dataclass

from .ir import ExecutionPlan
from .syntax import ProgramSpec


@dataclass(slots=True)
class TestSynthesizer:
    def synthesize(self, spec: ProgramSpec, plan: ExecutionPlan) -> list[dict[str, object]]:
        tests: list[dict[str, object]] = [{"name": "smoke", "description": "Program should execute end-to-end with representative input."}]
        if "deterministic" in spec.constraints:
            tests.append({"name": "determinism", "description": "Same input should yield the same output and graph trace shape."})
        if "sort" in spec.intent.lower():
            tests.append({"name": "empty_input", "description": "Sorting an empty dataset should return an empty dataset."})
            tests.append({"name": "duplicate_keys", "description": "Sorting should behave consistently for duplicate sort keys."})
        if spec.flow:
            tests.append({"name": "control_flow", "description": "Conditional and loop paths should preserve valid output contracts."})
        return tests
