from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProblemExample:
    description: str = ""
    bindings: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None


@dataclass(slots=True)
class ProblemSpec:
    name: str
    summary: str
    problem_type: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)
    correctness_conditions: list[str] = field(default_factory=list)
    examples: list[ProblemExample] = field(default_factory=list)
    algorithm_family_hint: str = ""
    algorithm_task_hint: str = ""
    notes: list[str] = field(default_factory=list)

    def primary_input(self) -> str:
        return next(iter(self.inputs.keys()), "input")
