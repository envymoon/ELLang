from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .compiler import Compiler
from .models import QwenLocalBackend
from .runtime import ExecutionEngine
from .syntax import (
    ActionStep,
    AppendStep,
    CallStep,
    ContinueStep,
    EmitStep,
    IfStep,
    InitializeStep,
    LoopStep,
    ModuleCallStep,
    ModuleSpec,
    ObjectSpec,
    ProgramSpec,
    RemoveLastStep,
    ReturnStep,
    AssignStep,
    render_program,
)
from .visualization import mermaid_flowchart, mermaid_trace


@dataclass(slots=True)
class IdeationResult:
    spec: ProgramSpec
    source: str
    diagnostics: list[str] = field(default_factory=list)


class IdeationEngine:
    def __init__(self, *, backend: QwenLocalBackend | None = None) -> None:
        self.backend = backend or QwenLocalBackend()

    def ideate(self, idea: str, bindings: dict[str, Any] | None = None) -> IdeationResult:
        bindings = bindings or {}
        draft = self._heuristic_spec(idea, bindings)
        diagnostics = ["Ideation used natural-language to ProgramSpec lowering."]
        diagnostics.append("Generated .ell is an editable mid-level view; the execution result comes from compiled runtime execution.")
        return IdeationResult(spec=draft, source=render_program(draft), diagnostics=diagnostics)

    def _heuristic_spec(self, idea: str, bindings: dict[str, Any]) -> ProgramSpec:
        program_name = _slug_to_program_name(idea)
        inputs = {key: _infer_binding_type(value) for key, value in bindings.items()}
        if not inputs:
            inputs = _infer_inputs_from_idea(idea)
        outputs = {"result": _infer_output_type(idea)}
        constraints = {"deterministic": "true"}

        lower = idea.lower()
        modules: dict[str, ModuleSpec] = {}
        objects: dict[str, ObjectSpec] = {}
        project: dict[str, str] = {}
        flow: list[Any]

        if "minstack" in lower or ("stack" in lower and "getmin" in lower):
            objects["MinStack"] = ObjectSpec(name="MinStack", fields={"stack": "list", "min_stack": "list"})
            modules = {
                "push": ModuleSpec(
                    name="push",
                    intent="Push a value and update the minimum stack when needed",
                    steps=[
                        AppendStep("value", "stack"),
                        IfStep(
                            condition="min_stack is empty OR value <= current minimum",
                            then_steps=[AppendStep("value", "min_stack")],
                        ),
                    ],
                    params=["value"],
                ),
                "pop": ModuleSpec(
                    name="pop",
                    intent="Pop the current top element and synchronize the minimum stack",
                    steps=[
                        IfStep(
                            condition="top of stack == top of min_stack",
                            then_steps=[RemoveLastStep("min_stack")],
                        ),
                        RemoveLastStep("stack"),
                    ],
                ),
                "top": ModuleSpec(name="top", intent="Read the top element from the stack", steps=[ReturnStep("last(stack)")]),
                "getMin": ModuleSpec(name="getMin", intent="Read the current minimum element", steps=[ReturnStep("last(min_stack)")]),
            }
            flow = [
                InitializeStep("MinStack"),
                LoopStep(
                    iterator="op",
                    source="operations",
                    body=[
                        IfStep(
                            condition='op == "MinStack"',
                            then_steps=[AppendStep("null", "results")],
                            else_steps=[
                                IfStep(
                                    condition='op == "push"',
                                    then_steps=[CallStep("push", ["values[next]"]), AppendStep("null", "results")],
                                    else_steps=[
                                        IfStep(
                                            condition='op == "pop"',
                                            then_steps=[CallStep("pop", []), AppendStep("null", "results")],
                                            else_steps=[
                                                IfStep(
                                                    condition='op == "top"',
                                                    then_steps=[AppendStep("call top()", "results")],
                                                    else_steps=[
                                                        IfStep(
                                                            condition='op == "getMin"',
                                                            then_steps=[AppendStep("call getMin()", "results")],
                                                        )
                                                    ],
                                                )
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                EmitStep("results"),
            ]
        elif "substring" in lower and "word" in lower:
            modules = {
                "prepare": ModuleSpec(
                    name="prepare",
                    intent="Compute the word length, total window length, and target frequency map",
                    steps=[
                        ActionStep("compute word length from words"),
                        ActionStep("compute total concatenation length"),
                        ActionStep("build target frequency map for words"),
                    ],
                ),
                "search": ModuleSpec(
                    name="search",
                    intent="Search every valid starting position in s",
                    steps=[
                        ActionStep("scan s with a sliding window"),
                        ActionStep("split each candidate window into equal-sized words"),
                        ActionStep("count candidate word frequencies"),
                        ActionStep("keep indices whose frequencies match the target map"),
                    ],
                ),
            }
            flow = [ModuleCallStep("prepare"), ModuleCallStep("search"), EmitStep("result")]
        elif (("triplet" in lower or "triplets" in lower) and "== 0" in lower) or "sum to zero" in lower or "3sum" in lower:
            project = {"algorithm_family": "array_two_pointers", "algorithm_task": "three_sum"}
            flow = [
                AssignStep("sorted_nums", "sorted(nums)"),
                AssignStep("results", "[]"),
                AssignStep("seen", "{}"),
                LoopStep(
                    iterator="i",
                    source="range(len(sorted_nums))",
                    body=[
                        LoopStep(
                            iterator="j",
                            source="range(i + 1, len(sorted_nums))",
                            body=[
                                LoopStep(
                                    iterator="k",
                                    source="range(j + 1, len(sorted_nums))",
                                    body=[
                                        IfStep(
                                            condition="sorted_nums[i] + sorted_nums[j] + sorted_nums[k] == 0",
                                            then_steps=[
                                                AssignStep("triplet", "[sorted_nums[i], sorted_nums[j], sorted_nums[k]]"),
                                                AssignStep("key", "str(triplet)"),
                                                IfStep(
                                                    condition="contains(seen, key) == false",
                                                    then_steps=[
                                                        AssignStep("seen[key]", "true"),
                                                        AppendStep("triplet", "results"),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                EmitStep("results"),
            ]
        elif "anagram" in lower and "group" in lower:
            project = {"algorithm_family": "hashmap_counting", "algorithm_task": "group_anagrams"}
            flow = [EmitStep("result")]
        elif "parentheses" in lower and "valid" in lower:
            project = {"algorithm_family": "stack_queue_heap", "algorithm_task": "valid_parentheses"}
            flow = [EmitStep("result")]
        elif "top k frequent" in lower:
            project = {"algorithm_family": "stack_queue_heap", "algorithm_task": "top_k_frequent"}
            flow = [EmitStep("result")]
        elif "reverse linked list" in lower or "reverse a linked list" in lower:
            project = {"algorithm_family": "linked_list", "algorithm_task": "reverse_list"}
            flow = [EmitStep("result")]
        elif "level order" in lower and "tree" in lower:
            project = {"algorithm_family": "tree_graph", "algorithm_task": "binary_tree_level_order"}
            flow = [EmitStep("result")]
        elif "number of islands" in lower or "num islands" in lower:
            project = {"algorithm_family": "tree_graph", "algorithm_task": "num_islands"}
            flow = [EmitStep("result")]
        elif "coin change" in lower:
            project = {"algorithm_family": "dp_backtracking", "algorithm_task": "coin_change"}
            flow = [EmitStep("result")]
        elif "subsets" in lower:
            project = {"algorithm_family": "dp_backtracking", "algorithm_task": "subsets"}
            flow = [EmitStep("result")]
        elif any(token in lower for token in ("sort", "top", "highest", "rank")):
            dataset_name = next(iter(inputs.keys()), "items")
            key_name = "score" if "score" in lower else "value"
            modules = {
                "plan": ModuleSpec(
                    name="plan",
                    intent="Prepare ranking and ordering strategy",
                    steps=[ActionStep(f"sort {dataset_name} by {key_name} descending")],
                )
            }
            flow = [ModuleCallStep("plan"), ActionStep(f"sort {dataset_name} by {key_name} descending and return the top 3"), EmitStep("result")]
        else:
            primary = next(iter(inputs.keys()), "input")
            modules = {
                "plan": ModuleSpec(
                    name="plan",
                    intent="Break the idea into deterministic operators when possible",
                    steps=[ActionStep(idea)],
                )
            }
            flow = [ModuleCallStep("plan"), EmitStep(primary if primary != "input" else "result")]

        return ProgramSpec(
            name=program_name,
            intent=_normalize_sentence(idea),
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
            objects=objects,
            modules=modules,
            project=project,
            flow=flow,
        )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m ellang.ideate \"<natural idea>\" [bindings.json] [--show-ell] [--write-ell path]")
        return 1

    args = sys.argv[1:]
    idea = args[0]
    bindings: dict[str, Any] = {}
    show_ell = False
    write_path: Path | None = None

    index = 1
    while index < len(args):
        item = args[index]
        if item == "--show-ell":
            show_ell = True
            index += 1
            continue
        if item == "--write-ell":
            if index + 1 >= len(args):
                raise SystemExit("--write-ell requires a target path.")
            write_path = Path(args[index + 1])
            index += 2
            continue
        bindings = json.loads(Path(item).read_text(encoding="utf-8"))
        index += 1

    ideation = IdeationEngine()
    drafted = ideation.ideate(idea, bindings)
    if write_path is not None:
        write_path.write_text(drafted.source, encoding="utf-8")

    plan = Compiler().compile(drafted.spec)
    result = ExecutionEngine().execute(plan, bindings or {})
    payload = {
        "idea": idea,
        "program": drafted.spec.name,
        "ell_source": drafted.source if show_ell or write_path is None else str(write_path),
        "ideation_diagnostics": drafted.diagnostics,
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


def _infer_binding_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "dataset" if value and isinstance(value[0], dict) else "list"
    if isinstance(value, dict):
        return "record"
    return "any"


def _infer_inputs_from_idea(idea: str) -> dict[str, str]:
    lower = idea.lower()
    guessed: dict[str, str] = {}
    if "string s" in lower or re.search(r"\bs\b", lower):
        guessed["s"] = "string"
    if "words" in lower:
        guessed["words"] = "dataset"
    if "operations" in lower:
        guessed["operations"] = "list"
    if "values" in lower:
        guessed["values"] = "list"
    if "students" in lower:
        guessed["students"] = "dataset"
    if not guessed:
        guessed["input"] = "record"
    return guessed


def _infer_output_type(idea: str) -> str:
    lower = idea.lower()
    if any(token in lower for token in ("indices", "results", "operations", "stack", "students", "list")):
        return "list"
    if "summary" in lower or "group" in lower:
        return "record"
    return "dataset"


def _slug_to_program_name(idea: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", idea)
    if not tokens:
        return "generated_program"
    return "_".join(token.lower() for token in tokens[:8])


def _normalize_sentence(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized[0].upper() + normalized[1:] if normalized else "Generated program"


if __name__ == "__main__":
    raise SystemExit(main())
