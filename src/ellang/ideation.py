from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .compiler import Compiler
from .algorithm_families import is_supported_algorithm_task
from .models import QwenLocalBackend
from .problem_spec import ProblemExample, ProblemSpec
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
    deserialize_program,
    parse_program,
    render_program,
)
from .visualization import mermaid_flowchart, mermaid_trace


@dataclass(slots=True)
class IdeationResult:
    problem_spec: ProblemSpec
    spec: ProgramSpec
    source: str
    diagnostics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ValidationAttempt:
    spec: ProgramSpec
    source: str
    diagnostics: list[str] = field(default_factory=list)


class IdeationEngine:
    def __init__(self, *, backend: QwenLocalBackend | None = None) -> None:
        self.backend = backend or QwenLocalBackend()

    def ideate(self, idea: str, bindings: dict[str, Any] | None = None) -> IdeationResult:
        bindings = bindings or {}
        problem_spec, draft, strategy = self._heuristic_spec(idea, bindings)
        diagnostics = [
            "Ideation used natural-language to ProblemSpec lowering.",
            f"Problem type: {problem_spec.problem_type}",
        ]
        first_attempt = _ValidationAttempt(spec=draft, source=render_program(draft))
        validated, validation_diags = self._validate_candidate(first_attempt, bindings)
        diagnostics.extend(validation_diags)
        if validated is not None and strategy != "generic_fallback":
            diagnostics.append("Generated .ell is an editable mid-level view; the execution result comes from compiled runtime execution.")
            return IdeationResult(problem_spec=problem_spec, spec=validated.spec, source=validated.source, diagnostics=diagnostics)

        if strategy == "generic_fallback":
            recovered_ok = False
            recovered = self._regenerate_from_planner_json(idea, bindings, problem_spec, diagnostics + validation_diags)
            if recovered is not None:
                recovered_problem, recovered_attempt, recovered_diags = recovered
                diagnostics.extend(recovered_diags)
                validated_recovered, recovered_validation = self._validate_candidate(recovered_attempt, bindings)
                diagnostics.extend(recovered_validation)
                if validated_recovered is not None:
                    recovered_ok = True
                    diagnostics.append("Generated .ell is an editable mid-level view; the execution result comes from compiled runtime execution.")
                    return IdeationResult(problem_spec=recovered_problem, spec=validated_recovered.spec, source=validated_recovered.source, diagnostics=diagnostics)

            if not recovered_ok:
                ell_regenerated = self._regenerate_from_ell(idea, bindings, diagnostics)
                if ell_regenerated is not None:
                    diagnostics.append("Structured planning could not fully validate, so constrained model regeneration produced a repaired .ell candidate.")
                    validated_ell, regenerated_diags = self._validate_candidate(ell_regenerated, bindings)
                    diagnostics.extend(regenerated_diags)
                    if validated_ell is not None:
                        diagnostics.append("Generated .ell is an editable mid-level view; the execution result comes from compiled runtime execution.")
                        return IdeationResult(problem_spec=problem_spec, spec=validated_ell.spec, source=validated_ell.source, diagnostics=diagnostics)

        diagnostics.append("Generated .ell is an editable mid-level view; the execution result comes from compiled runtime execution.")
        return IdeationResult(problem_spec=problem_spec, spec=draft, source=render_program(draft), diagnostics=diagnostics)

    def _heuristic_spec(self, idea: str, bindings: dict[str, Any]) -> tuple[ProblemSpec, ProgramSpec, str]:
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
            strategy = "matched_template"
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
            strategy = "matched_template"
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
            strategy = "matched_family"
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
        elif "rotate" in lower and "array" in lower and "right" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "array_manipulation", "algorithm_task": "rotate_array"}
            flow = [EmitStep("result")]
        elif "binary search" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "array_manipulation", "algorithm_task": "binary_search"}
            flow = [EmitStep("result")]
        elif "max subarray" in lower or "maximum subarray" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "array_manipulation", "algorithm_task": "max_subarray"}
            flow = [EmitStep("result")]
        elif "product except self" in lower or "product of array except self" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "array_manipulation", "algorithm_task": "product_except_self"}
            flow = [EmitStep("result")]
        elif "anagram" in lower and "group" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "hashmap_counting", "algorithm_task": "group_anagrams"}
            flow = [EmitStep("result")]
        elif ("frequency" in lower or "frequent" in lower or "count" in lower) and ("most frequent" in lower or "return the most frequent" in lower):
            strategy = "matched_family"
            project = {"algorithm_family": "hashmap_counting", "algorithm_task": "most_frequent_element"}
            flow = [EmitStep("result")]
        elif "parentheses" in lower and "valid" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "stack_queue_heap", "algorithm_task": "valid_parentheses"}
            flow = [EmitStep("result")]
        elif "top k frequent" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "stack_queue_heap", "algorithm_task": "top_k_frequent"}
            flow = [EmitStep("result")]
        elif "reverse linked list" in lower or "reverse a linked list" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "linked_list", "algorithm_task": "reverse_list"}
            flow = [EmitStep("result")]
        elif "level order" in lower and "tree" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "tree_graph", "algorithm_task": "binary_tree_level_order"}
            flow = [EmitStep("result")]
        elif "right side view" in lower and "tree" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "tree_graph", "algorithm_task": "binary_tree_right_side_view"}
            flow = [EmitStep("result")]
        elif "number of islands" in lower or "num islands" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "tree_graph", "algorithm_task": "num_islands"}
            flow = [EmitStep("result")]
        elif "coin change" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "dp_backtracking", "algorithm_task": "coin_change"}
            flow = [EmitStep("result")]
        elif "subsets" in lower:
            strategy = "matched_family"
            project = {"algorithm_family": "dp_backtracking", "algorithm_task": "subsets"}
            flow = [EmitStep("result")]
        elif "longest increasing subsequence" in lower or re.search(r"\blis\b", lower):
            strategy = "matched_family"
            project = {"algorithm_family": "dp_backtracking", "algorithm_task": "longest_increasing_subsequence"}
            flow = [EmitStep("result")]
        elif any(token in lower for token in ("sort", "top", "highest", "rank")):
            strategy = "matched_template"
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
            strategy = "generic_fallback"
            primary = next(iter(inputs.keys()), "input")
            modules = {
                "plan": ModuleSpec(
                    name="plan",
                    intent="Break the idea into deterministic operators when possible",
                    steps=[ActionStep(idea)],
                )
            }
            flow = [ModuleCallStep("plan"), EmitStep(primary if primary != "input" else "result")]

        problem_spec = ProblemSpec(
            name=program_name,
            summary=_normalize_sentence(idea),
            problem_type=_infer_problem_type(idea, project),
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
            correctness_conditions=_infer_correctness_conditions(idea, project),
            examples=[ProblemExample(description="user_bindings", bindings=bindings)] if bindings else [],
            algorithm_family_hint=project.get("algorithm_family", ""),
            algorithm_task_hint=project.get("algorithm_task", ""),
        )
        spec = ProgramSpec(
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
        return problem_spec, spec, strategy

    def _regenerate_from_planner_json(
        self,
        idea: str,
        bindings: dict[str, Any],
        fallback_problem: ProblemSpec,
        diagnostics: list[str],
    ) -> tuple[ProblemSpec, _ValidationAttempt, list[str]] | None:
        payload = self.backend.generate_problem_plan(idea, bindings, diagnostics)
        if not payload:
            return None
        try:
            problem_spec, program_spec = _problem_and_program_from_plan(payload, fallback_problem)
        except Exception as exc:
            diagnostics.append(f"Planner JSON could not be converted into ProgramSpec: {exc}")
            return None
        generated_source = render_program(program_spec)
        generated_diags = [
            "Heuristic ideation did not find a strong template, so local model planning produced a structured JSON plan.",
            f"Planner confidence: {payload.get('confidence', 0.0)}",
        ]
        return problem_spec, _ValidationAttempt(spec=program_spec, source=generated_source), generated_diags

    def _regenerate_from_ell(self, idea: str, bindings: dict[str, Any], diagnostics: list[str]) -> _ValidationAttempt | None:
        model_source = self.backend.generate_ell_program(idea, bindings, diagnostics)
        if not model_source:
            return None
        try:
            parsed = parse_program(model_source)
        except Exception:
            return None
        return _ValidationAttempt(spec=parsed, source=model_source)

    def _validate_candidate(self, attempt: _ValidationAttempt, bindings: dict[str, Any]) -> tuple[_ValidationAttempt | None, list[str]]:
        diagnostics: list[str] = []
        try:
            parsed = parse_program(attempt.source)
        except Exception as exc:
            diagnostics.append(f"Parse failed: {exc}")
            return None, diagnostics
        try:
            plan = Compiler().compile(parsed)
        except Exception as exc:
            diagnostics.append(f"Type check failed during compile: {exc}")
            return None, diagnostics
        diagnostics.append("Parse and type-check completed.")
        diagnostics.append(f"Sample tests synthesized: {len(plan.suggested_tests)}.")
        if bindings:
            missing_inputs = sorted(set(parsed.inputs.keys()) - set(bindings.keys()))
            if missing_inputs:
                diagnostics.append(f"Mismatch diagnosis: generated program expects missing inputs {missing_inputs}.")
                return None, diagnostics
            try:
                result = ExecutionEngine().execute(plan, bindings)
            except Exception as exc:
                diagnostics.append(f"Sample execution failed: {exc}")
                return None, diagnostics
            diagnostics.append("Sample execution completed.")
            diagnostics.extend(_diagnose_execution_mismatch(parsed, bindings, result.value))
            if parsed.constraints.get("deterministic", "").lower() == "true":
                second = ExecutionEngine().execute(plan, bindings)
                if second.value != result.value:
                    diagnostics.append("Determinism check failed: repeated sample execution produced a different result.")
                    return None, diagnostics
                diagnostics.append("Determinism check passed.")
            if any(item.startswith("Mismatch diagnosis:") for item in diagnostics):
                return None, diagnostics
        return _ValidationAttempt(spec=parsed, source=attempt.source), diagnostics


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
        "problem_spec": asdict(drafted.problem_spec),
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
        "backend_prototypes": plan.backend_prototypes,
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
    if any(token in lower for token in ("most frequent", "minimum number", "length of", "count of", "number of islands", "is valid", "right side view")):
        if "is valid" in lower:
            return "bool"
        if "right side view" in lower:
            return "list"
        return "int"
    if any(token in lower for token in ("rotate the array", "rotate array", "product except self")):
        return "list"
    if any(token in lower for token in ("binary search", "max subarray", "maximum subarray")):
        return "int"
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


def _infer_problem_type(idea: str, project: dict[str, str]) -> str:
    lower = idea.lower()
    if project.get("algorithm_family"):
        if any(token in project["algorithm_family"] for token in ("tree", "graph", "dp", "array", "hashmap", "linked")):
            return "algorithm"
    if "stack" in lower or "queue" in lower or "linked list" in lower:
        return "data_structure"
    if any(token in lower for token in ("transform", "group", "count", "sort", "select")):
        return "transformation"
    return "algorithm"


def _infer_correctness_conditions(idea: str, project: dict[str, str]) -> list[str]:
    conditions = ["Program output must satisfy the declared intent."]
    lower = idea.lower()
    if "duplicate" in lower:
        conditions.append("Output must not contain duplicates when the prompt forbids them.")
    if "any order" in lower:
        conditions.append("Output ordering may vary while preserving semantic correctness.")
    if project.get("algorithm_family"):
        conditions.append(f"Prefer deterministic lowering for {project['algorithm_family']}.")
    return conditions


def _problem_and_program_from_plan(payload: dict[str, Any], fallback_problem: ProblemSpec) -> tuple[ProblemSpec, ProgramSpec]:
    problem_payload = payload.get("problem_spec", {})
    if not isinstance(problem_payload, dict):
        raise ValueError("Planner JSON must contain a problem_spec object.")
    problem_spec = ProblemSpec(
        name=str(problem_payload.get("name", fallback_problem.name)),
        summary=str(problem_payload.get("summary", fallback_problem.summary)),
        problem_type=str(problem_payload.get("problem_type", fallback_problem.problem_type)),
        inputs=dict(problem_payload.get("inputs", fallback_problem.inputs)),
        outputs=dict(problem_payload.get("outputs", fallback_problem.outputs)),
        constraints={str(key): str(value) for key, value in dict(problem_payload.get("constraints", fallback_problem.constraints)).items()},
        correctness_conditions=[str(item) for item in problem_payload.get("correctness_conditions", fallback_problem.correctness_conditions)],
        examples=list(fallback_problem.examples),
        algorithm_family_hint=str(problem_payload.get("algorithm_family_hint", fallback_problem.algorithm_family_hint)),
        algorithm_task_hint=str(problem_payload.get("algorithm_task_hint", fallback_problem.algorithm_task_hint)),
    )
    program_payload = payload.get("program_spec", {})
    if not isinstance(program_payload, dict):
        raise ValueError("Planner JSON must contain a program_spec object.")
    modules: dict[str, dict[str, Any]] = {}
    for item in program_payload.get("modules", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        modules[name] = {
            "name": name,
            "intent": str(item.get("intent", "")),
            "params": [str(arg) for arg in item.get("params", [])],
            "steps": [step for step in item.get("steps", []) if isinstance(step, dict)],
        }
    project = {str(key): str(value) for key, value in dict(program_payload.get("project", {})).items()}
    if problem_spec.algorithm_family_hint and "algorithm_family" not in project:
        project["algorithm_family"] = problem_spec.algorithm_family_hint
    if problem_spec.algorithm_task_hint and "algorithm_task" not in project:
        project["algorithm_task"] = problem_spec.algorithm_task_hint
    if project.get("algorithm_family") and project.get("algorithm_task"):
        if not is_supported_algorithm_task(project["algorithm_family"], project["algorithm_task"]):
            project = {}
    assembled = deserialize_program(
        {
            "name": problem_spec.name,
            "intent": problem_spec.summary,
            "inputs": problem_spec.inputs,
            "outputs": problem_spec.outputs,
            "constraints": problem_spec.constraints,
            "objects": {},
            "modules": modules,
            "project": project,
            "flow": [step for step in program_payload.get("flow", []) if isinstance(step, dict)],
        }
    )
    return problem_spec, assembled


def _diagnose_execution_mismatch(spec: ProgramSpec, bindings: dict[str, Any], value: Any) -> list[str]:
    diagnostics: list[str] = []
    if not bindings:
        return diagnostics
    primary_name = next(iter(bindings.keys()))
    primary_value = bindings.get(primary_name)
    lower = spec.intent.lower()
    if value == primary_value and any(token in lower for token in ("count", "frequency", "most frequent", "group", "sort", "top", "sum", "minimum", "maximum", "triplet", "indices")):
        diagnostics.append("Mismatch diagnosis: execution echoed the primary input instead of producing a transformed result.")
    if isinstance(value, dict) and value.get("status") == "error":
        diagnostics.append(f"Mismatch diagnosis: runtime returned an intrinsic error: {value.get('message', 'unknown error')}.")
    if "most frequent" in lower and isinstance(primary_value, list) and value not in primary_value:
        diagnostics.append("Mismatch diagnosis: most-frequent result is not present in the input collection.")
    if ("triplet" in lower or "triplets" in lower) and isinstance(value, list) and not value:
        diagnostics.append("Mismatch diagnosis: triplet search returned an empty result on a non-trivial sample input.")
    return diagnostics


if __name__ == "__main__":
    raise SystemExit(main())
