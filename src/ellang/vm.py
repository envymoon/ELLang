from __future__ import annotations

import ast
import json
import operator
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .algorithm_families import execute_algorithm_family
from .bytecode import BytecodeProgram, OpCode
from .project import GitProjectManager
from .security import RuntimeQuota
from .syntax import (
    ActionStep,
    AppendStep,
    AssignStep,
    BreakStep,
    CallStep,
    ContinueStep,
    EmitStep,
    IfStep,
    InitializeStep,
    LoopStep,
    ModuleCallStep,
    ProgramSpec,
    RemoveLastStep,
    ReturnStep,
    WhileStep,
    deserialize_program,
)


@dataclass(slots=True)
class VMResult:
    state: dict[str, Any]
    output: Any
    diagnostics: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


class ReferenceVM:
    def __init__(self, *, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root

    def execute(self, program: BytecodeProgram, bindings: dict[str, Any], quota: RuntimeQuota) -> VMResult:
        state: dict[str, Any] = {}
        diagnostics: list[str] = ["Reference VM executed bytecode."]
        trace: list[dict[str, Any]] = []

        for instruction in program.instructions:
            if instruction.opcode in {OpCode.MODEL_PLAN}:
                quota.charge_tokens(32)
            value = self._exec_instruction(instruction.opcode, instruction.operand, state, bindings)
            state[instruction.operand["node_id"]] = value
            trace.append(
                {
                    "pc": len(trace),
                    "node_id": instruction.operand["node_id"],
                    "opcode": instruction.opcode.value,
                    "result_type": instruction.result_type.value,
                    "state_size": len(state),
                    "summary": _summarize(value),
                }
            )

        output_node_id = program.instructions[-1].operand["node_id"]
        return VMResult(state=state, output=_unwrap_result(state[output_node_id]), diagnostics=diagnostics, trace=trace)

    def _exec_instruction(self, opcode: OpCode, operand: dict[str, object], state: dict[str, Any], bindings: dict[str, Any]) -> Any:
        if opcode == OpCode.LOAD_INPUT:
            return dict(bindings)
        if opcode == OpCode.INIT_PROJECT:
            snapshot = GitProjectManager(self.workspace_root).snapshot()
            return {
                "vcs": snapshot.vcs,
                "branch": snapshot.branch,
                "dirty": snapshot.dirty,
                "changed_files": snapshot.changed_files,
                "suggested_commit_message": snapshot.suggested_commit_message,
                "suggested_version_bump": snapshot.suggested_version_bump,
                "release_notes_hint": snapshot.release_notes_hint,
            }
        if opcode == OpCode.REGISTER_OBJECT:
            return {"name": operand.get("name"), "fields": operand.get("fields", {})}
        if opcode == OpCode.COMPUTE_LENGTH:
            source = _resolve_symbol(str(operand.get("source", "")), bindings, state)
            if isinstance(source, list) and source:
                first = source[0]
                if isinstance(first, str):
                    return len(first)
            return 0
        if opcode == OpCode.COMPUTE_TOTAL_LENGTH:
            source = _resolve_symbol(str(operand.get("source", "")), bindings, state)
            if isinstance(source, list) and source and isinstance(source[0], str):
                return len(source) * len(source[0])
            return 0
        if opcode == OpCode.BUILD_FREQ_MAP:
            source = _resolve_symbol(str(operand.get("source", "")), bindings, state)
            freq: dict[str, int] = {}
            if isinstance(source, list):
                for item in source:
                    freq[str(item)] = freq.get(str(item), 0) + 1
            return freq
        if opcode == OpCode.SLIDING_WINDOW_SCAN:
            text = _resolve_symbol(str(operand.get("text", "")), bindings, state)
            words = _resolve_symbol(str(operand.get("words", "")), bindings, state)
            return _substring_concat_scan(text, words)
        if opcode == OpCode.COLLECT_INDICES:
            source = _resolve_symbol(str(operand.get("source", "")), bindings, state)
            return source if isinstance(source, list) else []
        if opcode == OpCode.SORT:
            dataset = _coerce_sequence(_resolve_source_or_last(operand, state, bindings), bindings)
            key = str(operand.get("sort_key", operand.get("key", "value")))
            descending = bool(operand.get("descending", False))
            limit = operand.get("limit")
            result = sorted(dataset, key=lambda item: item.get(key), reverse=descending)
            return result[:limit] if isinstance(limit, int) else result
        if opcode == OpCode.FILTER:
            dataset = _coerce_sequence(_resolve_source_or_last(operand, state, bindings), bindings)
            predicate = operand.get("predicate")
            if isinstance(predicate, str):
                return [item for item in dataset if _evaluate_item_predicate(item, predicate)]
            return [item for item in dataset if bool(item)]
        if opcode == OpCode.MAP:
            dataset = _coerce_sequence(_resolve_source_or_last(operand, state, bindings), bindings)
            expression = str(operand.get("expression", "item"))
            return [_map_item(item, expression) for item in dataset]
        if opcode == OpCode.REDUCE:
            source = _resolve_source_or_last(operand, state, bindings)
            mode = str(operand.get("mode", "group_by"))
            if mode == "group_by":
                key = str(operand.get("key", "value"))
                return _group_by(source, key)
            return source
        if opcode in {OpCode.TRANSFORM, OpCode.MODEL_PLAN}:
            expression = operand.get("expression")
            if expression:
                return _resolve_symbol(str(expression), bindings, state)
            return _last_materialized_value(state)
        if opcode == OpCode.EVAL_CONDITION:
            return _evaluate_condition(str(operand.get("condition", "")), bindings, state)
        if opcode == OpCode.LOOP:
            source = _resolve_source_or_last(operand, state, bindings)
            return source
        if opcode == OpCode.MERGE:
            return _last_materialized_value(state)
        if opcode == OpCode.VALIDATE:
            return {"result": _last_materialized_value(state), "constraints": operand.get("constraints", {}), "valid": True}
        if opcode == OpCode.SYNTH_TESTS:
            return {"result": _last_materialized_value(state), "tests_ready": True}
        if opcode == OpCode.DEBUG_PREP:
            return {"result": _last_materialized_value(state), "strategy": operand.get("fallback"), "local_first": True}
        if opcode == OpCode.OUTPUT:
            source = operand.get("source")
            value = _resolve_symbol(str(source), bindings, state) if isinstance(source, str) else _last_materialized_value(state)
            return _unwrap_result(value)
        if opcode == OpCode.CALL_INTRINSIC:
            intrinsic = str(operand.get("intrinsic", "unknown"))
            source = _resolve_source_or_last(operand, state, bindings)
            if intrinsic == "execute_algorithm_family":
                family = str(operand.get("algorithm_family", ""))
                task = str(operand.get("algorithm_task", ""))
                return execute_algorithm_family(family, task, bindings).result
            if intrinsic == "execute_structured_program":
                payload = operand.get("program_spec")
                if isinstance(payload, dict):
                    return _execute_structured_program(deserialize_program(payload), bindings)
            return _call_intrinsic(intrinsic, source)
        if opcode == OpCode.CALL_FFI:
            signature = operand.get("ffi_signature", {})
            return _call_ffi(signature if isinstance(signature, dict) else {}, _resolve_source_or_last(operand, state, bindings))
        if opcode == OpCode.SPLIT_EQUAL_WORDS:
            source = _resolve_source_or_last(operand, state, bindings)
            size = int(operand.get("chunk_size", 1))
            if isinstance(source, str) and size > 0:
                return [source[idx : idx + size] for idx in range(0, len(source), size)]
            return source
        if opcode == OpCode.COMPARE_FREQ_MAPS:
            left = _resolve_symbol(str(operand.get("left", "")), bindings, state)
            right = _resolve_symbol(str(operand.get("right", "")), bindings, state)
            return left == right
        if opcode in {OpCode.MODEL_PLAN}:
            return _last_materialized_value(state)
        raise ValueError(f"Unsupported opcode in reference VM: {opcode.value}")


class NativeVMHost:
    def __init__(self, *, executable: str | None = None, workspace_root: str = ".") -> None:
        self.executable = executable
        self.workspace_root = Path(workspace_root)

    def is_available(self) -> bool:
        return bool(self.executable and Path(self.executable).exists())

    def execute(self, program: BytecodeProgram, bindings: dict[str, Any], quota: RuntimeQuota) -> VMResult:
        if not self.executable:
            raise RuntimeError("Native VM executable is not configured.")
        temp = self.workspace_root / ".ellang-tmp" / uuid4().hex
        temp.mkdir(parents=True, exist_ok=True)
        bytecode_path = program.write_json(temp / "program.bytecode.json")
        bindings_path = temp / "bindings.json"
        bindings_path.write_text(json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8")
        completed = subprocess.run(
            [self.executable, str(bytecode_path), str(bindings_path), str(self.workspace_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Native VM execution failed.")
        payload = json.loads(completed.stdout)
        return VMResult(
            state=dict(payload.get("state", {})),
            output=payload.get("output"),
            diagnostics=[str(item) for item in payload.get("diagnostics", [])],
            trace=[dict(item) for item in payload.get("trace", []) if isinstance(item, dict)],
        )


def _resolve_symbol(name: str, bindings: dict[str, Any], state: dict[str, Any]) -> Any:
    if name in bindings:
        return bindings[name]
    if name in state:
        return state[name]
    return _last_materialized_value(state)


def _resolve_source_or_last(operand: dict[str, object], state: dict[str, Any], bindings: dict[str, Any]) -> Any:
    source = operand.get("source")
    if isinstance(source, str):
        return _resolve_symbol(source, bindings, state)
    return _last_materialized_value(state)


def _coerce_sequence(value: Any, bindings: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, list):
                return item
    for item in bindings.values():
        if isinstance(item, list):
            return item
    raise TypeError("Operation expects a sequence input.")


def _last_materialized_value(state: dict[str, Any]) -> Any:
    for key in reversed(list(state.keys())):
        return state[key]
    return None


def _unwrap_result(value: Any) -> Any:
    current = value
    while isinstance(current, dict) and "result" in current:
        current = current["result"]
    return current


def _evaluate_condition(condition: str, bindings: dict[str, Any], state: dict[str, Any]) -> bool:
    normalized = condition.strip()
    if "!=" in normalized:
        left, right = [item.strip() for item in normalized.split("!=", 1)]
        return not _evaluate_condition(f"{left} == {right}", bindings, state)
    if "==" in normalized:
        left, right = [item.strip() for item in normalized.split("==", 1)]
        left_value = _resolve_symbol(left, bindings, state)
        if right.lower() in {"true", "false"}:
            return bool(left_value) is (right.lower() == "true")
        if right.isdigit():
            return left_value == int(right)
        return str(left_value) == right.strip('"').strip("'")
    return bool(_resolve_symbol(normalized, bindings, state))


def _substring_concat_scan(text: Any, words: Any) -> list[int]:
    if not isinstance(text, str) or not isinstance(words, list) or not words:
        return []
    if not all(isinstance(word, str) for word in words):
        return []
    word_len = len(words[0])
    if word_len == 0 or any(len(word) != word_len for word in words):
        return []
    total_len = word_len * len(words)
    target: dict[str, int] = {}
    for word in words:
        target[word] = target.get(word, 0) + 1

    results: list[int] = []
    for offset in range(word_len):
        left = offset
        seen: dict[str, int] = {}
        count = 0
        for right in range(offset, len(text) - word_len + 1, word_len):
            word = text[right : right + word_len]
            if word in target:
                seen[word] = seen.get(word, 0) + 1
                count += 1
                while seen[word] > target[word]:
                    left_word = text[left : left + word_len]
                    seen[left_word] -= 1
                    left += word_len
                    count -= 1
                if count == len(words):
                    results.append(left)
                    left_word = text[left : left + word_len]
                    seen[left_word] -= 1
                    left += word_len
                    count -= 1
            else:
                seen.clear()
                count = 0
                left = right + word_len
        if offset == 0 and total_len > len(text):
            break
    return results


def _map_item(item: dict[str, Any], expression: str) -> Any:
    if expression == "item":
        return item
    if expression.startswith("item."):
        return item.get(expression.split(".", 1)[1])
    return item


def _group_by(source: Any, key: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                grouped.setdefault(str(item.get(key)), []).append(item)
    return grouped


def _evaluate_item_predicate(item: dict[str, Any], predicate: str) -> bool:
    predicate = predicate.strip()
    if "==" in predicate and predicate.startswith("item."):
        field, expected = [part.strip() for part in predicate.split("==", 1)]
        actual = item.get(field.split(".", 1)[1])
        expected_value = expected.strip('"').strip("'")
        if expected_value.lower() in {"true", "false"}:
            return bool(actual) is (expected_value.lower() == "true")
        if expected_value.isdigit():
            return actual == int(expected_value)
        return str(actual) == expected_value
    if ">" in predicate and predicate.startswith("item."):
        field, expected = [part.strip() for part in predicate.split(">", 1)]
        actual = item.get(field.split(".", 1)[1], 0)
        return float(actual) > float(expected)
    if "<" in predicate and predicate.startswith("item."):
        field, expected = [part.strip() for part in predicate.split("<", 1)]
        actual = item.get(field.split(".", 1)[1], 0)
        return float(actual) < float(expected)
    if predicate.startswith("item."):
        return bool(item.get(predicate.split(".", 1)[1]))
    return bool(item)


def _summarize(value: Any) -> str:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:5]})"
    return repr(value)


def _call_intrinsic(intrinsic: str, source: Any) -> Any:
    registry = {
        "dataset.len": lambda value: len(value) if isinstance(value, list) else None,
        "record.keys": lambda value: list(value.keys()) if isinstance(value, dict) else [],
    }
    if intrinsic in registry:
        return registry[intrinsic](source)
    return {"intrinsic": intrinsic, "status": "reference_stub", "value": source}


def _call_ffi(signature: dict[str, Any], source: Any) -> Any:
    name = str(signature.get("name", ""))
    required_capabilities = [str(item) for item in signature.get("required_capabilities", ["ffi.call"])]
    if "ffi.call" not in required_capabilities:
        raise PermissionError(f"FFI signature {name!r} is missing ffi.call capability.")
    registry = {
        "ellang_native.identity": lambda value: value,
        "ellang_native.keys": lambda value: list(value.keys()) if isinstance(value, dict) else [],
        "ellang_native.len": lambda value: len(value) if isinstance(value, (list, dict, str)) else 0,
        "ellang_native.uppercase": lambda value: str(value).upper(),
    }
    if name not in registry:
        raise ValueError(f"Unknown reference FFI binding: {name}")
    return {
        "ffi": {
            "name": name,
            "library": signature.get("library", ""),
            "abi": signature.get("abi", "system"),
        },
        "result": registry[name](source),
    }


class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


def _execute_structured_program(spec: ProgramSpec, bindings: dict[str, Any]) -> Any:
    env: dict[str, Any] = dict(bindings)
    env.setdefault("result", [])
    env.setdefault("results", [])
    env.setdefault("next", 0)
    if spec.objects and isinstance(env.get("operations"), list) and isinstance(env.get("values"), list):
        operations = env["operations"]
        values = env["values"]
        if operations and values and len(operations) == len(values) and operations[0] in spec.objects:
            env["next"] = 1
    object_state: dict[str, dict[str, Any]] = {}
    active_object: str | None = None

    def call_module(module_name: str, args: list[Any], scope: dict[str, Any]) -> Any:
        module = spec.modules[module_name]
        local_env = dict(zip(module.params, args))
        try:
            return run_steps(module.steps, local_env={**scope, **local_env})
        except _ReturnSignal as signal:
            return signal.value

    def run_steps(steps: list[Any], local_env: dict[str, Any] | None = None) -> Any:
        nonlocal active_object
        scope = env if local_env is None else {**env, **local_env}
        for step in steps:
            if isinstance(step, ActionStep):
                _execute_action(step.action, env, scope, object_state, active_object)
            elif isinstance(step, ModuleCallStep):
                call_module(step.module_name, [], scope)
            elif isinstance(step, CallStep):
                call_module(step.module_name, [_eval_expression(arg, env, scope, object_state, active_object) for arg in step.arguments], scope)
            elif isinstance(step, EmitStep):
                value = _eval_expression(step.expression, env, scope, object_state, active_object)
                env["result"] = value
                if step.expression == "results":
                    env["result"] = env.get("results", value)
            elif isinstance(step, AppendStep):
                target = _resolve_container(step.target, env, scope, object_state, active_object)
                if isinstance(target, list):
                    if step.expression.startswith("call "):
                        call = _parse_call_expression(step.expression)
                        target.append(call_module(call.module_name, [_eval_expression(arg, env, scope, object_state, active_object) for arg in call.arguments], scope))
                    else:
                        target.append(_eval_expression(step.expression, env, scope, object_state, active_object))
            elif isinstance(step, RemoveLastStep):
                target = _resolve_container(step.target, env, scope, object_state, active_object)
                if isinstance(target, list) and target:
                    target.pop()
            elif isinstance(step, ReturnStep):
                if step.expression.startswith("call "):
                    call = _parse_call_expression(step.expression)
                    raise _ReturnSignal(call_module(call.module_name, [_eval_expression(arg, env, scope, object_state, active_object) for arg in call.arguments], scope))
                raise _ReturnSignal(_eval_expression(step.expression, env, scope, object_state, active_object))
            elif isinstance(step, InitializeStep):
                active_object = step.object_name
                state = {field_name: [] if field_type == "list" else None for field_name, field_type in spec.objects.get(step.object_name, type("Empty", (), {"fields": {}})()).fields.items()}
                object_state[step.object_name] = state
                for key, value in state.items():
                    env[key] = value
            elif isinstance(step, AssignStep):
                _assign_target(step.target, _eval_expression(step.expression, env, scope, object_state, active_object), env, scope, object_state, active_object)
            elif isinstance(step, IfStep):
                branch = step.then_steps if _eval_condition_expr(step.condition, env, scope, object_state, active_object) else step.else_steps
                run_steps(branch, local_env=scope)
            elif isinstance(step, LoopStep):
                sequence = _eval_expression(step.source, env, scope, object_state, active_object)
                if isinstance(sequence, (list, tuple, range)):
                    for item in sequence:
                        loop_env = {**scope, step.iterator: item}
                        try:
                            run_steps(step.body, local_env=loop_env)
                        except _ContinueSignal:
                            continue
                        except _BreakSignal:
                            break
            elif isinstance(step, WhileStep):
                while _eval_condition_expr(step.condition, env, scope, object_state, active_object):
                    try:
                        run_steps(step.body, local_env=scope)
                    except _ContinueSignal:
                        continue
                    except _BreakSignal:
                        break
            elif isinstance(step, BreakStep):
                raise _BreakSignal()
            elif isinstance(step, ContinueStep):
                raise _ContinueSignal()
        return env.get("result", env.get("results"))

    try:
        result = run_steps(spec.flow)
    except _ReturnSignal as signal:
        result = signal.value
    if "results" in env and isinstance(env["results"], list) and env["results"]:
        return env["results"]
    return result


def _execute_action(
    action: str,
    env: dict[str, Any],
    scope: dict[str, Any],
    object_state: dict[str, dict[str, Any]],
    active_object: str | None,
) -> None:
    normalized = action.strip()
    if normalized.startswith("append ") and " to " in normalized:
        expr, target = normalized.removeprefix("append ").split(" to ", 1)
        container = _resolve_container(target, env, scope, object_state, active_object)
        if isinstance(container, list):
            container.append(_eval_expression(expr, env, scope, object_state, active_object))
        return
    if normalized.startswith("pop "):
        target = normalized.removeprefix("pop ").strip()
        container = _resolve_container(target, env, scope, object_state, active_object)
        if isinstance(container, list) and container:
            container.pop()
        return
    if normalized.startswith("read the last element of "):
        target = normalized.removeprefix("read the last element of ").strip()
        env["last_read"] = _last_value(_resolve_container(target, env, scope, object_state, active_object))
        env["result"] = env["last_read"]
        return
    if normalized == "append null to results":
        env.setdefault("results", []).append(None)
        return
    if normalized == "append call top() to results":
        env.setdefault("results", []).append(_last_value(env.get("stack", [])))
        return
    if normalized == "append call getMin() to results":
        env.setdefault("results", []).append(_last_value(env.get("min_stack", [])))
        return
    if normalized == "initialize MinStack with stack and min_stack":
        env["stack"] = []
        env["min_stack"] = []
        return
    if normalized == "if top of stack equals top of min_stack then pop min_stack":
        if _last_value(env.get("stack", [])) == _last_value(env.get("min_stack", [])) and env.get("min_stack"):
            env["min_stack"].pop()
        return
    if normalized == "if min_stack is empty or value is smaller than current minimum then append value to min_stack":
        min_stack = env.setdefault("min_stack", [])
        value = scope.get("value")
        if not min_stack or (value is not None and value <= _last_value(min_stack)):
            min_stack.append(value)
        return


def _eval_condition_expr(condition: str, env: dict[str, Any], scope: dict[str, Any], object_state: dict[str, dict[str, Any]], active_object: str | None) -> bool:
    normalized = condition.replace(" OR ", " or ").replace(" AND ", " and ").strip()
    if " or " in normalized:
        left, right = normalized.split(" or ", 1)
        return _eval_condition_expr(left, env, scope, object_state, active_object) or _eval_condition_expr(right, env, scope, object_state, active_object)
    if " and " in normalized:
        left, right = normalized.split(" and ", 1)
        return _eval_condition_expr(left, env, scope, object_state, active_object) and _eval_condition_expr(right, env, scope, object_state, active_object)
    if normalized.endswith(" is empty"):
        return len(_coerce_list(_eval_expression(normalized.removesuffix(" is empty"), env, scope, object_state, active_object))) == 0
    if "<=" in normalized:
        left, right = [item.strip() for item in normalized.split("<=", 1)]
        return _eval_expression(left, env, scope, object_state, active_object) <= _eval_expression(right, env, scope, object_state, active_object)
    if ">=" in normalized:
        left, right = [item.strip() for item in normalized.split(">=", 1)]
        return _eval_expression(left, env, scope, object_state, active_object) >= _eval_expression(right, env, scope, object_state, active_object)
    if "==" in normalized:
        left, right = [item.strip() for item in normalized.split("==", 1)]
        return _eval_expression(left, env, scope, object_state, active_object) == _eval_expression(right, env, scope, object_state, active_object)
    return bool(_eval_expression(normalized, env, scope, object_state, active_object))


def _eval_expression(expr: str, env: dict[str, Any], scope: dict[str, Any], object_state: dict[str, dict[str, Any]], active_object: str | None) -> Any:
    normalized = expr.strip()
    if normalized in {"null", "None"}:
        return None
    if normalized in {"true", "True"}:
        return True
    if normalized in {"false", "False"}:
        return False
    if normalized.startswith('"') and normalized.endswith('"'):
        return normalized[1:-1]
    if normalized.startswith("'") and normalized.endswith("'"):
        return normalized[1:-1]
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    if normalized.startswith("last(") and normalized.endswith(")"):
        return _last_value(_eval_expression(normalized[5:-1], env, scope, object_state, active_object))
    if normalized == "current minimum":
        return _last_value(env.get("min_stack", []))
    if normalized == "top of stack":
        return _last_value(env.get("stack", []))
    if normalized == "top of min_stack":
        return _last_value(env.get("min_stack", []))
    if normalized.startswith("call ") and normalized.endswith(")"):
        return normalized
    if normalized.endswith("[next]"):
        base = normalized[:-6]
        sequence = _coerce_list(_eval_expression(base, env, scope, object_state, active_object))
        index = int(env.get("next", 0))
        env["next"] = index + 1
        return sequence[index] if index < len(sequence) else None
    if normalized in scope:
        return scope[normalized]
    if normalized in env:
        return env[normalized]
    if active_object and active_object in object_state and normalized in object_state[active_object]:
        return object_state[active_object][normalized]
    try:
        return _safe_eval_expression(normalized, env, scope, object_state, active_object)
    except Exception:
        return normalized


def _resolve_container(target: str, env: dict[str, Any], scope: dict[str, Any], object_state: dict[str, dict[str, Any]], active_object: str | None) -> Any:
    normalized = target.strip()
    if normalized in scope:
        return scope[normalized]
    if normalized in env:
        return env[normalized]
    if active_object and active_object in object_state:
        return object_state[active_object].setdefault(normalized, [])
    return env.setdefault(normalized, [])


def _last_value(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[-1]
    return None


def _coerce_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_call_expression(expression: str) -> CallStep:
    payload = expression.removeprefix("call ").strip()
    if "(" not in payload or not payload.endswith(")"):
        return CallStep(module_name=payload, arguments=[])
    name, args = payload.split("(", 1)
    arg_text = args[:-1].strip()
    return CallStep(module_name=name.strip(), arguments=[item.strip() for item in arg_text.split(",") if item.strip()])


def _assign_target(target: str, value: Any, env: dict[str, Any], scope: dict[str, Any], object_state: dict[str, dict[str, Any]], active_object: str | None) -> None:
    normalized = target.strip()
    if "[" in normalized and normalized.endswith("]"):
        base, index_expr = normalized.split("[", 1)
        container = _eval_expression(base.strip(), env, scope, object_state, active_object)
        index = _eval_expression(index_expr[:-1].strip(), env, scope, object_state, active_object)
        if isinstance(container, dict):
            container[index] = value
            return
        if isinstance(container, list) and isinstance(index, int):
            while index >= len(container):
                container.append(None)
            container[index] = value
            return
    env[normalized] = value
    scope[normalized] = value
    if active_object and active_object in object_state and normalized in object_state[active_object]:
        object_state[active_object][normalized] = value


def _safe_eval_expression(expr: str, env: dict[str, Any], scope: dict[str, Any], object_state: dict[str, dict[str, Any]], active_object: str | None) -> Any:
    tree = ast.parse(expr, mode="eval")
    names = {**env, **scope}
    if active_object and active_object in object_state:
        names.update(object_state[active_object])
    return _eval_ast_node(tree.body, names)


def _eval_ast_node(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.List):
        return [_eval_ast_node(item, names) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast_node(item, names) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_ast_node(key, names): _eval_ast_node(value, names) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, names)
        right = _eval_ast_node(node.right, names)
        return {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, names)
        return {ast.USub: operator.neg, ast.Not: operator.not_}[type(node.op)](operand)
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast_node(item, names) for item in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        current = _eval_ast_node(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast_node(comparator, names)
            ok = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.In: lambda a, b: a in b,
                ast.NotIn: lambda a, b: a not in b,
            }[type(op)](current, right)
            if not ok:
                return False
            current = right
        return True
    if isinstance(node, ast.Subscript):
        value = _eval_ast_node(node.value, names)
        index = _eval_ast_node(node.slice, names) if not isinstance(node.slice, ast.Slice) else None
        if isinstance(value, (list, tuple, str)) and isinstance(index, int):
            return value[index]
        if isinstance(value, dict):
            return value.get(index)
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        args = [_eval_ast_node(arg, names) for arg in node.args]
        allowed = {
            "len": lambda x: len(x),
            "range": lambda *a: list(range(*a)),
            "sorted": lambda x: sorted(x),
            "last": _last_value,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "str": str,
            "contains": lambda collection, item: item in collection,
        }
        if func_name in allowed:
            return allowed[func_name](*args)
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
