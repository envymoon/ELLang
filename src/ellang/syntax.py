from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent


@dataclass(slots=True)
class ObjectSpec:
    name: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ActionStep:
    action: str


@dataclass(slots=True)
class ModuleCallStep:
    module_name: str


@dataclass(slots=True)
class CallStep:
    module_name: str
    arguments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmitStep:
    expression: str


@dataclass(slots=True)
class IfStep:
    condition: str
    then_steps: list["Statement"] = field(default_factory=list)
    else_steps: list["Statement"] = field(default_factory=list)


@dataclass(slots=True)
class LoopStep:
    iterator: str
    source: str
    body: list["Statement"] = field(default_factory=list)


@dataclass(slots=True)
class WhileStep:
    condition: str
    body: list["Statement"] = field(default_factory=list)


@dataclass(slots=True)
class AppendStep:
    expression: str
    target: str


@dataclass(slots=True)
class RemoveLastStep:
    target: str


@dataclass(slots=True)
class ReturnStep:
    expression: str


@dataclass(slots=True)
class InitializeStep:
    object_name: str


@dataclass(slots=True)
class AssignStep:
    target: str
    expression: str


@dataclass(slots=True)
class BreakStep:
    pass


@dataclass(slots=True)
class ContinueStep:
    pass


Statement = (
    ActionStep
    | ModuleCallStep
    | CallStep
    | EmitStep
    | IfStep
    | LoopStep
    | WhileStep
    | AppendStep
    | RemoveLastStep
    | ReturnStep
    | InitializeStep
    | AssignStep
    | BreakStep
    | ContinueStep
)


@dataclass(slots=True)
class ModuleSpec:
    name: str
    intent: str = ""
    params: list[str] = field(default_factory=list)
    steps: list[Statement] = field(default_factory=list)


@dataclass(slots=True)
class ProgramSpec:
    name: str
    intent: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)
    objects: dict[str, ObjectSpec] = field(default_factory=dict)
    modules: dict[str, ModuleSpec] = field(default_factory=dict)
    project: dict[str, str] = field(default_factory=dict)
    flow: list[Statement] = field(default_factory=list)


def parse_program(source: str) -> ProgramSpec:
    lines = _tokenize(dedent(source))
    name = "anonymous"
    intent = ""
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    constraints: dict[str, str] = {}
    objects: dict[str, ObjectSpec] = {}
    modules: dict[str, ModuleSpec] = {}
    project: dict[str, str] = {}
    flow: list[Statement] = []

    index = 0
    while index < len(lines):
        indent, line = lines[index]
        if indent != 0:
            raise ValueError(f"Unexpected indentation at top level: {line}")
        if line.startswith("program "):
            name = _strip_quoted(line.removeprefix("program ").strip())
            index += 1
            continue
        if line.startswith("intent "):
            intent = _strip_quoted(line.removeprefix("intent ").strip())
            index += 1
            continue
        if line.startswith("input "):
            key, value = _parse_decl(line.removeprefix("input ").strip())
            inputs[key] = value
            index += 1
            continue
        if line.startswith("output "):
            key, value = _parse_decl(line.removeprefix("output ").strip())
            outputs[key] = value
            index += 1
            continue
        if line.startswith("constraint "):
            key, value = _parse_decl(line.removeprefix("constraint ").strip(), separator="=")
            constraints[key] = value
            index += 1
            continue
        if line.startswith("object ") and line.endswith(":"):
            object_name = line.removeprefix("object ").removesuffix(":").strip()
            index += 1
            fields, index = _parse_object_body(lines, index, parent_indent=0)
            objects[object_name] = ObjectSpec(name=object_name, fields=fields)
            continue
        if line.startswith("module ") and line.endswith(":"):
            module_name, params = _parse_module_header(line)
            index += 1
            module_intent = ""
            if index < len(lines) and lines[index][0] == 2 and lines[index][1].startswith("intent "):
                module_intent = _strip_quoted(lines[index][1].removeprefix("intent ").strip())
                index += 1
            steps, index = _parse_steps(lines, index, parent_indent=0)
            modules[module_name] = ModuleSpec(name=module_name, intent=module_intent, params=params, steps=steps)
            continue
        if line == "project:":
            index += 1
            project, index = _parse_key_value_block(lines, index, parent_indent=0)
            continue
        if line == "flow:":
            index += 1
            flow, index = _parse_steps(lines, index, parent_indent=0)
            continue
        raise ValueError(f"Unsupported syntax line: {line}")

    if not intent:
        raise ValueError("Program intent is required.")

    return ProgramSpec(
        name=name,
        intent=intent,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        objects=objects,
        modules=modules,
        project=project,
        flow=flow,
    )


def render_program(spec: ProgramSpec) -> str:
    lines: list[str] = [f'program "{spec.name}"', f'intent "{spec.intent}"', ""]
    for name, value in spec.inputs.items():
        lines.append(f"input {name}: {value}")
    for name, value in spec.outputs.items():
        lines.append(f"output {name}: {value}")
    if spec.inputs or spec.outputs:
        lines.append("")
    for name, value in spec.constraints.items():
        lines.append(f"constraint {name} = {value}")
    if spec.constraints:
        lines.append("")
    for object_name, object_spec in spec.objects.items():
        lines.append(f"object {object_name}:")
        for field_name, field_type in object_spec.fields.items():
            lines.append(f"  field {field_name}: {field_type}")
        lines.append("")
    if spec.project:
        lines.append("project:")
        for key, value in spec.project.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    for module_name, module_spec in spec.modules.items():
        header = f"module {module_name}"
        if module_spec.params:
            header += f"({', '.join(module_spec.params)})"
        lines.append(f"{header}:")
        if module_spec.intent:
            lines.append(f'  intent "{module_spec.intent}"')
        lines.extend(_render_steps(module_spec.steps, indent=2))
        lines.append("")
    lines.append("flow:")
    lines.extend(_render_steps(spec.flow, indent=2))
    return "\n".join(lines).rstrip() + "\n"


def serialize_program(spec: ProgramSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "intent": spec.intent,
        "inputs": dict(spec.inputs),
        "outputs": dict(spec.outputs),
        "constraints": dict(spec.constraints),
        "objects": {name: {"name": obj.name, "fields": dict(obj.fields)} for name, obj in spec.objects.items()},
        "modules": {
            name: {"name": module.name, "intent": module.intent, "params": list(module.params), "steps": [_serialize_step(step) for step in module.steps]}
            for name, module in spec.modules.items()
        },
        "project": dict(spec.project),
        "flow": [_serialize_step(step) for step in spec.flow],
    }


def deserialize_program(payload: dict[str, object]) -> ProgramSpec:
    objects = {
        name: ObjectSpec(name=str(item.get("name", name)), fields=dict(item.get("fields", {})))
        for name, item in dict(payload.get("objects", {})).items()
        if isinstance(item, dict)
    }
    modules = {
        name: ModuleSpec(
            name=str(item.get("name", name)),
            intent=str(item.get("intent", "")),
            params=[str(arg) for arg in item.get("params", [])],
            steps=[_deserialize_step(step) for step in item.get("steps", []) if isinstance(step, dict)],
        )
        for name, item in dict(payload.get("modules", {})).items()
        if isinstance(item, dict)
    }
    return ProgramSpec(
        name=str(payload.get("name", "anonymous")),
        intent=str(payload.get("intent", "")),
        inputs=dict(payload.get("inputs", {})),
        outputs=dict(payload.get("outputs", {})),
        constraints=dict(payload.get("constraints", {})),
        objects=objects,
        modules=modules,
        project=dict(payload.get("project", {})),
        flow=[_deserialize_step(step) for step in payload.get("flow", []) if isinstance(step, dict)],
    )


def _tokenize(source: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Indentation must use multiples of 2 spaces: {raw_line}")
        tokens.append((indent, stripped))
    return tokens


def _parse_object_body(lines: list[tuple[int, str]], index: int, *, parent_indent: int) -> tuple[dict[str, str], int]:
    fields: dict[str, str] = {}
    while index < len(lines):
        indent, line = lines[index]
        if indent <= parent_indent:
            break
        if indent != parent_indent + 2:
            raise ValueError(f"Invalid object field indentation: {line}")
        if line.startswith("field "):
            key, value = _parse_decl(line.removeprefix("field ").strip())
        else:
            key, value = _parse_decl(line)
        fields[key] = value
        index += 1
    return fields, index


def _parse_key_value_block(lines: list[tuple[int, str]], index: int, *, parent_indent: int) -> tuple[dict[str, str], int]:
    payload: dict[str, str] = {}
    while index < len(lines):
        indent, line = lines[index]
        if indent <= parent_indent:
            break
        if indent != parent_indent + 2:
            raise ValueError(f"Unexpected indentation in project block: {line}")
        key, value = _parse_decl(line, separator=":")
        payload[key] = value
        index += 1
    return payload, index


def _parse_steps(lines: list[tuple[int, str]], index: int, *, parent_indent: int) -> tuple[list[Statement], int]:
    steps: list[Statement] = []
    while index < len(lines):
        indent, line = lines[index]
        if indent <= parent_indent:
            break
        if indent != parent_indent + 2:
            raise ValueError(f"Unexpected indentation in block: {line}")
        if line == "action:":
            index += 1
            nested_steps, index = _parse_steps(lines, index, parent_indent=indent)
            steps.extend(nested_steps)
            continue
        if line.startswith("use module "):
            steps.append(ModuleCallStep(module_name=line.removeprefix("use module ").strip()))
            index += 1
            continue
        if line.startswith("use "):
            steps.append(ActionStep(action=_strip_quoted(line.removeprefix("use ").strip())))
            index += 1
            continue
        if line.startswith("emit "):
            steps.append(EmitStep(expression=line.removeprefix("emit ").strip()))
            index += 1
            continue
        if line.startswith("append ") and " to " in line:
            expr, target = line.removeprefix("append ").split(" to ", 1)
            steps.append(AppendStep(expression=expr.strip(), target=target.strip()))
            index += 1
            continue
        if line.startswith("remove last(") and line.endswith(")"):
            steps.append(RemoveLastStep(target=line.removeprefix("remove last(").removesuffix(")").strip()))
            index += 1
            continue
        if line.startswith("return "):
            steps.append(ReturnStep(expression=line.removeprefix("return ").strip()))
            index += 1
            continue
        if line.startswith("initialize "):
            steps.append(InitializeStep(object_name=line.removeprefix("initialize ").strip()))
            index += 1
            continue
        if line.startswith("call "):
            steps.append(_parse_call_statement(line.removeprefix("call ").strip()))
            index += 1
            continue
        if line.startswith("set ") and "=" in line:
            target, expr = line.removeprefix("set ").split("=", 1)
            steps.append(AssignStep(target=target.strip(), expression=expr.strip()))
            index += 1
            continue
        if line.startswith("if ") and line.endswith(":"):
            condition = line.removeprefix("if ").removesuffix(":").strip()
            index += 1
            then_steps, index = _parse_steps(lines, index, parent_indent=indent)
            else_steps: list[Statement] = []
            if index < len(lines) and lines[index][0] == indent and lines[index][1] == "else:":
                index += 1
                else_steps, index = _parse_steps(lines, index, parent_indent=indent)
            steps.append(IfStep(condition=condition, then_steps=then_steps, else_steps=else_steps))
            continue
        if (line.startswith("loop ") or line.startswith("for each ")) and line.endswith(":"):
            payload = line.removeprefix("loop ").strip() if line.startswith("loop ") else line.removeprefix("for each ").strip()
            payload = payload.removesuffix(":").strip()
            if " in " not in payload:
                raise ValueError(f"Malformed loop statement: {line}")
            iterator, source = payload.split(" in ", 1)
            index += 1
            body, index = _parse_steps(lines, index, parent_indent=indent)
            steps.append(LoopStep(iterator=iterator.strip(), source=source.strip(), body=body))
            continue
        if line.startswith("while ") and line.endswith(":"):
            condition = line.removeprefix("while ").removesuffix(":").strip()
            index += 1
            body, index = _parse_steps(lines, index, parent_indent=indent)
            steps.append(WhileStep(condition=condition, body=body))
            continue
        if line == "break":
            steps.append(BreakStep())
            index += 1
            continue
        if line == "continue":
            steps.append(ContinueStep())
            index += 1
            continue
        raise ValueError(f"Unsupported statement: {line}")
    return steps, index


def _parse_decl(payload: str, separator: str = ":") -> tuple[str, str]:
    if separator not in payload:
        raise ValueError(f"Malformed declaration: {payload}")
    key, value = payload.split(separator, 1)
    return key.strip(), value.strip()


def _strip_quoted(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _render_steps(steps: list[Statement], *, indent: int) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for step in steps:
        if isinstance(step, ActionStep):
            lines.append(f'{prefix}use "{step.action}"')
        elif isinstance(step, ModuleCallStep):
            lines.append(f"{prefix}use module {step.module_name}")
        elif isinstance(step, CallStep):
            args = ", ".join(step.arguments)
            lines.append(f"{prefix}call {step.module_name}({args})")
        elif isinstance(step, EmitStep):
            lines.append(f"{prefix}emit {step.expression}")
        elif isinstance(step, AppendStep):
            lines.append(f"{prefix}append {step.expression} to {step.target}")
        elif isinstance(step, RemoveLastStep):
            lines.append(f"{prefix}remove last({step.target})")
        elif isinstance(step, ReturnStep):
            lines.append(f"{prefix}return {step.expression}")
        elif isinstance(step, InitializeStep):
            lines.append(f"{prefix}initialize {step.object_name}")
        elif isinstance(step, AssignStep):
            lines.append(f"{prefix}set {step.target} = {step.expression}")
        elif isinstance(step, IfStep):
            lines.append(f"{prefix}if {step.condition}:")
            lines.extend(_render_steps(step.then_steps, indent=indent + 2))
            if step.else_steps:
                lines.append(f"{prefix}else:")
                lines.extend(_render_steps(step.else_steps, indent=indent + 2))
        elif isinstance(step, LoopStep):
            lines.append(f"{prefix}loop {step.iterator} in {step.source}:")
            lines.extend(_render_steps(step.body, indent=indent + 2))
        elif isinstance(step, WhileStep):
            lines.append(f"{prefix}while {step.condition}:")
            lines.extend(_render_steps(step.body, indent=indent + 2))
        elif isinstance(step, BreakStep):
            lines.append(f"{prefix}break")
        elif isinstance(step, ContinueStep):
            lines.append(f"{prefix}continue")
    return lines


def _serialize_step(step: Statement) -> dict[str, object]:
    if isinstance(step, ActionStep):
        return {"kind": "ActionStep", "action": step.action}
    if isinstance(step, ModuleCallStep):
        return {"kind": "ModuleCallStep", "module_name": step.module_name}
    if isinstance(step, CallStep):
        return {"kind": "CallStep", "module_name": step.module_name, "arguments": list(step.arguments)}
    if isinstance(step, EmitStep):
        return {"kind": "EmitStep", "expression": step.expression}
    if isinstance(step, AppendStep):
        return {"kind": "AppendStep", "expression": step.expression, "target": step.target}
    if isinstance(step, RemoveLastStep):
        return {"kind": "RemoveLastStep", "target": step.target}
    if isinstance(step, ReturnStep):
        return {"kind": "ReturnStep", "expression": step.expression}
    if isinstance(step, InitializeStep):
        return {"kind": "InitializeStep", "object_name": step.object_name}
    if isinstance(step, AssignStep):
        return {"kind": "AssignStep", "target": step.target, "expression": step.expression}
    if isinstance(step, IfStep):
        return {
            "kind": "IfStep",
            "condition": step.condition,
            "then_steps": [_serialize_step(item) for item in step.then_steps],
            "else_steps": [_serialize_step(item) for item in step.else_steps],
        }
    if isinstance(step, LoopStep):
        return {
            "kind": "LoopStep",
            "iterator": step.iterator,
            "source": step.source,
            "body": [_serialize_step(item) for item in step.body],
        }
    if isinstance(step, WhileStep):
        return {"kind": "WhileStep", "condition": step.condition, "body": [_serialize_step(item) for item in step.body]}
    if isinstance(step, BreakStep):
        return {"kind": "BreakStep"}
    if isinstance(step, ContinueStep):
        return {"kind": "ContinueStep"}
    raise TypeError(f"Unsupported statement type: {type(step)!r}")


def _deserialize_step(payload: dict[str, object]) -> Statement:
    kind = payload.get("kind")
    if kind == "ActionStep":
        return ActionStep(action=str(payload.get("action", "")))
    if kind == "ModuleCallStep":
        return ModuleCallStep(module_name=str(payload.get("module_name", "")))
    if kind == "CallStep":
        return CallStep(module_name=str(payload.get("module_name", "")), arguments=[str(item) for item in payload.get("arguments", [])])
    if kind == "EmitStep":
        return EmitStep(expression=str(payload.get("expression", "")))
    if kind == "AppendStep":
        return AppendStep(expression=str(payload.get("expression", "")), target=str(payload.get("target", "")))
    if kind == "RemoveLastStep":
        return RemoveLastStep(target=str(payload.get("target", "")))
    if kind == "ReturnStep":
        return ReturnStep(expression=str(payload.get("expression", "")))
    if kind == "InitializeStep":
        return InitializeStep(object_name=str(payload.get("object_name", "")))
    if kind == "AssignStep":
        return AssignStep(target=str(payload.get("target", "")), expression=str(payload.get("expression", "")))
    if kind == "IfStep":
        return IfStep(
            condition=str(payload.get("condition", "")),
            then_steps=[_deserialize_step(item) for item in payload.get("then_steps", []) if isinstance(item, dict)],
            else_steps=[_deserialize_step(item) for item in payload.get("else_steps", []) if isinstance(item, dict)],
        )
    if kind == "LoopStep":
        return LoopStep(
            iterator=str(payload.get("iterator", "")),
            source=str(payload.get("source", "")),
            body=[_deserialize_step(item) for item in payload.get("body", []) if isinstance(item, dict)],
        )
    if kind == "WhileStep":
        return WhileStep(
            condition=str(payload.get("condition", "")),
            body=[_deserialize_step(item) for item in payload.get("body", []) if isinstance(item, dict)],
        )
    if kind == "BreakStep":
        return BreakStep()
    if kind == "ContinueStep":
        return ContinueStep()
    raise ValueError(f"Unsupported serialized statement: {payload}")


def _parse_module_header(line: str) -> tuple[str, list[str]]:
    body = line.removeprefix("module ").removesuffix(":").strip()
    if "(" not in body:
        return body, []
    name, remainder = body.split("(", 1)
    params = remainder.removesuffix(")").strip()
    parsed = [item.strip() for item in params.split(",") if item.strip()]
    return name.strip(), parsed


def _parse_call_statement(payload: str) -> CallStep:
    if "(" not in payload or not payload.endswith(")"):
        return CallStep(module_name=payload.strip(), arguments=[])
    name, remainder = payload.split("(", 1)
    args = remainder[:-1].strip()
    parsed = [item.strip() for item in args.split(",") if item.strip()]
    return CallStep(module_name=name.strip(), arguments=parsed)
