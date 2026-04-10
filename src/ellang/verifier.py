from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bytecode import BytecodeLowerer
from .typed_ir import (
    Capability,
    FFISignature,
    GenericOperator,
    JitTier,
    ResourceBudget,
    RuntimeConfig,
    RuntimeTarget,
    TypeSpec,
    TypedEdge,
    TypedNode,
    TypedProgram,
    ValueType,
)


@dataclass(slots=True)
class VerificationResult:
    typed_program: TypedProgram
    diagnostics: list[str]


OPERATOR_SIGNATURES: dict[GenericOperator, tuple[set[ValueType], set[ValueType]]] = {
    GenericOperator.SORT: ({ValueType.DATASET, ValueType.LIST, ValueType.ANY}, {ValueType.DATASET, ValueType.LIST}),
    GenericOperator.FILTER: ({ValueType.DATASET, ValueType.LIST, ValueType.ANY}, {ValueType.DATASET, ValueType.LIST}),
    GenericOperator.MAP: ({ValueType.DATASET, ValueType.LIST, ValueType.ANY}, {ValueType.DATASET, ValueType.LIST, ValueType.ANY}),
    GenericOperator.REDUCE: ({ValueType.DATASET, ValueType.LIST, ValueType.ANY}, {ValueType.RECORD, ValueType.MAP, ValueType.ANY}),
    GenericOperator.EVAL_CONDITION: ({ValueType.ANY, ValueType.BOOL}, {ValueType.BOOL}),
    GenericOperator.LOOP: ({ValueType.DATASET, ValueType.LIST, ValueType.ANY}, {ValueType.DATASET, ValueType.LIST, ValueType.ANY}),
    GenericOperator.CALL_INTRINSIC: ({ValueType.ANY, ValueType.RECORD, ValueType.DATASET, ValueType.LIST}, {ValueType.ANY, ValueType.INT, ValueType.BOOL, ValueType.LIST, ValueType.DATASET, ValueType.RECORD}),
    GenericOperator.CALL_FFI: ({ValueType.ANY, ValueType.RECORD, ValueType.DATASET, ValueType.LIST, ValueType.STRING}, {ValueType.ANY, ValueType.RECORD, ValueType.DATASET, ValueType.STRING}),
}


class TypedProgramVerifier:
    def verify(self, program: TypedProgram) -> VerificationResult:
        node_map = program.node_map()
        for edge in program.edges:
            if edge.source not in node_map or edge.target not in node_map:
                raise ValueError(f"Invalid typed IR edge: {edge.source} -> {edge.target}")
        for node in program.nodes:
            _validate_node(node)
        _validate_runtime(program.runtime)
        diagnostics = [
            "Capability and type verification completed.",
            f"Runtime target: {program.runtime.target.value}",
            f"JIT tier: {program.runtime.jit_tier.value}",
            f"FFI bindings: {len(program.ffi_bindings)}",
            f"Exported types: {len(program.exported_types)}",
        ]
        return VerificationResult(typed_program=program, diagnostics=diagnostics)


def lower_to_bytecode(program: TypedProgram) -> tuple[TypedProgram, object, list[str]]:
    verification = TypedProgramVerifier().verify(program)
    bytecode = BytecodeLowerer().lower(verification.typed_program)
    return verification.typed_program, bytecode, verification.diagnostics + bytecode.diagnostics


def typed_program_from_dict(payload: dict[str, object], *, program_name: str, budget: ResourceBudget) -> TypedProgram:
    nodes = [
        TypedNode(
            node_id=str(node["node_id"]),
            operator=GenericOperator(str(node["operator"])),
            input_types=[ValueType(item) for item in node.get("input_types", [])],
            output_type=ValueType(node.get("output_type", "any")),
            capabilities=[Capability(item) for item in node.get("capabilities", [])],
            deterministic=bool(node.get("deterministic", False)),
            config=dict(node.get("config", {})),
            type_spec=_parse_type_spec(node.get("type_spec")),
            intrinsic=str(node.get("intrinsic", "")),
            ffi_signature=_parse_ffi_signature(node.get("ffi_signature")),
        )
        for node in payload.get("nodes", [])
    ]
    edges = [
        TypedEdge(str(edge["source"]), str(edge["target"]), str(edge.get("label", "")))
        for edge in payload.get("edges", [])
    ]
    ffi_bindings = [_parse_ffi_signature(item) for item in payload.get("ffi_bindings", [])]
    ffi_bindings = [item for item in ffi_bindings if item is not None]
    required = sorted(
        {cap for node in nodes for cap in node.capabilities} | {cap for binding in ffi_bindings for cap in binding.required_capabilities},
        key=lambda item: item.value,
    )
    exported_types = {
        str(name): spec
        for name, spec in (
            (name, _parse_type_spec(spec)) for name, spec in dict(payload.get("exported_types", {})).items()
        )
        if spec is not None
    }
    runtime = _parse_runtime_config(payload.get("runtime"))
    return TypedProgram(
        program_name=program_name,
        nodes=nodes,
        edges=edges,
        budget=budget,
        required_capabilities=required,
        diagnostics=[str(item) for item in payload.get("diagnostics", [])],
        runtime=runtime,
        exported_types=exported_types,
        ffi_bindings=ffi_bindings,
    )


def _validate_node(node: TypedNode) -> None:
    if node.operator in {GenericOperator.SORT, GenericOperator.FILTER, GenericOperator.SLIDING_WINDOW_SCAN} and not node.deterministic:
        raise ValueError(f"Deterministic operator must be marked deterministic: {node.operator.value}")
    if node.operator in OPERATOR_SIGNATURES:
        allowed_inputs, allowed_outputs = OPERATOR_SIGNATURES[node.operator]
        if node.input_types and any(item not in allowed_inputs for item in node.input_types):
            raise ValueError(f"Invalid input signature for {node.operator.value}: {[item.value for item in node.input_types]}")
        if node.output_type not in allowed_outputs:
            raise ValueError(f"Invalid output type for {node.operator.value}: {node.output_type.value}")
    if node.operator == GenericOperator.CALL_INTRINSIC and not node.intrinsic:
        raise ValueError("call_intrinsic nodes must declare an intrinsic name.")
    if node.operator == GenericOperator.CALL_FFI:
        if node.ffi_signature is None:
            raise ValueError("call_ffi nodes must include an ffi signature.")
        if Capability.FFI_CALL not in node.capabilities:
            raise ValueError("call_ffi nodes must declare ffi.call capability.")
    if node.ffi_signature is not None and Capability.FFI_CALL not in node.capabilities:
        node.capabilities.append(Capability.FFI_CALL)


def _validate_runtime(runtime: RuntimeConfig) -> None:
    if runtime.target not in runtime.cross_platform_targets:
        raise ValueError(f"Primary runtime target {runtime.target.value} must be present in cross-platform targets.")
    if runtime.jit_tier == JitTier.AGGRESSIVE and not runtime.aot_enabled:
        raise ValueError("Aggressive JIT requires AOT metadata to stay enabled for fallback.")


def _parse_runtime_config(value: object) -> RuntimeConfig:
    if not isinstance(value, dict):
        return RuntimeConfig()
    targets = [RuntimeTarget(item) for item in value.get("cross_platform_targets", [RuntimeTarget.NATIVE.value])]
    return RuntimeConfig(
        target=RuntimeTarget(value.get("target", RuntimeTarget.NATIVE.value)),
        jit_tier=JitTier(value.get("jit_tier", JitTier.BASELINE.value)),
        aot_enabled=bool(value.get("aot_enabled", True)),
        hot_threshold=int(value.get("hot_threshold", 8)),
        cross_platform_targets=targets,
    )


def _parse_type_spec(value: object) -> TypeSpec | None:
    if value is None:
        return None
    if isinstance(value, str):
        return TypeSpec(kind=ValueType(value))
    if not isinstance(value, dict):
        return None
    return TypeSpec(
        kind=ValueType(value.get("kind", "any")),
        name=str(value.get("name", "")),
        fields={str(name): parsed for name, parsed in ((name, _parse_type_spec(item)) for name, item in value.get("fields", {}).items()) if parsed is not None},
        element_type=_parse_type_spec(value.get("element_type")),
        key_type=_parse_type_spec(value.get("key_type")),
        value_type=_parse_type_spec(value.get("value_type")),
        nullable=bool(value.get("nullable", False)),
    )


def _parse_ffi_signature(value: object) -> FFISignature | None:
    if not isinstance(value, dict):
        return None
    capabilities = [Capability(item) for item in value.get("required_capabilities", [Capability.FFI_CALL.value])]
    return FFISignature(
        name=str(value.get("name", "")),
        library=str(value.get("library", "")),
        abi=str(value.get("abi", "system")),
        args=[parsed for parsed in (_parse_type_spec(item) for item in value.get("args", [])) if parsed is not None],
        returns=_parse_type_spec(value.get("returns")) or TypeSpec(ValueType.ANY),
        required_capabilities=capabilities,
    )
