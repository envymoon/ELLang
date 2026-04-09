from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from .typed_ir import TypedProgram, ValueType


class OpCode(StrEnum):
    LOAD_INPUT = "load_input"
    MODEL_PLAN = "model_plan"
    INIT_PROJECT = "init_project"
    REGISTER_OBJECT = "register_object"
    BUILD_FREQ_MAP = "build_freq_map"
    COMPUTE_LENGTH = "compute_length"
    COMPUTE_TOTAL_LENGTH = "compute_total_length"
    SLIDING_WINDOW_SCAN = "sliding_window_scan"
    SPLIT_EQUAL_WORDS = "split_equal_words"
    COMPARE_FREQ_MAPS = "compare_freq_maps"
    COLLECT_INDICES = "collect_indices"
    SORT = "sort"
    FILTER = "filter"
    MAP = "map"
    REDUCE = "reduce"
    TRANSFORM = "transform"
    EVAL_CONDITION = "eval_condition"
    LOOP = "loop"
    MERGE = "merge"
    VALIDATE = "validate"
    SYNTH_TESTS = "synth_tests"
    DEBUG_PREP = "debug_prep"
    OUTPUT = "output"
    CALL_INTRINSIC = "call_intrinsic"
    CALL_FFI = "call_ffi"


@dataclass(slots=True)
class Instruction:
    opcode: OpCode
    operand: dict[str, object] = field(default_factory=dict)
    result_type: ValueType = ValueType.ANY


@dataclass(slots=True)
class BytecodeProgram:
    program_name: str
    instructions: list[Instruction]
    diagnostics: list[str] = field(default_factory=list)
    runtime: dict[str, object] = field(default_factory=dict)
    ffi_bindings: list[dict[str, object]] = field(default_factory=list)
    exported_types: dict[str, object] = field(default_factory=dict)
    backend_prototypes: dict[str, str] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, object]:
        return {
            "program_name": self.program_name,
            "instructions": [
                {
                    "opcode": instruction.opcode.value,
                    "operand": instruction.operand,
                    "result_type": instruction.result_type.value,
                }
                for instruction in self.instructions
            ],
            "diagnostics": self.diagnostics,
            "runtime": self.runtime,
            "ffi_bindings": self.ffi_bindings,
            "exported_types": self.exported_types,
            "backend_prototypes": self.backend_prototypes,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_serializable(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


class BytecodeLowerer:
    def lower(self, typed_program: TypedProgram) -> BytecodeProgram:
        instructions: list[Instruction] = []
        for node in typed_program.nodes:
            operand = {"node_id": node.node_id, **node.config}
            if node.intrinsic:
                operand["intrinsic"] = node.intrinsic
            if node.ffi_signature is not None:
                operand["ffi_signature"] = asdict(node.ffi_signature)
            instructions.append(
                Instruction(
                    opcode=OpCode(node.operator.value),
                    operand=operand,
                    result_type=node.output_type,
                )
            )
        return BytecodeProgram(
            program_name=typed_program.program_name,
            instructions=instructions,
            diagnostics=["Lowered typed IR to stable bytecode prototype.", "Embedded runtime/AOT/JIT/FFI metadata into bytecode."],
            runtime={
                "target": typed_program.runtime.target.value,
                "jit_tier": typed_program.runtime.jit_tier.value,
                "aot_enabled": typed_program.runtime.aot_enabled,
                "hot_threshold": typed_program.runtime.hot_threshold,
                "cross_platform_targets": [item.value for item in typed_program.runtime.cross_platform_targets],
                "budget": asdict(typed_program.budget),
            },
            ffi_bindings=[asdict(item) for item in typed_program.ffi_bindings],
            exported_types={name: _type_spec_to_dict(spec) for name, spec in typed_program.exported_types.items()},
            backend_prototypes={},
        )


def _type_spec_to_dict(spec: object) -> object:
    if hasattr(spec, "kind"):
        fields = getattr(spec, "fields", {})
        element_type = getattr(spec, "element_type", None)
        key_type = getattr(spec, "key_type", None)
        value_type = getattr(spec, "value_type", None)
        return {
            "kind": getattr(spec, "kind").value,
            "name": getattr(spec, "name", ""),
            "nullable": getattr(spec, "nullable", False),
            "fields": {name: _type_spec_to_dict(item) for name, item in fields.items()},
            "element_type": _type_spec_to_dict(element_type) if element_type is not None else None,
            "key_type": _type_spec_to_dict(key_type) if key_type is not None else None,
            "value_type": _type_spec_to_dict(value_type) if value_type is not None else None,
        }
    return spec
