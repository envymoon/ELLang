from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ValueType(StrEnum):
    ANY = "any"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    NUMBER = "number"
    STRING = "string"
    BYTES = "bytes"
    DATASET = "dataset"
    RECORD = "record"
    LIST = "list"
    MAP = "map"
    OBJECT = "object"
    MODULE = "module"
    FUNCTION = "function"
    TRACE = "trace"
    PROJECT = "project"
    TEST_REPORT = "test_report"
    DEBUG_REPORT = "debug_report"
    INDEX_LIST = "index_list"
    VOID = "void"


class Capability(StrEnum):
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    NETWORK = "network"
    PROCESS = "process"
    GIT_READ = "git.read"
    GIT_WRITE = "git.write"
    MODEL_INFER = "model.infer"
    DEBUG_ESCALATE = "debug.escalate"
    FFI_CALL = "ffi.call"


class GenericOperator(StrEnum):
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


class RuntimeTarget(StrEnum):
    REFERENCE = "reference"
    NATIVE = "native"
    JVM = "jvm"
    WASM = "wasm"


class JitTier(StrEnum):
    OFF = "off"
    BASELINE = "baseline"
    HOTSPOT = "hotspot"
    AGGRESSIVE = "aggressive"


@dataclass(slots=True)
class TypeSpec:
    kind: ValueType
    name: str = ""
    fields: dict[str, "TypeSpec"] = field(default_factory=dict)
    element_type: "TypeSpec | None" = None
    key_type: "TypeSpec | None" = None
    value_type: "TypeSpec | None" = None
    nullable: bool = False

    def describe(self) -> str:
        if self.kind == ValueType.LIST and self.element_type is not None:
            return f"list[{self.element_type.describe()}]"
        if self.kind == ValueType.MAP and self.key_type is not None and self.value_type is not None:
            return f"map[{self.key_type.describe()}, {self.value_type.describe()}]"
        if self.kind == ValueType.OBJECT and self.name:
            return self.name
        return self.kind.value


@dataclass(slots=True)
class FFISignature:
    name: str
    library: str
    abi: str = "system"
    args: list[TypeSpec] = field(default_factory=list)
    returns: TypeSpec = field(default_factory=lambda: TypeSpec(ValueType.ANY))
    required_capabilities: list[Capability] = field(default_factory=lambda: [Capability.FFI_CALL])


@dataclass(slots=True)
class RuntimeConfig:
    target: RuntimeTarget = RuntimeTarget.NATIVE
    jit_tier: JitTier = JitTier.BASELINE
    aot_enabled: bool = True
    hot_threshold: int = 8
    cross_platform_targets: list[RuntimeTarget] = field(default_factory=lambda: [RuntimeTarget.NATIVE])


@dataclass(slots=True)
class ResourceBudget:
    max_tokens: int = 2048
    max_vram_mb: int = 6144
    max_cpu_ms: int = 5000
    max_wall_ms: int = 10000
    max_writes: int = 0
    max_network_calls: int = 0
    max_ffi_calls: int = 0


@dataclass(slots=True)
class TypedNode:
    node_id: str
    operator: GenericOperator
    input_types: list[ValueType] = field(default_factory=list)
    output_type: ValueType = ValueType.ANY
    capabilities: list[Capability] = field(default_factory=list)
    deterministic: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    type_spec: TypeSpec | None = None
    intrinsic: str = ""
    ffi_signature: FFISignature | None = None


@dataclass(slots=True)
class TypedEdge:
    source: str
    target: str
    label: str = ""


@dataclass(slots=True)
class TypedProgram:
    program_name: str
    nodes: list[TypedNode]
    edges: list[TypedEdge]
    budget: ResourceBudget
    required_capabilities: list[Capability]
    diagnostics: list[str] = field(default_factory=list)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    exported_types: dict[str, TypeSpec] = field(default_factory=dict)
    ffi_bindings: list[FFISignature] = field(default_factory=list)

    def node_map(self) -> dict[str, TypedNode]:
        return {node.node_id: node for node in self.nodes}
