from __future__ import annotations

from dataclasses import dataclass

from .ir import ExecutionPlan, IREdge, IRNode, NodeKind
from .models import BackendResult, LocalModelBackend, QwenLocalBackend
from .optimizer import GraphOptimizer
from .syntax import ProgramSpec, serialize_program
from .testing import TestSynthesizer
from .typed_ir import GenericOperator, JitTier, ResourceBudget, RuntimeConfig, RuntimeTarget, TypeSpec, ValueType
from .verifier import lower_to_bytecode, typed_program_from_dict


@dataclass(slots=True)
class Compiler:
    backend: LocalModelBackend | None = None
    optimizer: GraphOptimizer | None = None
    test_synthesizer: TestSynthesizer | None = None

    def compile(self, spec: ProgramSpec) -> ExecutionPlan:
        if _requires_structured_execution(spec):
            plan_result = _structured_intrinsic_plan(spec)
        else:
            backend = self.backend or QwenLocalBackend()
            plan_result = backend.plan_typed_program(spec)
        diagnostics = list(plan_result.diagnostics)

        budget = ResourceBudget(
            max_tokens=1024 if spec.constraints.get("deterministic", "").lower() == "true" else 2048,
            max_vram_mb=4096,
            max_cpu_ms=5000,
            max_wall_ms=10000,
            max_writes=0,
            max_network_calls=0,
        )
        typed_program = typed_program_from_dict(plan_result.typed_program_payload, program_name=spec.name, budget=budget)
        typed_program.runtime = RuntimeConfig(
            target=RuntimeTarget.NATIVE,
            jit_tier=JitTier.HOTSPOT,
            aot_enabled=True,
            hot_threshold=6,
            cross_platform_targets=[RuntimeTarget.NATIVE, RuntimeTarget.WASM, RuntimeTarget.JVM],
        )
        typed_program.exported_types.update(
            {
                name: TypeSpec(
                    kind=ValueType.OBJECT,
                    name=name,
                    fields={field_name: TypeSpec(kind=_field_value_type(field_type)) for field_name, field_type in object_spec.fields.items()},
                )
                for name, object_spec in spec.objects.items()
            }
        )
        typed_program, bytecode, lowering_diagnostics = lower_to_bytecode(typed_program)
        diagnostics.extend(lowering_diagnostics)

        nodes = [_to_ir_node(node) for node in typed_program.nodes]
        edges = [IREdge(edge.source, edge.target, edge.label) for edge in typed_program.edges]
        output_nodes = [node.node_id for node in typed_program.nodes if node.operator == GenericOperator.OUTPUT]
        plan = ExecutionPlan(
            program_name=spec.name,
            intent=spec.intent,
            nodes=nodes,
            edges=edges,
            entrypoint=typed_program.nodes[0].node_id if typed_program.nodes else "input.main",
            outputs=output_nodes or ([typed_program.nodes[-1].node_id] if typed_program.nodes else []),
            diagnostics=diagnostics,
        )
        synthesizer = self.test_synthesizer or TestSynthesizer()
        plan.suggested_tests = synthesizer.synthesize(spec, plan)
        plan.typed_program = typed_program
        plan.bytecode = bytecode

        optimizer = self.optimizer or GraphOptimizer()
        optimized_plan, report = optimizer.optimize(plan)
        optimized_plan.diagnostics.extend(report.diagnostics)
        return optimized_plan


def _to_ir_node(node: object) -> IRNode:
    from .typed_ir import TypedNode

    typed = node if isinstance(node, TypedNode) else None
    if typed is None:
        raise TypeError("Expected TypedNode while building IR view.")
    return IRNode(
        node_id=typed.node_id,
        kind=_kind_from_operator(typed.operator),
        title=_title_from_operator(typed.operator),
        config={
            **typed.config,
            "operator": typed.operator.value,
            "output_type": typed.output_type.value,
            "deterministic": typed.deterministic,
            "capabilities": [cap.value for cap in typed.capabilities],
        },
    )


def _kind_from_operator(operator: GenericOperator) -> NodeKind:
    return {
        GenericOperator.LOAD_INPUT: NodeKind.INPUT,
        GenericOperator.MODEL_PLAN: NodeKind.INFER,
        GenericOperator.INIT_PROJECT: NodeKind.PROJECT,
        GenericOperator.REGISTER_OBJECT: NodeKind.OBJECT,
        GenericOperator.EVAL_CONDITION: NodeKind.CONDITION,
        GenericOperator.LOOP: NodeKind.LOOP,
        GenericOperator.SORT: NodeKind.SORT,
        GenericOperator.FILTER: NodeKind.FILTER,
        GenericOperator.VALIDATE: NodeKind.VALIDATE,
        GenericOperator.SYNTH_TESTS: NodeKind.TEST,
        GenericOperator.DEBUG_PREP: NodeKind.DEBUG,
        GenericOperator.OUTPUT: NodeKind.OUTPUT,
        GenericOperator.MERGE: NodeKind.MERGE,
    }.get(operator, NodeKind.TRANSFORM)


def _title_from_operator(operator: GenericOperator) -> str:
    return {
        GenericOperator.LOAD_INPUT: "Load Input",
        GenericOperator.MODEL_PLAN: "Model Plan",
        GenericOperator.INIT_PROJECT: "Project Runtime",
        GenericOperator.REGISTER_OBJECT: "Register Object",
        GenericOperator.BUILD_FREQ_MAP: "Build Frequency Map",
        GenericOperator.COMPUTE_LENGTH: "Compute Length",
        GenericOperator.COMPUTE_TOTAL_LENGTH: "Compute Total Length",
        GenericOperator.SLIDING_WINDOW_SCAN: "Sliding Window Scan",
        GenericOperator.SPLIT_EQUAL_WORDS: "Split Equal Words",
        GenericOperator.COMPARE_FREQ_MAPS: "Compare Frequency Maps",
        GenericOperator.COLLECT_INDICES: "Collect Indices",
        GenericOperator.SORT: "Deterministic Sort",
        GenericOperator.FILTER: "Deterministic Filter",
        GenericOperator.MAP: "Map",
        GenericOperator.REDUCE: "Reduce",
        GenericOperator.TRANSFORM: "Transform",
        GenericOperator.EVAL_CONDITION: "Evaluate Condition",
        GenericOperator.LOOP: "Loop",
        GenericOperator.MERGE: "Merge",
        GenericOperator.VALIDATE: "Validate Output Contracts",
        GenericOperator.SYNTH_TESTS: "Synthesize Verification Tests",
        GenericOperator.DEBUG_PREP: "Prepare AI Debug Strategy",
        GenericOperator.OUTPUT: "Publish Result",
        GenericOperator.CALL_INTRINSIC: "Call Native Intrinsic",
        GenericOperator.CALL_FFI: "Call FFI",
    }[operator]


def _field_value_type(field_type: str) -> ValueType:
    return {
        "string": ValueType.STRING,
        "number": ValueType.NUMBER,
        "int": ValueType.INT,
        "float": ValueType.FLOAT,
        "bool": ValueType.BOOL,
        "dataset": ValueType.DATASET,
        "record": ValueType.RECORD,
    }.get(field_type, ValueType.ANY)


def _output_type_from_spec(spec: ProgramSpec) -> ValueType:
    if not spec.outputs:
        return ValueType.ANY
    first = next(iter(spec.outputs.values()))
    return {
        "dataset": ValueType.DATASET,
        "string": ValueType.STRING,
        "bool": ValueType.BOOL,
        "record": ValueType.RECORD,
        "list": ValueType.LIST,
        "int": ValueType.INT,
        "float": ValueType.FLOAT,
    }.get(first, ValueType.ANY)


def _requires_structured_execution(spec: ProgramSpec) -> bool:
    return bool(spec.project.get("algorithm_family") and spec.project.get("algorithm_task")) or any(module.params for module in spec.modules.values()) or _serialized_shape_contains_structured_ops(serialize_program(spec))


def _serialized_shape_contains_structured_ops(payload: dict[str, object]) -> bool:
    structured_kinds = {"CallStep", "AppendStep", "RemoveLastStep", "ReturnStep", "InitializeStep", "AssignStep", "WhileStep", "BreakStep", "ContinueStep"}
    for module in payload.get("modules", {}).values():
        if isinstance(module, dict):
            if any(_step_contains_kind(step, structured_kinds) for step in module.get("steps", []) if isinstance(step, dict)):
                return True
    return any(_step_contains_kind(step, structured_kinds) for step in payload.get("flow", []) if isinstance(step, dict))


def _step_contains_kind(step: dict[str, object], kinds: set[str]) -> bool:
    if str(step.get("kind", "")) in kinds:
        return True
    for key in ("then_steps", "else_steps", "body"):
        nested = step.get(key)
        if isinstance(nested, list) and any(_step_contains_kind(item, kinds) for item in nested if isinstance(item, dict)):
            return True
    return False


def _structured_intrinsic_plan(spec: ProgramSpec) -> BackendResult:
    intrinsic_name = "execute_algorithm_family" if spec.project.get("algorithm_family") and spec.project.get("algorithm_task") else "execute_structured_program"
    object_nodes = [
        {
            "node_id": f"object.{name}",
            "operator": GenericOperator.REGISTER_OBJECT.value,
            "input_types": [ValueType.RECORD.value],
            "output_type": ValueType.OBJECT.value,
            "capabilities": [],
            "deterministic": True,
            "config": {"name": name, "fields": object_spec.fields},
            "type_spec": {
                "kind": ValueType.OBJECT.value,
                "name": name,
                "fields": {field_name: {"kind": _field_value_type(field_type).value} for field_name, field_type in object_spec.fields.items()},
            },
        }
        for name, object_spec in spec.objects.items()
    ]
    execute_source = object_nodes[-1]["node_id"] if object_nodes else "input.main"
    nodes = [
        {
            "node_id": "input.main",
            "operator": GenericOperator.LOAD_INPUT.value,
            "input_types": [],
            "output_type": ValueType.RECORD.value,
            "capabilities": [],
            "deterministic": True,
            "config": {"bindings": spec.inputs},
        },
        *object_nodes,
        {
            "node_id": "op.execute.0",
            "operator": GenericOperator.CALL_INTRINSIC.value,
            "input_types": [ValueType.ANY.value],
            "output_type": _output_type_from_spec(spec).value,
            "capabilities": [],
            "deterministic": True,
            "intrinsic": intrinsic_name,
            "config": {
                "source": execute_source,
                "program_spec": serialize_program(spec),
                "algorithm_family": spec.project.get("algorithm_family", ""),
                "algorithm_task": spec.project.get("algorithm_task", ""),
            },
        },
        {
            "node_id": "validate.output",
            "operator": GenericOperator.VALIDATE.value,
            "input_types": [ValueType.ANY.value],
            "output_type": ValueType.RECORD.value,
            "capabilities": [],
            "deterministic": True,
            "config": {"constraints": spec.constraints},
        },
        {
            "node_id": "test.output",
            "operator": GenericOperator.SYNTH_TESTS.value,
            "input_types": [ValueType.RECORD.value],
            "output_type": ValueType.TEST_REPORT.value,
            "capabilities": [],
            "deterministic": True,
            "config": {"intent": spec.intent},
        },
        {
            "node_id": "debug.output",
            "operator": GenericOperator.DEBUG_PREP.value,
            "input_types": [ValueType.TEST_REPORT.value],
            "output_type": ValueType.DEBUG_REPORT.value,
            "capabilities": ["debug.escalate"],
            "deterministic": False,
            "config": {"fallback": "external_api_if_local_debug_fails"},
        },
        {
            "node_id": "output.main",
            "operator": GenericOperator.OUTPUT.value,
            "input_types": [ValueType.ANY.value],
            "output_type": _output_type_from_spec(spec).value,
            "capabilities": [],
            "deterministic": True,
            "config": {"bindings": spec.outputs, "source": "op.execute.0"},
        },
    ]
    edges = []
    previous = "input.main"
    for object_node in object_nodes:
        edges.append({"source": previous, "target": object_node["node_id"], "label": "object"})
        previous = object_node["node_id"]
    edges.extend(
        [
            {"source": previous, "target": "op.execute.0", "label": "execute"},
            {"source": "op.execute.0", "target": "validate.output", "label": "validate"},
            {"source": "validate.output", "target": "test.output", "label": "test"},
            {"source": "test.output", "target": "debug.output", "label": "debug"},
            {"source": "debug.output", "target": "output.main", "label": "output"},
        ]
    )
    return BackendResult(
        typed_program_payload={
            "nodes": nodes,
            "edges": edges,
            "diagnostics": [f"Structured program lowered to {intrinsic_name} intrinsic."],
        },
        diagnostics=["Compiler selected structured intrinsic execution path."],
    )
