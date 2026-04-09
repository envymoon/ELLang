from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from pathlib import Path
from typing import Any

from .debugging import AIDebugger, DebugReport
from .ir import ExecutionPlan
from .memory import HeapMemory
from .replay import build_replay_envelope
from .security import CapabilityPolicy, CapabilityVerifier, RuntimeQuota
from .tracing import TraceRecorder
from .typed_ir import Capability
from .vm import NativeVMHost, ReferenceVM


@dataclass(slots=True)
class ExecutionResult:
    value: Any
    memory: dict[str, Any]
    diagnostics: list[str] = field(default_factory=list)
    trace: TraceRecorder | None = None
    debug_report: DebugReport | None = None
    replay: dict[str, Any] | None = None
    vm_backend: str = "reference"


class ExecutionEngine:
    def __init__(self, *, workspace_root: str = ".", native_vm_path: str | None = None) -> None:
        self.workspace_root = workspace_root
        self.native_vm = NativeVMHost(executable=native_vm_path or _default_native_vm_path(), workspace_root=workspace_root)
        self.reference_vm = ReferenceVM(workspace_root=workspace_root)

    def execute(self, plan: ExecutionPlan, bindings: dict[str, Any]) -> ExecutionResult:
        typed_program = plan.typed_program
        if typed_program is None or plan.bytecode is None:
            raise ValueError("Execution plan must be verified and lowered before execution.")

        policy = CapabilityPolicy(
            granted=[Capability.MODEL_INFER, Capability.GIT_READ, Capability.DEBUG_ESCALATE]
        )
        diagnostics = list(plan.diagnostics)
        diagnostics.extend(CapabilityVerifier().verify(typed_program, policy))
        quota = RuntimeQuota(typed_program.budget)
        heap = HeapMemory()
        trace = TraceRecorder()

        started = perf_counter()
        if self.native_vm.is_available():
            vm_result = self.native_vm.execute(plan.bytecode, bindings, quota)
            vm_backend = "native"
        else:
            vm_result = self.reference_vm.execute(plan.bytecode, bindings, quota)
            vm_backend = "reference"
        finished = perf_counter()

        diagnostics.extend(vm_result.diagnostics)
        quota.charge_cpu_ms(int((finished - started) * 1000))
        quota.charge_wall_ms(int((finished - started) * 1000))

        trace_entries = vm_result.trace or [{"node_id": node_id, "opcode": "bytecode", "summary": _summarize_value(value)} for node_id, value in vm_result.state.items()]
        for entry in trace_entries:
            node_id = str(entry.get("node_id", "unknown"))
            value = vm_result.state.get(node_id)
            heap.store(node_id, value)
            trace.record(
                node_id=node_id,
                title=node_id,
                operation=str(entry.get("opcode", "bytecode")),
                input_summary="vm-managed",
                output_summary=str(entry.get("summary", _summarize_value(value))),
                metadata={"vm_backend": vm_backend},
                started=started,
                finished=finished,
            )

        result = ExecutionResult(
            value=vm_result.output,
            memory={"state": vm_result.state, "heap": heap.snapshot()},
            diagnostics=diagnostics,
            trace=trace,
            vm_backend=vm_backend,
        )
        git_version = None
        project_state = next((value for key, value in vm_result.state.items() if "project" in key), None)
        if isinstance(project_state, dict):
            branch = project_state.get("branch")
            git_version = branch if isinstance(branch, str) else None
        result.replay = asdict(
            build_replay_envelope(
                bindings=bindings,
                model_version=_model_version(plan),
                graph_version=plan.replay_schema_version,
                git_version=git_version,
                metadata={"diagnostics": diagnostics, "vm_backend": vm_backend},
            )
        )
        result.debug_report = AIDebugger().analyze(result)
        return result


def _summarize_value(value: Any) -> str:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:5]})"
    return repr(value)


def _model_version(plan: ExecutionPlan) -> str:
    if plan.typed_program is None:
        return "unknown"
    planner = next((node for node in plan.typed_program.nodes if Capability.MODEL_INFER in node.capabilities), None)
    if planner is None:
        return "deterministic-typed-ir"
    model_id = planner.config.get("model_id", "typed-planner")
    quantization = planner.config.get("quantization", "unknown")
    return f"{model_id}:{quantization}"


def _default_native_vm_path() -> str | None:
    candidate = Path(__file__).resolve().parents[2] / "native" / "runtime-rs" / "target" / "release" / "ellang-runtime.exe"
    return str(candidate) if candidate.exists() else None
