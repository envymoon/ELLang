from __future__ import annotations

from dataclasses import dataclass, field

from .ir import ExecutionPlan, IRNode, NodeKind


@dataclass(slots=True)
class OptimizationReport:
    applied_passes: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


class GraphOptimizer:
    """
    Lightweight optimizer passes.

    This is where the future executable-model runtime can lower planner output
    into increasingly deterministic kernels, fuse validation, and prepare hot
    paths for native execution.
    """

    def optimize(self, plan: ExecutionPlan) -> tuple[ExecutionPlan, OptimizationReport]:
        report = OptimizationReport()
        self._annotate_hot_path(plan, report)
        self._mark_deterministic_segments(plan, report)
        return plan, report

    def _annotate_hot_path(self, plan: ExecutionPlan, report: OptimizationReport) -> None:
        for node in plan.nodes:
            if node.kind in {NodeKind.SORT, NodeKind.FILTER, NodeKind.TRANSFORM, NodeKind.MODULE, NodeKind.LOOP}:
                node.config.setdefault("hot_path", True)
        report.applied_passes.append("annotate_hot_path")
        report.diagnostics.append("Marked deterministic operator nodes as hot-path candidates.")

    def _mark_deterministic_segments(self, plan: ExecutionPlan, report: OptimizationReport) -> None:
        for node in plan.nodes:
            if node.kind in {NodeKind.SORT, NodeKind.FILTER, NodeKind.VALIDATE, NodeKind.TEST}:
                node.config.setdefault("execution_mode", "deterministic")
            elif node.kind in {NodeKind.INFER, NodeKind.DEBUG}:
                node.config.setdefault("execution_mode", "model_guided")
        report.applied_passes.append("mark_deterministic_segments")
        report.diagnostics.append("Annotated node execution modes for future lowering and fusion.")
