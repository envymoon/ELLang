from __future__ import annotations

from .ir import ExecutionPlan
from .tracing import TraceRecorder


def mermaid_flowchart(plan: ExecutionPlan) -> str:
    lines = ["flowchart TD"]
    for node in plan.nodes:
        label = f"{node.title}\\n[{node.kind.value}]"
        lines.append(f'    {sanitize(node.node_id)}["{label}"]')
    for edge in plan.edges:
        edge_label = f"|{edge.label}|" if edge.label else ""
        lines.append(f"    {sanitize(edge.source)} -->{edge_label} {sanitize(edge.target)}")
    return "\n".join(lines)


def mermaid_trace(trace: TraceRecorder) -> str:
    lines = ["flowchart LR"]
    for idx, frame in enumerate(trace.frames):
        node_id = f"step_{idx}"
        label = f"{frame.title}\\n{frame.operation}\\n{frame.output_summary}"
        lines.append(f'    {node_id}["{label}"]')
        if idx > 0:
            lines.append(f"    step_{idx - 1} --> step_{idx}")
    return "\n".join(lines)


def sanitize(node_id: str) -> str:
    return node_id.replace(".", "_").replace("-", "_")
