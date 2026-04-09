from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class TraceFrame:
    node_id: str
    title: str
    operation: str
    started_at_ms: float
    finished_at_ms: float
    input_summary: str
    output_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self) -> None:
        self._t0 = perf_counter()
        self.frames: list[TraceFrame] = []

    def record(
        self,
        *,
        node_id: str,
        title: str,
        operation: str,
        input_summary: str,
        output_summary: str,
        metadata: dict[str, Any] | None = None,
        started: float,
        finished: float,
    ) -> None:
        self.frames.append(
            TraceFrame(
                node_id=node_id,
                title=title,
                operation=operation,
                started_at_ms=(started - self._t0) * 1000,
                finished_at_ms=(finished - self._t0) * 1000,
                input_summary=input_summary,
                output_summary=output_summary,
                metadata=metadata or {},
            )
        )
