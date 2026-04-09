from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ReplayEnvelope:
    created_at_utc: str
    input_summary: dict[str, str]
    model_version: str
    graph_version: str
    git_version: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_replay_envelope(
    *,
    bindings: dict[str, Any],
    model_version: str,
    graph_version: str,
    git_version: str | None,
    metadata: dict[str, Any] | None = None,
) -> ReplayEnvelope:
    return ReplayEnvelope(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_summary={key: _summarize(value) for key, value in bindings.items()},
        model_version=model_version,
        graph_version=graph_version,
        git_version=git_version,
        metadata=metadata or {},
    )


def _summarize(value: Any) -> str:
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:5]})"
    return type(value).__name__
