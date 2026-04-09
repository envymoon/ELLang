from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HeapObject:
    object_id: str
    kind: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class HeapMemory:
    def __init__(self) -> None:
        self._counter = 0
        self.objects: dict[str, HeapObject] = {}
        self.bindings: dict[str, str] = {}

    def store(self, symbol: str, value: Any, *, kind: str | None = None) -> str:
        object_id = f"obj_{self._counter}"
        self._counter += 1
        object_kind = kind or _infer_kind(value)
        self.objects[object_id] = HeapObject(
            object_id=object_id,
            kind=object_kind,
            value=value,
            metadata={"symbol": symbol},
        )
        self.bindings[symbol] = object_id
        return object_id

    def load(self, symbol: str) -> Any:
        object_id = self.bindings[symbol]
        return self.objects[object_id].value

    def snapshot(self) -> dict[str, Any]:
        return {
            symbol: {
                "object_id": object_id,
                "kind": self.objects[object_id].kind,
            }
            for symbol, object_id in self.bindings.items()
        }


def _infer_kind(value: Any) -> str:
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "record"
    return type(value).__name__
