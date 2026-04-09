from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .syntax import ProgramSpec


@dataclass(slots=True)
class CacheHit:
    key: str
    payload: dict[str, Any]
    metadata: dict[str, Any]


class TypedIRCache:
    def __init__(self, root: str | Path | None = None, *, ttl_hours: int = 168, max_entries: int = 256) -> None:
        self.root = Path(root) if root else Path.cwd() / ".ellang-cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "typed_ir_cache.json"
        self.stats_path = self.root / "stats.json"
        self.ttl = timedelta(hours=ttl_hours)
        self.max_entries = max_entries
        if not self.index_path.exists():
            self.index_path.write_text("{}", encoding="utf-8")
        if not self.stats_path.exists():
            self.stats_path.write_text(json.dumps({"hits": 0, "misses": 0, "stores": 0, "evictions": 0}, indent=2), encoding="utf-8")

    def make_key(self, spec: ProgramSpec, *, planner_profile: str, planner_version: str) -> str:
        canonical = canonicalize_spec(spec)
        canonical["planner_profile"] = planner_profile
        canonical["planner_version"] = planner_version
        digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def load(self, key: str) -> CacheHit | None:
        index = self._read_index()
        entry = index.get(key)
        if not isinstance(entry, dict):
            self._bump_stat("misses")
            return None
        metadata = dict(entry.get("metadata", {}))
        if self._expired(metadata):
            index.pop(key, None)
            self._write_index(index)
            self._bump_stat("evictions")
            self._bump_stat("misses")
            return None
        self._bump_stat("hits")
        metadata["last_accessed_utc"] = _now_iso()
        entry["metadata"] = metadata
        self._write_index(index)
        payload = entry.get("payload", {})
        return CacheHit(key=key, payload=dict(payload) if isinstance(payload, dict) else {}, metadata=metadata)

    def store(self, key: str, payload: dict[str, Any], *, canonical: dict[str, Any] | None = None) -> None:
        index = self._read_index()
        index[key] = {
            "payload": canonicalize_payload(payload),
            "metadata": {
                "created_at_utc": _now_iso(),
                "last_accessed_utc": _now_iso(),
                "canonical_spec": canonical or {},
            },
        }
        index = self._prune_index(index)
        self._write_index(index)
        self._bump_stat("stores")

    def stats(self) -> dict[str, Any]:
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        hits = int(data.get("hits", 0))
        misses = int(data.get("misses", 0))
        total = hits + misses
        data["entries"] = len(self._read_index())
        data["hit_rate"] = round(hits / total, 4) if total else 0.0
        data["ttl_hours"] = int(self.ttl.total_seconds() // 3600)
        data["max_entries"] = self.max_entries
        return data

    def _expired(self, metadata: dict[str, Any]) -> bool:
        created = metadata.get("created_at_utc")
        if not isinstance(created, str):
            return False
        try:
            created_at = datetime.fromisoformat(created)
        except ValueError:
            return False
        return datetime.now(timezone.utc) - created_at > self.ttl

    def _read_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        return data if isinstance(data, dict) else {}

    def _write_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _bump_stat(self, key: str) -> None:
        stats = self.stats()
        stats[key] = int(stats.get(key, 0)) + 1
        self.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _prune_index(self, index: dict[str, Any]) -> dict[str, Any]:
        if len(index) <= self.max_entries:
            return index
        sortable: list[tuple[str, str]] = []
        for key, entry in index.items():
            metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            accessed = metadata.get("last_accessed_utc") if isinstance(metadata, dict) else None
            sortable.append((key, accessed if isinstance(accessed, str) else ""))
        sortable.sort(key=lambda item: item[1])
        while len(index) > self.max_entries and sortable:
            victim, _ = sortable.pop(0)
            index.pop(victim, None)
            self._bump_stat("evictions")
        return index


def canonicalize_spec(spec: ProgramSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "intent": " ".join(spec.intent.lower().split()),
        "inputs": dict(sorted(spec.inputs.items())),
        "outputs": dict(sorted(spec.outputs.items())),
        "constraints": dict(sorted(spec.constraints.items())),
        "objects": {name: dict(sorted(obj.fields.items())) for name, obj in sorted(spec.objects.items())},
        "project": dict(sorted(spec.project.items())),
        "modules": sorted(spec.modules.keys()),
        "flow_summary": _normalize(_flow_summary(spec)),
    }


def canonicalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _normalize(payload)


def _flow_summary(spec: ProgramSpec) -> list[dict[str, Any]]:
    from .models.backend import _program_flow_summary

    return _program_flow_summary(spec)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
