from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(slots=True)
class DebugReport:
    status: str
    summary: str
    suggestions: list[str]
    escalated: bool = False


@dataclass(slots=True)
class ExternalDebugConfig:
    endpoint: str | None = None
    api_key_env: str = "ELLANG_DEBUG_API_KEY"
    model: str = "gpt-5.4"


class AIDebugger:
    def __init__(self, external: ExternalDebugConfig | None = None) -> None:
        self.external = external or ExternalDebugConfig()

    def analyze(self, result: Any) -> DebugReport:
        local = self._local_debug(result)
        if local.status != "needs_escalation":
            return local
        external = self._external_debug(result)
        return external or local

    def _local_debug(self, result: Any) -> DebugReport:
        suggestions: list[str] = []
        trace = result.trace.frames if result.trace else []
        if not result.value:
            suggestions.append("Check whether the flow is missing a final `emit` or output-producing transform.")
        if any("valid" in frame.output_summary for frame in trace):
            suggestions.append("Contracts validated, so inspect branch conditions and loop source bindings next.")
        if not suggestions:
            return DebugReport(
                status="needs_escalation",
                summary="Local debugger could not isolate a precise root cause.",
                suggestions=["Escalate to an external high-capacity debug model with trace and diagnostics."],
            )
        return DebugReport(
            status="resolved_locally",
            summary="Local debugger produced heuristic remediation guidance.",
            suggestions=suggestions,
        )

    def _external_debug(self, result: Any) -> DebugReport | None:
        endpoint = self.external.endpoint or os.getenv("ELLANG_DEBUG_API_ENDPOINT")
        api_key = os.getenv(self.external.api_key_env)
        if not endpoint or not api_key:
            return None

        payload = {
            "model": self.external.model,
            "messages": [
                {"role": "system", "content": "You are a precise debugger for an executable AI programming language. Return JSON only."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "value": result.value,
                            "diagnostics": result.diagnostics,
                            "trace": [frame.__dict__ for frame in (result.trace.frames if result.trace else [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

        content = _extract_message_content(body)
        if not isinstance(content, dict):
            return None
        return DebugReport(
            status="escalated",
            summary=str(content.get("summary", "External debugger responded.")),
            suggestions=[str(item) for item in content.get("suggestions", [])],
            escalated=True,
        )


def _extract_message_content(body: dict[str, Any]) -> dict[str, Any] | None:
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception:
        return None
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None
