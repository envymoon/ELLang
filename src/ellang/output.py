from __future__ import annotations

import json
from typing import Any


def is_error_result(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "error"


def error_message(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    intrinsic = value.get("intrinsic")
    message = value.get("message", "unknown error")
    if intrinsic:
        return f"{intrinsic}: {message}"
    return str(message)


def print_result(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def print_error(value: Any, *, extend_hint: bool = True) -> None:
    print(f"Error: {error_message(value)}")
    if extend_hint:
        print("Hint: rerun with --extend to view the full record.")
