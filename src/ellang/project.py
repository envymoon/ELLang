from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectSnapshot:
    vcs: str
    branch: str | None
    dirty: bool
    changed_files: list[str]
    suggested_commit_message: str | None
    suggested_version_bump: str | None
    release_notes_hint: str | None


class GitProjectManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def snapshot(self) -> ProjectSnapshot:
        if not (self.root / ".git").exists():
            return ProjectSnapshot("git", None, False, [], None, None, None)

        branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip() or None
        changed = [line.strip() for line in self._run_git(["status", "--short"]).splitlines() if line.strip()]
        version_bump = self._suggest_version_bump(changed)
        return ProjectSnapshot(
            vcs="git",
            branch=branch,
            dirty=bool(changed),
            changed_files=changed,
            suggested_commit_message=(f"ellang: update runtime and compiler across {len(changed)} changed paths" if changed else None),
            suggested_version_bump=version_bump,
            release_notes_hint=(f"Prepare a {version_bump} release note covering runtime, compiler, and model-backend changes." if version_bump else None),
        )

    def _suggest_version_bump(self, changed: list[str]) -> str | None:
        if not changed:
            return None
        if any("syntax.py" in line or "compiler.py" in line or "runtime.py" in line for line in changed):
            return "minor"
        return "patch"

    def _run_git(self, args: list[str]) -> str:
        try:
            completed = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout
