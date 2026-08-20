"""Lightweight success-plan index for MemCon PLANINJECT."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


_CONFIG_DIR = ".nanoterminal"
_PLANS_NAME = "plans.json"

# Instance-specific tokens → placeholders (MemCon-style generalization).
_GENERIC_PATTERNS = (
    (re.compile(r"\b[A-Za-z]:\\[^\s]+"), "[path]"),
    (re.compile(r"/(?:home|Users|app|tmp|var)/[^\s]+"), "[path]"),
    (re.compile(r"\b\d{1,5}\b"), "[n]"),
    (re.compile(r"\b[0-9a-f]{7,40}\b", re.I), "[id]"),
)


def generalize_step(text: str) -> str:
    out = text.strip()
    for pattern, repl in _GENERIC_PATTERNS:
        out = pattern.sub(repl, out)
    return out


class PlanIndex:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            base = Path.home() / _CONFIG_DIR
            base.mkdir(parents=True, exist_ok=True)
            self.path = str(base / _PLANS_NAME)
        else:
            self.path = path
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._plans: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            self._plans = {}
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._plans = data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError):
            self._plans = {}

    def save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = json.dumps(self._plans, indent=2)
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    def has_plan(self, goal_type: str) -> bool:
        plan = self._plans.get(goal_type)
        return bool(plan and plan.get("steps"))

    def get_plan(self, goal_type: str) -> Optional[dict]:
        plan = self._plans.get(goal_type)
        if not plan or not plan.get("steps"):
            return None
        return plan

    def upsert_plan(
        self,
        goal_type: str,
        steps: list[str],
        summary: str = "",
    ) -> None:
        cleaned = [generalize_step(s) for s in steps if s and s.strip()]
        if not cleaned:
            return
        self._plans[goal_type] = {
            "steps": cleaned[:12],
            "summary": summary.strip(),
        }
        self.save()

    def format_injection(self, goal_type: str) -> str:
        plan = self.get_plan(goal_type)
        if not plan:
            return ""
        lines = [f"\n[Proven plan for '{goal_type}' tasks]"]
        for i, step in enumerate(plan["steps"], start=1):
            lines.append(f"{i}. {step}")
        lines.append("Adapt paths/ids to the current task.\n")
        return "\n".join(lines)
