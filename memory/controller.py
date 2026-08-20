from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from memory.state import (
    DEFAULT_ACTIONS,
    ControllerState,
    MemoryAction,
    MemoryOp,
    allowable_actions,
    prior_for,
)

# GPT-4.1-mini defaults from MemCon Appendix C, Table 6.
ALPHA = 0.15
GAMMA = 0.9
UCB_C = 1.4
R_SUCC = 1.0
R_FAIL = 0.5
LAMBDA_EFF = 0.3
T_MAX = 30
FLUSH_INTERVAL = 5

_CONFIG_DIR = ".nanoterminal"
_QTABLE_NAME = "memcon_qtable.json"


@dataclass(frozen=True)
class _Decision:
    state: ControllerState
    action: MemoryAction


class MemConBandit:
    def __init__(
        self,
        alpha: float = ALPHA,
        gamma: float = GAMMA,
        c_param: float = UCB_C,
        t_max: int = T_MAX,
        flush_interval: int = FLUSH_INTERVAL,
        persist_path: Optional[str] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.c_param = c_param
        self.t_max = max(1, t_max)
        self.flush_interval = max(1, flush_interval)
        self.all_actions = DEFAULT_ACTIONS

        if persist_path is None:
            base_dir = Path.home() / _CONFIG_DIR
            base_dir.mkdir(parents=True, exist_ok=True)
            self.persist_path = str(base_dir / _QTABLE_NAME)
        else:
            self.persist_path = persist_path
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)

        self.q_table: dict[str, dict[str, float]] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self._episode: list[_Decision] = []
        self._updates_since_flush = 0

        self.load()

    def begin_episode(self) -> None:
        self._episode.clear()

    def select_action(self, state: ControllerState) -> MemoryAction:
        """Pick a feasible action via UCB; unvisited arms win, ties broken by prior."""
        state_key = state.to_key()
        self._ensure_state(state_key)

        feasible = allowable_actions(state)
        if not feasible:
            action = next(a for a in self.all_actions if a.op is MemoryOp.NOOP)
            self._record(state, action)
            return action

        unvisited = [
            action
            for action in feasible
            if self.counts[state_key][action.to_key()] == 0
        ]
        if unvisited:
            action = max(unvisited, key=prior_for)
            self._record(state, action)
            return action

        total_visits = self._state_visits(state_key)
        log_n = math.log(max(1, total_visits))
        best_action = feasible[0]
        best_score = float("-inf")
        for action in feasible:
            a_key = action.to_key()
            n_sa = self.counts[state_key][a_key]
            bonus = self.c_param * math.sqrt(log_n / n_sa)
            score = self.q_table[state_key][a_key] + bonus
            if score > best_score:
                best_score = score
                best_action = action

        self._record(state, best_action)
        return best_action

    def end_episode(self, success: bool, steps: int) -> float:
        """Apply reverse-discounted Monte-Carlo updates from the task outcome."""
        reward = self.compute_reward(success, steps)
        episode = list(self._episode)
        length = len(episode)
        for index, decision in enumerate(episode):
            credit = (self.gamma ** (length - index - 1)) * reward
            self._apply_q_update(decision.state, decision.action, credit)

        self._episode.clear()
        self._updates_since_flush += 1
        if self._updates_since_flush >= self.flush_interval:
            self.save()
            self._updates_since_flush = 0
        return reward

    def compute_reward(self, success: bool, steps: int) -> float:
        """Table 6 / Eq. (5): success bonus, failure penalty, step-efficiency term."""
        safe_steps = max(0, int(steps))
        efficiency = LAMBDA_EFF * max(0.0, 1.0 - (safe_steps / self.t_max))
        if success:
            return R_SUCC + efficiency
        return efficiency - R_FAIL

    def save(self) -> None:
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "counts": self.counts,
        }
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def load(self) -> None:
        path = Path(self.persist_path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            self.q_table = {}
            self.counts = {}
            return

        raw_q = data.get("q_table") if isinstance(data, dict) else {}
        raw_n = data.get("counts") if isinstance(data, dict) else {}
        if not isinstance(raw_q, dict):
            raw_q = {}
        if not isinstance(raw_n, dict):
            raw_n = {}

        self.q_table = {}
        self.counts = {}
        for state_key in set(raw_q) | set(raw_n):
            q_row = raw_q.get(state_key) or {}
            n_row = raw_n.get(state_key) or {}
            if not isinstance(q_row, dict):
                q_row = {}
            if not isinstance(n_row, dict):
                n_row = {}
            self.q_table[state_key] = {}
            self.counts[state_key] = {}
            for action in self.all_actions:
                a_key = action.to_key()
                try:
                    self.q_table[state_key][a_key] = float(q_row[a_key]) if a_key in q_row else prior_for(action)
                except (TypeError, ValueError):
                    self.q_table[state_key][a_key] = prior_for(action)
                try:
                    self.counts[state_key][a_key] = max(0, int(n_row.get(a_key, 0)))
                except (TypeError, ValueError):
                    self.counts[state_key][a_key] = 0

    def _ensure_state(self, state_key: str) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if state_key not in self.counts:
            self.counts[state_key] = {}
        for action in self.all_actions:
            a_key = action.to_key()
            if a_key not in self.q_table[state_key]:
                self.q_table[state_key][a_key] = prior_for(action)
            if a_key not in self.counts[state_key]:
                self.counts[state_key][a_key] = 0

    def _state_visits(self, state_key: str) -> int:
        return sum(self.counts[state_key].values())

    def _record(self, state: ControllerState, action: MemoryAction) -> None:
        state_key = state.to_key()
        a_key = action.to_key()
        self._ensure_state(state_key)
        self.counts[state_key][a_key] += 1
        self._episode.append(_Decision(state, action))

    def _apply_q_update(
        self,
        state: ControllerState,
        action: MemoryAction,
        target: float,
    ) -> None:
        state_key = state.to_key()
        a_key = action.to_key()
        self._ensure_state(state_key)
        old_q = self.q_table[state_key][a_key]
        self.q_table[state_key][a_key] = old_q + self.alpha * (target - old_q)
