"""MemoryEngine: MemCon read controller + Lychee write loop."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, Tuple

from memory.buffer import SessionBuffer
from memory.controller import MemConBandit
from memory.embeddings import EmbeddingModel
from memory.plans import PlanIndex
from memory.schemas import MemoryRecord
from memory.segmenter import LycheeSegmenter
from memory.state import (
    IntentType,
    MemoryAction,
    MemoryOp,
    extract_state,
    is_shell_command,
)
from memory.store import MemoryStore

_INSIGHT_TYPES = ("constraint", "failure_pattern", "preference", "fact")
_ALT_QUERY_SUFFIX = " alternative approach different strategy recovery fix"


class MemoryEngine:
    def __init__(
        self,
        llm_client,
        db_path: Optional[str] = None,
        qtable_path: Optional[str] = None,
        plans_path: Optional[str] = None,
    ):
        self.embedder = EmbeddingModel()
        self.store = MemoryStore(db_path=db_path)
        self.buffer = SessionBuffer()
        self.segmenter = LycheeSegmenter(
            embedder=self.embedder,
            store=self.store,
            llm_client=llm_client,
        )
        self.bandit = MemConBandit(persist_path=qtable_path)
        self.plans = PlanIndex(path=plans_path)

        self.step_index = 0
        self.task_index = 0
        self.consecutive_failures = 0
        self.last_exit_code: Optional[int] = None
        self.last_command: Optional[str] = None
        self.current_command: Optional[str] = None
        self.unique_cwds = 1
        self._seen_cwds: Set[str] = set()
        self._episode_commands: list[str] = []
        self._current_goal: str = ""
        self._current_intent: str = IntentType.CHAT_QUERY.value

        self.bandit.begin_episode()

    def note_cwd(self, cwd: str) -> None:
        """Track unique working directories for φ(s).cwd_bin."""
        if not cwd:
            return
        normalized = str(Path(cwd))
        if normalized not in self._seen_cwds:
            self._seen_cwds.add(normalized)
            self.unique_cwds = max(1, len(self._seen_cwds))

    def observe_turn(
        self,
        role: str,
        content: str,
        exit_code: Optional[int] = None,
        command: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> List[MemoryRecord]:
        self.last_exit_code = exit_code
        if exit_code is not None and exit_code != 0:
            self.consecutive_failures += 1
        elif exit_code == 0:
            self.consecutive_failures = 0

        if command:
            self.last_command = self.current_command
            self.current_command = command
            self._episode_commands.append(command)

        if cwd:
            self.note_cwd(cwd)

        emb = self.embedder.embed_text(content)
        self.buffer.add_turn(role=role, content=content, embedding=emb)

        extracted: List[MemoryRecord] = []
        if self.segmenter.should_segment(self.buffer.get_turns()):
            extracted = self.segmenter.extract_and_store(self.buffer.get_turns())
            self.buffer.clear()

        return extracted

    def prepare_context(self, user_query: str) -> Tuple[str, MemoryAction]:
        """MemCon read step: select action and materialize context / maintenance."""
        self.step_index += 1
        self._current_goal = user_query
        store_size = self.store.count()

        # Peek intent for plan lookup without mutating state twice.
        probe = extract_state(
            user_query=user_query,
            total_records_in_store=store_size,
            last_exit_code=self.last_exit_code,
            consecutive_failures=self.consecutive_failures,
        )
        self._current_intent = probe.intent_type
        plan_available = self.plans.has_plan(probe.intent_type)

        state = extract_state(
            user_query=user_query,
            total_records_in_store=store_size,
            last_exit_code=self.last_exit_code,
            consecutive_failures=self.consecutive_failures,
            step_index=self.step_index,
            last_command=self.last_command,
            current_command=self.current_command or user_query,
            plan_available=plan_available,
            task_index=self.task_index,
            unique_cwds=self.unique_cwds,
        )

        action = self.bandit.select_action(state)
        context = self._execute_action(action, user_query)
        return context, action

    def _execute_action(self, action: MemoryAction, user_query: str) -> str:
        op = action.op

        if op is MemoryOp.NOOP:
            return ""

        if op is MemoryOp.CONSOLIDATE:
            self.store.consolidate()
            return ""

        if op is MemoryOp.FORGET:
            self.store.forget()
            return ""

        if op is MemoryOp.PLANINJECT:
            return self.plans.format_injection(self._current_intent)

        if op is MemoryOp.RE_RETRIEVE:
            query = f"{user_query}{_ALT_QUERY_SUFFIX}"
            return self._retrieve_context(action, query, insight_only=False)

        if op is MemoryOp.RETRIEVE:
            insight_only = action.label == "insight" or action.hop == 0
            return self._retrieve_context(action, user_query, insight_only=insight_only)

        return ""

    def _retrieve_context(
        self,
        action: MemoryAction,
        query: str,
        *,
        insight_only: bool,
    ) -> str:
        if self.store.count() == 0:
            return ""

        k = action.top_k or 2
        hop = action.hop if action.hop is not None else 1
        # Deeper hop → slightly lower similarity floor (wider net).
        min_score = {0: 0.35, 1: 0.40, 2: 0.32}.get(hop, 0.40)
        types = _INSIGHT_TYPES if insight_only else None

        query_emb = self.embedder.embed_text(query)
        scored = self.store.retrieve_similar(
            query_emb,
            k=k,
            min_score=min_score,
            memory_types=types,
        )
        if not scored and insight_only:
            # Fall back to any type if no insight hits.
            scored = self.store.retrieve_similar(
                query_emb, k=k, min_score=min_score, memory_types=None
            )
        if not scored:
            return ""

        insight_k = action.insight_k or k
        lines = ["\n[RELEVANT MEMORY CONTEXT]"]
        for record, score in scored[:insight_k]:
            lines.append(
                f"- ({record.memory_type.upper()} · {score:.2f}) {record.text}"
            )
        lines.append("[END MEMORY CONTEXT]\n")
        return "\n".join(lines)

    def complete_task(self, success: bool, goal: Optional[str] = None) -> None:
        if success:
            self._maybe_store_plan(goal or self._current_goal)

        self.bandit.end_episode(success=success, steps=self.step_index)
        self.task_index += 1
        self.step_index = 0
        self.consecutive_failures = 0
        self.last_command = None
        self.current_command = None
        self._episode_commands.clear()
        self.bandit.begin_episode()

    def _maybe_store_plan(self, goal: str) -> None:
        if not goal or not self._episode_commands:
            return
        intent = self._current_intent
        if intent == IntentType.CHAT_QUERY.value and is_shell_command(goal):
            intent = IntentType.COMMAND.value
        steps = self._episode_commands[-8:]
        summary = goal.strip()[:160]
        self.plans.upsert_plan(intent, steps, summary=summary)

    def shutdown(self) -> None:
        if not self.buffer.is_empty():
            try:
                self.segmenter.extract_and_store(self.buffer.get_turns())
            except Exception as exc:
                print(f"[memory] shutdown extract skipped: {exc}")
            self.buffer.clear()
        self.bandit.save()
        self.plans.save()
