

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


STEP_EARLY_LT = 8
STEP_MID_LT = 18
LEARNING_COLD_MAX_TASKS = 15
STUCK_FAIL_THRESHOLD = 2
MEM_SIZE_DIVISOR = 10
MEM_SIZE_BIN_CAP = 5
CWD_DIVISOR = 3
CWD_BIN_CAP = 4

# ---------------------------------------------------------------------------
# Intent / phase vocabularies
# ---------------------------------------------------------------------------


class IntentType(str, Enum):
    COMMAND = "command"
    ERROR_RECOVERY = "error_recovery"
    CHAT_QUERY = "chat_query"


class StepPhase(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"


class LearningPhase(str, Enum):
    COLD = "cold"
    WARM = "warm"


class MemoryOp(str, Enum):
    RETRIEVE = "RETRIEVE"
    PLANINJECT = "PLANINJECT"
    RE_RETRIEVE = "RE_RETRIEVE"
    CONSOLIDATE = "CONSOLIDATE"
    FORGET = "FORGET"
    NOOP = "NOOP"


# ---------------------------------------------------------------------------
# Action space (MemCon Appendix C, Table 7 - preset 'default')
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryAction:
    """A memory operation together with its retrieval parameters."""

    op: MemoryOp
    top_k: Optional[int] = None
    insight_k: Optional[int] = None
    hop: Optional[int] = None
    label: str = ""

    def to_key(self) -> str:
        return f"{self.op.value}:{self.label or '-'}:{self.top_k}:{self.insight_k}:{self.hop}"


DEFAULT_ACTIONS: tuple[MemoryAction, ...] = (
    MemoryAction(MemoryOp.RETRIEVE, top_k=1, insight_k=3, hop=1, label="shallow"),
    MemoryAction(MemoryOp.RETRIEVE, top_k=2, insight_k=5, hop=1, label="medium"),
    MemoryAction(MemoryOp.RETRIEVE, top_k=3, insight_k=8, hop=2, label="deep"),
    MemoryAction(MemoryOp.PLANINJECT, top_k=1, insight_k=3, label="plan"),
    MemoryAction(MemoryOp.RE_RETRIEVE, top_k=2, insight_k=5, hop=2, label="alt"),
    MemoryAction(MemoryOp.CONSOLIDATE, label="maintain"),
    MemoryAction(MemoryOp.FORGET, label="evict"),
    MemoryAction(MemoryOp.RETRIEVE, top_k=1, insight_k=2, hop=0, label="insight"),
    MemoryAction(MemoryOp.NOOP, label="skip"),
)

# Warm-start priors Q0(s, a) by operation family (Table 6).
ACTION_PRIORS: dict[MemoryOp, float] = {
    MemoryOp.RETRIEVE: 0.5,
    MemoryOp.PLANINJECT: 0.3,
    MemoryOp.RE_RETRIEVE: 0.1,
    MemoryOp.CONSOLIDATE: 0.0,
    MemoryOp.FORGET: -0.1,
    MemoryOp.NOOP: -0.2,
}


def _index_actions_by_op(
    actions: tuple[MemoryAction, ...],
) -> dict[MemoryOp, tuple[MemoryAction, ...]]:
    grouped: dict[MemoryOp, tuple[MemoryAction, ...]] = {}
    for action in actions:
        grouped[action.op] = grouped.get(action.op, ()) + (action,)
    return grouped


_ACTIONS_BY_OP = _index_actions_by_op(DEFAULT_ACTIONS)


# ---------------------------------------------------------------------------
# Hashable controller state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerState:
    """Discretized MemCon state phi(s) for a terminal session."""

    intent_type: str
    step_phase: str
    is_stuck: bool
    cwd_bin: int
    mem_size_bin: int
    store_empty: bool
    plan_available: bool
    learning_phase: str

    def to_key(self) -> str:
        stuck = "1" if self.is_stuck else "0"
        plan = "1" if self.plan_available else "0"
        filled = "0" if self.store_empty else "1"
        return (
            f"{self.intent_type}:{self.step_phase}:{stuck}:"
            f"{self.cwd_bin}:{self.mem_size_bin}:{filled}:{plan}:{self.learning_phase}"
        )


# ---------------------------------------------------------------------------
# Shell-intent classifier
# ---------------------------------------------------------------------------

_QUESTION_PREFIXES = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "whose",
    "whom",
    "can",
    "could",
    "should",
    "would",
    "is",
    "are",
    "do",
    "does",
    "did",
    "please",
    "explain",
    "describe",
    "tell",
    "help",
)

_UNAMBIGUOUS_BUILTINS = frozenset({
    ".",
    "alias",
    "bg",
    "cd",
    "clear",
    "dirs",
    "echo",
    "exit",
    "export",
    "fg",
    "getopts",
    "history",
    "kill",
    "logout",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "source",
    "ulimit",
    "unalias",
    "unset",
})

_AMBIGUOUS_BUILTINS = frozenset({
    "break",
    "builtin",
    "command",
    "continue",
    "declare",
    "enable",
    "eval",
    "exec",
    "false",
    "hash",
    "help",
    "jobs",
    "let",
    "local",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "test",
    "time",
    "times",
    "trap",
    "true",
    "type",
    "typeset",
    "umask",
    "wait",
})

_WRAPPERS = frozenset({
    "sudo",
    "doas",
    "nohup",
    "nice",
    "env",
    "command",
    "builtin",
    "stdbuf",
    "timeout",
    "time",
    "watch",
})

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PIPE = re.compile(r"\|\||&&|(?<!\|)\|(?!\|)")
_REDIRECT = re.compile(r">>|2>>|2>|\d>")
_SUBST = re.compile(r"\$\(")
_SEQ = re.compile(r";\s+\S")
_BACKTICK = re.compile(r"`")
_FLAG_TOKEN = re.compile(r"^-[-\w]")
_PATH_PREFIXES = ("./", "../", "/", "~/", ".\\", "..\\", "~\\")
_PROMPT_PREFIX = re.compile(r"^[#$>]\s+")
_LEADING_FENCE = re.compile(r"^`+")
_TRAILING_FENCE = re.compile(r"`+$")


def _has_strong_shell_syntax(text: str) -> bool:
    return bool(
        _PIPE.search(text)
        or _REDIRECT.search(text)
        or _SUBST.search(text)
        or _SEQ.search(text)
        or _BACKTICK.search(text)
    )


def _normalize_query(query: Optional[str]) -> str:
    if query is None:
        return ""
    text = str(query).strip()
    if not text:
        return ""
    text = _PROMPT_PREFIX.sub("", text)
    text = _LEADING_FENCE.sub("", text)
    text = _TRAILING_FENCE.sub("", text).strip()
    return text


def _looks_like_question(query: str) -> bool:
    lowered = query.lower()
    if lowered.endswith("?"):
        return True
    first = lowered.split(None, 1)[0] if lowered else ""
    return first in _QUESTION_PREFIXES or lowered.startswith(("tell me", "help me"))


def _has_drive_path(token: str) -> bool:
    return len(token) >= 3 and token[1] == ":" and token[0].isalpha() and token[2] in {"\\", "/"}


def _looks_like_path(token: str) -> bool:
    return (
        token.startswith(_PATH_PREFIXES)
        or _has_drive_path(token)
        or ("/" in token and not token.startswith("http"))
        or "\\" in token
    )


def _bash_which(token: str) -> bool:
    if not token or token in {".", ".."}:
        return False
    if shutil.which(token) is not None:
        return True

    bash = shutil.which("bash")
    if not bash:
        return False

    bash_dir = Path(bash).resolve().parent
    candidates = (
        bash_dir / token,
        bash_dir / f"{token}.exe",
        bash_dir.parent / "usr" / "bin" / token,
        bash_dir.parent / "usr" / "bin" / f"{token}.exe",
        bash_dir.parent / "bin" / token,
        bash_dir.parent / "bin" / f"{token}.exe",
    )
    return any(path.is_file() for path in candidates)


def _remainder_looks_like_shell(tokens: list[str]) -> bool:
    if len(tokens) <= 1:
        return False
    rest = tokens[1:]
    return any(
        _FLAG_TOKEN.match(tok)
        or _looks_like_path(tok)
        or tok.startswith("$")
        or "=" in tok
        or tok in {"|", "||", "&&", ";", ">", ">>", "<"}
        for tok in rest
    )


def is_shell_command(query: str) -> bool:
    clean = _normalize_query(query)
    if not clean:
        return False

    if _looks_like_question(clean) and not _has_strong_shell_syntax(clean):
        if not _looks_like_path(clean.split()[0]):
            return False

    if _has_strong_shell_syntax(clean):
        return True

    tokens = clean.split()
    while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
        if len(tokens) == 1:
            return True
        tokens = tokens[1:]
    if not tokens:
        return False

    while len(tokens) > 1 and tokens[0] in _WRAPPERS:
        tokens = tokens[1:]

    first = os.path.basename(tokens[0].replace("\\", "/"))
    if not first:
        return False

    if _looks_like_path(tokens[0]) or _looks_like_path(first):
        return True

    if first in _UNAMBIGUOUS_BUILTINS:
        return True

    if first in _AMBIGUOUS_BUILTINS:
        return len(tokens) == 1 or _remainder_looks_like_shell(tokens)

    return _bash_which(first)


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


def _clamp_non_negative(value: object, default: int = 0) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, number)


def bin_step_phase(step_index: int) -> StepPhase:
    if step_index < STEP_EARLY_LT:
        return StepPhase.EARLY
    if step_index < STEP_MID_LT:
        return StepPhase.MID
    return StepPhase.LATE


def bin_learning_phase(task_index: int) -> LearningPhase:
    if task_index <= LEARNING_COLD_MAX_TASKS:
        return LearningPhase.COLD
    return LearningPhase.WARM


def bin_mem_size(total_records: int) -> int:
    return min(total_records // MEM_SIZE_DIVISOR, MEM_SIZE_BIN_CAP)


def bin_cwds(unique_cwds: int) -> int:
    return min(unique_cwds // CWD_DIVISOR, CWD_BIN_CAP)


def _is_stuck(
    consecutive_failures: int,
    last_command: Optional[str],
    current_command: Optional[str],
) -> bool:
    if consecutive_failures >= STUCK_FAIL_THRESHOLD:
        return True
    if last_command is None or current_command is None:
        return False
    left = last_command.strip()
    right = current_command.strip()
    return bool(left) and left == right


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_state(
    user_query: str,
    total_records_in_store: int,
    last_exit_code: Optional[int] = None,
    consecutive_failures: int = 0,
    *,
    step_index: int = 0,
    last_command: Optional[str] = None,
    current_command: Optional[str] = None,
    plan_available: bool = False,
    task_index: int = 0,
    unique_cwds: int = 0,
) -> ControllerState:
    records = _clamp_non_negative(total_records_in_store)
    fails = _clamp_non_negative(consecutive_failures)
    step = _clamp_non_negative(step_index)
    tasks = _clamp_non_negative(task_index)
    cwds = _clamp_non_negative(unique_cwds)
    clean_query = _normalize_query(user_query)

    if last_exit_code is not None and last_exit_code != 0:
        intent = IntentType.ERROR_RECOVERY
    elif is_shell_command(clean_query):
        intent = IntentType.COMMAND
    else:
        intent = IntentType.CHAT_QUERY

    action_for_stuck = current_command if current_command is not None else clean_query
    return ControllerState(
        intent_type=intent.value,
        step_phase=bin_step_phase(step).value,
        is_stuck=_is_stuck(fails, last_command, action_for_stuck),
        cwd_bin=bin_cwds(cwds),
        mem_size_bin=bin_mem_size(records),
        store_empty=records == 0,
        plan_available=bool(plan_available),
        learning_phase=bin_learning_phase(tasks).value,
    )


def allowable_actions(state: ControllerState) -> tuple[MemoryAction, ...]:
    allowed: list[MemoryAction] = list(_ACTIONS_BY_OP[MemoryOp.NOOP])

    if state.plan_available:
        allowed.extend(_ACTIONS_BY_OP[MemoryOp.PLANINJECT])

    if not state.store_empty:
        allowed.extend(_ACTIONS_BY_OP[MemoryOp.RETRIEVE])
        allowed.extend(_ACTIONS_BY_OP[MemoryOp.CONSOLIDATE])
        allowed.extend(_ACTIONS_BY_OP[MemoryOp.FORGET])
        if state.is_stuck:
            allowed.extend(_ACTIONS_BY_OP[MemoryOp.RE_RETRIEVE])

    seen: set[str] = set()
    ordered: list[MemoryAction] = []
    for action in allowed:
        key = action.to_key()
        if key not in seen:
            seen.add(key)
            ordered.append(action)
    return tuple(ordered)


def prior_for(action: MemoryAction) -> float:
    return ACTION_PRIORS[action.op]