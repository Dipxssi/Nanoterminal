"""Dual memory cost bench: Lychee write calls + MemCon read tokens."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.table import Table

from memory.engine import MemoryEngine
from memory.schemas import MemoryRecord
from memory.segmenter import EXTRACTION_SYSTEM_PROMPT

console = Console()

# Approximate tokens for naive "always retrieve top-5" injection.
NAIVE_RETRIEVE_TOKENS_PER_TURN = 250

WORKLOAD: list[tuple[str, str, int]] = [
    ("ls -la", "command", 0),
    ("pwd", "command", 0),
    ("git status", "command", 0),
    ("how is python packaging configured?", "chat_query", 0),
    ("poetry run pytest", "command", 1),
    ("fix test failures with poetry", "error_recovery", 1),
    ("git commit -m 'fix'", "command", 0),
    ("git push origin main", "command", 0),
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate used by both read and write sides."""
    if not text or not text.strip():
        return 0
    return max(1, len(text.split()) * 2)


class CountingLLM:
    """Wraps extract LLM and records call count + prompt tokens."""

    def __init__(self, respond: Callable[[str], str] | None = None):
        self.calls = 0
        self.prompt_tokens = 0
        self._respond = respond or (
            lambda _prompt: (
                '[{"text": "Project uses Poetry for package management",'
                ' "memory_type": "fact"}]'
            )
        )

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompt_tokens += estimate_tokens(prompt)
        return self._respond(prompt)


def _eager_extract_prompt(user_text: str, assistant_text: str) -> str:
    transcript = (
        f"Turn 0 [USER]: {user_text}\n"
        f"Turn 1 [ASSISTANT]: {assistant_text}"
    )
    return f"{EXTRACTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nJSON Output:"


def _seed_store(engine: MemoryEngine, n: int = 12) -> None:
    for i in range(n):
        emb = engine.embedder.embed_text(f"Dummy rule #{i}")
        engine.store.save_record(
            MemoryRecord(
                text=f"Rule #{i}: Set environment variable VAR_{i}=true",
                memory_type="constraint",
                embedding=emb,
            )
        )


def _measure_read(engine: MemoryEngine) -> dict[str, Any]:
    naive_tokens = 0
    memcon_tokens = 0
    rows: list[tuple[str, str, str, int]] = []

    for prompt, _intent, exit_code in WORKLOAD:
        naive_tokens += NAIVE_RETRIEVE_TOKENS_PER_TURN
        ctx, action = engine.prepare_context(prompt)
        tokens_used = estimate_tokens(ctx)
        memcon_tokens += tokens_used
        rows.append(
            (
                prompt,
                "RETRIEVE (k=5)",
                f"{action.op.value}:{action.label}",
                max(0, NAIVE_RETRIEVE_TOKENS_PER_TURN - tokens_used),
            )
        )
        # Observe without counting write LLM here — write measured separately.
        engine.observe_turn("user", prompt)
        engine.observe_turn("assistant", f"Executed {prompt}", exit_code=exit_code)

    engine.complete_task(success=True)
    reduction = (
        ((naive_tokens - memcon_tokens) / naive_tokens) * 100 if naive_tokens else 0.0
    )
    return {
        "naive_tokens": naive_tokens,
        "memcon_tokens": memcon_tokens,
        "reduction_pct": reduction,
        "rows": rows,
    }


def _measure_write() -> dict[str, Any]:
    """Compare eager per-exchange extract vs Lychee segment extract."""
    db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    q_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    plans_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    lychee_llm = CountingLLM()
    engine = MemoryEngine(
        llm_client=lychee_llm,
        db_path=db_tmp,
        qtable_path=q_tmp,
        plans_path=plans_tmp,
    )

    eager_calls = 0
    eager_tokens = 0

    for prompt, _intent, exit_code in WORKLOAD:
        assistant = f"Executed {prompt}"
        # Eager baseline: one extract LLM call after every user/assistant exchange.
        eager_calls += 1
        eager_tokens += estimate_tokens(_eager_extract_prompt(prompt, assistant))

        engine.observe_turn("user", prompt)
        engine.observe_turn("assistant", assistant, exit_code=exit_code)

    engine.shutdown()  # flush any remaining segment

    lychee_calls = lychee_llm.calls
    lychee_tokens = lychee_llm.prompt_tokens
    reduction = (
        ((eager_tokens - lychee_tokens) / eager_tokens) * 100 if eager_tokens else 0.0
    )
    return {
        "eager_extract_calls": eager_calls,
        "lychee_extract_calls": lychee_calls,
        "eager_tokens": eager_tokens,
        "lychee_tokens": lychee_tokens,
        "reduction_pct": reduction,
    }


def run_dual_benchmark() -> dict[str, Any]:
    """Return structured dual metrics for tests and CLI."""
    db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    q_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    plans_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    # Mute write LLM during read measurement (observe still runs Lychee path).
    read_llm = CountingLLM()
    engine = MemoryEngine(
        llm_client=read_llm,
        db_path=db_tmp,
        qtable_path=q_tmp,
        plans_path=plans_tmp,
    )
    _seed_store(engine)
    read = _measure_read(engine)
    engine.shutdown()

    write = _measure_write()
    return {
        "workload_turns": len(WORKLOAD),
        "read": read,
        "write": write,
    }


def run_benchmark() -> dict[str, Any]:
    """CLI entry: print dual tables and return the same metrics dict."""
    result = run_dual_benchmark()
    read = result["read"]
    write = result["write"]

    read_table = Table(
        title="Read path · MemCon vs Naive Retrieval",
        show_header=True,
    )
    read_table.add_column("Turn / Prompt", style="cyan", width=38)
    read_table.add_column("Naive Strategy", style="red")
    read_table.add_column("MemCon Action", style="green")
    read_table.add_column("Tokens Saved", style="yellow", justify="right")
    for prompt, naive, action, saved in read["rows"]:
        read_table.add_row(prompt, naive, action, f"+{saved}")
    console.print(read_table)

    write_table = Table(
        title="Write path · Lychee vs Eager Extract",
        show_header=True,
    )
    write_table.add_column("Metric", style="cyan")
    write_table.add_column("Eager (per exchange)", style="red", justify="right")
    write_table.add_column("Lychee (segment)", style="green", justify="right")
    write_table.add_row(
        "Extract LLM calls",
        str(write["eager_extract_calls"]),
        str(write["lychee_extract_calls"]),
    )
    write_table.add_row(
        "Construction tokens (est.)",
        str(write["eager_tokens"]),
        str(write["lychee_tokens"]),
    )
    console.print(write_table)

    console.print(
        f"\n[bold]Read[/bold]  naive={read['naive_tokens']}  "
        f"memcon={read['memcon_tokens']}  "
        f"reduction={read['reduction_pct']:.1f}%"
    )
    console.print(
        f"[bold]Write[/bold] eager_calls={write['eager_extract_calls']}  "
        f"lychee_calls={write['lychee_extract_calls']}  "
        f"eager_tok={write['eager_tokens']}  "
        f"lychee_tok={write['lychee_tokens']}  "
        f"reduction={write['reduction_pct']:.1f}%\n"
    )
    return result


if __name__ == "__main__":
    run_benchmark()
