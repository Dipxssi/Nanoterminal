import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.table import Table

from memory.engine import MemoryEngine
from memory.schemas import MemoryRecord

console = Console()


def run_benchmark():
    db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    q_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    mock_llm = lambda prompt: '[{"text": "Project uses Poetry for package management", "memory_type": "fact"}]'
    engine = MemoryEngine(llm_client=mock_llm, db_path=db_tmp, qtable_path=q_tmp)

    # Seed 12 constraint records into the store
    for i in range(12):
        emb = engine.embedder.embed_text(f"Dummy rule #{i}")
        engine.store.save_record(
            MemoryRecord(
                text=f"Rule #{i}: Set environment variable VAR_{i}=true",
                memory_type="constraint",
                embedding=emb,
            )
        )

    # Representative developer session workload
    workload = [
        ("ls -la", "command", 0),
        ("pwd", "command", 0),
        ("git status", "command", 0),
        ("how is python packaging configured?", "chat_query", 0),
        ("poetry run pytest", "command", 1),
        ("fix test failures with poetry", "error_recovery", 1),
        ("git commit -m 'fix'", "command", 0),
        ("git push origin main", "command", 0),
    ]

    naive_tokens = 0
    memcon_tokens = 0

    table = Table(title="MemCon vs. Naive Retrieval Benchmark", show_header=True)
    table.add_column("Turn / Prompt", style="cyan", width=38)
    table.add_column("Naive Strategy", style="red")
    table.add_column("MemCon Action", style="green")
    table.add_column("Tokens Saved", style="yellow", justify="right")

    for prompt, intent, exit_code in workload:
        # Naive baseline: blindly injects top 5 records (~250 tokens) on every single turn
        naive_tokens += 250

        # MemCon controller: dynamically chooses whether and how much to retrieve
        ctx, action = engine.prepare_context(prompt)
        tokens_used = len(ctx.split()) * 2 if ctx else 0
        memcon_tokens += tokens_used

        saved = max(0, 250 - tokens_used)
        table.add_row(
            prompt,
            "RETRIEVE (k=5)",
            f"{action.op.value}:{action.label}",
            f"+{saved}",
        )

        engine.observe_turn("user", prompt)
        engine.observe_turn("assistant", f"Executed {prompt}", exit_code=exit_code)

    engine.complete_task(success=True)
    engine.shutdown()

    console.print(table)

    reduction = ((naive_tokens - memcon_tokens) / naive_tokens) * 100
    console.print(f"\n[bold green]Total Naive Tokens:[/bold green] {naive_tokens}")
    console.print(f"[bold green]Total MemCon Tokens:[/bold green] {memcon_tokens}")
    console.print(f"[bold bold cyan]Overall Token Reduction:[/bold cyan] {reduction:.1f}%\n")


if __name__ == "__main__":
    run_benchmark()