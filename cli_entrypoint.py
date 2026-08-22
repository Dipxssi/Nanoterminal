import argparse
import os
import subprocess
import time

from rich.console import Console

import executor
from llm import ask_gemini, ask_gemini_raw, get_model_name
from logger import TrajectoryLogger
from memory.engine import MemoryEngine

try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except Exception:
    langfuse = None

console = Console()

__version__ = "0.1.0"

DEFAULT_MAX_TURNS = 80

BOOTSTRAP_COMMAND = (
    "pwd; ls -la; echo '---TOOLS---'; "
    "command -v python3 python node npm gcc g++ make git curl wget jq 2>/dev/null; "
    "echo '---LANGS---'; "
    "(python3 --version 2>/dev/null || true); "
    "(node --version 2>/dev/null || true); "
    "echo '---OS---'; uname -a 2>/dev/null || true"
)

STUCK_RECOVERY_HINT = (
    "STUCK RECOVERY: You appear stuck (repeated failures or same command). "
    "Replan from the original goal. Inspect the environment with "
    "`which`/`command -v`, check paths and file contents, and try a different approach. "
    "Do not blindly repeat the same failing command."
)

FINISH_GATE_MESSAGE = (
    "FINISH REJECTED: Premature `finish_task`. Before finishing you MUST verify "
    "against the ORIGINAL user goal:\n"
    "1) Re-read the goal and list each concrete requirement.\n"
    "2) Run checks (tests, file reads, command outputs) that prove each requirement.\n"
    "3) Only then call `finish_task` with a summary of WHAT you verified and HOW.\n"
    "Continue with verification commands now."
)


FINISH_NUDGE = (
    "TIME BUDGET: Few turns remain. If the original goal is already satisfied and "
    "verified, call `finish_task` NOW with a short verification summary. "
    "Do not keep re-checking the same files."
)


def _scaffold_hardened() -> bool:
    raw = os.environ.get("NANOTERMINAL_SCAFFOLD", "hardened").strip().lower()
    return raw not in ("legacy", "0", "false", "off", "no")


def _skip_lychee() -> bool:
    raw = os.environ.get("NANOTERMINAL_SKIP_LYCHEE", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    # Harbor Linux trials: prefer not burning extract quota mid-eval.
    return os.environ.get("NANOTERMINAL_ENV", "").lower() == "linux"


def _bootstrap_environment(history: list[str], debug: bool = False) -> None:
    if debug:
        console.print("[bold cyan][BOOTSTRAP][/bold cyan] Snapshotting environment…")
    try:
        stdout, stderr, code = executor.run_command(BOOTSTRAP_COMMAND)
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "") if not isinstance(e.stdout, bytes) else e.stdout.decode(
            "utf-8", errors="replace"
        )
        stderr = f"Bootstrap timed out after {e.timeout} seconds."
        code = 124

    snapshot = (
        f"Environment bootstrap (exit {code}):\n"
        f"STDOUT:\n{stdout if stdout else '(empty)'}\n"
        f"STDERR:\n{stderr if stderr else '(empty)'}"
    )
    history.append(snapshot)
    if debug:
        console.print(f"[dim cyan]{snapshot[:800]}[/dim cyan]")


def run_task(
    task_prompt: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    log_dir: str = "trajectories",
    debug_memory: bool = False,
) -> bool:
    model_name = get_model_name()
    hardened = _scaffold_hardened()
    console.print(
        f"[bold blue]Autopilot Task Started:[/bold blue] {task_prompt}\n"
        f"[dim]model={model_name} scaffold={'hardened' if hardened else 'legacy'} "
        f"max_turns={max_turns}[/dim]\n"
    )

    logger = TrajectoryLogger(log_dir=log_dir)
    logger.set_goal(task_prompt)

    # Initialize Memory Engine (optionally skip Lychee LLM extract under Harbor)
    memory_engine = MemoryEngine(llm_client=ask_gemini_raw)
    if _skip_lychee():
        memory_engine.segmenter.llm_client = lambda _prompt: ""

    trace = None
    if langfuse:
        try:
            trace = langfuse.start_trace(
                name="nanoterminal_task",
                input=task_prompt,
                metadata={
                    "agent": "NanoTerminal",
                    "model": model_name,
                    "scaffold": "hardened" if hardened else "legacy",
                },
            )
        except Exception:
            trace = None

    history: list[str] = []

    mem_context, action = memory_engine.prepare_context(task_prompt)
    if debug_memory:
        console.print(
            f"[bold cyan][MEMCON][/bold cyan] Action: {action.op.value} | "
            f"Label: {action.label}"
        )
        if mem_context:
            console.print(f"[dim cyan]{mem_context.strip()}[/dim cyan]")

    if mem_context:
        history.append(mem_context)

    history.append(f"User Goal: {task_prompt}")
    memory_engine.observe_turn(role="user", content=task_prompt)

    if hardened:
        _bootstrap_environment(history, debug=debug_memory)

    task_success = False
    finish_attempts = 0
    turns_since_finish_reject = 0

    try:
        for turn in range(max_turns):
            console.print(f"[dim]--- Turn {turn + 1} of {max_turns} ---[/dim]")
            time.sleep(1)

            if turn >= max(1, int(max_turns * 0.7)):
                history.append(FINISH_NUDGE)

            if memory_engine.consecutive_failures >= 2:
                if hardened:
                    history.append(STUCK_RECOVERY_HINT)
                    if debug_memory:
                        console.print(
                            "[bold yellow][STUCK][/bold yellow] Injected recovery hint"
                        )
                stuck_ctx, stuck_action = memory_engine.prepare_context(task_prompt)
                if debug_memory:
                    console.print(
                        f"[bold cyan][MEMCON re-read][/bold cyan] "
                        f"{stuck_action.op.value}:{stuck_action.label}"
                    )
                if stuck_ctx:
                    history.append(stuck_ctx)

            start_time = time.time()
            try:
                func_name, func_args = ask_gemini(history)
            except Exception as e:
                console.print(f"[bold red]LLM Error:[/bold red] {e}")
                logger.finalize(status="LLM_ERROR")
                return False

            if func_name == "finish_task":
                summary = func_args.get("summary", "Task completed.")
                if hardened and finish_attempts == 0:
                    finish_attempts += 1
                    turns_since_finish_reject = 0
                    console.print(
                        "[bold yellow]Finish gated — requiring verification "
                        "against the original goal.[/bold yellow]"
                    )
                    history.append(f"Agent attempted finish_task: {summary}")
                    history.append(FINISH_GATE_MESSAGE)
                    continue

                if hardened and turns_since_finish_reject < 1:
                    # Second finish with no intervening command — reject again.
                    finish_attempts += 1
                    console.print(
                        "[bold yellow]Finish gated — run at least one verification "
                        "command first.[/bold yellow]"
                    )
                    history.append(f"Agent attempted finish_task again: {summary}")
                    history.append(FINISH_GATE_MESSAGE)
                    continue

                console.print(
                    f"\n[bold green][OK] Agent Finished Task:[/bold green] {summary}"
                )
                logger.finalize(status="SUCCESS")
                task_success = True
                if trace:
                    try:
                        trace.update(
                            output={"status": "SUCCESS", "summary": summary}
                        )
                    except Exception:
                        pass
                return True

            cmd = func_args.get("command", "").strip()
            console.print(f"[dim]Executing:[/dim] [cyan]{cmd}[/cyan]")
            history.append(f"Agent Action: {cmd}")
            if finish_attempts > 0:
                turns_since_finish_reject += 1

            try:
                stdout, stderr, code = executor.run_command(cmd)
            except subprocess.TimeoutExpired as e:
                stdout = (
                    (e.stdout or "")
                    if not isinstance(e.stdout, bytes)
                    else e.stdout.decode("utf-8", errors="replace")
                )
                stderr = f"Command timed out after {e.timeout} seconds."
                code = 124

            elapsed = time.time() - start_time

            memory_engine.observe_turn(
                role="assistant",
                content=(
                    f"Ran `{cmd}` -> Exit {code}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                ),
                exit_code=code,
                command=cmd,
                cwd=executor.CURRENT_CWD,
            )

            logger.log_turn(
                turn_num=turn + 1,
                command=cmd,
                stdout=stdout,
                stderr=stderr,
                exit_code=code,
                latency_seconds=elapsed,
            )

            if code == 0:
                console.print("[bold green][OK] Command Succeeded[/bold green]")
                if stdout:
                    console.print(f"[green]{stdout.strip()}[/green]")
                output_msg = (
                    f"Command Succeeded (Exit Code 0).\nSTDOUT:\n"
                    f"{stdout if stdout else '(no output)'}"
                )
            else:
                console.print(
                    f"[bold red][FAIL] Command Failed (Exit Code {code})[/bold red]"
                )
                if stderr:
                    console.print(f"[red]{stderr.strip()}[/red]")
                output_msg = f"Command Failed (Exit Code {code}).\nSTDERR:\n{stderr}"

            history.append(f"Observation:\n{output_msg}")

        console.print(
            "\n[bold yellow]Task reached max turns without explicit completion."
            "[/bold yellow]"
        )
        logger.finalize(status="MAX_TURNS_EXCEEDED")
        return False

    finally:
        memory_engine.complete_task(success=task_success, goal=task_prompt)
        memory_engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoTerminal Non-Interactive Runner"
    )
    parser.add_argument("--prompt", type=str, help="Task prompt to execute")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum agent turns before stopping",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="trajectories",
        help="Directory for trajectory JSON logs",
    )
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Print MemCon retrieval actions and decisions",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"nanoterminal {__version__}",
    )
    args = parser.parse_args()

    if args.prompt:
        ok = run_task(
            args.prompt,
            max_turns=args.max_turns,
            log_dir=args.log_dir,
            debug_memory=args.debug_memory,
        )
        raise SystemExit(0)
