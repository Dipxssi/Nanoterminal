import argparse
import subprocess
import time

from rich.console import Console

import executor
from llm import ask_gemini, ask_gemini_raw
from logger import TrajectoryLogger
from memory.engine import MemoryEngine
from memory.state import MemoryOp

try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except Exception:
    langfuse = None

console = Console()

__version__ = "0.1.0"


def run_task(
    task_prompt: str,
    max_turns: int = 30,
    log_dir: str = "trajectories",
    debug_memory: bool = False
) -> bool:
    console.print(
        f"[bold blue]Autopilot Task Started:[/bold blue] {task_prompt}\n"
    )

    logger = TrajectoryLogger(log_dir=log_dir)
    logger.set_goal(task_prompt)

    # Initialize Memory Engine
    memory_engine = MemoryEngine(llm_client=ask_gemini_raw)

    trace = None
    if langfuse:
        try:
            trace = langfuse.start_trace(
                name="nanoterminal_task",
                input=task_prompt,
                metadata={"agent": "NanoTerminal", "model": "gemini-2.5-flash"},
            )
        except Exception:
            trace = None

    history = []

    # MemCon context retrieval
    mem_context, action = memory_engine.prepare_context(task_prompt)
    if debug_memory:
        console.print(f"[bold cyan][MEMCON][/bold cyan] Action: {action.op.value} | Label: {action.label}")
        if mem_context:
            console.print(f"[dim cyan]{mem_context.strip()}[/dim cyan]")

    if mem_context:
        history.append(mem_context)

    history.append(f"User Goal: {task_prompt}")
    memory_engine.observe_turn(role="user", content=task_prompt)

    task_success = False

    try:
        for turn in range(max_turns):
            console.print(f"[dim]--- Turn {turn + 1} of {max_turns} ---[/dim]")
            time.sleep(1)

            if memory_engine.consecutive_failures >= 2:
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
                console.print(
                    f"\n[bold green]✓ Agent Finished Task:[/bold green] {summary}"
                )
                logger.finalize(status="SUCCESS")
                task_success = True
                if trace:
                    try:
                        trace.update(output={"status": "SUCCESS", "summary": summary})
                    except Exception:
                        pass
                return True

            cmd = func_args.get("command", "").strip()
            console.print(f"[dim]Executing:[/dim] [cyan]{cmd}[/cyan]")
            history.append(f"Agent Action: {cmd}")

            try:
                stdout, stderr, code = executor.run_command(cmd)
            except subprocess.TimeoutExpired as e:
                stdout = (e.stdout or "") if not isinstance(e.stdout, bytes) else e.stdout.decode("utf-8", errors="replace")
                stderr = f"Command timed out after {e.timeout} seconds."
                code = 124

            elapsed = time.time() - start_time

            memory_engine.observe_turn(
                role="assistant",
                content=f"Ran `{cmd}` -> Exit {code}\nSTDOUT: {stdout}\nSTDERR: {stderr}",
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
                console.print("[bold green]✓ Command Succeeded[/bold green]")
                if stdout:
                    console.print(f"[green]{stdout.strip()}[/green]")
                output_msg = (
                    f"Command Succeeded (Exit Code 0).\nSTDOUT:\n"
                    f"{stdout if stdout else '(no output)'}"
                )
            else:
                console.print(
                    f"[bold red]✗ Command Failed (Exit Code {code})[/bold red]"
                )
                if stderr:
                    console.print(f"[red]{stderr.strip()}[/red]")
                output_msg = f"Command Failed (Exit Code {code}).\nSTDERR:\n{stderr}"

            history.append(f"Observation:\n{output_msg}")

        console.print(
            "\n[bold yellow]Task reached max turns without explicit completion.[/bold yellow]"
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
    parser.add_argument(
        "--prompt", type=str, help="Task prompt to execute"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
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
        run_task(
            args.prompt,
            max_turns=args.max_turns,
            log_dir=args.log_dir,
            debug_memory=args.debug_memory
        )