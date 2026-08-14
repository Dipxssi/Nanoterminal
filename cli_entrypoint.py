import argparse
import subprocess
import time

from rich.console import Console

from executor import run_command
from llm import ask_gemini
from logger import TrajectoryLogger

try:
    from langfuse import Langfuse

    langfuse = Langfuse()
except Exception:
    langfuse = None

console = Console()

__version__ = "0.1.0"


def run_task(task_prompt: str, max_turns: int = 30, log_dir: str = "trajectories") -> bool:
    console.print(
        f"[bold blue]Autopilot Task Started:[/bold blue] {task_prompt}\n"
    )

    logger = TrajectoryLogger(log_dir=log_dir)
    logger.set_goal(task_prompt)

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

    history = [f"User Goal: {task_prompt}"]

    for turn in range(max_turns):
        console.print(f"[dim]--- Turn {turn + 1} of {max_turns} ---[/dim]")

        # Pacing delay to respect API rate limits
        time.sleep(1)

        start_time = time.time()
        try:
            func_name, func_args = ask_gemini(history)
        except Exception as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
            logger.finalize(status="LLM_ERROR")
            return False

        # Handle explicit task completion signal
        if func_name == "finish_task":
            summary = func_args.get("summary", "Task completed.")
            console.print(
                f"\n[bold green]✓ Agent Finished Task:[/bold green] {summary}"
            )
            logger.finalize(status="SUCCESS")
            if trace:
                try:
                    trace.update(output={"status": "SUCCESS", "summary": summary})
                except Exception:
                    pass
            return True

        # Handle bash execution command
        cmd = func_args.get("command", "").strip()
        console.print(f"[dim]Executing:[/dim] [cyan]{cmd}[/cyan]")
        history.append(f"Agent Action: {cmd}")

        try:
            stdout, stderr, code = run_command(cmd)
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") if not isinstance(e.stdout, bytes) else e.stdout.decode("utf-8", errors="replace")
            stderr = f"Command timed out after {e.timeout} seconds."
            code = 124
        elapsed = time.time() - start_time

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
        "--version",
        action="version",
        version=f"nanoterminal {__version__}",
    )
    args = parser.parse_args()

    if args.prompt:
        run_task(args.prompt, max_turns=args.max_turns, log_dir=args.log_dir)
