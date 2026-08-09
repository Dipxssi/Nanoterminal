import argparse
import time
from logger import TrajectoryLogger
from rich.console import Console
from executor import run_command
from llm import ask_gemini

# Safely initialize Langfuse if installed and configured
try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except Exception:
    langfuse = None

console = Console()


def run_task(task_prompt: str, max_turns: int = 5) -> bool:
    console.print(
        f"[bold blue]Autopilot Task Started:[/bold blue] {task_prompt}\n"
    )

    logger = TrajectoryLogger()
    logger.set_goal(task_prompt)

    # 1. Initialize Langfuse Trace inside the task runner
    trace = None
    if langfuse:
        trace = langfuse.trace(
            name="nanoterminal_task",
            input=task_prompt,
            metadata={"agent": "NanoTerminal", "model": "gemini-2.5-flash"},
        )

    history = [f"User Goal: {task_prompt}"]

    for turn in range(max_turns):
        console.print(f"[dim]--- Turn {turn + 1} of {max_turns} ---[/dim]")

        start_time = time.time()
        try:
            cmd = ask_gemini(history).strip()
        except Exception as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
            logger.finalize(status="LLM_ERROR")
            if trace:
                trace.update(output={"error": str(e)}, status_message="LLM_ERROR")
            return False

        console.print(f"[dim]Executing:[/dim] [cyan]{cmd}[/cyan]")
        history.append(f"Agent Action: {cmd}")

        stdout, stderr, code = run_command(cmd)
        elapsed = time.time() - start_time

        # Log turn locally
        logger.log_turn(
            turn_num=turn + 1,
            command=cmd,
            stdout=stdout,
            stderr=stderr,
            exit_code=code,
            latency_seconds=elapsed,
        )

        # 2. Log turn span to Langfuse
        if trace:
            trace.span(
                name=f"turn_{turn + 1}",
                input={"cmd": cmd},
                output={"stdout": stdout, "stderr": stderr, "exit_code": code},
            )

        if code == 0:
            console.print("[bold green]✓ Command Succeeded[/bold green]")
            if stdout:
                console.print(f"[green]{stdout.strip()}[/green]")

            output_msg = f"Command Succeeded (Exit Code 0).\nSTDOUT:\n{stdout if stdout else '(no output)'}"
            history.append(f"Observation:\n{output_msg}")

            console.print(
                "\n[bold green]Task Execution Completed Successfully.[/bold green]"
            )
            logger.finalize(status="SUCCESS")
            if trace:
                trace.update(output={"status": "SUCCESS"})
            return True
        else:
            console.print(f"[bold red]✗ Failed (Exit Code {code})[/bold red]")
            if stderr:
                console.print(f"[red]{stderr.strip()}[/red]")

            output_msg = f"Command Failed (Exit Code {code}).\nSTDERR:\n{stderr}"
            history.append(f"Observation:\n{output_msg}")

    console.print("\n[bold red]Task hit max turns without succeeding.[/bold red]")
    logger.finalize(status="MAX_TURNS_EXCEEDED")
    if trace:
        trace.update(output={"status": "MAX_TURNS_EXCEEDED"})
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoTerminal Non-Interactive Runner"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Task prompt to execute"
    )
    args = parser.parse_args()

    run_task(args.prompt)