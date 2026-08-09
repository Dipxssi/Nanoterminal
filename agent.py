from rich.console import Console
from rich.panel import Panel
from executor import run_command
from llm import ask_gemini
from guardrails import analyze_command_risk
from logger import TrajectoryLogger

console = Console()

console.print(
    Panel(
        "[bold green]NanoTerminal Active[/bold green]\nType [bold cyan]'exit'[/bold cyan] or [bold cyan]'quit'[/bold cyan] to stop.",
        title="Welcome",
    )
)

history = []

while True:
    try:
        goal = console.input("\n[bold magenta]nano>[/bold magenta] ")

        if goal.lower().strip() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not goal.strip():
            continue

        logger = TrajectoryLogger()
        logger.set_goal(goal)

        history.append(f"User Goal: {goal}")

        for attempt in range(3):
            try:
                cmd = ask_gemini(history).strip()
            except Exception as e:
                console.print(f"[bold red]LLM Error: [/bold red] {e}")
                logger.finalize(status="LLM_ERROR")
                break

            console.print(f"[dim]Executing:[/dim] [cyan]{cmd}[/cyan]")

            # High-risk guardrail check
            is_high_risk, reason = analyze_command_risk(cmd)

            if is_high_risk:
                console.print(
                    f"[bold yellow]⚠️ HIGH-RISK COMMAND DETECTED ({reason})[/bold yellow]"
                )
                confirm = console.input(
                    "[bold red]Do you want to execute this command? [y/N]: [/bold red]"
                )

                if confirm.lower().strip() not in ["y", "yes"]:
                    console.print(
                        "[bold yellow]Command execution cancelled by user.[/bold yellow]"
                    )
                    history.append(f"Agent Action: {cmd}")
                    history.append(
                        "Observation: Command cancelled by user due to high-risk safety guardrail."
                    )
                    logger.finalize(status="CANCELLED_BY_USER")
                    break

            history.append(f"Agent Action: {cmd}")
            stdout, stderr, code = run_command(cmd)

            logger.log_turn(
                turn_num=attempt + 1,
                command=cmd,
                stdout=stdout,
                stderr=stderr,
                exit_code=code,
                latency_seconds=0.0,
                is_high_risk=is_high_risk,
            )

            if code == 0:
                console.print("[bold green]✓ Command Succeeded[/bold green]")
                if stdout:
                    console.print(f"[green]{stdout}[/green]")

                output_msg = f"Command Succeeded (Exit Code 0).\nSTDOUT:\n{stdout if stdout else '(no output)'}"
                history.append(f"Observation:\n{output_msg}")
                logger.finalize(status="SUCCESS")
                break
            else:
                console.print(
                    f"[bold red]✗ Failed (Exit Code {code})[/bold red]"
                )
                if stderr:
                    console.print(f"[red]{stderr}[/red]")

                output_msg = (
                    f"Command Failed (Exit Code {code}).\nSTDERR:\n{stderr}"
                )
                history.append(f"Observation:\n{output_msg}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Type 'exit' to quit.[/yellow]")