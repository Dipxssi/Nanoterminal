import sys
from rich.console import Console
from rich.panel import Panel
import executor
from llm import ask_gemini, ask_gemini_raw
from guardrails import analyze_command_risk
from logger import TrajectoryLogger
from memory.engine import MemoryEngine
from memory.state import MemoryOp

console = Console()

console.print(
    Panel(
        "[bold green]NanoTerminal Active (MemCon + Lychee Memory Enabled)[/bold green]\nType [bold cyan]'exit'[/bold cyan] or [bold cyan]'quit'[/bold cyan] to stop.",
        title="Welcome",
    )
)

# Initialize Unified Memory Engine
memory_engine = MemoryEngine(llm_client=ask_gemini_raw)
history = []

try:
    while True:
        try:
            goal = console.input("\n[bold magenta]nano>[/bold magenta] ")

            if goal.lower().strip() in ["exit", "quit"]:
                console.print("[yellow]Saving memory & goodbye![/yellow]")
                break

            if not goal.strip():
                continue

            logger = TrajectoryLogger()
            logger.set_goal(goal)

            # 1. MemCon Read Step: Retrieve / plan / maintain as needed
            memory_context, action = memory_engine.prepare_context(goal)
            if memory_context:
                console.print(
                    f"[dim blue]🧠 MemCon [{action.op.value}:{action.label}] Context Injected[/dim blue]"
                )
                history.append(memory_context)
            elif action.op in (MemoryOp.CONSOLIDATE, MemoryOp.FORGET):
                console.print(
                    f"[dim blue]🧠 MemCon [{action.op.value}] maintenance[/dim blue]"
                )

            # Observe user turn for Lychee segmentation
            memory_engine.observe_turn(role="user", content=goal)
            history.append(f"User Goal: {goal}")

            task_success = False

            for attempt in range(3):
                # Re-consult MemCon when stuck so RE_RETRIEVE can fire mid-task.
                if memory_engine.consecutive_failures >= 2:
                    stuck_ctx, stuck_action = memory_engine.prepare_context(goal)
                    if stuck_ctx:
                        console.print(
                            f"[dim blue]🧠 MemCon re-read [{stuck_action.op.value}:{stuck_action.label}][/dim blue]"
                        )
                        history.append(stuck_ctx)

                try:
                    cmd_res = ask_gemini(history)
                    if isinstance(cmd_res, tuple):
                        func_name, func_args = cmd_res
                        if func_name == "finish_task":
                            summary = func_args.get("summary", "Done")
                            console.print(f"[bold green]✓ Completed:[/bold green] {summary}")
                            task_success = True
                            break
                        cmd = func_args.get("command", "").strip()
                    else:
                        cmd = str(cmd_res).strip()
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
                stdout, stderr, code = executor.run_command(cmd)

                # Observe tool turn for memory extraction
                memory_engine.observe_turn(
                    role="assistant",
                    content=f"Ran `{cmd}` with exit code {code}.\nOutput: {stdout}\nErrors: {stderr}",
                    exit_code=code,
                    command=cmd,
                    cwd=executor.CURRENT_CWD,
                )

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
                    task_success = True
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

            # Complete MemCon episodic reinforcement learning update
            memory_engine.complete_task(success=task_success, goal=goal)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Type 'exit' to quit.[/yellow]")

finally:
    memory_engine.shutdown()