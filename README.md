# NanoTerminal 

An autonomous, lightweight terminal execution agent powered by **Gemini 2.5 Flash**, featuring native function calling, persistent subshell execution, context sliding-window truncation, human-in-the-loop safety guardrails, and automated evaluation harnesses.


---

## Key Features & Production Design Patterns

### 1. Gemini Native Function Calling
Instead of relying on unstructured text output and fragile regex parsing, NanoTerminal uses Gemini's native function calling (`types.Tool`) with `mode="ANY"` enforcement to guarantee 100% structured JSON outputs.

### 2. Subshell Persistence & Context Truncation
* **Stateful Execution:** Captures directory updates (`pwd`) across tool calls so stateful operations like `cd` persist natively throughout multi-turn workflows.
* **Sliding Window Output Truncation:** Commands producing massive terminal output streams are truncated using a head-and-tail window strategy (keeping the first and last 1,000 characters). This prevents context-window bloat and protects against API rate limits (`429 RESOURCE_EXHAUSTED`).

### 3. Human-in-the-Loop (HITL) Safety Guardrails
Before invoking system commands, NanoTerminal evaluates the command string against regex risk rules. Destructive commands (`rm -rf`, `git reset --hard`, disk operations) trigger interactive user confirmation prompts (`[y/N]`) before execution.

### 4. Continuous Observability & Trajectory Auditing
Every run exports structured JSON trace artifacts to `./trajectories/run_<timestamp>.json`, capturing step turns, execution commands, stdout/stderr logs, exit codes, and latencies. Integrated support for **Langfuse** provides token usage and cost metrics.

### 5. Automated Benchmarking Harness
Equipped with a non-interactive CLI entry point (`cli_entrypoint.py`), allowing NanoTerminal to run as an autonomous agent in benchmark execution frameworks like **Terminal-Bench 2.0** (via **Harbor**).

---

## Project Structure

```text
nanoterminal/
├── agent.py            # Interactive REPL CLI interface
├── cli_entrypoint.py   # Non-interactive autopilot entry point for benchmark runners
├── executor.py         # Subshell execution engine with cwd persistence & output truncation
├── guardrails.py       # Regex-based command risk classifier
├── llm.py              # Gemini 2.5 Flash API client with tool-calling schema
├── logger.py           # Structured JSON trajectory logging engine
├── benchmark.py        # Local 5-task automated evaluation suite
├── harbor.toml         # Harbor execution configuration for Terminal-Bench 2.0
├── pyproject.toml      # Project dependencies managed via UV
└── README.md           # Project documentation
