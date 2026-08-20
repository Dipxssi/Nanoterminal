# NanoTerminal

An autonomous terminal agent powered by **Gemini 2.5 Flash**, with an **adaptive dual-loop memory architecture**: Lychee hierarchical episodic ingestion on the write path and a MemCon reinforcement-learning retrieval controller on the read path.

---

## Why NanoTerminal

Most terminal agents either dump every past turn into the prompt or always retrieve a fixed top‑k from memory. NanoTerminal treats memory as a **learned control problem**: at each step a lightweight tabular bandit decides whether to retrieve (and how deeply), inject a plan, consolidate, forget, or do nothing — **without an extra LLM call for the policy itself**.

---

## Features

| Area | What you get |
|---|---|
| **Agent loop** | Gemini native function calling (`execute_bash` / `finish_task`) |
| **Execution** | Persistent cwd across turns, head/tail output truncation, timeouts |
| **Safety** | HITL prompts for high-risk commands (`rm -rf`, hard resets, …) |
| **MemCon (read)** | φ(s) discretization, 9-action UCB1, feasibility mask, episode credit |
| **Lychee (write)** | Session buffer, centroid topic boundaries, batch JSON extraction |
| **Plans** | Success-plan index for `PLANINJECT` (`~/.nanoterminal/plans.json`) |
| **Observability** | JSON trajectories; optional Langfuse |
| **Eval** | Harbor / Terminal-Bench 2.0 + local MemCon token benchmark |

---

## Adaptive dual-loop memory

Inspired by [*Memory as a Controlled Process* (MemCon)](https://arxiv.org/abs/2607.13591).

```text
┌─────────────────────────────────────────────────────────────┐
│                      User / Shell Input                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  MemCon Read Controller (MDP)               │
│  • State φ(s): intent, step phase, stuck, cwd, store, plan  │
│  • 9-Action UCB1 + warm-start priors Q₀                     │
└──────────────┬──────────────────────────────┬───────────────┘
               │ NOOP / PlanInject / Maintain │ RETRIEVE / RE-RETRIEVE
               ▼                              ▼
     [Zero or plan context]         [Vector search in SQLite]
               │                              │
               └──────────────┬───────────────┘
                              │ Injected context
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution Loop                     │
│          (LLM function calls + shell tool execution)        │
└──────────────────────────────┬──────────────────────────────┘
                               │ Live turns (role, content, exit_code, cwd)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Lychee Write Engine (segmentation)            │
│  • SessionBuffer · centroid cosine < 0.50 · 3–10 turns      │
│  • Batch structured LLM extraction (JSON schema)            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Durable facts
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              SQLite + FastEmbed (BGE-small) + plans.json    │
└─────────────────────────────────────────────────────────────┘
```

### Core components

**Lychee write loop** (`buffer` → `segmenter` → `store`)

- Buffers user / assistant / tool turns in RAM (embedding cost only; no extraction LLM until a boundary).
- Compares the newest turn to the centroid of prior **user** turns; segments when cosine similarity &lt; **0.50**, or at turn caps (**3–10**).
- Batch-extracts durable constraints, facts, and failure patterns; filters ephemeral CLI noise (`cd`, `ls`, typos).

**MemCon read loop** (`state` → `controller` → `engine`)

- Discretizes session signals into φ(s): intent, early/mid/late phase, stuck flag, `cwd_bin`, memory size, `plan_available`, learning phase.
- Selects among 9 actions with UCB1 and warm-start priors, filtered by a feasibility mask.
- **Executes** the chosen action (not just logs it):

| Action | Behavior |
|---|---|
| `RETRIEVE` shallow/medium/deep/insight | Vector search with `top_k` / hop / type filters |
| `RE_RETRIEVE` | Alt-query retrieve when the agent is stuck |
| `PLANINJECT` | Prepend a stored success plan for this intent |
| `CONSOLIDATE` | Merge near-duplicate SQLite records |
| `FORGET` | Evict oldest records |
| `NOOP` | Inject nothing |

Mid-task, if `consecutive_failures ≥ 2`, the agent **re-consults** MemCon so `RE_RETRIEVE` can fire before the next LLM step.

**Learning loop** (`complete_task`)

- Task reward with step efficiency:
  \(R = R_{\mathrm{succ/fail}} \pm \lambda_{\mathrm{eff}}\cdot\max(0, 1 - T/T_{\max})\)
- Reverse-discounted credit: \(\gamma^{L-t-1}\cdot R\)
- On success, upserts a generalized plan from the command trace into `plans.json`

**Vector engine & persistence**

- FastEmbed `BAAI/bge-small-en-v1.5` (ONNX; no external vector DB)
- SQLite cosine scan; Q-table / plans saved atomically (`.tmp` + `os.replace`)

### Key hyperparameters (Appendix C defaults)

| Symbol | Value | Role |
|---|---|---|
| \(\alpha\) | 0.15 | Q-learning step size |
| \(\gamma\) | 0.90 | Within-episode discount |
| \(c\) | 1.40 | UCB exploration |
| \(T_{\max}\) | 30 | Efficiency horizon |
| \(\lambda_{\mathrm{eff}}\) | 0.30 | Efficiency weight |
| Segment threshold | 0.50 | Lychee cosine boundary |

Artifacts: `~/.nanoterminal/memory.db`, `memcon_qtable.json`, `plans.json`.

### Token efficiency (local benchmark)

Naive “always retrieve top‑5” vs MemCon adaptive retrieval:

![MemCon vs Naive Retrieval Benchmark](images/img.jpeg)

| Strategy | Behavior | Tokens | Reduction |
|---|---|---|---|
| Naive RAG | Static `top_k=5` every turn | 2000 | 0% |
| **MemCon + Lychee** | Adaptive NOOP / shallow / medium / deep | **288** | **85.6%** |

```bash
uv run python -m benchmarks.eval_memory
```

---

## Quick start

### Requirements

- Python **3.12+**
- [uv](https://github.com/astral-sh/uv)
- Docker (for Harbor / Terminal-Bench)
- A Gemini API key

### Setup

```bash
cd nanoterminal
uv sync
```

Create `.env` in the project root:

```bash
GEMINI_API_KEY=your_key_here
# optional Langfuse keys...
```

### Interactive REPL

```bash
uv run python agent.py
```

### Single non-interactive task

```bash
uv run python cli_entrypoint.py --prompt "List files in /app" --debug-memory
```

### Tests

```bash
# Recommended (ensures the memory package imports)
PYTHONPATH=. uv run pytest tests/ -v

# Focused suites
PYTHONPATH=. uv run pytest tests/test_memory.py tests/test_memory_controller.py tests/test_memory_state.py -v
```

On Git Bash / PowerShell, same idea: set `PYTHONPATH` to the repo root, then run pytest.

---

## Harbor / Terminal-Bench 2.0

`harbor_agent.py` installs the agent **and** the full `memory/` package (including `plans.py`) into the task container, with FastEmbed warmup.

Config: [`harbor.yaml`](harbor.yaml) (YAML/JSON only — not TOML).

```bash
# Loads GEMINI_API_KEY from .env via harbor_agent
uv run harbor run -c harbor.yaml

# Explicit 3-task smoke run
uv run harbor run -a harbor_agent:NanoTerminalAgent -d "terminal-bench@2.0" -l 3 -n 1
```

Results: `jobs/<timestamp>/`. Inspect with `harbor view jobs`.

**Note:** Harbor **mean** is verifier pass rate (reward `1`/`0`), not MemCon’s internal bandit reward. Memory cuts prompt tokens; solving the task still drives the score.

---

## Project layout

```text
nanoterminal/
├── agent.py                 # Interactive REPL (HITL + memory)
├── cli_entrypoint.py        # Autopilot runner (Harbor / scripts)
├── harbor_agent.py          # Harbor adapter (uploads memory/)
├── harbor.yaml              # Terminal-Bench job config
├── executor.py              # Shell runner (cwd persistence, truncation)
├── guardrails.py            # High-risk command classifier
├── llm.py                   # Gemini client + tool schemas
├── logger.py                # Trajectory JSON logger
├── memory/
│   ├── engine.py            # Dual-loop facade (execute all 9 actions)
│   ├── state.py             # φ(s) + feasibility mask
│   ├── controller.py        # MemCon UCB bandit + episode credit
│   ├── plans.py             # Success-plan index (PLANINJECT)
│   ├── store.py             # SQLite + consolidate / forget
│   ├── buffer.py            # Session turn buffer
│   ├── segmenter.py         # Lychee boundary + extraction
│   ├── embeddings.py        # FastEmbed BGE + cosine
│   └── schemas.py           # Turn / MemoryRecord
├── images/img.jpeg          # Benchmark screenshot
├── benchmarks/eval_memory.py
└── tests/
```

---

## Memory API

```python
from memory.engine import MemoryEngine
from llm import ask_gemini_raw

engine = MemoryEngine(llm_client=ask_gemini_raw)

# Read: MemCon selects + executes an action
context, action = engine.prepare_context(user_query)

# Write: Lychee observes turns (optional command/cwd for φ(s))
engine.observe_turn("user", user_query)
engine.observe_turn(
    "assistant",
    observation,
    exit_code=code,
    command=cmd,
    cwd=cwd,
)

# Learn + optional plan upsert
engine.complete_task(success=True, goal=user_query)
engine.shutdown()
```

---

## Configuration

| Item | Default |
|---|---|
| API key | `.env` → `GEMINI_API_KEY` |
| Memory DB | `~/.nanoterminal/memory.db` |
| Q-table | `~/.nanoterminal/memcon_qtable.json` |
| Plans | `~/.nanoterminal/plans.json` |
| Max turns | `--max-turns` (default `30`) |
| MemCon debug | `--debug-memory` |

---

## License

Add your preferred license here.
