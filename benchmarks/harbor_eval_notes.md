# Harbor / local eval notes (Win Terminal Bench plan)

## API reality (this key)

| Model | Status |
|---|---|
| `gemini-2.5-flash` | Works |
| `gemini-2.5-pro` | 404 — retired for new users |
| `gemini-3.1-pro-preview` | 429 — free-tier quota 0 (needs billing) |
| `gemini-3.5-flash` | Works (default agent model) |
| `gemini-3.6-flash` | Works but very tight free RPD |

Default agent model: **`gemini-3.5-flash`**. Lychee extract: **`gemini-2.5-flash`**.

## Baseline Harbor (Flash + legacy)

- Job: `jobs/tb-baseline-flash-legacy`
- Config: `harbor.baseline.yaml` (`n_tasks: 1`, task `gpt2-codegolf`)
- Env: `NANOTERMINAL_MODEL=gemini-2.5-flash`, `SCAFFOLD=legacy`, `THINKING=0`
- Result: **mean 0.0**, exception `AgentTimeoutError` (agent timed out before verifier pass)
- Reward: `0.0`

## Compare Harbor (3.5 Flash + hardened)

- Job: `jobs/tb-compare-35flash-hardened`
- Config: `harbor.compare.yaml` + `--timeout-multiplier 2.0`
- Env: `NANOTERMINAL_MODEL=gemini-3.5-flash`, `SCAFFOLD=hardened`, `THINKING=1`
- Result: **mean 0.0**, **0 agent exceptions** (completed cleanly vs baseline timeout)
- Reward: `0.0` (task still failed verifier — `gpt2-codegolf` is a hard TB task)

## Local smoke (`benchmarks/eval_compare.py`)

Prompt: create `hello_tb.txt` with `nanoterminal-ok` then finish.

| Config | Model | Scaffold | Result |
|---|---|---|---|
| flash_legacy | gemini-2.5-flash | legacy | **success=True** (~17s) |
| pro_legacy | gemini-3.5-flash | legacy | success=False — free-tier 429 (20 RPD) + no finish |
| pro_hardened | gemini-3.5-flash | hardened | success=False — free-tier 429 mid-run |

See `benchmarks/eval_compare_results.json`.

**Quota note:** Free-tier caps (~20 RPD per model) make multi-config Harbor evals unreliable. Enable Gemini API billing and set `NANOTERMINAL_MODEL=gemini-3.1-pro-preview` for serious TB runs.

## How to re-compare on Harbor

```powershell
# Flash legacy baseline
$env:NANOTERMINAL_MODEL='gemini-2.5-flash'; $env:NANOTERMINAL_SCAFFOLD='legacy'; $env:NANOTERMINAL_THINKING='0'
uv run harbor run -c harbor.baseline.yaml --env-file .env --job-name tb-baseline-flash-legacy

# 3.5 Flash + hardened (default path)
$env:NANOTERMINAL_MODEL='gemini-3.5-flash'; $env:NANOTERMINAL_SCAFFOLD='hardened'; $env:NANOTERMINAL_THINKING='1'
uv run harbor run -c harbor.compare.yaml --env-file .env --job-name tb-compare-35flash-hardened --timeout-multiplier 2.0
```
