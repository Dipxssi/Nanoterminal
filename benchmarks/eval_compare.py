#!/usr/bin/env python3
"""Compare Flash vs Pro vs Pro+hardened on a local smoke prompt (not full TB).

For Harbor / Terminal-Bench A/B (same task slice, record verifier rewards):

  # 1) Flash + legacy scaffold
  $env:NANOTERMINAL_MODEL='gemini-2.5-flash'
  $env:NANOTERMINAL_SCAFFOLD='legacy'
  $env:NANOTERMINAL_THINKING='0'
  uv run harbor run -c harbor.yaml --env-file .env

  # 2) Higher Gemini + legacy scaffold
  $env:NANOTERMINAL_MODEL='gemini-3.5-flash'
  $env:NANOTERMINAL_SCAFFOLD='legacy'
  $env:NANOTERMINAL_THINKING='1'
  uv run harbor run -c harbor.yaml --env-file .env

  # 3) Higher Gemini + hardened scaffold (default)
  $env:NANOTERMINAL_MODEL='gemini-3.5-flash'
  $env:NANOTERMINAL_SCAFFOLD='hardened'
  $env:NANOTERMINAL_THINKING='1'
  uv run harbor run -c harbor.yaml --env-file .env

  # Optional: Pro when billing/quota allows
  # $env:NANOTERMINAL_MODEL='gemini-3.1-pro-preview'

Claim leaderboard numbers with >=5 attempts per task on the full dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Ensure project root imports work when run as a script.
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_entrypoint import run_task  # noqa: E402


CONFIGS = (
    {
        "name": "flash_legacy",
        "NANOTERMINAL_MODEL": "gemini-2.5-flash",
        "NANOTERMINAL_SCAFFOLD": "legacy",
        "NANOTERMINAL_THINKING": "0",
    },
    {
        "name": "pro_legacy",
        "NANOTERMINAL_MODEL": "gemini-3.5-flash",
        "NANOTERMINAL_SCAFFOLD": "legacy",
        "NANOTERMINAL_THINKING": "1",
    },
    {
        "name": "pro_hardened",
        "NANOTERMINAL_MODEL": "gemini-3.5-flash",
        "NANOTERMINAL_SCAFFOLD": "hardened",
        "NANOTERMINAL_THINKING": "1",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash vs Pro vs Pro+hardened smoke")
    parser.add_argument(
        "--prompt",
        default=(
            "Create a file named hello_tb.txt containing exactly the text "
            "'nanoterminal-ok' and then finish."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument(
        "--out",
        type=str,
        default="benchmarks/eval_compare_results.json",
        help="Where to write the comparison JSON",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated config names to run (default: all)",
    )
    args = parser.parse_args()

    selected = {s.strip() for s in args.only.split(",") if s.strip()} or {
        c["name"] for c in CONFIGS
    }

    results = []
    for cfg in CONFIGS:
        if cfg["name"] not in selected:
            continue
        for key, value in cfg.items():
            if key.startswith("NANOTERMINAL_"):
                os.environ[key] = value

        log_dir = f"trajectories/eval_compare/{cfg['name']}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running {cfg['name']} ===")
        t0 = time.time()
        ok = run_task(
            args.prompt,
            max_turns=args.max_turns,
            log_dir=log_dir,
            debug_memory=False,
        )
        elapsed = time.time() - t0
        row = {
            "name": cfg["name"],
            "model": cfg["NANOTERMINAL_MODEL"],
            "scaffold": cfg["NANOTERMINAL_SCAFFOLD"],
            "thinking": cfg["NANOTERMINAL_THINKING"],
            "success": bool(ok),
            "elapsed_sec": round(elapsed, 2),
        }
        results.append(row)
        print(f"=== {cfg['name']}: success={ok} elapsed={elapsed:.1f}s ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    for row in results:
        print(
            f"  {row['name']}: success={row['success']} "
            f"({row['elapsed_sec']}s)"
        )


if __name__ == "__main__":
    main()
