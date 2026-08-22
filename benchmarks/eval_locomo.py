"""Run LoCoMo QA evaluation against NanoTerminal memory (Lychee + MemCon)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from google.genai import types
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.locomo.dataset import (  # noqa: E402
    DEFAULT_DATA_PATH,
    build_qa_prompt,
    download_locomo,
    format_question,
    ingest_conversation,
    load_locomo,
    parse_cat5_answer,
)
from benchmarks.locomo.metrics import (  # noqa: E402
    LOCOMO_CATEGORY_NAMES,
    aggregate_by_category,
    eval_question_answering,
)
from llm import (  # noqa: E402
    active_text_model_label,
    ask_grok_raw,
    ask_groq_raw,
    ask_text_raw,
    get_client,
    get_llm_provider,
    get_model_name,
    require_grok_api_key,
    require_groq_api_key,
    _groq_qa_max_tokens,
)
from dotenv import load_dotenv  # noqa: E402
from memory.engine import MemoryEngine  # noqa: E402

console = Console()


def ask_eval_model(prompt: str, *, provider: str | None = None) -> str:
    name = (provider or get_llm_provider()).strip().lower()
    if name in ("grok", "xai"):
        return ask_grok_raw(prompt)
    if name == "groq":
        return ask_groq_raw(prompt, max_tokens=_groq_qa_max_tokens())
    model = get_model_name()
    res = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return (res.text or "").strip()


def mock_answer_fn(prompt: str) -> str:
    """Deterministic stub for offline smoke runs."""
    if "Where does Alice work" in prompt:
        return "Acme Corp"
    if "When did Alice ship" in prompt or "approximate date" in prompt:
        return "10 Feb 2024"
    if "Select the correct answer" in prompt and "Mars" in prompt:
        return "Not mentioned in the conversation"
    if "team is Alice on" in prompt:
        return "payments team"
    return "unknown"


def run_locomo_eval(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    out_file: Path | None = None,
    max_samples: int | None = None,
    max_questions: int | None = None,
    max_sessions: int | None = None,
    sample_id: str | None = None,
    ingest_mode: str = "lychee",
    mock_extract: bool = False,
    seed: int = 42,
    provider: str | None = None,
    answer_fn=None,
) -> dict[str, Any]:
    provider_name = (provider or get_llm_provider()).strip().lower()
    if answer_fn is None:
        answer_fn = lambda prompt: ask_eval_model(prompt, provider=provider_name)

    samples = load_locomo(data_path)
    if sample_id:
        samples = [s for s in samples if s["sample_id"] == sample_id]
    if max_samples is not None:
        samples = samples[:max_samples]

    rng = random.Random(seed)
    all_qas: list[dict[str, Any]] = []
    results_by_sample: list[dict[str, Any]] = []

    extract_llm = (
        (lambda _prompt: '[{"text": "Conversation fact", "memory_type": "fact"}]')
        if mock_extract
        else (lambda prompt: ask_text_raw(prompt, provider=provider_name, max_tokens=512))
    )

    if (
        ingest_mode == "lychee"
        and not mock_extract
        and provider_name == "groq"
    ):
        console.print(
            "[yellow]Groq free tier: ~8000 TPM. Prefer --ingest-mode dialog, "
            "or keep --max-sessions low. Retries auto-wait on 429.[/yellow]"
        )

    if (
        ingest_mode == "lychee"
        and not mock_extract
        and provider_name == "gemini"
        and max_samples is None
        and max_questions is None
    ):
        console.print(
            "[yellow]Warning: full LoCoMo + lychee on Gemini free tier will hit "
            "~20 RPD. Use --provider groq, --ingest-mode dialog, or cap with "
            "--max-samples / --max-questions.[/yellow]"
        )

    for sample in samples:
        db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        q_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        plans_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

        engine = MemoryEngine(
            llm_client=extract_llm,
            db_path=db_tmp,
            qtable_path=q_tmp,
            plans_path=plans_tmp,
        )

        turn_count = ingest_conversation(
            engine,
            sample["conversation"],
            mode=ingest_mode,
            max_sessions=max_sessions,
        )

        qa_rows = list(sample["qa"])
        if max_questions is not None:
            qa_rows = qa_rows[:max_questions]

        evaluated: list[dict[str, Any]] = []
        for qa in qa_rows:
            question, cat5_key = format_question(qa, rng=rng.random)
            memory_context, action = engine.prepare_context(question)
            prompt = build_qa_prompt(sample["conversation"], question, memory_context)
            prediction = answer_fn(prompt)
            if cat5_key is not None:
                prediction = parse_cat5_answer(prediction, cat5_key)

            row = dict(qa)
            row["prediction"] = prediction
            row["memcon_action"] = f"{action.op.value}:{action.label}"
            row["memory_context_chars"] = len(memory_context)
            evaluated.append(row)

        engine.complete_task(success=True)
        engine.shutdown()

        scores, _recall = eval_question_answering(evaluated, prediction_key="prediction")
        for row, score in zip(evaluated, scores):
            row["f1"] = round(score, 3)

        all_qas.extend(evaluated)
        results_by_sample.append(
            {
                "sample_id": sample["sample_id"],
                "turns_ingested": turn_count,
                "records_in_store": engine.store.count(),
                "questions": len(evaluated),
                "mean_f1": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "qa": evaluated,
            }
        )

    overall_scores, _ = eval_question_answering(all_qas, prediction_key="prediction")
    summary = {
        "data_path": str(data_path),
        "ingest_mode": ingest_mode,
        "provider": provider_name,
        "model": active_text_model_label(provider_name)
        if provider_name in ("grok", "xai", "groq")
        else get_model_name(),
        "samples": len(results_by_sample),
        "questions": len(all_qas),
        "mean_f1": round(float(sum(overall_scores) / len(overall_scores)), 3)
        if overall_scores
        else 0.0,
        "by_category": aggregate_by_category(all_qas, overall_scores),
        "category_names": LOCOMO_CATEGORY_NAMES,
        "results": results_by_sample,
    }

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title="LoCoMo QA · NanoTerminal memory", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("F1", justify="right", style="green")
    for name, value in summary["by_category"].items():
        table.add_row(name, f"{value:.3f}")
    console.print(table)
    console.print(
        f"\nSamples={summary['samples']}  Questions={summary['questions']}  "
        f"Mean F1={summary['mean_f1']:.3f}  Ingest={summary['ingest_mode']}  "
        f"Model={summary['model']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo QA eval for NanoTerminal memory")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to locomo10.json (auto-downloads if missing)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download locomo10.json and exit",
    )
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--ingest-mode",
        choices=("lychee", "dialog"),
        default="lychee",
        help="lychee=segment+extract; dialog=store raw turns (fast baseline)",
    )
    parser.add_argument(
        "--mock-extract",
        action="store_true",
        help="Skip real Lychee extract LLM (uses canned JSON)",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq", "grok"),
        default=None,
        help="LLM backend for extract + QA (default: NANOTERMINAL_LLM_PROVIDER or gemini)",
    )
    parser.add_argument(
        "--mock-answers",
        action="store_true",
        help="Use deterministic stub answers (no Gemini QA calls)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    provider = (args.provider or get_llm_provider()).strip().lower()
    if provider in ("grok", "xai") and not args.mock_extract and not args.mock_answers:
        require_grok_api_key()
    if provider == "groq" and not args.mock_extract and not args.mock_answers:
        require_groq_api_key()

    answer_fn = mock_answer_fn if args.mock_answers else None

    if args.download_only:
        path = download_locomo(args.data_file)
        console.print(f"Downloaded LoCoMo dataset to {path}")
        return

    summary = run_locomo_eval(
        data_path=args.data_file,
        out_file=args.out_file,
        max_samples=args.max_samples,
        max_questions=args.max_questions,
        max_sessions=args.max_sessions,
        sample_id=args.sample_id,
        ingest_mode=args.ingest_mode,
        mock_extract=args.mock_extract,
        seed=args.seed,
        provider=args.provider,
        answer_fn=answer_fn,
    )
    _print_summary(summary)
    if args.out_file:
        console.print(f"Wrote {args.out_file}")


if __name__ == "__main__":
    main()
