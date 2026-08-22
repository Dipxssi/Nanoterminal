"""LoCoMo QA scoring (ported from snap-research/locomo task_eval/evaluation.py)."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any

import numpy as np

# Official code mapping (see snap-research/locomo issue #6).
LOCOMO_CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def normalize_answer(text: str) -> str:
    text = text.replace(",", "")

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(
        np.mean(
            [
                max(f1_score(pred, gt) for pred in predictions)
                for gt in ground_truths
            ]
        )
    )


def score_qa_item(qa: dict[str, Any], prediction: str) -> float:
    """Return F1 (or adversarial accuracy) for one QA row."""
    answer = str(qa["answer"])
    category = int(qa["category"])
    output = prediction or ""

    if category == 3:
        answer = answer.split(";")[0].strip()

    if category in (2, 3, 4):
        return f1_score(output, answer)
    if category == 1:
        return f1_multi_answer(output, answer)
    if category == 5:
        lowered = output.lower()
        if "no information available" in lowered or "not mentioned" in lowered:
            return 1.0
        return 0.0
    raise ValueError(f"Unknown LoCoMo category: {category}")


def eval_question_answering(
    qas: list[dict[str, Any]],
    *,
    prediction_key: str = "prediction",
) -> tuple[list[float], list[float]]:
    """Score a list of QA dicts; returns (f1_scores, recall_scores)."""
    scores: list[float] = []
    recall_scores: list[float] = []

    for qa in qas:
        prediction = qa.get(prediction_key, "")
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else ""
        prediction = str(prediction)
        scores.append(score_qa_item(qa, prediction))

        ctx_key = prediction_key + "_context"
        if ctx_key in qa and qa.get("evidence"):
            ctx = qa[ctx_key]
            if ctx and isinstance(ctx[0], str) and ctx[0].startswith("S"):
                sessions = {e[1:] for e in ctx if isinstance(e, str)}
                recall = sum(
                    ev.split(":")[0][1:] in sessions for ev in qa["evidence"]
                ) / len(qa["evidence"])
            else:
                recall = sum(ev in ctx for ev in qa["evidence"]) / len(qa["evidence"])
            recall_scores.append(float(recall))
        else:
            recall_scores.append(1.0)

    return scores, recall_scores


def aggregate_by_category(qas: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {name: [] for name in LOCOMO_CATEGORY_NAMES.values()}
    for qa, score in zip(qas, scores):
        name = LOCOMO_CATEGORY_NAMES[int(qa["category"])]
        buckets[name].append(score)
    return {
        name: (float(np.mean(values)) if values else 0.0)
        for name, values in buckets.items()
    }
