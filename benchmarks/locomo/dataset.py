"""LoCoMo dataset loading and conversation ingestion."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from memory.engine import MemoryEngine
from memory.schemas import MemoryRecord

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "locomo10.json"

CONV_START_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {question} Short answer:
"""


def download_locomo(dest: Path = DEFAULT_DATA_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LOCOMO_URL, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_locomo(path: Path | str = DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        download_locomo(path)
    return json.loads(path.read_text(encoding="utf-8"))


def session_numbers(conversation: dict[str, Any]) -> list[int]:
    nums: list[int] = []
    for key in conversation:
        match = re.fullmatch(r"session_(\d+)", key)
        if match:
            nums.append(int(match.group(1)))
    return sorted(nums)


def format_turn(dialog: dict[str, Any], *, date_time: str) -> str:
    text = dialog.get("text", "").strip()
    speaker = dialog.get("speaker", "Unknown")
    line = f'{speaker} said, "{text}"'
    caption = dialog.get("blip_caption")
    if caption:
        line += f" and shared {caption}"
    return f"DATE: {date_time}\n{line}"


def iter_turns(
    conversation: dict[str, Any],
    *,
    max_sessions: int | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Yield (speaker, formatted_text, dia_id) in chronological order."""
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    sessions = session_numbers(conversation)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    for session_num in sessions:
        date_key = f"session_{session_num}_date_time"
        turn_key = f"session_{session_num}"
        date_time = conversation.get(date_key, "")
        for dialog in conversation.get(turn_key, []):
            yield (
                dialog.get("speaker", speaker_a),
                format_turn(dialog, date_time=date_time),
                dialog.get("dia_id", ""),
            )


def ingest_conversation(
    engine: MemoryEngine,
    conversation: dict[str, Any],
    *,
    mode: str = "lychee",
    max_sessions: int | None = None,
) -> int:
    """Ingest turns into MemoryEngine. Returns turn count."""
    speaker_a = conversation.get("speaker_a", "Speaker A")
    turns = 0

    for speaker, text, _dia_id in iter_turns(
        conversation, max_sessions=max_sessions
    ):
        role = "user" if speaker == speaker_a else "assistant"
        if mode == "dialog":
            emb = engine.embedder.embed_text(text)
            engine.store.save_record(
                MemoryRecord(
                    text=text,
                    memory_type="event",
                    embedding=emb,
                )
            )
        else:
            engine.observe_turn(role=role, content=text)
        turns += 1

    if mode == "lychee":
        engine.shutdown()
    return turns


def format_question(qa: dict[str, Any], *, rng: Callable[[], float] | None = None) -> tuple[str, dict[str, str] | None]:
    """Return (question_text, adversarial_key_or_none)."""
    question = qa["question"]
    category = int(qa["category"])

    if category == 2:
        return (
            question + " Use DATE of CONVERSATION to answer with an approximate date.",
            None,
        )

    if category == 5:
        flip = (rng or (lambda: 0.0))() < 0.5
        if flip:
            prompt = (
                question
                + " Select the correct answer: (a) Not mentioned in the conversation (b) "
                + qa["answer"]
                + ". "
            )
            return prompt, {"a": "Not mentioned in the conversation", "b": qa["answer"]}
        prompt = (
            question
            + " Select the correct answer: (a) "
            + qa["answer"]
            + " (b) Not mentioned in the conversation. "
        )
        return prompt, {"a": qa["answer"], "b": "Not mentioned in the conversation"}

    return question, None


def parse_cat5_answer(prediction: str, answer_key: dict[str, str]) -> str:
    text = prediction.strip().lower()
    if len(text) == 1:
        return answer_key["a" if "a" in text else "b"]
    if len(text) == 3:
        return answer_key["a" if "(a)" in text else "b"]
    return prediction


def build_qa_prompt(
    conversation: dict[str, Any],
    question: str,
    memory_context: str,
) -> str:
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    header = CONV_START_PROMPT.format(speaker_a=speaker_a, speaker_b=speaker_b)
    context_block = memory_context.strip() or "No additional memory context retrieved."
    if int(question.count("Select the correct answer")) > 0:
        body = QA_PROMPT_CAT_5.format(question=question)
    else:
        body = QA_PROMPT.format(question=question)
    return f"{header}{context_block}\n\n{body}"
