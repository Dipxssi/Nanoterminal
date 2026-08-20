import json
import numpy as np
from typing import List, Optional
from memory.schemas import Turn, MemoryRecord
from memory.embeddings import EmbeddingModel, cosine_similarity
from memory.store import MemoryStore


EXTRACTION_SYSTEM_PROMPT = """You are a precise Memory Extraction Engine for a developer terminal assistant.
Your task is to analyze the conversation turns and extract ONLY durable, important facts, constraints, failure patterns, and commitments.

Ignore trivial ephemeral commands (e.g., 'ls', 'cd', 'clear', 'pwd', typing errors).

Output a strict JSON array of objects with this structure:
[
  {
    "text": "Clear, standalone statement of fact, error fix, or constraint.",
    "memory_type": "fact" | "preference" | "event" | "constraint" | "failure_pattern"
  }
]

If there is nothing worth remembering, output an empty JSON array: []
"""


class LycheeSegmenter:
    def __init__(
        self,
        embedder: EmbeddingModel,
        store: MemoryStore,
        llm_client,
        similarity_threshold: float = 0.50,
        min_turns_per_segment: int = 3,
        max_turns_per_segment: int = 10,
    ):
        self.embedder = embedder
        self.store = store
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self.min_turns_per_segment = min_turns_per_segment
        self.max_turns_per_segment = max_turns_per_segment

    def should_segment(self, buffer_turns: List[Turn]) -> bool:
        """Determines if the active buffer has reached a logical boundary."""
        num_turns = len(buffer_turns)
        if num_turns < self.min_turns_per_segment:
            return False

        if num_turns >= self.max_turns_per_segment:
            return True

        latest_turn = buffer_turns[-1]
        if latest_turn.embedding is None:
            return False

        # Prefer prior *user* turns: short assistant replies ("done", "ok")
        # dilute the centroid and hide real topic shifts.
        prior_user = [
            t for t in buffer_turns[:-1]
            if t.role == "user" and t.embedding is not None
        ]
        prior_turns = prior_user or [
            t for t in buffer_turns[:-1] if t.embedding is not None
        ]
        if not prior_turns:
            return False

        prior_vectors = np.array([t.embedding for t in prior_turns])
        centroid = np.mean(prior_vectors, axis=0).tolist()
        sim = cosine_similarity(latest_turn.embedding, centroid)
        return sim < self.similarity_threshold

    def extract_and_store(self, buffer_turns: List[Turn]) -> List[MemoryRecord]:
        """Runs batch extraction on finalized segment and persists records into SQLite."""
        if not buffer_turns:
            return []

        transcript_lines = []
        turn_indices = []
        for idx, turn in enumerate(buffer_turns):
            transcript_lines.append(f"Turn {idx} [{turn.role.upper()}]: {turn.content}")
            turn_indices.append(idx)

        transcript = "\n".join(transcript_lines)
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}\n\nJSON Output:"

        raw_response = self.llm_client(prompt)

        extracted_records: List[MemoryRecord] = []
        try:
            cleaned_resp = raw_response.strip()
            if cleaned_resp.startswith("```json"):
                cleaned_resp = cleaned_resp[7:]
            if cleaned_resp.startswith("```"):
                cleaned_resp = cleaned_resp[3:]
            if cleaned_resp.endswith("```"):
                cleaned_resp = cleaned_resp[:-3]

            data = json.loads(cleaned_resp.strip())

            for item in data:
                text = item.get("text", "").strip()
                mtype = item.get("memory_type", "fact")
                if not text:
                    continue

                emb = self.embedder.embed_text(text)
                record = MemoryRecord(
                    text=text,
                    memory_type=mtype,
                    embedding=emb,
                    source_turn_indices=turn_indices
                )
                self.store.save_record(record)
                extracted_records.append(record)

        except Exception as e:
            print(f"[Memory Extraction Warning] Failed to parse JSON response: {e}")

        return extracted_records 