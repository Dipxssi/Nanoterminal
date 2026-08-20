import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from memory.schemas import MemoryRecord
from memory.embeddings import cosine_similarity


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path.home() / ".nanoterminal"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(base_dir / "memory.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.__init_db__()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def __init_db__(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_records (
                 id TEXT PRIMARY KEY,
                 text TEXT NOT NULL,
                 memory_type TEXT NOT NULL,
                 timestamp TEXT NOT NULL,
                 embedding_json TEXT,
                 source_turn_indices_json TEXT
               )
               """)

    def save_record(self, record: MemoryRecord):
        """Inserts or updates a memory record in SQLite."""
        emb_json = json.dumps(record.embedding) if record.embedding else None
        turns_json = json.dumps(record.source_turn_indices)
        ts = record.timestamp
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_records
                (id, text, memory_type, timestamp, embedding_json, source_turn_indices_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.text,
                record.memory_type,
                ts_str,
                emb_json,
                turns_json,
            ))

    def delete_record(self, record_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_records WHERE id = ?", (record_id,)
            )
            return cursor.rowcount > 0

    def delete_records(self, record_ids: Sequence[str]) -> int:
        if not record_ids:
            return 0
        placeholders = ",".join("?" for _ in record_ids)
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memory_records WHERE id IN ({placeholders})",
                tuple(record_ids),
            )
            return cursor.rowcount

    def count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM memory_records").fetchone()
            return int(row["n"])

    def get_all_records(self) -> List[MemoryRecord]:
        """Loads all records from SQLite disk storage."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM memory_records")
            rows = cursor.fetchall()

        records = []
        for row in rows:
            rec = MemoryRecord(
                id=row["id"],
                text=row["text"],
                memory_type=row["memory_type"],
                timestamp=row["timestamp"],
                embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
                source_turn_indices=json.loads(row["source_turn_indices_json"]) if row["source_turn_indices_json"] else [],
            )
            records.append(rec)
        return records

    def retrieve_similar(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_score: float = 0.5,
        memory_types: Optional[Sequence[str]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Scans records in SQLite and retrieves top-k matches using cosine similarity."""
        records = self.get_all_records()
        type_filter = set(memory_types) if memory_types else None
        scored_records: List[Tuple[MemoryRecord, float]] = []

        for rec in records:
            if not rec.embedding:
                continue
            if type_filter is not None and rec.memory_type not in type_filter:
                continue
            sim = cosine_similarity(query_embedding, rec.embedding)
            if sim >= min_score:
                scored_records.append((rec, sim))

        scored_records.sort(key=lambda item: item[1], reverse=True)
        return scored_records[:k]

    def consolidate(self, similarity_threshold: float = 0.92) -> int:
        """Merge near-duplicate records; keep the longer text. Returns deletions."""
        records = [r for r in self.get_all_records() if r.embedding]
        if len(records) < 2:
            return 0

        records.sort(key=lambda r: str(r.timestamp), reverse=True)
        keep: list[MemoryRecord] = []
        to_delete: list[str] = []

        for rec in records:
            merged = False
            for kept in keep:
                if kept.embedding is None or rec.embedding is None:
                    continue
                if cosine_similarity(rec.embedding, kept.embedding) >= similarity_threshold:
                    if len(rec.text) > len(kept.text):
                        kept.text = rec.text
                        self.save_record(kept)
                    to_delete.append(rec.id)
                    merged = True
                    break
            if not merged:
                keep.append(rec)

        return self.delete_records(to_delete)

    def forget(self, max_delete: int = 3) -> int:
        """Evict oldest records (maintenance hook). Returns deletions."""
        records = self.get_all_records()
        if not records:
            return 0
        records.sort(key=lambda r: str(r.timestamp))
        n = min(max_delete, max(1, len(records) // 5), len(records))
        return self.delete_records([r.id for r in records[:n]])
