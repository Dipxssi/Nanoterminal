import json
import tempfile
from pathlib import Path

from benchmarks.eval_locomo import run_locomo_eval
from benchmarks.locomo.dataset import ingest_conversation, iter_turns, load_locomo
from memory.engine import MemoryEngine

FIXTURE = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "locomo_fixture.json"


def test_iter_turns_fixture():
    sample = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    turns = list(iter_turns(sample["conversation"]))
    assert len(turns) == 5
    assert "Acme Corp" in turns[0][1]


def test_ingest_dialog_mode():
    sample = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    q = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    plans = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    engine = MemoryEngine(llm_client=lambda p: "[]", db_path=db, qtable_path=q, plans_path=plans)
    count = ingest_conversation(engine, sample["conversation"], mode="dialog")
    assert count == 5
    assert engine.store.count() == 5


def test_run_locomo_eval_fixture_mock():
    summary = run_locomo_eval(
        data_path=FIXTURE,
        max_samples=1,
        max_questions=2,
        ingest_mode="dialog",
        mock_extract=True,
        answer_fn=lambda prompt: "Acme Corp" if "Where does Alice work" in prompt else "10 Feb 2024",
    )
    assert summary["samples"] == 1
    assert summary["questions"] == 2
    assert summary["mean_f1"] > 0.0
    assert "single_hop" in summary["by_category"]


def test_load_locomo_downloads_when_missing(tmp_path, monkeypatch):
    dest = tmp_path / "locomo10.json"

    def fake_download(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return path

    monkeypatch.setattr("benchmarks.locomo.dataset.download_locomo", fake_download)
    data = load_locomo(dest)
    assert data == []
