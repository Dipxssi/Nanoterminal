import json
from pathlib import Path

from benchmarks.locomo.metrics import (
    LOCOMO_CATEGORY_NAMES,
    eval_question_answering,
    f1_score,
    score_qa_item,
)


def test_f1_exact_match():
    assert f1_score("Acme Corp", "Acme Corp") == 1.0
    assert f1_score("totally wrong", "Acme Corp") == 0.0


def test_adversarial_scoring():
    qa = {"answer": "Not mentioned in the conversation", "category": 5}
    assert score_qa_item(qa, "Not mentioned in the conversation") == 1.0
    assert score_qa_item(qa, "Alice visited Mars") == 0.0


def test_eval_question_answering_batch():
    qas = [
        {"question": "q1", "answer": "foo bar", "category": 4, "prediction": "foo bar"},
        {"question": "q2", "answer": "x, y", "category": 1, "prediction": "x, y"},
    ]
    scores, recall = eval_question_answering(qas, prediction_key="prediction")
    assert len(scores) == 2
    assert all(s == 1.0 for s in scores)
    assert len(recall) == 2


def test_category_mapping_has_five_types():
    assert set(LOCOMO_CATEGORY_NAMES) == {1, 2, 3, 4, 5}


def test_fixture_dataset_loads():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "locomo_fixture.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert len(data[0]["qa"]) == 4
