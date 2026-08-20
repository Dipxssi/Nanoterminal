import tempfile
import pytest
from memory.schemas import MemoryRecord, Turn
from memory.embeddings import EmbeddingModel
from memory.store import MemoryStore
from memory.buffer import SessionBuffer
from memory.segmenter import LycheeSegmenter
from memory.state import extract_state, MemoryOp
from memory.controller import MemConBandit
from memory.engine import MemoryEngine
from memory.plans import PlanIndex


@pytest.fixture
def temp_env():
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    q_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    plans_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    return db_file.name, q_file.name, plans_file.name


def test_sqlite_vector_store(temp_env):
    db_path, _, _ = temp_env
    store = MemoryStore(db_path=db_path)
    embedder = EmbeddingModel()

    vec = embedder.embed_text("FastAPI backend port 8000")
    record = MemoryRecord(
        text="FastAPI backend port 8000",
        memory_type="constraint",
        embedding=vec
    )
    store.save_record(record)

    all_records = store.get_all_records()
    assert len(all_records) == 1
    assert all_records[0].text == "FastAPI backend port 8000"

    query_vec = embedder.embed_text("what port is fastapi?")
    results = store.retrieve_similar(query_vec, k=1, min_score=0.3)
    assert len(results) == 1
    assert results[0][0].text == "FastAPI backend port 8000"


def test_store_consolidate_and_forget(temp_env):
    db_path, _, _ = temp_env
    store = MemoryStore(db_path=db_path)
    embedder = EmbeddingModel()
    text = "Use Poetry for package management"
    for _ in range(3):
        store.save_record(
            MemoryRecord(
                text=text,
                memory_type="constraint",
                embedding=embedder.embed_text(text),
            )
        )
    store.save_record(
        MemoryRecord(
            text="Totally unrelated lasagna recipe tip",
            memory_type="fact",
            embedding=embedder.embed_text("Totally unrelated lasagna recipe tip"),
        )
    )
    assert store.count() == 4
    deleted = store.consolidate(similarity_threshold=0.95)
    assert deleted >= 2
    assert store.count() <= 2
    before = store.count()
    forgotten = store.forget(max_delete=1)
    assert forgotten >= 1
    assert store.count() == before - forgotten


def test_session_buffer():
    buf = SessionBuffer()
    assert buf.is_empty()
    buf.add_turn(role="user", content="git status")
    buf.add_turn(role="tool", content="clean working tree")
    assert len(buf.get_turns()) == 2
    buf.clear()
    assert buf.is_empty()


def test_lychee_boundary_segmenter(temp_env):
    db_path, _, _ = temp_env
    embedder = EmbeddingModel()
    store = MemoryStore(db_path=db_path)
    mock_llm = lambda prompt: '[{"text": "Python 3.11 installed", "memory_type": "fact"}]'

    segmenter = LycheeSegmenter(embedder=embedder, store=store, llm_client=mock_llm)

    t1 = Turn(role="user", content="install python 3.11", embedding=embedder.embed_text("install python 3.11"))
    t2 = Turn(role="assistant", content="done", embedding=embedder.embed_text("done"))
    t3 = Turn(role="user", content="how to make lasagna?", embedding=embedder.embed_text("how to make lasagna?"))

    assert segmenter.should_segment([t1, t2, t3]) is True
    records = segmenter.extract_and_store([t1, t2])
    assert len(records) == 1
    assert records[0].text == "Python 3.11 installed"


def test_memcon_bandit_learning(temp_env):
    _, q_path, _ = temp_env
    bandit = MemConBandit(persist_path=q_path, flush_interval=1)
    bandit.begin_episode()

    state = extract_state("run pytest", total_records_in_store=5, last_exit_code=0)
    action = bandit.select_action(state)
    assert action is not None

    reward = bandit.end_episode(success=True, steps=1)
    assert reward > 1.0
    assert bandit.counts[state.to_key()][action.to_key()] == 1


def test_plan_index(temp_env):
    _, _, plans_path = temp_env
    plans = PlanIndex(path=plans_path)
    assert not plans.has_plan("command")
    plans.upsert_plan("command", ["cd /app", "pytest -q"], summary="run tests")
    assert plans.has_plan("command")
    text = plans.format_injection("command")
    assert "Proven plan" in text
    assert "pytest" in text


def test_engine_full_actions(temp_env):
    db_path, q_path, plans_path = temp_env
    mock_llm = lambda prompt: '[{"text": "Production uses PostgreSQL", "memory_type": "fact"}]'
    engine = MemoryEngine(
        llm_client=mock_llm,
        db_path=db_path,
        qtable_path=q_path,
        plans_path=plans_path,
    )

    # Seed store so retrieve/consolidate/forget are feasible.
    emb = engine.embedder.embed_text("Production uses PostgreSQL on port 5432")
    for i in range(4):
        engine.store.save_record(
            MemoryRecord(
                text=f"Production uses PostgreSQL on port 5432 #{i}",
                memory_type="constraint",
                embedding=emb,
            )
        )

    engine.plans.upsert_plan(
        "chat_query",
        ["Check DATABASE_URL", "Connect to Postgres on 5432"],
        summary="db lookup",
    )

    engine.observe_turn("user", "We configured Postgres for production")
    engine.observe_turn(
        "assistant",
        "Postgres ready on 5432",
        exit_code=0,
        command="pg_isready",
        cwd="/app",
    )
    engine.observe_turn("user", "Let us write some poetry")

    # Force stuck + populated so many actions are legal across selects.
    engine.consecutive_failures = 2
    seen_ops = set()
    for _ in range(12):
        ctx, action = engine.prepare_context("Where is production database?")
        seen_ops.add(action.op)
        if action.op in (MemoryOp.RETRIEVE, MemoryOp.RE_RETRIEVE, MemoryOp.PLANINJECT):
            # Context may be empty if similarity is low; action still must be valid.
            assert isinstance(ctx, str)

    assert MemoryOp.RETRIEVE in seen_ops or MemoryOp.PLANINJECT in seen_ops
    assert engine.unique_cwds >= 1

    engine.complete_task(success=True, goal="Where is production database?")
    assert engine.task_index == 1
    engine.shutdown()


def test_engine_facade(temp_env):
    db_path, q_path, plans_path = temp_env
    mock_llm = lambda prompt: '[{"text": "Production uses PostgreSQL", "memory_type": "fact"}]'
    engine = MemoryEngine(
        llm_client=mock_llm,
        db_path=db_path,
        qtable_path=q_path,
        plans_path=plans_path,
    )

    engine.observe_turn("user", "We configured Postgres for production")
    engine.observe_turn("assistant", "Postgres ready on 5432")
    engine.observe_turn("user", "Let us write some poetry")

    ctx, action = engine.prepare_context("Where is production database?")
    assert action.op in {
        MemoryOp.RETRIEVE,
        MemoryOp.NOOP,
        MemoryOp.PLANINJECT,
        MemoryOp.CONSOLIDATE,
        MemoryOp.FORGET,
        MemoryOp.RE_RETRIEVE,
    }
    engine.complete_task(success=True)
    engine.shutdown()
