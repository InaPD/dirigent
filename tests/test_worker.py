"""The worker's lease discipline and its failure paths.

These are the paths Phase 5's sweeper depends on, so they are worth pinning now.
"""

import pytest

import ra.nodes.canned as canned
from ra.deps import Deps
from ra.graph import build_graph
from ra.schemas import RunState
from ra.worker import enqueue_run, job_id, run_graph
from tests.factories import make_state
from tests.fakes import FakePool


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


@pytest.fixture
def ctx(deps: Deps):
    return {"store": deps.store, "worker_id": "worker-a", "graph": build_graph(deps)}


class ExplodingGraph:
    async def ainvoke(self, *_args, **_kwargs):
        raise RuntimeError("recursion limit reached")


def test_job_id_includes_the_attempt():
    assert job_id(make_state(attempt=0)) == "run_test000001:0"
    assert job_id(make_state(attempt=3)) == "run_test000001:3"


async def test_enqueue_uses_the_attempt_suffixed_job_id():
    pool = FakePool()
    await enqueue_run(pool, make_state(attempt=2))
    ((args, kwargs),) = pool.jobs
    assert args == ("run_graph", "run_test000001")
    assert kwargs["_job_id"] == "run_test000001:2"


async def test_happy_path_marks_the_run_running_then_done(ctx):
    store = ctx["store"]
    state = make_state(status="queued", started_at=None)
    await store.save(state)

    await run_graph(ctx, state.run_id)

    final = await store.load(state.run_id)
    assert final.status == "done"
    assert final.worker_id == "worker-a"
    assert final.started_at is not None
    assert await store.lease_holder(state.run_id) is None


async def test_a_run_held_by_another_worker_is_left_alone(ctx):
    store = ctx["store"]
    state = make_state(status="queued")
    await store.save(state)
    await store.acquire_lease(state.run_id, "worker-b")

    await run_graph(ctx, state.run_id)

    assert await store.load(state.run_id) == state  # untouched
    assert await store.lease_holder(state.run_id) == "worker-b"  # and not stolen


async def test_a_missing_run_document_is_survivable(ctx):
    await run_graph(ctx, "run_does_not_exist")
    assert await ctx["store"].lease_holder("run_does_not_exist") is None


@pytest.mark.parametrize("status", ["done", "failed", "budget_exceeded"])
async def test_a_finished_run_is_not_restarted(ctx, status):
    store = ctx["store"]
    state = make_state(status=status)
    await store.save(state)

    await run_graph(ctx, state.run_id)

    assert await store.load(state.run_id) == state
    assert await store.lease_holder(state.run_id) is None


async def test_a_graph_level_failure_is_recorded_and_the_lease_released(deps):
    store = deps.store
    state = make_state(status="queued")
    await store.save(state)
    ctx = {"store": store, "worker_id": "worker-a", "graph": ExplodingGraph()}

    await run_graph(ctx, state.run_id)

    final: RunState = await store.load(state.run_id)
    assert final.status == "failed"
    assert "recursion limit reached" in final.error
    assert final.finished_at is not None
    assert await store.lease_holder(state.run_id) is None
    # a failed run must not sit in the active set waiting for a sweeper
    assert await store.active_runs() == set()


async def test_the_lease_is_released_even_when_loading_fails(deps, monkeypatch):
    store = deps.store
    state = make_state(status="queued")
    await store.save(state)

    async def broken_load(_run_id):
        raise ConnectionError("redis went away")

    monkeypatch.setattr(store, "load", broken_load)
    ctx = {"store": store, "worker_id": "worker-a", "graph": build_graph(deps)}

    with pytest.raises(ConnectionError):
        await run_graph(ctx, state.run_id)

    assert await store.lease_holder(state.run_id) is None


def test_build_deps_skips_the_model_client_without_a_key(settings):
    from ra.worker import build_deps

    deps = build_deps(settings, store=None)
    assert deps.llm is None


def test_build_deps_builds_the_model_client_when_a_key_is_set():
    from ra.config import Settings
    from ra.worker import build_deps

    with_key = Settings(anthropic_api_key="sk-ant-test", tavily_api_key=None)
    deps = build_deps(with_key, store=None)
    assert deps.llm is not None
