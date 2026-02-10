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


# -- the sweeper ---------------------------------------------------------------


@pytest.fixture
def sweep_ctx(deps):
    return {"store": deps.store, "pool": FakePool(), "worker_id": "worker-sweeper"}


async def test_an_orphaned_run_is_re_enqueued(sweep_ctx):
    from ra.worker import sweep

    store = sweep_ctx["store"]
    state = make_state(status="running")  # in runs:active, with no lease behind it
    await store.save(state)

    await sweep(sweep_ctx)

    ((args, kwargs),) = sweep_ctx["pool"].jobs
    assert args == ("run_graph", state.run_id)
    assert kwargs["_job_id"] == f"{state.run_id}:1"  # a fresh attempt, not the dead one
    assert (await store.load(state.run_id)).attempt == 1


async def test_a_run_with_a_live_lease_is_left_alone(sweep_ctx):
    from ra.worker import sweep

    store = sweep_ctx["store"]
    state = make_state(status="running")
    await store.save(state)
    await store.acquire_lease(state.run_id, "worker-a")

    await sweep(sweep_ctx)

    assert sweep_ctx["pool"].jobs == []
    assert (await store.load(state.run_id)).attempt == 0


@pytest.mark.parametrize("status", ["done", "failed", "budget_exceeded", "queued"])
async def test_a_run_that_is_not_running_is_dropped_from_the_active_set(sweep_ctx, status):
    from ra.worker import sweep

    store = sweep_ctx["store"]
    state = make_state(status="running")
    await store.save(state)
    # the status changed without the active set being updated
    await store.client.set(
        f"run:{state.run_id}", state.model_copy(update={"status": status}).model_dump_json()
    )

    await sweep(sweep_ctx)

    assert sweep_ctx["pool"].jobs == []
    assert await store.active_runs() == set()


async def test_a_run_whose_document_vanished_is_forgotten(sweep_ctx):
    from ra.store import ACTIVE_SET
    from ra.worker import sweep

    store = sweep_ctx["store"]
    await store.client.sadd(ACTIVE_SET, "run_ghost")

    await sweep(sweep_ctx)

    assert sweep_ctx["pool"].jobs == []
    assert await store.active_runs() == set()


async def test_sweeping_an_empty_set_does_nothing(sweep_ctx):
    from ra.worker import sweep

    await sweep(sweep_ctx)

    assert sweep_ctx["pool"].jobs == []


@pytest.mark.parametrize(
    "interval, expected",
    [(30, [0, 30]), (60, [0]), (15, [0, 15, 30, 45]), (0, 60), (120, [0])],
    ids=["default", "once a minute", "four times", "clamped up", "clamped down"],
)
def test_the_sweep_schedule_is_clamped_to_a_minute(interval, expected):
    from ra.worker import sweep_schedule

    schedule = sweep_schedule(interval)
    if isinstance(expected, int):
        assert len(schedule) == expected
    else:
        assert sorted(schedule) == expected


# -- the lease heartbeat -------------------------------------------------------


async def test_the_heartbeat_keeps_a_slow_node_from_being_swept(deps, monkeypatch):
    """A node can outlive the lease TTL. The heartbeat is what stops that looking like death."""
    import asyncio

    from ra.worker import _hold_lease

    store = deps.store
    await store.acquire_lease("run_slow", "worker-a")
    task = asyncio.create_task(_hold_lease(store, "run_slow", "worker-a"))
    try:
        await asyncio.sleep(1.2)  # longer than lease_ttl_s / 3 for the test's ttl of 2
        assert await store.lease_holder("run_slow") == "worker-a"
    finally:
        task.cancel()


async def test_the_heartbeat_stops_when_the_lease_is_lost(deps):
    """If another worker has taken over, stop refreshing and let it own the run."""
    import asyncio

    from ra.worker import _hold_lease

    store = deps.store
    await store.acquire_lease("run_lost", "worker-a")
    task = asyncio.create_task(_hold_lease(store, "run_lost", "worker-a"))

    await store.release_lease("run_lost", "worker-a")
    await store.acquire_lease("run_lost", "worker-b")

    await asyncio.wait_for(task, timeout=3)

    assert await store.lease_holder("run_lost") == "worker-b"


async def test_shutdown_closes_the_search_client(deps):
    """The Tavily client owns an httpx session, so the worker has to close it."""
    from ra.worker import on_shutdown
    from tests.fakes import FakeSearch

    class ClosableSearch(FakeSearch):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def aclose(self):
            self.closed = True

    search = ClosableSearch()
    ctx = {"deps": Deps(store=deps.store, settings=deps.settings, search=search)}

    await on_shutdown(ctx)

    assert search.closed is True
