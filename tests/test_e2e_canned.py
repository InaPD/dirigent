"""The Phase 1 exit criterion.

A run travels POST -> arq queue -> worker -> graph -> Redis -> GET, using a real Redis and
a real arq worker. Only the four work nodes are fake. If this passes, every later phase is
swapping out node bodies inside a frame that already works.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import create_worker

import ra.nodes.canned as canned
from ra.api import create_app
from ra.config import get_settings
from ra.store import RunStore
from ra.worker import WorkerSettings
from tests.conftest import TEST_REDIS_URL

pytestmark = pytest.mark.redis

GOOD_QUESTION = "What is the current state of durable agent execution?"
EXPECTED_NODES = ["plan", "research", "research", "review", "write"]
DRAIN_TIMEOUT_S = 30


@pytest.fixture(autouse=True)
def fast_nodes(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


@pytest.fixture(autouse=True)
def test_redis_env(monkeypatch):
    """Point every component, including the worker's own on_startup, at the test database.

    The keys are blanked so the worker selects the canned nodes. This test is about the
    pipeline, and it has to pass on a machine with a real .env and in CI without one.
    """
    monkeypatch.setenv("REDIS_URL", TEST_REDIS_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def api(real_store: RunStore):
    app = create_app()
    app.state.settings = get_settings()
    app.state.store = real_store
    app.state.redis = real_store._r
    app.state.pool = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await app.state.pool.aclose()


async def drain_the_queue() -> None:
    """Run a real arq worker in burst mode: it takes what is queued, then stops."""
    worker = create_worker(
        WorkerSettings,
        redis_settings=RedisSettings.from_dsn(TEST_REDIS_URL),
        burst=True,
        handle_signals=False,
        poll_delay=0.05,
    )
    try:
        await asyncio.wait_for(worker.async_run(), timeout=DRAIN_TIMEOUT_S)
    finally:
        await worker.close()


async def test_a_canned_run_completes_through_the_real_queue(api, real_store: RunStore):
    posted = await api.post("/research", json={"question": GOOD_QUESTION})
    assert posted.status_code == 202
    run_id = posted.json()["data"]["run_id"]

    # before the worker touches it
    pending = (await api.get(f"/research/{run_id}")).json()["data"]
    assert pending["status"] == "queued"
    assert pending["report_markdown"] is None

    await drain_the_queue()

    done = (await api.get(f"/research/{run_id}")).json()["data"]
    assert done["status"] == "done"
    assert done["report_markdown"].startswith("# ")
    assert done["started_at"] is not None
    assert done["finished_at"] is not None
    assert done["error"] is None


async def test_the_trace_is_readable_over_http(api, real_store: RunStore):
    run_id = (await api.post("/research", json={"question": GOOD_QUESTION})).json()["data"][
        "run_id"
    ]
    await drain_the_queue()

    steps = (await api.get(f"/research/{run_id}/trace")).json()["data"]["steps"]
    assert [s["node"] for s in steps] == EXPECTED_NODES
    assert [s["seq"] for s in steps] == [1, 2, 3, 4, 5]
    assert all(s["status"] == "ok" for s in steps)
    assert all(len(s["input_digest"]) == 12 for s in steps)


async def test_the_worker_releases_its_lease_and_clears_the_active_set(api, real_store: RunStore):
    run_id = (await api.post("/research", json={"question": GOOD_QUESTION})).json()["data"][
        "run_id"
    ]
    await drain_the_queue()

    assert await real_store.lease_holder(run_id) is None
    assert await real_store.active_runs() == set()
    state = await real_store.load(run_id)
    assert state.worker_id is not None


async def test_running_the_same_run_twice_is_a_no_op(api, real_store: RunStore):
    """A duplicate delivery must not append a second copy of every step."""
    run_id = (await api.post("/research", json={"question": GOOD_QUESTION})).json()["data"][
        "run_id"
    ]
    await drain_the_queue()
    first = await real_store.load(run_id)

    pool = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
    await pool.enqueue_job("run_graph", run_id, _job_id=f"{run_id}:replay")
    await pool.aclose()
    await drain_the_queue()

    again = await real_store.load(run_id)
    assert [s.seq for s in again.steps] == [s.seq for s in first.steps]
    assert again.report_markdown == first.report_markdown
