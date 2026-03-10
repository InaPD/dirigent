"""HTTP surface: envelope shape, validation, and that POST really enqueues."""

import httpx
import pytest
import pytest_asyncio

from ra.api import create_app
from ra.store import RunStore
from tests.factories import make_state
from tests.fakes import FakePool

GOOD_QUESTION = "What is the current state of durable agent execution?"


@pytest_asyncio.fixture
async def client(store: RunStore, settings):
    """The app with its lifespan dependencies injected, so no Redis or arq is needed."""
    app = create_app()
    app.state.settings = settings
    app.state.store = store
    app.state.redis = store.client
    app.state.pool = FakePool()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app
        yield c


async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "data": {"ok": True, "redis": True}, "error": None}


async def test_post_creates_a_queued_run_and_enqueues_it(client, store: RunStore):
    r = await client.post("/research", json={"question": GOOD_QUESTION})
    assert r.status_code == 202

    body = r.json()
    assert body["ok"] is True
    run_id = body["data"]["run_id"]
    assert body["data"]["status"] == "queued"
    assert run_id.startswith("run_")

    state = await store.load(run_id)
    assert state is not None
    assert state.status == "queued"
    assert state.question == GOOD_QUESTION
    assert state.steps == []

    # queued runs are not in runs:active; only a running one is a sweeper candidate
    assert await store.active_runs() == set()

    ((args, kwargs),) = client.app.state.pool.jobs
    assert args == ("run_graph", run_id)
    assert kwargs["_job_id"] == f"{run_id}:0"


async def test_post_trims_the_question(client, store: RunStore):
    r = await client.post("/research", json={"question": f"   {GOOD_QUESTION}   "})
    state = await store.load(r.json()["data"]["run_id"])
    assert state.question == GOOD_QUESTION


async def test_post_accepts_custom_budgets(client, store: RunStore):
    r = await client.post(
        "/research", json={"question": GOOD_QUESTION, "budgets": {"max_subquestions": 2}}
    )
    state = await store.load(r.json()["data"]["run_id"])
    assert state.budgets.max_subquestions == 2


@pytest.mark.parametrize("question", ["", "   ", "too short", "x" * 1001])
async def test_post_rejects_bad_questions(client, question):
    r = await client.post("/research", json={"question": question})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert body["data"] is None
    assert "question" in body["error"]


async def test_post_rejects_a_missing_body(client):
    r = await client.post("/research", json={})
    assert r.status_code == 422
    assert r.json()["ok"] is False


async def test_get_unknown_run_is_404_in_the_envelope(client):
    r = await client.get("/research/run_nope")
    assert r.status_code == 404
    assert r.json() == {"ok": False, "data": None, "error": "no such run"}


async def test_get_trace_unknown_run_is_404(client):
    r = await client.get("/research/run_nope/trace")
    assert r.status_code == 404
    assert r.json()["ok"] is False


async def test_get_returns_the_run_summary(client, store: RunStore):
    state = make_state(status="done", report_markdown="# Report\n", cost_usd=0.0123)
    await store.save(state)

    data = (await client.get(f"/research/{state.run_id}")).json()["data"]
    assert data["run_id"] == state.run_id
    assert data["status"] == "done"
    assert data["report_markdown"] == "# Report\n"
    assert data["cost_usd"] == pytest.approx(0.0123)
    # the summary route must not ship the whole trace
    assert "steps" not in data


async def test_get_trace_returns_steps(client, store: RunStore):
    from ra.schemas import StepRecord

    step = StepRecord(
        seq=1,
        node="plan",
        started_at=make_state().created_at,
        duration_ms=7,
        status="ok",
        input_digest="a" * 12,
        output_digest="b" * 12,
    )
    state = make_state(steps=[step])
    await store.save(state)

    data = (await client.get(f"/research/{state.run_id}/trace")).json()["data"]
    assert data["run_id"] == state.run_id
    assert len(data["steps"]) == 1
    assert data["steps"][0]["node"] == "plan"
    assert data["steps"][0]["duration_ms"] == 7


async def test_oversized_body_is_rejected_before_parsing(client):
    r = await client.post(
        "/research",
        content=b"x" * 20_000,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert r.json()["error"] == "request body too large"


async def test_rate_limit_returns_429(client):
    limit = client.app.state.settings.rate_limit_per_min
    codes = [(await client.get("/research/run_nope")).status_code for _ in range(limit + 2)]
    assert 429 in codes
    assert codes.count(404) == limit


async def test_healthz_is_exempt_from_the_rate_limit(client):
    limit = client.app.state.settings.rate_limit_per_min
    for _ in range(limit + 5):
        assert (await client.get("/healthz")).status_code == 200


async def test_an_oversized_chunked_body_is_still_refused(client):
    """A Content-Length header is whatever the client says. The bytes are what count."""

    async def chunks():
        for _ in range(40):
            yield b"x" * 1024

    r = await client.post(
        "/research", content=chunks(), headers={"content-type": "application/json"}
    )

    assert r.status_code == 413
    assert r.json()["error"] == "request body too large"


async def test_an_understated_content_length_does_not_get_through(client):
    r = await client.request(
        "POST",
        "/research",
        content=b"x" * 20_000,
        headers={"content-type": "application/json", "content-length": "10"},
    )

    assert r.status_code in (413, 422)


async def test_a_normal_body_is_unaffected(client, store):
    r = await client.post("/research", json={"question": GOOD_QUESTION})

    assert r.status_code == 202


@pytest.mark.parametrize(
    "budgets",
    [
        {"max_searches_per_sq": 100_000},
        {"max_tavily_credits": 10**9},
        {"max_total_tokens": 10**12},
        {"max_subquestions": 5_000},
        {"max_extracts_per_sq": -1},
    ],
    ids=["searches", "credits", "tokens", "sub-questions", "negative"],
)
async def test_a_caller_cannot_raise_its_own_budget_past_the_ceiling(client, budgets):
    """POST /research starts paid work, so the caps it accepts are bounded here."""
    r = await client.post("/research", json={"question": GOOD_QUESTION, "budgets": budgets})

    assert r.status_code == 422
    assert r.json()["ok"] is False


async def test_a_caller_may_still_lower_a_budget(client, store):
    r = await client.post(
        "/research", json={"question": GOOD_QUESTION, "budgets": {"max_subquestions": 2}}
    )

    assert r.status_code == 202
    state = await store.load(r.json()["data"]["run_id"])
    assert state.budgets.max_subquestions == 2


# -- GET /runs -----------------------------------------------------------------


async def seed_runs(store, count: int = 3, **overrides):
    from datetime import timedelta

    from ra.clock import now

    base = now()
    made = []
    for i in range(count):
        state = make_state(
            run_id=f"run_seed{i:06d}",
            status="done",
            cost_usd=0.01,
            tokens_in=100,
            tokens_out=20,
            tavily_credits=2,
            created_at=base - timedelta(hours=i),
            **overrides,
        )
        await store.save(state)
        made.append(state)
    return made


async def test_runs_is_empty_before_anything_has_run(client):
    body = (await client.get("/runs")).json()

    assert body["ok"] is True
    assert body["data"]["runs"] == []
    assert body["data"]["window"]["runs"] == 0
    assert body["data"]["spend"]["cost_usd"] == 0


async def test_runs_lists_newest_first_with_totals(client, store):
    await seed_runs(store, 3)

    data = (await client.get("/runs")).json()["data"]

    assert [r["run_id"] for r in data["runs"]] == [
        "run_seed000000",
        "run_seed000001",
        "run_seed000002",
    ]
    assert data["window"]["runs"] == 3
    assert data["spend"]["cost_usd"] == pytest.approx(0.03)
    assert data["spend"]["tavily_credits"] == 6
    assert data["by_status"] == {"done": 3}


async def test_the_totals_describe_the_window_they_cover(client, store):
    await seed_runs(store, 5)

    data = (await client.get("/runs?limit=2")).json()["data"]

    assert len(data["runs"]) == 2
    assert data["window"]["runs"] == 2
    assert data["window"]["truncated"] is True
    assert data["spend"]["cost_usd"] == pytest.approx(0.02)


async def test_the_time_window_excludes_older_runs(client, store):
    await seed_runs(store, 5)  # one per hour going back

    data = (await client.get("/runs?hours=2")).json()["data"]

    assert len(data["runs"]) == 2
    assert data["window"]["since"] is not None


async def test_runs_can_be_filtered_by_status(client, store):
    await seed_runs(store, 2)
    await store.save(make_state(run_id="run_broken", status="failed", error="boom"))

    data = (await client.get("/runs?status=failed")).json()["data"]

    assert [r["run_id"] for r in data["runs"]] == ["run_broken"]
    assert data["runs"][0]["error"] == "boom"


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=501", "hours=0", "hours=99999", "status=nonsense"],
)
async def test_runs_rejects_nonsense_parameters(client, query):
    r = await client.get(f"/runs?{query}")

    assert r.status_code == 422
    assert r.json()["ok"] is False


async def test_runs_reports_where_things_go_wrong(client, store):
    from ra.schemas import StepRecord

    def step(node, status, seq=1):
        return StepRecord(
            seq=seq,
            node=node,
            started_at=make_state().created_at,
            duration_ms=10,
            status=status,
            input_digest="a" * 12,
            output_digest="b" * 12,
        )

    await store.save(
        make_state(
            run_id="run_withsteps",
            status="failed",
            steps=[step("research", "error"), step("plan", "ok", 2)],
        )
    )

    nodes = (await client.get("/runs")).json()["data"]["nodes"]

    assert nodes[0]["node"] == "research"
    assert nodes[0]["error"] == 1


async def test_a_posted_run_shows_up_in_the_listing(client, store):
    posted = await client.post("/research", json={"question": GOOD_QUESTION})
    run_id = posted.json()["data"]["run_id"]

    data = (await client.get("/runs")).json()["data"]

    assert run_id in [r["run_id"] for r in data["runs"]]
    assert data["by_status"] == {"queued": 1}


async def test_the_listing_survives_a_run_whose_document_expired(client, store):
    """The index outlives a document only if someone deletes one, but it must not 500."""
    await seed_runs(store, 2)
    await store.client.delete("run:run_seed000000")

    data = (await client.get("/runs")).json()["data"]

    assert [r["run_id"] for r in data["runs"]] == ["run_seed000001"]
    assert data["window"]["runs"] == 1
