# Implementation plan - Multi-Agent Research Assistant

Technical breakdown of [research-agent-plan.md](research-agent-plan.md) into build phases.
The source plan fixes *what* and *why*; this document fixes *how*: files, signatures,
key mechanisms, tests, and exit criteria per phase. Phases map 1:1 to the sessions in the
source plan. Order is fixed. Dates can slide.

Conventions used throughout:

- Package is `ra` under `src/ra/`. Import as `from ra.schemas import RunState`.
- Async everywhere (FastAPI, arq, `redis.asyncio`, `httpx.AsyncClient`, `anthropic.AsyncAnthropic`).
- State is never mutated in place. Nodes return a new `RunState` via `state.model_copy(update=...)`.
- Every external failure becomes a status on a record, never an exception out of a node.
- Tests live in `tests/`, mirror the module name (`tests/test_store.py`), and use a real Redis
  (compose service locally, service container in CI). Unit tests that do not need Redis use
  `fakeredis` so the suite runs without Docker.
- Target 80% line coverage from Phase 1 onward. `make test` fails below it.

---

## Phase 0 - Pre-flight (30 min, before Phase 1)

Nothing here needs a design decision. Do it in one sitting.

**Accounts**

1. Anthropic Console: new workspace `research-agent`, spend limit $20/month, alert at 50%.
   Create the API key inside that workspace. Never paste it anywhere but `.env`.
2. Tavily: free-tier key. Note the credit costs in `search.py` constants later:
   basic search = 1 credit, advanced = 2, extract = 1 per 5 URLs (verify on the pricing page
   when you write `search.py`, do not trust this line).

**Local toolchain** (checked on this machine on 14 Sep 2026)

| Tool | State | Action |
|---|---|---|
| Python 3.12.3 | present | none |
| Docker 28 + Compose v2.35 | present | Redis runs in compose, no local install needed |
| GNU Make | present | none |
| `uv` | missing | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `gitleaks` | missing | install binary, then `pre-commit` hook (see below) |
| `redis-cli` | missing | optional; `docker compose exec redis redis-cli` covers it |

**Repo**

The git remote still points at `InaPD/ecomm-app`. Either rename that GitHub repo to
`research-agent` and keep history, or create a fresh repo and `git remote set-url origin`.
Decide before the first push. Both work with this plan.

First commit, in this order so no key can ever land in history:

```
.gitignore            .env, .venv/, __pycache__/, .pytest_cache/, .coverage, htmlcov/
.env.example          ANTHROPIC_API_KEY=sk-ant-..., TAVILY_API_KEY=tvly-..., REDIS_URL=redis://redis:6379/0
.pre-commit-config.yaml   gitleaks hook (repo: gitleaks/gitleaks, id: gitleaks)
pyproject.toml        see below
compose.yml           api, worker, redis services (bodies filled in Phase 1)
Makefile              targets stubbed: up, down, test, demo, lint
README.md             one line placeholder
docs/                 the two plan files
```

`pyproject.toml` skeleton (pin everything; refresh pins on the day):

```toml
[project]
name = "ra"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi", "uvicorn[standard]", "arq", "redis",
  "langgraph", "anthropic", "httpx", "pydantic>=2", "pydantic-settings",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "pytest-cov", "respx", "fakeredis", "ruff", "pre-commit"]
[tool.pytest.ini_options]
asyncio_mode = "auto"
[tool.ruff]
line-length = 100
```

Run `uv lock` immediately and commit `uv.lock`. Then `uv add` pins the exact versions into
the lock file, which is what "pin it" means here. LangGraph is the one that drifts; if a
minor bump breaks the `Command` API, stay on the locked version.

**Exit:** `uv sync` works, `pre-commit run --all-files` runs gitleaks, `.env` is ignored.

---

## Phase 1 - Skeleton end to end (Session 1, Mon 14 Sep)

Goal: a canned run travels API -> queue -> worker -> graph -> Redis -> API with no
intelligence anywhere. Everything later slots into this frame.

### 1.1 `config.py`

`Settings(BaseSettings)` with `anthropic_api_key: SecretStr`, `tavily_api_key: SecretStr`,
`redis_url: str`, `worker_id: str = hostname:pid`, `lease_ttl_s: int = 60`,
`sweeper_interval_s: int = 30`, plus a `models: ModelConfig` object:

```python
class ModelConfig(BaseModel):
    planner: str = "claude-haiku-4-5"
    reviewer: str = "claude-haiku-4-5"
    researcher: str = "claude-sonnet-5"
    writer: str = "claude-sonnet-5"
```

Model IDs verified against the Claude API skill reference on 14 Sep 2026. They are complete
as written; never append date suffixes. One `get_settings()` with `lru_cache`. Missing keys
fail at startup with a clear message (Settings raises on missing required fields).
Keys are only required by the worker; the API process needs Redis only. Split into
`ApiSettings` and `WorkerSettings` if the single class becomes awkward.

### 1.2 `schemas.py`

Exactly the models from the source plan, plus two fields on `RunState` that the resume
mechanism needs:

```python
attempt: int = 0  # incremented on every (re)enqueue; part of the arq job_id
worker_id: str | None = None
```

Add `ids.py` with `new_run_id() -> "run_" + 12 hex`, `new_finding_id() -> "f_" + 6 hex`,
`sq_id(n) -> f"sq_{n:02d}"`. Keep IDs short; they appear in traces and prompts.

Test: round trip every model through `model_dump_json` / `model_validate_json`. Assert
`RunState` with defaults serialises to under 2 KB (guards against accidentally storing
raw page content on the state later).

### 1.3 `store.py`

One class, `RunStore(redis: Redis)`, all methods async:

| Method | Redis ops | Notes |
|---|---|---|
| `save(state)` | `SET run:{id} <json>` | also `SADD runs:active` if running, `SREM` otherwise |
| `load(run_id) -> RunState \| None` | `GET` | `None` on miss, never raises |
| `acquire_lease(run_id, worker_id) -> bool` | `SET run:{id}:lease worker_id NX EX ttl` | false if held |
| `refresh_lease(run_id, worker_id) -> bool` | Lua: if value == worker_id then EXPIRE | never refresh a lease you do not own |
| `release_lease(run_id, worker_id)` | Lua: if value == worker_id then DEL | same |
| `lease_holder(run_id) -> str \| None` | `GET` | used by sweeper and tests |
| `active_runs() -> set[str]` | `SMEMBERS runs:active` | |
| `cache_get/set(key, value, ttl)` | `GET` / `SET EX` | used by `search.py` in Phase 3 |

The two Lua scripts are the only subtle part: refresh and release must be compare-and-act,
otherwise a worker that lost its lease during a long node could refresh a lease now held by
the replacement worker. Register them once with `redis.register_script`.

Tests (real Redis or fakeredis with Lua support): save/load round trip, lease NX semantics,
refresh by non-owner returns false and leaves TTL untouched, release by non-owner is a no-op,
`runs:active` membership tracks status.

### 1.4 `graph.py`

Thin LangGraph. State schema is `RunState` (LangGraph accepts a Pydantic model as the
state type). Nodes: `router`, `plan`, `research`, `review`, `write`. Wiring:

```
START -> router
router -> Command(goto=one of plan|research|review|write|END)
plan|research|review|write -> router      (plain edges)
```

Every work node returns the *whole* updated `RunState` as a dict (`state.model_dump()`);
do not rely on per-field reducers, they add nothing here and complicate resume.

Router in Phase 1 is already the final router (it is pure state, so write it once):

```python
def route(state: RunState) -> str:
    if state.status in ("done", "failed"):
        return END
    if state.report is not None:
        return END
    if not state.plan:
        return "plan"
    if next_open_subquestion(state) is not None:
        return "research"
    if not state.reviewed:
        return "review"
    return "write"
```

`next_open_subquestion` returns the first sub-question with status `pending` or
`needs_one_more_pass`, or `None`. Add `reviewed: bool = False` to `RunState`. The research
node reads `next_open_subquestion(state)` itself, so the router passes nothing.

`build_graph(deps: Deps) -> CompiledStateGraph`. `Deps` is a small dataclass holding
`store`, `llm`, `search`, `settings`, so nodes are constructed with their dependencies and
tests can inject stubs. Compile with `recursion_limit=100` at invoke time: a run visits
`router` once per node, so 4 sub-questions with one revision each is already ~20 steps.

Built and verified on LangGraph 1.2.11. Dispatch uses `add_conditional_edges("router",
route, path_map)` rather than a router node returning `Command(goto=...)`. Same graph, same
pure-state routing, but `add_node` / `add_edge` / `add_conditional_edges` are the oldest and
most stable surface LangGraph has, which is what the "pin it, it drifts" note is really
asking for. The router stays a node so Phase 2 can record a budget step from it.

Phase 1 nodes: each sleeps 1 s and writes canned data (a 2-item plan, one finding per
sub-question, a one-section report). Nodes save state through `deps.store.save` before
returning. The `@traced` decorator arrives in Phase 2; in Phase 1 nodes append a bare
`StepRecord` by hand so the trace endpoint has data.

### 1.5 `worker.py`

arq `WorkerSettings` with `functions=[run_graph]`, `cron_jobs` empty until Phase 5,
`redis_settings` from `REDIS_URL`, `max_jobs=4`, `job_timeout=max_wall_clock_s + 60`.

```python
async def run_graph(ctx, run_id: str) -> None:
    store = ctx["store"]
    worker_id = ctx["worker_id"]
    if not await store.acquire_lease(run_id, worker_id):
        return  # someone else has it
    state = await store.load(run_id)
    if state is None or state.status in ("done", "failed", "budget_exceeded"):
        await store.release_lease(run_id, worker_id)
        return
    state = state.model_copy(
        update={
            "status": "running",
            "started_at": state.started_at or now(),
            "worker_id": worker_id,
        }
    )
    await store.save(state)
    try:
        await graph.ainvoke(state.model_dump(), config={"recursion_limit": 100})
    except Exception as e:  # graph-level failure, not node-level
        final = (await store.load(run_id)).model_copy(
            update={"status": "failed", "error": repr(e)[:500], "finished_at": now()}
        )
        await store.save(final)
    finally:
        await store.release_lease(run_id, worker_id)
```

Enqueue helper `enqueue_run(pool, state)` uses `_job_id=f"{run_id}:{attempt}"`. arq refuses a
duplicate `job_id` while the job is queued or running, which gives idempotent enqueue for
free; the attempt suffix lets the sweeper re-enqueue after a crash without colliding with the
stale job record from the dead worker.

`on_startup` builds `ctx["store"]`, `ctx["graph"]`, `ctx["worker_id"]`.

### 1.6 `api.py`

FastAPI with lifespan that opens the Redis client and the arq pool.

| Route | Body / response | Notes |
|---|---|---|
| `POST /research` | `{question: str, budgets?: Budgets}` -> 202 `{run_id, status}` | question length 10..1000 after strip, validated by a Pydantic request model |
| `GET /research/{run_id}` | `{run_id, status, question, report_markdown, cost_usd, tokens_in, tokens_out, created_at, finished_at, error}` | 404 if unknown |
| `GET /research/{run_id}/trace` | `{run_id, steps: [StepRecord]}` | 404 if unknown |
| `GET /healthz` | `{ok: true, redis: true}` | compose healthcheck |

Response envelope per the house rules: `{ok: bool, data: ..., error: str | null}`.
Error responses never include stack traces. A simple per-IP token bucket middleware
(in-memory, 30 req/min) satisfies the rate-limit rule without a dependency; note in the
README that a real deployment moves this to the gateway.

### 1.7 `compose.yml`

```yaml
services:
  redis:   { image: redis:7-alpine, healthcheck: redis-cli ping, ports: ["6379:6379"] }
  api:     { build: ., command: uvicorn ra.api:app --host 0.0.0.0 --port 8000, env_file: .env, depends_on: redis healthy, ports: ["8000:8000"] }
  worker:  { build: ., command: arq ra.worker.WorkerSettings, env_file: .env, depends_on: redis healthy }
```

One `Dockerfile` (python:3.12-slim, install `uv`, `uv sync --frozen --no-dev`, copy `src/`).
`redis` publishes 6379 so tests on the host can use `REDIS_URL=redis://localhost:6379/1`
(db 1 for tests, db 0 for the app, so `make test` can `FLUSHDB` safely).

### 1.8 Tests for Phase 1

- `test_store.py` as in 1.3.
- `test_router.py`: table-driven, one row per state shape -> expected node. This is the
  most valuable test in the repo; keep it exhaustive.
- `test_api.py`: `httpx.AsyncClient(app=...)`, POST creates a `run:{id}` in Redis with
  status `queued`, GET on unknown id is 404, question too short is 422.
- `test_e2e_canned.py`: run a real arq worker in burst mode against a real Redis, POST,
  drain the queue, assert the report is present and `len(steps) == 5` (plan, research x2,
  review, write; the router runs between each one but records no step).

**Exit:** `docker compose up`, `curl -X POST localhost:8000/research -d '{"question":"..."}'`,
poll GET, canned report appears. All Phase 1 tests green.

---

## Phase 2 - Trace and budgets (Session 2, Tue 15 Sep)

Goal: every step is measured from real `usage`, and every cap provably stops a run.

### 2.1 `pricing.py`

```python
PRICES_USD_PER_MTOK = {
    "claude-sonnet-5":  Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-4-5": Price(input=1.00, output=5.00,  cache_write=1.25, cache_read=0.10),
}
def cost_usd(model: str, usage: Usage) -> float
```

Input/output rates are from the Claude API reference dated 24 Jun 2026 (Sonnet 5 $2/$10,
Haiku 4.5 $1/$5 per million). Cache multipliers (1.25x write, 0.1x read) are the documented
defaults; confirm both on the pricing page the day you write this file and record the date
in a module docstring. Unknown model -> raise at import of the config, not at run time.

### 2.2 `llm.py`

Verified against the installed SDK: `anthropic` 1.5.0 exposes
`AsyncAnthropic().messages.parse` with both `output_format` and `output_config`.

**The SDK runs on `httpx2`, not `httpx`, so respx cannot intercept it.** Tests pass a
`httpx2.MockTransport` through the `http_client` argument instead, pointed at a fake host so
a mock that fails to match cannot reach the real API. This applies only to the Anthropic
client. Phase 3's `search.py` uses `httpx` directly, so respx works there as planned.

`class LLM` wrapping `anthropic.AsyncAnthropic`. One public method:

```python
async def structured(self, *, model: str, system: str, user: str,
                     schema: type[T], max_tokens: int = 4096) -> LLMResult[T]
```

Uses `client.messages.parse(model=..., max_tokens=..., system=..., messages=[...],
output_format=schema)` and returns `LLMResult(parsed: T, usage: Usage, model: str,
latency_ms: int)`. `Usage` copies `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens` from `response.usage`. Nothing is
ever estimated. Check `response.stop_reason`; `max_tokens` or `refusal` -> raise
`LLMError(kind=...)` so the node records `status="error"`.

Thinking: leave defaults. Sonnet 5 runs adaptive thinking when the parameter is omitted;
Haiku 4.5 runs without thinking unless given `budget_tokens`, and none of these steps need
it. Do not pass `temperature`; it is rejected on Sonnet 5.

Retries: SDK default `max_retries=2` handles 429/5xx. Do not add another retry layer.

`FakeLLM` in `tests/fakes.py`: takes a dict `schema -> instance or callable`, returns
fixed usage numbers so cost assertions are exact.

### 2.3 `trace.py`

```python
def traced(node: str):
    def deco(fn: Callable[[RunState, Deps], Awaitable[NodeOutcome]]): ...
```

The decorated node returns `NodeOutcome(state: RunState, status, model, usage, tool_calls,
sub_question_id, error)`. The decorator:

1. `t0 = perf_counter()`, `input_digest = sha256(state.model_dump_json(exclude={"steps"}))[:12]`.
2. Calls the node inside `try`. An exception becomes `status="error"`, `error=repr(e)[:500]`,
   state unchanged.
3. Builds `StepRecord(seq=len(state.steps)+1, ...)`, `output_digest` over the new state.
4. Appends the step, adds usage and cost to the run totals (immutably), `await deps.store.save`,
   `await deps.store.refresh_lease`.
5. Returns the new state as a dict for LangGraph.

Digests exclude `steps` so re-running the same node on the same state gives the same
`input_digest`. That is what makes the kill test's "exactly once" assertion meaningful.

### 2.4 `budgets.py`

```python
class BudgetVerdict(BaseModel):
    ok: bool
    exceeded: list[str]        # e.g. ["max_total_tokens", "max_wall_clock_s"]

def check_budgets(state: RunState, now: datetime) -> BudgetVerdict
```

Checks: `tokens_in + tokens_out + writer_reserve_tokens > max_total_tokens` (skipped when
the current node *is* `write`), `now - started_at > max_wall_clock_s`, `tavily_credits >=
max_tavily_credits`. Per-sub-question caps (`max_searches_per_sq`, `max_extracts_per_sq`)
are enforced inside `research`, not here. `max_subquestions` is enforced in `plan`.

Router change: before dispatching, check the caps. When one trips, record a
`StepRecord(node="budget", status="budget_exceeded", error=", ".join(exceeded))` and set a
**`budget_stopped: bool` flag on `RunState`**, then route to `write` if there are findings,
else end. The write node runs once more with the reserve and sets the final status.

The flag is separate from `status` on purpose. If the router set `status="budget_exceeded"`
directly, a client polling for a terminal status would see the run finish a full node before
the report existed, and would read `report_markdown: null` from a run that was about to
produce one. So a stopped run with findings stays `running` until the writer is done, and
`route()` dispatches on `budget_stopped` rather than on the status. A stopped run with no
findings has nothing to write, so it does end immediately.

If write itself trips the wall clock it still runs; the cap is advisory for the final write.

### 2.5 Tests

- `test_pricing.py`: known usage -> exact cost, unknown model raises.
- `test_llm.py`: `respx` mock of `/v1/messages` returning a structured-output body; assert
  usage copied verbatim and `stop_reason="max_tokens"` raises.
- `test_trace.py`: decorator appends one step, totals increase, save and refresh called,
  exception inside node -> `status="error"` and run continues.
- `test_budgets.py`: parametrised over each cap with spend already past it -> run ends
  `budget_exceeded`, `report_markdown` is not `None` when findings existed, `None` otherwise,
  and a `budget` step appears in the trace naming every cap that tripped. Note that a cap
  measures spend, not intent: a token cap of 1 does not trip a run that has spent nothing,
  so the parametrised cases seed the spend they are testing.

**Exit:** `GET /research/{id}/trace` shows model, tokens, cost, latency per step.
Caps provably stop a run.

---

## Phase 3 - Researcher with real search (Session 3, Wed 16 Sep)

This is the first risky phase. Budget the whole evening for it.

### 3.1 `search.py`

`class Tavily(api_key: str, store: RunStore, client: httpx.AsyncClient | None)` using
`httpx` directly (not the Tavily SDK) so `respx` can mock it and so the cache layer sits in
one place.

Request and response shapes verified against the Tavily API reference on 14 Sep 2026:
base `https://api.tavily.com`, auth `Authorization: Bearer tvly-...`. Search takes `query`,
`search_depth` (basic 1 credit, advanced 2) and `max_results`; each result carries `title`,
`url`, `content` and `score`. Extract takes `urls` (max 20) and `extract_depth`, returning
`results[].raw_content` plus `failed_results[].error`. Basic extraction costs 1 credit per
5 successful URLs.

**Pass `include_usage: true` on both calls.** The response then carries `usage.credits`, so
credits are read rather than computed, the same rule the token accounting follows. The
documented rates stay in the code only as a fallback for a response with no usage block.

```python
async def search(self, query: str, *, depth="basic", max_results=5) -> SearchResult
async def extract(self, urls: list[str]) -> list[ExtractResult]
```

`SearchOutcome(status, hits: list[Hit], credits, cached, error, tool_call)` and
`ExtractBatch(results: list[ExtractOutcome], credits, tool_calls)`. Credits are charged per
request, not per URL, so extraction returns a batch rather than a bare list.

Failure mapping, exhaustive, in one function `classify_response(resp | exc) -> status`:

| Condition | Status |
|---|---|
| 200, empty results | `empty`, which the node records as `skipped` |
| 200, extract body empty or under 200 chars | `paywalled` |
| 200, extract failed entry in `failed_results` | `error` |
| 200, a requested URL in neither list | `error` |
| 429 | honour `Retry-After` once (cap 10 s), then `rate_limited` |
| 429 with an unparseable or oversized `Retry-After` | `error`, with no wait |
| `httpx.TimeoutException` | `timeout` |
| any other 4xx/5xx, connection error, or malformed JSON | `error` |

**`extraction_status` describes the content held, not the reason it fell short.** `full` is
the page body; `snippet_only` means the search snippet was used instead, whatever the
reason; `paywalled`, `timeout` and `error` mean nothing usable came back at all. The reason
lives on the `ToolCall`, where it belongs. Keeping the two apart is what lets the writer
rank sources on quality without knowing anything about HTTP. One function,
`resolve_extraction(extract, hit)`, makes that call.

Cache keys `cache:search:{sha256(query|depth|max_results)}` and `cache:extract:{sha256(url)}`,
TTL 7 days. Cache hits report `credits=0`. Every call appends a `ToolCall` to the current
node's outcome (`args_digest` = first 12 hex of the sha256 of the args).

### 3.2 `nodes/research.py`

One sub-question per invocation:

1. `sq = next_open_subquestion(state)`.
2. Up to `max_searches_per_sq` searches. First query is `sq.text`; if results are under 3,
   one reformulation via the researcher model (structured `{query: str}`), counted against
   the cap. Stop early when enough unique URLs.
3. Dedupe URLs against `{f.source_url for f in state.findings}` and within the batch.
4. Extract top `max_extracts_per_sq` URLs.
5. One LLM call, model `researcher`, schema `list[FindingDraft]` where `FindingDraft` has
   `claim`, `source_url`, `snippet` (max 400 chars). Prompt: "Return at most 5 findings that
   directly answer the sub-question. Each claim must be supported by the quoted snippet
   from the given source. Skip sources that do not help." Content per source is truncated
   to 6,000 characters before prompting; record the truncation as `truncated:N` in the
   step's `note` field, not `error`, because it is not a failure.
6. Each draft becomes a `Finding` with `extraction_status` copied from its source's
   `ExtractResult` and `id=new_finding_id()`. Drafts pointing at URLs not in the batch are
   dropped (the LLM invented a URL) and counted in the step's `note` field as `dropped:N`.
7. `sq.passes += 1`, `sq.status = "answered"` if any finding else `"unanswerable"`.
   The review node may later flip `answered` to `needs_one_more_pass`.
8. `tavily_credits` incremented by the sum of `ToolCall.credits`.

Sub-question status is written only at the end of the node, together with the findings,
in one `store.save`. That is the atomicity the kill test depends on: either the whole
sub-question landed or none of it did.

### 3.3 Tests

- `test_search.py` with `respx`: 429 with `Retry-After: 1` -> one retry then success; 429
  twice -> `rate_limited`; an oversized `Retry-After` -> no wait at all; timeout ->
  `timeout`; 4xx, 5xx, connection failure and malformed JSON -> `error`; short body ->
  `paywalled`; second identical call -> `cached=True, credits=0`. A blank
  `ANTHROPIC_API_KEY` or `TAVILY_API_KEY` counts as absent, so an exported-but-empty
  variable falls back to the canned node rather than failing on the first call.
- `test_research_node.py`: `FakeLLM` returns drafts including one invented URL; assert it
  is dropped, the rest have correct `extraction_status`, URL dedupe against prior findings,
  `passes` incremented, per-sq caps respected (count `ToolCall`s).
- One opt-in live test `@pytest.mark.live` with a hardcoded 3-sub-question plan against real
  Tavily and Sonnet. Skipped without keys. Run it once by hand and read the findings.

**Exit:** a hardcoded three-sub-question plan produces real findings with provenance.

---

## Phase 4 - Planner and writer (Session 4, Thu 17 Sep)

### 4.1 `nodes/plan.py`

Model `planner`. Schema `Plan(sub_questions: list[str])`. Prompt asks for 2 to N
independently searchable, non-overlapping sub-questions, N = `max_subquestions`, one line
each, no numbering. Truncate to N if the model over-delivers. Assign `sq_01..`.

**No plan means the run fails here, in the planner.** The original plan routed an empty plan
to `write`, but the router sends a run with no plan straight back to the planner, so anything
short of ending the run is an infinite loop. Measured before the fix: a bad API key produced
50 planner attempts before the graph hit its recursion limit, each one a paid call. The node
therefore catches every exception, not just `LLMError`, and any failure to produce
sub-questions sets `status="failed"` with the reason.

### 4.2 `nodes/write.py`

Model `writer`. Schema is `Report` minus `generated_at`. Prompt structure:

```
Question: ...
Findings (cite by id only, never by URL):
f_a1b2c3 [full]         claim ... (source title)
f_d4e5f6 [snippet_only] claim ...
Rules: every claim cites at least one finding id from the list; prefer [full] findings;
do not introduce facts that are not in a finding; 3 to 6 sections.
```

Validation `validate_citations(draft, findings) -> list[str]` returns the complaints, and
an empty list is the only thing that permits rendering. It catches two faults: an id that is
not in `state.findings`, and a claim that cites nothing at all. The second matters as much as
the first, since an uncited claim is exactly the unsupported assertion the whole design is
meant to make impossible.

Non-empty -> one retry, with the offending ids named in the prompt. Still non-empty -> step
`status="error"`, run `status="failed"`, error `"citation validation failed: ..."`. Never
render an unvalidated report.

If `state.budget_stopped`, the prompt gains one line: "Research was cut short by a budget
cap; say so in a short closing note." The report title gets a `(partial)` suffix in the
renderer, never from the model.

### 4.3 `render.py`

Pure function `render_markdown(report, findings, *, partial: bool = False) -> str`. The
`partial` flag is what adds the title suffix; replay passes `state.budget_stopped` for it.
Footnote numbering is by first appearance of a finding id, stable across re-renders.
Output: title, sections with claims as paragraphs each ending in `[n]` markers, then a
`## Sources` list `[n] title - url (retrieved YYYY-MM-DD, status)`. Deterministic: no
timestamps other than the ones inside the records, sorted output where any set is
involved. Replay in Phase 6 asserts byte identity, so keep this function boring.

### 4.4 Tests

- `test_plan_node.py`: over-delivery truncated to `max_subquestions`; empty plan -> step error.
- `test_write_node.py`: `FakeLLM` first returns an invented id, then a valid report ->
  two LLM calls, step `ok`; always invalid -> two calls, step `error`, run `failed`.
- `test_render.py`: golden-file test against `tests/golden/report_small.md`.
- Live: one real question end to end (`@pytest.mark.live`). Read the report yourself.

**Exit:** real question -> real cited report, once, end to end.

---

## Phase 5 - Supervisor and resume (Session 5, Fri 18 Sep)

Second risky phase. The kill test is the deliverable; the reviewer is secondary.

### 5.1 `nodes/review.py`

Model `reviewer`. One call for the whole plan: schema
`Review(verdicts: list[Verdict])`, `Verdict(sub_question_id, verdict: Literal[...],
reason: str)`. Input is each sub-question with its findings as `id [status]: claim`.
Apply: `needs_one_more_pass` is honoured only while `revisions_used < max_revisions` and
`sq.passes < 2`; otherwise it is recorded in the step but downgraded to `answered` or
`unanswerable` by whether findings exist. Set `reviewed = True`, and increment
`revisions_used` if any sub-question was reopened. The step's `error` field is not used for
reasons; add `note: str | None = None` to `StepRecord` and put the joined reasons there.

Two rules the reviewer does not get to break.

**A verdict of `answered` on a sub-question with no findings is downgraded to
`unanswerable`.** The verdict still reaches the trace, but the plan stays honest: a
sub-question with nothing behind it is not answered, whatever the model says.

**A reviewer that fails does not fail the run.** Any exception sets `reviewed = True` and
records the reason, so the writer still gets its turn. A report from unreviewed findings
beats no report. It also has to change the state, or the Phase 4 stall guard would see a
node erroring without progressing and end the run.

Time-box prompt tuning to one hour. The trace makes a mediocre reviewer visible, which is
enough for this week.

Router addition: after research of a reopened sub-question, `reviewed` must be reset so the
run goes through review again. Simplest rule: `review` sets `reviewed=True`; `research`
sets `reviewed=False`. The router table test from Phase 1 gains rows for this.

### 5.2 Sweeper

arq `cron(sweep, second={0, 30})` in `WorkerSettings.cron_jobs`:

```python
async def sweep(ctx):
    store = ctx["store"]
    for run_id in await store.active_runs():
        state = await store.load(run_id)
        if state is None:
            await store.forget_active(run_id)
            continue
        if state.status != "running":
            await store.forget_active(run_id)
            continue
        if await store.lease_holder(run_id) is not None:
            continue
        bumped = state.model_copy(update={"attempt": state.attempt + 1})
        await store.save(bumped)
        await enqueue_run(ctx["pool"], bumped)
```

Two workers can both sweep at the same second; `acquire_lease` inside `run_graph` makes
the second one a no-op, and the `attempt`-suffixed `job_id` means both enqueues target the
same id, so arq drops the duplicate. Use `run_at_startup=True`, so a worker starting after a
crash sweeps immediately rather than waiting for the next slot.

**The sweeper needs the lease heartbeat to exist first.** A node can legitimately outlive the
60s TTL, and without a heartbeat the sweeper re-enqueues healthy but slow runs. `run_graph`
starts an `asyncio.Task` that refreshes every `lease_ttl_s / 3` and cancels it in `finally`.
The heartbeat stops on its own if the refresh fails, which means another worker has taken
over and this one should let go. `test_a_live_run_is_not_swept_out_from_under_its_worker`
holds a run at a gate for three lease lifetimes and asserts it is never re-enqueued.

**Where the stubs live.** `src/ra/nodes/stubs.py`, not `tests/`, because the crash-resume
test runs real worker processes and a subprocess can only import what is installed. They are
selected with `RA_STUB`, which nothing else sets, and `select_nodes` applies them last so
they win. `stubs.py` is omitted from coverage: it runs inside the worker subprocesses, where
the coverage run in the test process cannot see it.

### 5.3 `test_resume_after_kill.py`

Deterministic by construction, no `sleep` for synchronisation:

1. Fixture starts Redis-backed store on db 1, `FLUSHDB`.
2. Spawn worker A as a subprocess: `python -m arq ra.worker.WorkerSettings` with env
   `RA_STUB=slow_research`, `RA_LEASE_TTL_S=2`, `RA_SWEEP_SECONDS=1` (so the test finishes
   in seconds; these env overrides exist only for this purpose and are documented).
3. The `slow_research` stub node (selected via `Deps` when `RA_STUB` is set): finishes
   `sq_01` normally with a canned finding, then for `sq_02` sets Redis key
   `test:gate:reached` and blocks on `BLPOP test:gate:release 0`.
4. Test creates a run with a canned 2-sub-question plan already in state, enqueues it,
   then `BLPOP`-waits on `test:gate:reached` (timeout 20 s -> fail, not hang).
5. `os.kill(worker_a.pid, SIGKILL)`. Assert `store.load(run_id).status == "running"` and
   `sq_01.status == "answered"`, `sq_02.status == "pending"`.
6. Spawn worker B with `RA_STUB=fast_research` (no gate). Its sweeper sees the lease expire
   within 2 s and re-enqueues.
7. Poll `store.load` every 100 ms until `done` (timeout 30 s).
8. Assert: exactly one `StepRecord` with `node="research", sub_question_id="sq_01"`;
   exactly one for `sq_02`; findings for `sq_01` unchanged (same ids as before the kill);
   `attempt == 1`; `runs:active` is empty.

Kill worker B in the fixture teardown. Mark the test `@pytest.mark.redis` and run it in CI.

### 5.4 CI

`.github/workflows/ci.yml`: ubuntu-latest, `services: redis: image: redis:7` with a
health check, `astral-sh/setup-uv`, `uv sync --frozen`, `uv run ruff check`, `uv run pytest
--cov=ra --cov-fail-under=80 -m "not live"`, `gitleaks/gitleaks-action` on the full history.
Fake keys in env so Settings validates; live tests are excluded by marker.

**Exit:** the kill test is green in CI.

---

## Phase 6 - Replay and fixtures (Session 6, Sat 19 Sep)

### 6.1 Record

Script `scripts/record_run.py "question"`: POST, poll, then dump the final `RunState` to
`fixtures/runs/{slug}.json` with `indent=2, sort_keys=True`. Run three questions of
different shapes (a comparison, a "what is the current state of", a "why did"). Skim each
report; if one is bad, keep it anyway and say so in the README. Bad runs with honest traces
are part of the story.

Check fixture size. If a fixture exceeds ~300 KB, the `snippet` cap in Phase 3 was not
enforced somewhere; fix that rather than trimming the fixture. There is a test for this.

**Still outstanding: the committed fixture is a canned run, not a real one.** Recording a
real run needs keys, so `fixtures/runs/canned-example.json` is a placeholder produced by the
real graph with the canned nodes. It makes `make demo` work on a clean clone today, and its
trace is genuine, but every token count and cost in it is zero because no model was called.
Record three real questions with `make record Q="..."` once keys are in `.env`, delete the
placeholder, and take the README's trace excerpt from one of those. Do not quote the
placeholder's numbers anywhere; they are not measurements of anything.

### 6.2 `replay.py`

```python
def load_fixture(path) -> RunState
def replay(state: RunState, *, speed: float = 20.0, out: TextIO = sys.stdout) -> str
```

Prints one line per step: `seq node sq model tokens_in/out $cost ms status`, sleeping
`duration_ms / speed / 1000` between lines (total under 30 s for a 5-minute run at 20x).
Then prints `render_markdown(state.report, state.findings)` and returns it.
`python -m ra.replay fixtures/runs/x.json --speed 0` for tests.

Test `test_replay.py`: block `socket.socket` outright rather than patching an HTTP client,
replay every fixture at `speed=0`, and assert the returned markdown equals
`state.report_markdown` byte for byte. This pins `render.py`: any rendering change must
regenerate the fixtures deliberately.

That byte-identity test caught a real inconsistency on its first run. The canned writer had
kept the placeholder renderer it was given in Phase 1, so its output disagreed with
`render.py` in footnotes and sources. Every node now renders through `render.py`, which is
what makes a canned run a valid golden file for it.

### 6.3 `Makefile`

```
up:    docker compose up --build
down:  docker compose down -v
test:  uv run pytest --cov=ra --cov-fail-under=80 -m "not live"
lint:  uv run ruff check . && uv run ruff format --check .
demo:  uv run python -m ra.replay $(FIXTURE)
record: uv run python scripts/record_run.py "$(Q)"
trace: python3 -m http.server 8111   # then open /demo/trace.html
```

`FIXTURE` defaults to the first file in `fixtures/runs/`, so the demo keeps working once the
placeholder is replaced with real recordings.

`make demo` must work right after `git clone` with only `uv` installed: it imports
`ra.schemas`, `ra.render`, `ra.replay` and nothing that touches Redis or the network.
Enforce this with an import test that fails if `ra.replay` transitively imports `redis`,
`anthropic`, or `arq`.

### 6.4 `demo/trace.html`

A single static file, no build step and no dependencies, that loads a fixture and renders
the run header, the step table and the report. It escapes every value before rendering,
because a fixture is data, not markup. Browsers will not fetch local files, so `make trace`
serves the repo root and prints the URL; `?fixture=` picks a different run.

**Not yet opened in a browser.** The file serves and the fixture it points at resolves, but
the rendering itself is unverified. Check it before taking the README screenshot.

**Exit:** `git clone && make demo` works on a machine with no keys.

---

## Phase 7 - README, hardening, buffer (Session 7, Sun 20 Sep)

1. Finish whatever overran from Phases 3 to 5. Kill test and replay test are the two that
   must be green before anything else.
2. README per the skeleton in the source plan. Paste a verbatim trace excerpt from a
   fixture, 8 to 10 rows.
3. Hardening pass, one hour: run the `security-reviewer` and `python-reviewer` agents on
   `src/ra`, fix CRITICAL and HIGH. Expected items: response envelope on every error path,
   no secrets in logs (`SecretStr` covers config; grep for `api_key` in log calls), request
   body size limit on `POST /research`.
4. `gitleaks detect --source . --log-opts="--all"` over the full history. Only then flip the
   repo public.
5. Save the three fixture questions and the workspace spend after the week to the README
   "what it cost" line. Real numbers are the most convincing part.

---

## Cross-cutting notes

**The stall guard (`progress.py`).** The router deriving the next node from state is what
makes resume trivial, and it is also what makes a failing node loop: a node that errors
without changing anything gets sent straight back in. So if the last three steps are all
errors whose `input_digest` equals their `output_digest`, the run is failed with the node
named. Three, because each attempt can be a paid call. The planner's own check above ends a
run on the first failure; this guard is the net under every other node, including the ones
Phase 5 adds.

**Immutability in nodes.** `state.model_copy(update={...})` for scalar fields;
`[*state.findings, *new]` for lists; never `state.findings.append`. The trace decorator
relies on the pre-node state object staying unchanged to compute `input_digest`.

**Where LangGraph could fight you.** Three places: Pydantic state with full-state returns
(if a version rejects it, switch the state type to a `TypedDict` with one key `run: RunState`
and adapt the nodes; one hour), `Command(goto=...)` import path, and `recursion_limit`.
If a third thing breaks, replace the graph with a 15-line `while True: node = route(state)`
loop. Nothing in the deliverables depends on LangGraph itself.

**Lease TTL versus node duration.** A node can legitimately run longer than 60 s (a slow
extract batch plus a Sonnet call). Refresh happens only after the node in Phase 2. Add a
background `asyncio.Task` in `run_graph` that refreshes every `lease_ttl_s / 3` while the
graph runs and is cancelled in `finally`. Do this in Phase 5 when writing the sweeper;
without it the sweeper can re-enqueue a healthy but slow run.

**Cost guard rails during development.** Every live test and every recorded run uses the
default `Budgets`; at Sonnet 5 and Haiku 4.5 prices a full run is well under $0.50. The $20
workspace cap is the real backstop. Check the Console spend once a day during the week.

**Deferred, with the reason** (unchanged from the source plan): LangGraph checkpointer,
SSE, per-domain caps, revision depth beyond one, gateway-level key routing, any UI beyond
the optional static page.
