# Multi-Agent Research Assistant — one-week build plan

**Target:** public GitHub repo, demoable by Sun 20 Sep 2026.
**Stack:** Python 3.12 · FastAPI · arq · Redis 7 (plain) · LangGraph (thin) · Anthropic SDK · Tavily · Pydantic v2 · pytest · uv · Docker Compose

---

## Done when

- `make demo` replays a committed real run offline — no API keys, under 30 seconds.
- `docker compose up` + keys → `POST /research` with a real question → fully cited report in under 5 minutes.
- `kill -9` on the worker mid-run → new worker finishes the run without re-executing completed sub-questions. This is a pytest, and it runs in CI.
- `GET /research/{id}/trace` returns node, tokens, cost, latency, and status for every step.
- Every citation in every report resolves to a real `Finding` record.

**Resume line:** Built a supervisor/worker multi-agent research system with explicit durable run state in Redis, per-run token and cost caps, crash-resume with a lease/sweeper, and step-level tracing — plus offline replay from recorded traces.

---

## Decisions (and the one-line reason, for the README)

| Decision | Why |
|---|---|
| Graph runs in an arq worker, API only enqueues and reads | A `BackgroundTask` dies with the process; the whole point is surviving that |
| Own `run:{id}` document in plain Redis, no LangGraph checkpointer | Status, trace, and resume all need the state queryable; the checkpointer blob isn't, and it needs Redis Stack modules |
| Router node at START derives the next step from state | Makes resume trivial — always enter at START, the router skips what's done |
| Hard spend cap is an Anthropic workspace limit, not code | Code caps die with the process; the workspace cap doesn't |
| Hand-written `StateGraph`, no `langgraph-supervisor`, no subagents-as-tools | The package is unmaintained; the tools pattern hides routing inside a message list, and visible routing is the point |
| Writer emits `finding_ids`, never URLs | Citation validation becomes set membership |
| Budget exceeded → jump to `write` with what you have, flagged | A partial cited report beats an exception |
| Replay mode is the demo | Readers can't hit a service; they can run `make demo` |
| Polling, not SSE | 20 minutes vs. half a session; nothing in the story depends on it |

---

## Pre-flight (30 min, before session 1)

- [ ] Anthropic Console: new workspace for this project → Spend limits tab → cap at $20/month, alert at 50%. Create the API key inside that workspace.
- [ ] Tavily: API key (free tier, 1,000 credits/month — advanced search costs 2).
- [ ] New repo. First commit contains `.gitignore` with `.env`, and `.env.example` with placeholder values. Install `gitleaks` as a pre-commit hook before any key exists on disk.
- [ ] `uv init`, pin versions in `pyproject.toml` (LangGraph moves fast — pin it).
- [ ] `compose.yml` with `api`, `worker`, `redis` services.

---

## Architecture

```
POST /research ──► API ──► arq queue ──► worker ──► StateGraph
                    │                       │           │
                    │                       │      router ─► plan ─► research (×N) ─► review ─► write
                    │                       │           │                    ▲            │
                    │                       │           │                    └── revise ◄─┘ (once)
                    ▼                       ▼           ▼
              GET /research/{id}        lease key    run:{id} JSON    ◄── written after every node
              GET /research/{id}/trace  (TTL 60s)         │
                                                      sweeper: lease expired + status=running → re-enqueue
```

**Redis keys**

| Key | Contents | TTL |
|---|---|---|
| `run:{id}` | full `RunState` JSON — plan, findings, steps, report, spend | none |
| `run:{id}:lease` | worker id | 60s, refreshed after each node |
| `runs:active` | set of run ids with `status=running` | none |
| `cache:search:{sha256(query,depth)}` | Tavily search response | 7d |
| `cache:extract:{sha256(url)}` | extracted page content | 7d |

**Model tiering.** Planner and reviewer on Haiku; researcher extraction and writer on Sonnet. Verify current model strings against the models page before hardcoding; put them in one config object so the trace can record which model each step used.

---

## Schemas (session 1 — write these first)

```python
class SubQuestion(BaseModel):
    id: str  # sq_01 …
    text: str
    status: Literal["pending", "answered", "needs_one_more_pass", "unanswerable"] = "pending"
    passes: int = 0


class Finding(BaseModel):
    id: str  # f_a1b2c3
    sub_question_id: str
    claim: str
    source_url: str
    source_title: str | None
    snippet: str
    retrieved_at: datetime
    extraction_status: Literal["full", "snippet_only", "paywalled", "timeout", "error"]


class ToolCall(BaseModel):
    tool: str  # tavily.search / tavily.extract
    args_digest: str
    status: str
    duration_ms: int
    credits: int = 0


class StepRecord(BaseModel):
    seq: int
    node: str
    sub_question_id: str | None = None
    started_at: datetime
    duration_ms: int
    status: Literal["ok", "error", "skipped", "budget_exceeded"]
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_calls: list[ToolCall] = []
    input_digest: str
    output_digest: str
    error: str | None = None


class Claim(BaseModel):
    text: str
    finding_ids: list[str]  # validated against RunState.findings before render


class Section(BaseModel):
    heading: str
    claims: list[Claim]


class Report(BaseModel):
    title: str
    sections: list[Section]
    generated_at: datetime


class Budgets(BaseModel):
    max_subquestions: int = 4
    max_searches_per_sq: int = 3
    max_extracts_per_sq: int = 5
    max_revisions: int = 1
    max_total_tokens: int = 150_000
    writer_reserve_tokens: int = 20_000  # always leave room for one write pass
    max_wall_clock_s: int = 300
    max_tavily_credits: int = 20


class RunState(BaseModel):
    run_id: str
    question: str
    status: Literal["queued", "running", "done", "failed", "budget_exceeded"]
    budgets: Budgets
    plan: list[SubQuestion] = []
    findings: list[Finding] = []
    steps: list[StepRecord] = []
    report: Report | None = None
    report_markdown: str | None = None
    revisions_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tavily_credits: int = 0
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
```

Every node: `check_budgets(state)` first → do work → append `StepRecord` → `store.save(state)` → refresh lease → return `Command(goto=…)`.

---

## Sessions

Order is fixed; dates can slide. Sessions 3 and 5 are the risky ones — if weekday evenings are short, swap so one of them lands on Saturday.

### Session 1 — Mon 14 Sep — Skeleton end to end

- `schemas.py` as above. `store.py`: `save/load(run_id)`, lease acquire/refresh/release, `runs:active`.
- `graph.py`: `StateGraph(RunState)` with `router`, `plan`, `research`, `review`, `write`. All four work nodes return canned data and sleep 1s.
- `api.py`: `POST /research` → create run, enqueue, return `{run_id}`. `GET /research/{id}` → status + report when done.
- `worker.py`: arq job `run_graph(run_id)` — acquire lease, load state, invoke graph, release.
- `compose.yml` works: `docker compose up` → curl POST → poll GET → canned report.

**Exit:** a fake run completes through the real queue, worker, and Redis. Nothing intelligent yet.

### Session 2 — Tue 15 Sep — Trace and budgets

- `trace.py`: a `@traced("node_name")` decorator that times the node, computes digests, appends `StepRecord`, saves state.
- `llm.py`: Anthropic client wrapper. Reads `usage` from every response (never estimate tokens), looks up cost in a price table (`pricing.py` — fill from the current pricing page, don't guess), increments `state.tokens_*`, `state.cost_usd`.
- `budgets.py`: `check_budgets(state) -> BudgetVerdict`. Tokens check includes `writer_reserve_tokens`. Exceeded → status `budget_exceeded`, `Command(goto="write")` if any findings exist, else `END`.
- `GET /research/{id}/trace` → `state.steps`.
- Test: set every cap to 1 in turn, assert run stops with `budget_exceeded` and a partial report.

**Exit:** trace shows every step with cost; caps provably stop a run.

### Session 3 — Wed 16 Sep — Researcher with real search

- `search.py`: Tavily `/search` (basic depth by default — advanced doubles credits) then `/extract` on the top results. Cache both in Redis by hash. Count credits into `state.tavily_credits`.
- Map every failure to `extraction_status`, not an exception: empty results → node records `skipped`; 429 → `error` with retry-after honoured once; paywall/empty body → `paywalled`; timeout → `timeout`. The supervisor sees statuses, never stack traces.
- Dedupe by URL across the whole run. (Per-domain caps: cut this week.)
- `research` node: one sub-question per invocation, controlled by the router. LLM turns extracted content into `Finding` records with a structured output schema. A finding with `extraction_status != "full"` is allowed but the writer is told to prefer full ones.
- Test: mock Tavily to return 429 / timeout / empty → correct statuses, run continues.

**Exit:** a hardcoded three-sub-question plan produces real findings with provenance.

### Session 4 — Thu 17 Sep — Planner and writer

- `plan` node: question → `list[SubQuestion]`, capped at `max_subquestions`. Structured output. Prompt asks for sub-questions that are independently searchable and non-overlapping; nothing about being a world-class anything.
- `write` node: findings → `Report`. The prompt includes the finding list as `id: claim (source)` and the schema forces `finding_ids`. Validation: every id ∈ `state.findings`. Failure → one retry with the offending ids listed → hard fail with a clear error in the trace.
- `render.py`: `Report` → markdown with `[n]` footnotes mapping to `source_url`. Rendering happens *after* validation.
- Test: writer stub returns an invented id → retry → fail path exercised.

**Exit:** real question → real cited report, end to end, once.

### Session 5 — Fri 18 Sep — Supervisor and resume

- `review` node: for each sub-question with findings, structured verdict `answered | needs_one_more_pass | unanswerable`, with a one-line reason recorded in the step. `needs_one_more_pass` only honoured while `revisions_used < max_revisions`. Time-box the prompt tuning to one hour — the trace makes a mediocre reviewer forgivable.
- `router`: reads state and dispatches — no plan → `plan`; any `pending`/`needs_one_more_pass` sub-question → `research(that one)`; all researched, not reviewed → `review`; reviewed → `write`; report exists → `END`. Because the router is pure state, resume is "invoke the graph again."
- `worker.py` sweeper: every 30s, for each id in `runs:active`, if `run:{id}:lease` is missing and status is `running` → re-enqueue.
- Test `test_resume_after_kill`: start a run with a slow stubbed researcher, `os.kill(worker_pid, SIGKILL)` after sub-question 1 completes, start a new worker, assert run finishes and sub-question 1's step appears exactly once.
- CI: GitHub Actions with a Redis service container, `pytest`.

**Exit:** the kill test is green in CI.

### Session 6 — Sat 19 Sep — Replay and fixtures

- Run three real questions end to end. Commit them to `fixtures/runs/*.json` — full `RunState` including every step and finding.
- `replay.py`: load a fixture, re-run `render.py`, and stream the steps to stdout with their original timings compressed (e.g. 20×). No network calls; the test blocks `httpx` and asserts the report is byte-identical.
- `Makefile`: `make demo` (replay), `make up`, `make test`.
- Optional, if time: `demo/trace.html` — single static file that loads a fixture and renders the step table and report. Build it for the README screenshot.

**Exit:** `git clone && make demo` works on a machine with no keys.

### Session 7 — Sun 20 Sep — README and buffer

Something in 3–5 overran. Finish it first. Then the README (skeleton below). Then a last `gitleaks detect` over full history before flipping the repo public.

---

## Repo layout

```
research-agent/
├── README.md
├── Makefile
├── compose.yml
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
├── src/ra/
│   ├── api.py          FastAPI app
│   ├── worker.py       arq worker + sweeper
│   ├── graph.py        StateGraph wiring, router
│   ├── nodes/          plan.py  research.py  review.py  write.py
│   ├── schemas.py
│   ├── budgets.py
│   ├── trace.py
│   ├── store.py        Redis run doc, lease, cache
│   ├── search.py       Tavily + failure mapping + cache
│   ├── llm.py          Anthropic wrapper: usage, pricing, structured output
│   ├── pricing.py
│   ├── render.py
│   └── replay.py
├── fixtures/runs/
├── tests/
└── demo/trace.html     (optional)
```

---

## README skeleton

1. **One paragraph.** What it does, and that the interesting parts are durable state, budgets, and the trace — not the agents.
2. **`make demo`** — first thing after the intro. Replay a real run offline.
3. **Architecture** — the ASCII diagram above.
4. **A real trace excerpt** — 8–10 steps, pasted verbatim: node, model, tokens, cost, latency, status.
5. **Budgets** — the `Budgets` table plus the workspace cap, and what happens when a cap trips.
6. **Resume** — the lease/sweeper mechanism and a link to `test_resume_after_kill`.
7. **Citations** — the `finding_ids` validation in three sentences.
8. **Deliberately not done** — checkpointer, SSE, per-domain caps, gateway-level budgets — each with its one-line reason from the decisions table.
9. **Run it for real** — keys, `docker compose up`, curl.

---

## Cut this week

- LangGraph Redis checkpointer (needs Redis Stack; own run doc covers it)
- SSE / progress streaming (poll `GET /research/{id}` every 2s)
- Per-domain result caps (URL dedupe only)
- Revision depth beyond one
- Routing through a gateway key (README mentions it as the multi-tenant move)
- Any UI beyond the optional static trace page

---

## Where time actually goes

- **Supervisor prompt tuning.** One hour, then stop. Structured verdict + reason in the trace is enough.
- **Extract quality on JS-heavy pages.** Don't fight it. `snippet_only` is a legitimate status; let the writer down-rank it.
- **Token accounting.** Always from `usage` in the response. The moment you estimate, `cost_usd` stops being trustworthy.
- **LangGraph API drift.** Pin versions. If a prebuilt helper fights you, drop it — four nodes and `Command` are all you need.
- **The kill test flaking.** Use a stubbed slow researcher and explicit sync points (a Redis flag the test waits on), not `sleep`.
