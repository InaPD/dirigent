# dirigent

A multi-agent research assistant: ask a question, get back a report where every claim cites
a source the system actually read.

The agents are the least interesting part. What is worth looking at is underneath them. Run
state is an explicit document in Redis, written after every step, so a run survives its
worker being killed and picks up where it left off. Every step records its model, token
counts, dollar cost and latency, so you can see what a run did and what it cost. Per-run
budgets stop a run before it overspends, and a stopped run still produces a report rather
than an exception. Citations are validated as set membership before anything is rendered,
so a claim cannot point at a source that does not exist.

## Try it without keys

```bash
git clone https://github.com/InaPD/dirigent && cd dirigent
make demo
```

That replays a recorded run from its own trace: each step at a fraction of its original
pace, then the report. No API keys, no Redis, no network. The replay module imports nothing
that could need any of those, and there is a test that fails if it ever starts to.

> The fixture currently committed is a placeholder: a real run of the real graph, but with
> the model and search clients stubbed out. Its timings and statuses are genuine; its token
> and cost columns read zero because nothing was called. Record a real one with
> `make record` once you have keys.

## How it works

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

The API only enqueues and reads. The graph runs in an arq worker, because a background task
inside the web process dies with the web process, and surviving exactly that is the point.

The router at the top derives the next node from the run state and nothing else. That is
what makes resume trivial: a resumed run re-enters at the router, which skips whatever is
already done. It is also the single place the supervisor's decisions are visible, which is
why the routing is not hidden inside a message list.

| Redis key | Contents | TTL |
|---|---|---|
| `run:{id}` | the whole run document: plan, findings, steps, report, spend | none |
| `run:{id}:lease` | the id of the worker currently running it | 60s, refreshed while it runs |
| `runs:active` | the ids of runs with `status=running` | none |
| `cache:search:{hash}` | a Tavily search response | 7 days |
| `cache:extract:{hash}` | an extracted page | 7 days |

Planner and reviewer run on Haiku, researcher and writer on Sonnet. Which model served each
step is recorded in the trace.

## The trace

`GET /research/{id}/trace` returns every step with its node, sub-question, model, tokens,
cost, latency and status. `make demo` prints the same thing:

```
seq  node      sq     model                 in    out     cost     ms  status
  1  plan      -      -                      0      0  0.00000   1001  ok
  2  research  sq_01  -                      0      0  0.00000   1001  ok
  3  research  sq_02  -                      0      0  0.00000   1001  ok
  4  review    -      -                      0      0  0.00000   1000  ok
     canned reviewer accepted every sub-question
  5  write     -      -                      0      0  0.00000   1001  ok
```

Token counts always come from the API response's own usage block and are never estimated.
Search credits come from Tavily's usage block the same way. The moment either is estimated,
the cost column stops being a measurement.

`demo/trace.html` renders the same data as a page. `make trace` serves it.

## Budgets

Every run carries its own caps, and `POST /research` can override them.

| Cap | Default | What it limits |
|---|---|---|
| `max_subquestions` | 4 | how far the planner may split the question |
| `max_searches_per_sq` | 3 | searches per sub-question, including one reworded retry |
| `max_extracts_per_sq` | 5 | pages fetched per sub-question |
| `max_revisions` | 1 | how many times the reviewer may reopen a sub-question |
| `max_total_tokens` | 150,000 | tokens across the whole run |
| `writer_reserve_tokens` | 20,000 | held back so a stopped run can still be written up |
| `max_wall_clock_s` | 300 | how long a run may take |
| `max_tavily_credits` | 20 | search credits per run |

A tripped cap is not an exception. The run is marked, the reason is written into the trace,
and it goes to the writer with whatever findings it has. The writer reserve exists so there
is always room for that pass, and the report is titled `(partial)`. Only a run with nothing
found at all stops without a report.

The hard spending limit is not in this code. It is a spend cap on the Anthropic workspace,
because a cap in code dies with the process and a cap on the workspace does not.

## Surviving a dead worker

A worker takes a lease on a run and refreshes it while it works. If the worker dies, the
lease expires. A sweeper on every worker looks for runs still marked running with no lease
behind them and puts them back on the queue, under a fresh attempt number so the dead
worker's stale job cannot collide with the new one.

The heartbeat matters as much as the sweeper. A node can legitimately outlive the lease, so
a background task refreshes it while the graph runs. Without that, the sweeper would
re-enqueue runs that are perfectly healthy and merely slow.

[`tests/test_resume_after_kill.py`](tests/test_resume_after_kill.py) is the proof. It starts
two real worker processes against a real Redis and a real queue, and sends a real `SIGKILL`
at the exact moment one sub-question is finished and the next is in flight. It then asserts
the replacement worker finishes the run and that the completed sub-question is researched
exactly once. Synchronisation runs through Redis gates rather than sleeps, so it cannot
flake on timing. It runs in CI.

## Citations

The writer is given findings as ids and cites ids, never URLs. Validation is therefore set
membership: every id in the report must be an id in `state.findings`, and every claim must
cite at least one. A report that fails gets one retry with the offending ids named; a second
failure fails the run with no report at all, because no report beats a report whose
provenance does not resolve. Rendering happens only after validation passes.

A finding whose page could not be fetched keeps the search snippet and is labelled
`snippet_only`, and the writer is told to prefer `full` sources. Pages that cannot be read
are a fact about the web, not a failure.

## Run it for real

You need an Anthropic key and a Tavily key. Put the Anthropic one in a workspace with a
spend limit on it.

```bash
cp .env.example .env      # then fill in the two keys
make up                   # redis, api and worker

curl -X POST localhost:8000/research \
  -H 'content-type: application/json' \
  -d '{"question": "What is the current state of durable agent execution?"}'

curl localhost:8000/research/<run_id>          # poll until finished_at is set
curl localhost:8000/research/<run_id>/trace
```

Poll on `finished_at`, not on `status`. A run that trips a budget cap is marked before the
writer has had its turn, so a status alone can be terminal while the report is still coming.

`make record Q="your question"` does the same thing and saves the finished run as a fixture
that `make demo` can replay.

## Before you put this on the internet

There is no authentication. `POST /research` is open and it starts work that costs money,
and `GET /research/{id}` will hand the question, the report and the full trace to anyone who
has the id. The rate limiter is per process and keyed on the client address, which makes it
a courtesy rather than a control: behind a proxy every caller shares one address, and it
resets when the process restarts. Put a gateway with real authentication in front of this
before it faces anything public.

What is enforced, because none of it depends on a gateway:

- Budgets arrive from the request body, so every field has a ceiling. A caller may lower a
  cap but cannot raise one past what the server allows.
- The research loop checks the run's own caps between searches. Budgets are otherwise only
  evaluated between nodes, which is too coarse to stop a loop inside one.
- The request body limit counts the bytes that arrive rather than believing a header.
- Error text stored on a run is redacted before it is saved, so a connection string or a key
  inside an exception does not reach whoever can read the run.
- A source URL must be http or https before it can become a citation.
- The real spending backstop is a spend limit on the Anthropic workspace, because a cap in
  code dies with the process.

`report_markdown` contains text a model wrote after reading pages it was sent to. Treat it
as untrusted: escape or sanitise it before rendering it as HTML, as `demo/trace.html` does.

## Deliberately not done

| Left out | Why |
|---|---|
| LangGraph's Redis checkpointer | status, trace and resume all need the state queryable; a checkpointer blob is not, and it needs Redis Stack |
| `langgraph-supervisor` and subagents-as-tools | the package is unmaintained, and the tools pattern hides routing inside a message list when visible routing is the point |
| Server-sent events | polling costs twenty minutes to build and nothing here depends on streaming |
| Per-domain result caps | URL dedupe across the run covers the case that actually came up |
| More than one revision pass | a second pass rarely changed the answer and always cost money |
| Budgets at a gateway | the multi-tenant move, and a different project |
| Authentication, and a shared rate limiter | see the section above; this belongs at a gateway, not in the app |
| A real UI | one static page renders a trace, which is all the story needs |

## Development

```bash
make test    # the suite, with an 80% coverage floor; needs Redis on localhost
make lint    # ruff, check and format
make fmt     # apply both
```

Tests that need real keys are marked `live` and excluded by default. Everything else,
including the crash-resume test, runs in CI against a Redis service container with blank
keys, so CI can never make a paid call.

Python 3.12, FastAPI, arq, Redis 7, LangGraph, Pydantic v2, uv, Docker Compose.
