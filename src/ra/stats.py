"""Runs as a set, rather than one at a time.

Every run is individually legible through its trace. This is the other half: what a batch of
runs cost, which node fails most often, and how much of what gets cited was a full page
rather than a search snippet.

A pure function over run documents, so the arithmetic is testable without Redis and without
a server. Everything it reports is a total over the runs it was given, not over all time.
"""

from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from ra.schemas import RunState

# Long enough to be recognisable in a list, short enough not to dominate it.
QUESTION_PREVIEW = 90


class Window(BaseModel):
    """What was actually aggregated. Every number below is a total over exactly this."""

    runs: int = 0
    since: datetime | None = None
    oldest: datetime | None = None
    newest: datetime | None = None
    truncated: bool = False


class Spend(BaseModel):
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tavily_credits: int = 0


class NodeStats(BaseModel):
    """One row per node, which is what answers "where does this thing go wrong"."""

    node: str
    runs: int = 0
    ok: int = 0
    error: int = 0
    skipped: int = 0
    budget_exceeded: int = 0
    cost_usd: float = 0.0
    total_ms: int = 0
    slowest_ms: int = 0

    @property
    def calls(self) -> int:
        return self.ok + self.error + self.skipped + self.budget_exceeded


class Overview(BaseModel):
    window: Window
    by_status: dict[str, int] = Field(default_factory=dict)
    spend: Spend = Field(default_factory=Spend)
    nodes: list[NodeStats] = Field(default_factory=list)
    findings_by_quality: dict[str, int] = Field(default_factory=dict)
    reports_written: int = 0
    findings: int = 0


class RunSummary(BaseModel):
    run_id: str
    question: str
    status: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    tavily_credits: int
    findings: int
    steps: int
    has_report: bool
    duration_s: float | None
    created_at: datetime
    error: str | None = None


def summarise(states: list[RunState], *, since: datetime | None = None, truncated: bool = False):
    """Totals across the given runs. Ordering is the caller's business."""
    overview = Overview(
        window=Window(
            runs=len(states),
            since=since,
            oldest=min((s.created_at for s in states), default=None),
            newest=max((s.created_at for s in states), default=None),
            truncated=truncated,
        )
    )

    status_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    nodes: dict[str, NodeStats] = defaultdict(lambda: NodeStats(node=""))

    for state in states:
        status_counts[state.status] += 1
        overview.spend.cost_usd += state.cost_usd
        overview.spend.tokens_in += state.tokens_in
        overview.spend.tokens_out += state.tokens_out
        overview.spend.tavily_credits += state.tavily_credits
        overview.findings += len(state.findings)
        overview.reports_written += 1 if state.report is not None else 0

        for finding in state.findings:
            quality_counts[finding.extraction_status] += 1

        for node_name in {step.node for step in state.steps}:
            entry = nodes[node_name]
            entry.node = node_name
            entry.runs += 1

        for step in state.steps:
            entry = nodes[step.node]
            entry.node = step.node
            setattr(entry, step.status, getattr(entry, step.status) + 1)
            entry.cost_usd += step.cost_usd
            entry.total_ms += step.duration_ms
            entry.slowest_ms = max(entry.slowest_ms, step.duration_ms)

    overview.spend.cost_usd = round(overview.spend.cost_usd, 6)
    overview.by_status = dict(status_counts.most_common())
    overview.findings_by_quality = dict(quality_counts.most_common())
    # Noisiest first, because the question this answers is where things go wrong.
    overview.nodes = sorted(
        (n.model_copy(update={"cost_usd": round(n.cost_usd, 6)}) for n in nodes.values()),
        key=lambda n: (-n.error, -n.calls, n.node),
    )
    return overview


def summarise_run(state: RunState) -> RunSummary:
    """One line per run, enough to spot the one worth opening."""
    duration = None
    if state.started_at and state.finished_at:
        duration = round((state.finished_at - state.started_at).total_seconds(), 2)

    question = state.question
    if len(question) > QUESTION_PREVIEW:
        question = question[: QUESTION_PREVIEW - 1].rstrip() + "…"

    return RunSummary(
        run_id=state.run_id,
        question=question,
        status=state.status,
        cost_usd=state.cost_usd,
        tokens_in=state.tokens_in,
        tokens_out=state.tokens_out,
        tavily_credits=state.tavily_credits,
        findings=len(state.findings),
        steps=len(state.steps),
        has_report=state.report is not None,
        duration_s=duration,
        created_at=state.created_at,
        error=state.error,
    )
