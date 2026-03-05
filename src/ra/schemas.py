"""The durable run document and everything inside it.

RunState is the single source of truth for a run. It is stored whole at run:{id} in Redis
after every node, which is what makes status, trace and resume all work off one object.

Nothing in here mutates. Nodes build a new state with state.model_copy(update=...).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ra.clock import now

ExtractionStatus = Literal["full", "snippet_only", "paywalled", "timeout", "error"]

HTTP_SCHEMES = ("http://", "https://")


def is_http_url(url: str) -> bool:
    """Only http(s) may become provenance.

    A source URL is chosen by a model from search results and ends up as a link in the
    rendered report. Anything else, a file: path or a javascript: URI, has no business
    being there whoever put it in front of us.
    """
    return isinstance(url, str) and url.startswith(HTTP_SCHEMES)


StepStatus = Literal["ok", "error", "skipped", "budget_exceeded"]
SubQuestionStatus = Literal["pending", "answered", "needs_one_more_pass", "unanswerable"]
RunStatus = Literal["queued", "running", "done", "failed", "budget_exceeded"]


class SubQuestion(BaseModel):
    id: str
    text: str
    status: SubQuestionStatus = "pending"
    passes: int = 0


class Finding(BaseModel):
    id: str
    sub_question_id: str
    claim: str
    source_url: str
    source_title: str | None = None
    snippet: str
    retrieved_at: datetime
    extraction_status: ExtractionStatus

    @field_validator("source_url")
    @classmethod
    def _must_be_http(cls, v: str) -> str:
        if not is_http_url(v):
            raise ValueError(f"source_url must be http or https, got {v[:40]!r}")
        return v


class ToolCall(BaseModel):
    tool: str
    args_digest: str
    status: str
    duration_ms: int
    credits: int = 0


class Usage(BaseModel):
    """Token counts copied verbatim from an Anthropic response. Never estimated."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class StepRecord(BaseModel):
    seq: int
    node: str
    sub_question_id: str | None = None
    started_at: datetime
    duration_ms: int
    status: StepStatus
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_digest: str
    output_digest: str
    error: str | None = None
    # Free-text detail that is not a failure, such as a reviewer's reason. Used from Phase 5.
    note: str | None = None


class Claim(BaseModel):
    text: str
    # Validated against RunState.findings before the report is ever rendered.
    finding_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    claims: list[Claim] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    sections: list[Section] = Field(default_factory=list)
    generated_at: datetime


class Budgets(BaseModel):
    """Per-run caps. A caller may lower these; the ceilings are not negotiable.

    POST /research takes budgets from the request body, and that endpoint starts paid work.
    Without an upper bound a caller could ask for a hundred thousand searches, which the
    per-run token and credit caps would not catch in time because they are only evaluated
    between nodes. So every field is bounded here, at the boundary, where the value arrives.
    """

    max_subquestions: int = Field(4, ge=1, le=10)
    max_searches_per_sq: int = Field(3, ge=1, le=5)
    max_extracts_per_sq: int = Field(5, ge=1, le=10)
    max_revisions: int = Field(1, ge=0, le=3)
    max_total_tokens: int = Field(150_000, ge=0, le=2_000_000)
    # Always leave room for one write pass, so a tripped cap still produces a report.
    writer_reserve_tokens: int = Field(20_000, ge=0, le=200_000)
    max_wall_clock_s: int = Field(300, ge=0, le=1_800)
    max_tavily_credits: int = Field(20, ge=0, le=200)


class RunState(BaseModel):
    run_id: str
    question: str
    status: RunStatus
    budgets: Budgets = Field(default_factory=Budgets)
    plan: list[SubQuestion] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    report: Report | None = None
    report_markdown: str | None = None
    reviewed: bool = False
    # Set when a cap trips. The run stays "running" until the writer has had its turn, so a
    # client polling on status never sees a terminal status with no report behind it.
    budget_stopped: bool = False
    revisions_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tavily_credits: int = 0
    # Incremented on every enqueue. Part of the arq job id, so a sweeper re-enqueue after a
    # crash does not collide with the stale job record left by the dead worker.
    attempt: int = 0
    worker_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "failed", "budget_exceeded")


class NodeOutcome(BaseModel):
    """What a work node hands back to the tracing wrapper.

    The node owns the new state. The wrapper owns the StepRecord, the run totals, the save
    and the lease refresh, so no node has to remember to do those.
    """

    state: RunState
    status: StepStatus = "ok"
    model: str | None = None
    usage: Usage | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sub_question_id: str | None = None
    error: str | None = None
    note: str | None = None
