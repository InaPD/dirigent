"""Per-run caps, and what happens when one trips.

A tripped cap is not an exception. The run is marked budget_exceeded and sent to the writer
with whatever findings it already has, because a short cited report beats a stack trace.
The writer reserve exists so there is always room for that final pass.

The hard dollar backstop is not here. It is a spend limit on the Anthropic workspace,
because code caps die with the process and the workspace cap does not.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from ra.clock import now as clock_now
from ra.schemas import RunState, StepRecord
from ra.trace import digest

BUDGET_NODE = "budget"


class BudgetVerdict(BaseModel):
    ok: bool
    exceeded: list[str] = Field(default_factory=list)

    @property
    def reason(self) -> str:
        return ", ".join(self.exceeded)


def check_budgets(
    state: RunState, now: datetime | None = None, *, next_node: str | None = None
) -> BudgetVerdict:
    """Which caps, if any, this run has passed.

    The token check keeps writer_reserve_tokens in hand, except when the next node is the
    writer itself. That is the whole point of the reserve: stop early enough to still write.
    """
    now = now or clock_now()
    exceeded: list[str] = []
    b = state.budgets

    spent = state.tokens_in + state.tokens_out
    reserve = 0 if next_node == "write" else b.writer_reserve_tokens
    if spent + reserve > b.max_total_tokens:
        exceeded.append("max_total_tokens")

    if state.started_at is not None:
        elapsed = (now - state.started_at).total_seconds()
        if elapsed > b.max_wall_clock_s:
            exceeded.append("max_wall_clock_s")

    if state.tavily_credits >= b.max_tavily_credits:
        exceeded.append("max_tavily_credits")

    return BudgetVerdict(ok=not exceeded, exceeded=exceeded)


def apply_budget_stop(state: RunState, verdict: BudgetVerdict) -> RunState:
    """Mark the run, record why, and leave it pointed at the writer if it has anything to say."""
    from ra.trace import append_step

    step = StepRecord(
        seq=len(state.steps) + 1,
        node=BUDGET_NODE,
        started_at=clock_now(),
        duration_ms=0,
        status="budget_exceeded",
        input_digest=digest(state),
        output_digest=digest(state),
        error=verdict.reason,
    )
    # With no findings there is nothing to write, so the run really does end here. With
    # findings it stays running until the writer has spent the reserve.
    finishing = not state.findings
    stopped = state.model_copy(
        update={
            "budget_stopped": True,
            "status": "budget_exceeded" if finishing else state.status,
            "finished_at": clock_now() if finishing else state.finished_at,
        }
    )
    return append_step(stopped, step)
