"""The reviewer: does each sub-question actually have an answer?

One call for the whole plan, one structured verdict per sub-question, each with a one-line
reason that goes into the trace. Time-box the prompt tuning on this node. A mediocre reviewer
whose reasoning is visible in the trace is worth more than a clever one that is not.

A reviewer that fails does not fail the run. It marks the plan reviewed and lets the writer
have its turn, because a report from unreviewed findings still beats no report.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ra.deps import Deps
from ra.errors import safe_detail
from ra.schemas import NodeOutcome, RunState, SubQuestion

log = logging.getLogger("ra.nodes.review")

# A sub-question gets at most two research passes, whatever the reviewer wants.
MAX_PASSES = 2

REVIEWER_SYSTEM = (
    "You judge whether each sub-question has been answered by the findings gathered for it.\n"
    "answered: the findings support an answer.\n"
    "needs_one_more_pass: the findings are close but thin, and another search would "
    "plausibly close the gap.\n"
    "unanswerable: no amount of further searching is likely to help.\n"
    "Give one short reason for each verdict."
)


class Verdict(BaseModel):
    sub_question_id: str
    verdict: Literal["answered", "needs_one_more_pass", "unanswerable"]
    reason: str = ""


class Review(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


async def review(state: RunState, deps: Deps) -> NodeOutcome:
    try:
        result = await deps.llm.structured(
            model=deps.settings.models.reviewer,
            system=REVIEWER_SYSTEM,
            user=_prompt(state),
            schema=Review,
        )
    except Exception as exc:
        # Never block the writer on a broken reviewer. Mark the plan reviewed so the run
        # progresses, and let the trace carry the reason it was not properly checked.
        log.warning("run %s could not be reviewed: %s", state.run_id, exc)
        detail = safe_detail(exc)
        return NodeOutcome(
            state=state.model_copy(update={"reviewed": True}),
            status="error",
            error=f"reviewer failed: {detail}",
        )

    by_id = {v.sub_question_id: v for v in result.parsed.verdicts}
    plan = [_applied(sq, by_id.get(sq.id), state) for sq in state.plan]
    reopened = sum(
        1
        for before, after in zip(state.plan, plan, strict=True)
        if after.status == "needs_one_more_pass" and before.status != "needs_one_more_pass"
    )

    return NodeOutcome(
        state=state.model_copy(
            update={
                "plan": plan,
                "reviewed": True,
                "revisions_used": state.revisions_used + (1 if reopened else 0),
            }
        ),
        model=result.model,
        usage=result.usage,
        note=_note(result.parsed.verdicts, reopened),
    )


def _applied(sq: SubQuestion, verdict: Verdict | None, state: RunState) -> SubQuestion:
    """Honour the verdict, within the run's revision budget and the evidence on hand."""
    if verdict is None:
        return sq

    has_findings = any(f.sub_question_id == sq.id for f in state.findings)

    if verdict.verdict == "answered" and not has_findings:
        # A sub-question with nothing behind it is not answered, whatever the reviewer says.
        # The verdict still reaches the trace; the plan stays honest.
        return sq.model_copy(update={"status": "unanswerable"})

    if verdict.verdict != "needs_one_more_pass":
        return sq.model_copy(update={"status": verdict.verdict})

    affordable = state.revisions_used < state.budgets.max_revisions and sq.passes < MAX_PASSES
    if affordable:
        return sq.model_copy(update={"status": "needs_one_more_pass"})

    # Out of revisions. The verdict is still recorded in the trace, but the plan settles on
    # whatever the findings already support.
    return sq.model_copy(update={"status": "answered" if has_findings else "unanswerable"})


def _prompt(state: RunState) -> str:
    lines = [f"Question: {state.question}", ""]
    for sq in state.plan:
        lines.append(f"{sq.id}: {sq.text}")
        findings = [f for f in state.findings if f.sub_question_id == sq.id]
        if not findings:
            lines.append("  (no findings)")
        for f in findings:
            lines.append(f"  {f.id} [{f.extraction_status}] {f.claim}")
        lines.append("")
    lines.append("Return one verdict for every sub-question listed above.")
    return "\n".join(lines)


def _note(verdicts: list[Verdict], reopened: int) -> str | None:
    if not verdicts:
        return None
    reasons = "; ".join(f"{v.sub_question_id} {v.verdict}: {v.reason}".strip() for v in verdicts)
    prefix = f"reopened {reopened}. " if reopened else ""
    return f"{prefix}{reasons}"
