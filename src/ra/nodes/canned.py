"""Phase 1 stand-ins for the four work nodes.

They sleep and return fixed data. Their only job is to prove that a run travels the API,
the queue, the worker, the graph and Redis and comes back out. Each is replaced by a real
node in Phases 3 to 5, keeping the same (state, deps) -> NodeOutcome signature.
"""

import asyncio

from ra.clock import now
from ra.deps import Deps
from ra.ids import new_finding_id, sq_id
from ra.routing import next_open_subquestion
from ra.schemas import (
    Claim,
    Finding,
    NodeOutcome,
    Report,
    RunState,
    Section,
    SubQuestion,
)

NODE_DELAY_S = 1.0

CANNED_SUBQUESTIONS = [
    "What is the current state of the art?",
    "What are the main open problems?",
]


async def plan(state: RunState, deps: Deps) -> NodeOutcome:
    await asyncio.sleep(NODE_DELAY_S)
    texts = CANNED_SUBQUESTIONS[: state.budgets.max_subquestions]
    plan = [SubQuestion(id=sq_id(i), text=t) for i, t in enumerate(texts, start=1)]
    return NodeOutcome(state=state.model_copy(update={"plan": plan}))


async def research(state: RunState, deps: Deps) -> NodeOutcome:
    await asyncio.sleep(NODE_DELAY_S)
    sq = next_open_subquestion(state)
    if sq is None:
        return NodeOutcome(state=state, status="skipped", error="no open sub-question")

    finding = Finding(
        id=new_finding_id(),
        sub_question_id=sq.id,
        claim=f"A canned finding for {sq.id}.",
        source_url=f"https://example.invalid/{sq.id}",
        source_title="Example source",
        snippet="Canned snippet standing in for extracted page content.",
        retrieved_at=now(),
        extraction_status="full",
    )
    answered = sq.model_copy(update={"status": "answered", "passes": sq.passes + 1})
    plan = [answered if item.id == sq.id else item for item in state.plan]

    return NodeOutcome(
        state=state.model_copy(
            update={
                "plan": plan,
                "findings": [*state.findings, finding],
                "reviewed": False,
            }
        ),
        sub_question_id=sq.id,
    )


async def review(state: RunState, deps: Deps) -> NodeOutcome:
    await asyncio.sleep(NODE_DELAY_S)
    return NodeOutcome(
        state=state.model_copy(update={"reviewed": True}),
        note="canned reviewer accepted every sub-question",
    )


async def write(state: RunState, deps: Deps) -> NodeOutcome:
    await asyncio.sleep(NODE_DELAY_S)
    sections = [
        Section(
            heading=sq.text,
            claims=[
                Claim(text=f.claim, finding_ids=[f.id])
                for f in state.findings
                if f.sub_question_id == sq.id
            ],
        )
        for sq in state.plan
    ]
    report = Report(title=state.question, sections=sections, generated_at=now())
    return NodeOutcome(
        state=state.model_copy(
            update={
                "report": report,
                "report_markdown": _canned_markdown(report),
                "status": "done",
                "finished_at": now(),
            }
        )
    )


def _canned_markdown(report: Report) -> str:
    """Placeholder renderer. Replaced by render.py in Phase 4."""
    lines = [f"# {report.title}", ""]
    for section in report.sections:
        lines += [f"## {section.heading}", ""]
        lines += [f"{claim.text}" for claim in section.claims]
        lines.append("")
    return "\n".join(lines).strip() + "\n"


CANNED_NODES = {"plan": plan, "research": research, "review": review, "write": write}
