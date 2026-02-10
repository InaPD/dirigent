"""Researcher stand-ins used only by the crash-resume test.

They live here, not in tests/, because that test runs real worker processes and a subprocess
can only import what is installed. Selected with RA_STUB, which nothing else sets.

The slow stub exists so the test has an exact moment to kill a worker at. It signals through
Redis and then blocks, rather than sleeping, so the test never races a timer.
"""

import logging

from ra.clock import now
from ra.deps import Deps
from ra.ids import new_finding_id
from ra.routing import next_open_subquestion
from ra.schemas import Finding, NodeOutcome, RunState

log = logging.getLogger("ra.nodes.stubs")

GATE_REACHED = "test:gate:reached"
GATE_RELEASE = "test:gate:release"
BLOCK_AFTER = "sq_01"
GATE_TIMEOUT_S = 60


def _finding(sq_id: str) -> Finding:
    return Finding(
        id=new_finding_id(),
        sub_question_id=sq_id,
        claim=f"A stubbed finding for {sq_id}.",
        source_url=f"https://stub.invalid/{sq_id}",
        source_title="Stub source",
        snippet="stubbed snippet",
        retrieved_at=now(),
        extraction_status="full",
    )


def _answered(state: RunState, sq) -> NodeOutcome:
    updated = sq.model_copy(update={"status": "answered", "passes": sq.passes + 1})
    return NodeOutcome(
        state=state.model_copy(
            update={
                "plan": [updated if item.id == sq.id else item for item in state.plan],
                "findings": [*state.findings, _finding(sq.id)],
                "reviewed": False,
            }
        ),
        sub_question_id=sq.id,
    )


async def fast_research(state: RunState, deps: Deps) -> NodeOutcome:
    """Answers immediately. The replacement worker uses this so the test finishes."""
    sq = next_open_subquestion(state)
    if sq is None:
        return NodeOutcome(state=state, status="skipped", error="no open sub-question")
    return _answered(state, sq)


async def slow_research(state: RunState, deps: Deps) -> NodeOutcome:
    """Answers the first sub-question, then blocks on the second until released.

    The test waits for the gate, which is the only point at which the run is provably half
    finished, and kills the worker there. No sleeps, so nothing can flake on timing.
    """
    sq = next_open_subquestion(state)
    if sq is None:
        return NodeOutcome(state=state, status="skipped", error="no open sub-question")
    if sq.id == BLOCK_AFTER:
        return _answered(state, sq)

    redis = deps.store.client
    log.warning("stub reached the gate on %s", sq.id)
    await redis.rpush(GATE_REACHED, sq.id)
    await redis.blpop(GATE_RELEASE, GATE_TIMEOUT_S)
    return _answered(state, sq)


STUBS = {
    "slow_research": {"research": slow_research},
    "fast_research": {"research": fast_research},
}
