"""Adapter between a work node and LangGraph.

A node is written as (state, deps) -> NodeOutcome and never has to think about timing,
step records, run totals, saving, or the lease. This wrapper does all of that, so every
step in the trace is recorded the same way.

Phase 2 replaces the body of record_step with the @traced decorator in trace.py and adds
cost lookup. The node signature does not change.
"""

import hashlib
from collections.abc import Awaitable, Callable
from time import perf_counter

from ra.clock import now
from ra.deps import Deps
from ra.schemas import NodeOutcome, RunState, StepRecord

NodeFn = Callable[[RunState, Deps], Awaitable[NodeOutcome]]

DIGEST_LEN = 12
ERROR_LEN = 500


def digest(state: RunState) -> str:
    """Fingerprint of a state, ignoring the trace itself.

    Steps are excluded so that running the same node on the same inputs twice produces the
    same input_digest. That is what makes the resume test's "exactly once" claim checkable.
    """
    payload = state.model_dump_json(exclude={"steps"})
    return hashlib.sha256(payload.encode()).hexdigest()[:DIGEST_LEN]


async def record_step(node: str, fn: NodeFn, state: RunState, deps: Deps) -> RunState:
    """Run one node, append its StepRecord, update totals, save, refresh the lease."""
    started_at = now()
    t0 = perf_counter()
    input_digest = digest(state)

    try:
        outcome = await fn(state, deps)
    except Exception as exc:  # a node bug must not kill the run
        outcome = NodeOutcome(state=state, status="error", error=repr(exc)[:ERROR_LEN])

    duration_ms = int((perf_counter() - t0) * 1000)
    new_state = outcome.state
    usage = outcome.usage

    step = StepRecord(
        seq=len(state.steps) + 1,
        node=node,
        sub_question_id=outcome.sub_question_id,
        started_at=started_at,
        duration_ms=duration_ms,
        status=outcome.status,
        model=outcome.model,
        tokens_in=usage.input_tokens if usage else 0,
        tokens_out=usage.output_tokens if usage else 0,
        cost_usd=0.0,  # priced in Phase 2
        tool_calls=outcome.tool_calls,
        input_digest=input_digest,
        output_digest=digest(new_state),
        error=outcome.error,
        note=outcome.note,
    )

    new_state = new_state.model_copy(
        update={
            "steps": [*new_state.steps, step],
            "tokens_in": new_state.tokens_in + step.tokens_in,
            "tokens_out": new_state.tokens_out + step.tokens_out,
            "cost_usd": round(new_state.cost_usd + step.cost_usd, 6),
        }
    )

    await deps.store.save(new_state)
    if new_state.worker_id:
        await deps.store.refresh_lease(new_state.run_id, new_state.worker_id)
    return new_state


def as_graph_node(node: str, fn: NodeFn, deps: Deps) -> Callable[[RunState], Awaitable[dict]]:
    """Wrap a node so LangGraph can call it. Returns the whole state as a dict."""

    async def _run(state: RunState) -> dict:
        new_state = await record_step(node, fn, state, deps)
        return new_state.model_dump()

    _run.__name__ = f"node_{node}"
    return _run
