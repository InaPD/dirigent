"""Step-level tracing.

Every work node is wrapped by @traced. The wrapper owns timing, digests, the StepRecord,
the run totals, the cost lookup, the save and the lease refresh, so no node has to remember
any of it and every step in the trace is recorded the same way.

A node that raises becomes a step with status "error" and the run carries on. The router
decides what to do about it. The supervisor sees statuses, never stack traces.
"""

import hashlib
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from ra.clock import now
from ra.deps import Deps
from ra.errors import safe_detail
from ra.pricing import cost_usd
from ra.schemas import NodeOutcome, RunState, StepRecord

log = logging.getLogger("ra.trace")

NodeFn = Callable[[RunState, Deps], Awaitable[NodeOutcome]]
TracedFn = Callable[[RunState, Deps], Awaitable[RunState]]

DIGEST_LEN = 12


def digest(state: RunState) -> str:
    """Fingerprint of a state, ignoring the trace itself.

    Steps are excluded so that running the same node on the same inputs twice produces the
    same input_digest. That is what makes the resume test's "exactly once" claim checkable.
    """
    payload = state.model_dump_json(exclude={"steps"})
    return hashlib.sha256(payload.encode()).hexdigest()[:DIGEST_LEN]


def append_step(state: RunState, step: StepRecord) -> RunState:
    """Add a step and roll its numbers into the run totals. Returns a new state."""
    return state.model_copy(
        update={
            "steps": [*state.steps, step],
            "tokens_in": state.tokens_in + step.tokens_in,
            "tokens_out": state.tokens_out + step.tokens_out,
            "cost_usd": round(state.cost_usd + step.cost_usd, 6),
        }
    )


def traced(node: str) -> Callable[[NodeFn], TracedFn]:
    """Wrap a (state, deps) -> NodeOutcome node so that it records a step and saves."""

    def decorate(fn: NodeFn) -> TracedFn:
        async def wrapper(state: RunState, deps: Deps) -> RunState:
            started_at = now()
            t0 = perf_counter()
            input_digest = digest(state)

            try:
                outcome = await fn(state, deps)
            except Exception as exc:  # a node bug must not kill the run
                log.exception("node %s failed on run %s", node, state.run_id)
                outcome = NodeOutcome(state=state, status="error", error=safe_detail(exc))

            duration_ms = int((perf_counter() - t0) * 1000)
            new_state = outcome.state
            usage = outcome.usage
            cost = cost_usd(outcome.model, usage) if outcome.model and usage else 0.0

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
                cost_usd=cost,
                tool_calls=outcome.tool_calls,
                input_digest=input_digest,
                output_digest=digest(new_state),
                error=outcome.error,
                note=outcome.note,
            )

            new_state = append_step(new_state, step)
            await deps.store.save(new_state)
            if new_state.worker_id:
                await deps.store.refresh_lease(new_state.run_id, new_state.worker_id)
            return new_state

        wrapper.__name__ = f"traced_{node}"
        return wrapper

    return decorate


def as_graph_node(node: str, fn: NodeFn, deps: Deps) -> Callable[[RunState], Awaitable[dict]]:
    """Bind deps and adapt to what LangGraph calls: state in, whole state dict out."""
    traced_fn = traced(node)(fn)

    async def _run(state: RunState) -> dict:
        new_state = await traced_fn(state, deps)
        return new_state.model_dump()

    _run.__name__ = f"node_{node}"
    return _run
