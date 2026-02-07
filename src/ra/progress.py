"""A stall guard.

The router derives the next node from the state. That is what makes resume trivial, but it
also means a node that fails without changing anything gets sent straight back in, forever.
Left alone, the graph spins until its recursion limit, and every spin can be a paid call.

So: if the last few steps are all errors that left the state untouched, the run is not going
to recover by trying again. Stop, and say so in the trace.
"""

import logging

from ra.clock import now
from ra.schemas import RunState, StepRecord
from ra.trace import append_step, digest

log = logging.getLogger("ra.progress")

STALL_NODE = "stalled"
MAX_STALLED_STEPS = 3


def stalled_steps(state: RunState) -> int:
    """How many steps in a row failed without moving the run forward."""
    count = 0
    for step in reversed(state.steps):
        if step.status != "error" or step.input_digest != step.output_digest:
            break
        count += 1
    return count


def is_stalled(state: RunState) -> bool:
    return stalled_steps(state) >= MAX_STALLED_STEPS


def apply_stall_stop(state: RunState) -> RunState:
    """End the run, naming the node that would not progress."""
    node = state.steps[-1].node if state.steps else "unknown"
    last_error = state.steps[-1].error if state.steps else None
    error = f"{node} made no progress in {stalled_steps(state)} attempts: {last_error}"
    log.warning("run %s stalled on %s", state.run_id, node)

    step = StepRecord(
        seq=len(state.steps) + 1,
        node=STALL_NODE,
        started_at=now(),
        duration_ms=0,
        status="error",
        input_digest=digest(state),
        output_digest=digest(state),
        error=error,
    )
    stopped = state.model_copy(update={"status": "failed", "error": error, "finished_at": now()})
    return append_step(stopped, step)
