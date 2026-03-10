"""Two guards against a run that cannot make progress.

The router derives the next node from the state. That is what makes resume trivial, but it
also means a node that fails without changing anything gets sent straight back in, forever.
Left alone, the graph spins until its recursion limit, and every spin can be a paid call.
So: if the last few steps are all errors that left the state untouched, stop.

The sweeper has the same shape one level up. It puts a run back on the queue whenever its
worker disappears, which is exactly right when the worker died for its own reasons, and
exactly wrong when the run is what killed it. An out-of-memory kill, or a crash in a
dependency on one particular document, repeats: worker dies, run is re-enqueued, worker
dies. The run document outlives every worker, so that cycle survives restarts and deploys.
So: count the re-enqueues, and stop once a run has had its share.

Both guards end the run and write the reason into the trace.
"""

import logging

from ra.clock import now
from ra.schemas import RunState, StepRecord
from ra.trace import append_step, digest

log = logging.getLogger("ra.progress")

STALL_NODE = "stalled"
MAX_STALLED_STEPS = 3

ABANDONED_NODE = "abandoned"


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


def is_exhausted(state: RunState, max_attempts: int) -> bool:
    """Has this run used up the times it may be put back on the queue?"""
    return state.attempt >= max_attempts


def apply_abandoned(state: RunState) -> RunState:
    """End a run that keeps taking its worker down with it.

    The last thing the trace shows is whatever the run got through before the worker went,
    which is the useful part: it says where the crash happens.
    """
    error = (
        f"abandoned after {state.attempt} re-enqueues: the run did not survive a worker. "
        f"Last step: {state.steps[-1].node if state.steps else 'none'}"
    )
    log.error("run %s abandoned after %d attempts", state.run_id, state.attempt)

    step = StepRecord(
        seq=len(state.steps) + 1,
        node=ABANDONED_NODE,
        started_at=now(),
        duration_ms=0,
        status="error",
        input_digest=digest(state),
        output_digest=digest(state),
        error=error,
    )
    stopped = state.model_copy(update={"status": "failed", "error": error, "finished_at": now()})
    return append_step(stopped, step)
