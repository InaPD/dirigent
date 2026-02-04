"""The router's decision function.

This is the whole resume mechanism. The next step is derived from the state and nothing
else, so restarting a run is just invoking the graph again: the router skips whatever is
already done. Keep it pure, and keep test_router.py exhaustive.
"""

from langgraph.graph import END

from ra.schemas import RunState, SubQuestion

OPEN_STATUSES = ("pending", "needs_one_more_pass")


def next_open_subquestion(state: RunState) -> SubQuestion | None:
    """First sub-question still needing research, or None when they are all settled."""
    for sq in state.plan:
        if sq.status in OPEN_STATUSES:
            return sq
    return None


def route(state: RunState) -> str:
    """Name of the next node, or END."""
    if state.is_terminal:
        return END
    if state.report is not None:
        return END
    if not state.plan:
        return "plan"
    if next_open_subquestion(state) is not None:
        return "research"
    if not state.reviewed:
        return "review"
    return "write"
