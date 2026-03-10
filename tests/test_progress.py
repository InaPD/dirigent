"""The stall guard: a run that cannot move forward must stop, not spin."""

import pytest

import ra.nodes.canned as canned
from ra.deps import Deps
from ra.graph import RECURSION_LIMIT, build_graph
from ra.progress import MAX_STALLED_STEPS, apply_stall_stop, is_stalled, stalled_steps
from ra.schemas import RunState, StepRecord
from tests.factories import make_state


def step(seq: int, *, node="plan", status="error", same_state=True, error="boom") -> StepRecord:
    return StepRecord(
        seq=seq,
        node=node,
        started_at=make_state().created_at,
        duration_ms=1,
        status=status,
        input_digest="a" * 12,
        output_digest="a" * 12 if same_state else "b" * 12,
        error=error,
    )


def test_a_fresh_run_is_not_stalled():
    assert stalled_steps(make_state()) == 0
    assert not is_stalled(make_state())


def test_errors_that_changed_the_state_are_progress():
    """The researcher errors but still counts a pass, so the run moves on."""
    state = make_state(steps=[step(i, same_state=False) for i in range(1, 6)])

    assert stalled_steps(state) == 0
    assert not is_stalled(state)


def test_a_successful_step_resets_the_count():
    state = make_state(
        steps=[step(1), step(2), step(3, status="ok"), step(4)],
    )

    assert stalled_steps(state) == 1


def test_only_the_trailing_run_of_failures_counts():
    state = make_state(steps=[step(1, status="ok"), step(2), step(3)])

    assert stalled_steps(state) == 2


def test_the_guard_trips_at_the_threshold():
    below = make_state(steps=[step(i) for i in range(1, MAX_STALLED_STEPS)])
    at = make_state(steps=[step(i) for i in range(1, MAX_STALLED_STEPS + 1)])

    assert not is_stalled(below)
    assert is_stalled(at)


def test_the_stop_names_the_node_and_the_last_error():
    state = make_state(steps=[step(i, node="review", error="model exploded") for i in range(1, 4)])

    stopped = apply_stall_stop(state)

    assert stopped.status == "failed"
    assert stopped.finished_at is not None
    assert "review" in stopped.error
    assert "3 attempts" in stopped.error
    assert "model exploded" in stopped.error
    assert stopped.steps[-1].node == "stalled"


async def test_a_node_that_never_progresses_stops_the_run(deps):
    """Without this the graph spins to its recursion limit, paying for every attempt."""

    class AlwaysFailingLLM:
        def __init__(self):
            self.calls = 0

        async def structured(self, **kwargs):
            self.calls += 1
            raise PermissionError("401 - API key is invalid")

    llm = AlwaysFailingLLM()
    d = Deps(store=deps.store, settings=deps.settings, llm=llm)
    state = make_state(worker_id="worker-test")
    await deps.store.save(state)

    out = await build_graph(d).ainvoke(
        state.model_dump(), config={"recursion_limit": RECURSION_LIMIT}
    )
    final = RunState.model_validate(out)

    assert final.status == "failed"
    assert "planner failed" in final.error
    # the planner ends the run on its first failure, so exactly one paid attempt
    assert llm.calls == 1
    assert [s.node for s in final.steps] == ["plan"]


async def test_the_guard_catches_a_node_the_planner_check_would_not(deps, monkeypatch):
    """Belt and braces: any future node that fails without progressing is stopped too."""

    async def useless_review(state, _deps):
        raise RuntimeError("reviewer is broken")

    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)
    monkeypatch.setitem(canned.CANNED_NODES, "review", useless_review)

    state = make_state(
        worker_id="worker-test",
        plan=[],
        steps=[],
    )
    from tests.factories import make_plan

    state = state.model_copy(update={"plan": make_plan(1, status="answered")})
    await deps.store.save(state)

    out = await build_graph(deps).ainvoke(
        state.model_dump(), config={"recursion_limit": RECURSION_LIMIT}
    )
    final = RunState.model_validate(out)

    assert final.status == "failed"
    assert "review" in final.error
    assert [s.node for s in final.steps].count("review") == MAX_STALLED_STEPS
    assert final.steps[-1].node == "stalled"


@pytest.mark.parametrize("limit", [MAX_STALLED_STEPS])
def test_the_threshold_is_small_enough_to_be_cheap(limit):
    """Each stalled attempt can be a paid call, so the guard must not be generous."""
    assert limit <= 3


# -- the re-enqueue guard ------------------------------------------------------


def test_a_fresh_run_has_attempts_left():
    from ra.progress import is_exhausted

    assert not is_exhausted(make_state(attempt=0), 3)
    assert not is_exhausted(make_state(attempt=2), 3)


def test_a_run_is_exhausted_at_its_limit():
    from ra.progress import is_exhausted

    assert is_exhausted(make_state(attempt=3), 3)
    assert is_exhausted(make_state(attempt=9), 3)


def test_abandoning_names_the_count_and_the_last_step():
    from ra.progress import apply_abandoned

    state = make_state(attempt=3, steps=[step(1, node="research", status="ok")])

    stopped = apply_abandoned(state)

    assert stopped.status == "failed"
    assert stopped.finished_at is not None
    assert "3 re-enqueues" in stopped.error
    assert "research" in stopped.error
    assert stopped.steps[-1].node == "abandoned"


def test_abandoning_a_run_that_never_got_anywhere():
    from ra.progress import apply_abandoned

    stopped = apply_abandoned(make_state(attempt=3, steps=[]))

    assert stopped.status == "failed"
    assert "none" in stopped.error
