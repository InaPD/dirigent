"""The wrapper that turns a node into a traced, priced, saved step."""

import pytest

from ra.deps import Deps
from ra.schemas import NodeOutcome, RunState, StepRecord, ToolCall, Usage
from ra.trace import append_step, digest, traced
from tests.factories import make_plan, make_state


async def ok_node(state: RunState, deps: Deps) -> NodeOutcome:
    return NodeOutcome(
        state=state.model_copy(update={"plan": make_plan(1)}),
        model="claude-haiku-4-5",
        usage=Usage(input_tokens=100, output_tokens=20),
        sub_question_id="sq_01",
        note="a reason",
    )


async def free_node(state: RunState, deps: Deps) -> NodeOutcome:
    """A node that calls no model, such as one served entirely from cache."""
    return NodeOutcome(state=state)


async def boom_node(state: RunState, deps: Deps) -> NodeOutcome:
    raise RuntimeError("the node exploded")


async def run(node: str, fn, state: RunState, deps: Deps) -> RunState:
    return await traced(node)(fn)(state, deps)


async def test_step_is_appended_with_the_measured_numbers(deps):
    new = await run("plan", ok_node, make_state(worker_id="worker-test"), deps)

    assert len(new.steps) == 1
    step = new.steps[0]
    assert step.seq == 1
    assert step.node == "plan"
    assert step.status == "ok"
    assert step.model == "claude-haiku-4-5"
    assert step.tokens_in == 100
    assert step.tokens_out == 20
    assert step.sub_question_id == "sq_01"
    assert step.note == "a reason"
    assert step.duration_ms >= 0


async def test_cost_comes_from_the_price_table(deps):
    """100 input and 20 output on Haiku 4.5, at $1 and $5 per million."""
    new = await run("plan", ok_node, make_state(worker_id="worker-test"), deps)

    expected = (100 * 1.00 + 20 * 5.00) / 1_000_000
    assert new.steps[0].cost_usd == pytest.approx(expected)
    assert new.cost_usd == pytest.approx(expected)


async def test_a_node_that_calls_no_model_costs_nothing(deps):
    new = await run("review", free_node, make_state(worker_id="worker-test"), deps)

    assert new.steps[0].cost_usd == 0.0
    assert new.steps[0].model is None
    assert new.cost_usd == 0.0


async def test_totals_accumulate_across_steps(deps):
    state = make_state(worker_id="worker-test")
    state = await run("plan", ok_node, state, deps)
    state = await run("plan", ok_node, state, deps)

    assert state.tokens_in == 200
    assert state.tokens_out == 40
    assert state.cost_usd == pytest.approx(2 * (100 * 1.00 + 20 * 5.00) / 1_000_000)
    assert [s.seq for s in state.steps] == [1, 2]


async def test_state_is_saved_after_every_node(deps):
    new = await run("plan", ok_node, make_state(worker_id="worker-test"), deps)
    assert await deps.store.load(new.run_id) == new


async def test_lease_is_refreshed_after_a_node(deps):
    state = make_state(worker_id="worker-test")
    await deps.store.acquire_lease(state.run_id, "worker-test")
    await deps.store.save(state)

    await run("plan", ok_node, state, deps)

    assert await deps.store.lease_holder(state.run_id) == "worker-test"


async def test_a_node_exception_becomes_an_error_step_not_a_crash(deps):
    new = await run("research", boom_node, make_state(worker_id="worker-test"), deps)

    assert new.steps[0].status == "error"
    assert "the node exploded" in new.steps[0].error
    # the state is otherwise untouched, so the router can still decide what to do
    assert new.plan == []
    assert new.status == "running"


async def test_tool_calls_ride_along(deps):
    async def node(state, _deps):
        return NodeOutcome(
            state=state,
            tool_calls=[
                ToolCall(
                    tool="tavily.search",
                    args_digest="abc123",
                    status="ok",
                    duration_ms=3,
                    credits=1,
                )
            ],
        )

    new = await run("research", node, make_state(), deps)

    assert new.steps[0].tool_calls[0].tool == "tavily.search"
    assert new.steps[0].tool_calls[0].credits == 1


def a_step(seq: int = 1, **overrides) -> StepRecord:
    base = {
        "seq": seq,
        "node": "plan",
        "started_at": make_state().created_at,
        "duration_ms": 1,
        "status": "ok",
        "input_digest": "a" * 12,
        "output_digest": "b" * 12,
    }
    return StepRecord(**(base | overrides))


def test_append_step_rolls_numbers_into_the_totals():
    state = make_state()
    new = append_step(state, a_step(tokens_in=5, tokens_out=3, cost_usd=0.001))

    assert new.tokens_in == 5
    assert new.tokens_out == 3
    assert new.cost_usd == pytest.approx(0.001)


def test_digest_ignores_the_trace():
    """Same inputs must give the same input_digest however long the trace already is."""
    state = make_state()
    assert digest(state) == digest(state.model_copy(update={"steps": [a_step()]}))


def test_digest_changes_when_the_state_changes():
    state = make_state()
    assert digest(state) != digest(state.model_copy(update={"plan": make_plan(1)}))


def test_digest_is_stable_across_repeated_calls():
    state = make_state()
    assert digest(state) == digest(state)
    assert len(digest(state)) == 12


async def test_the_lease_is_not_refreshed_for_a_run_with_no_worker(deps):
    """A run being replayed or inspected outside a worker has no lease to refresh."""
    state = make_state(worker_id=None)
    new = await run("plan", ok_node, state, deps)

    assert new.steps[0].status == "ok"
    assert await deps.store.lease_holder(state.run_id) is None
