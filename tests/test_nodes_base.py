"""The wrapper that turns a node into a traced graph step."""

import pytest

from ra.deps import Deps
from ra.nodes.base import digest, record_step
from ra.schemas import NodeOutcome, RunState, ToolCall, Usage
from tests.factories import make_plan, make_state


async def ok_node(state: RunState, deps: Deps) -> NodeOutcome:
    return NodeOutcome(
        state=state.model_copy(update={"plan": make_plan(1)}),
        model="claude-haiku-4-5",
        usage=Usage(input_tokens=100, output_tokens=20),
        sub_question_id="sq_01",
        note="a reason",
    )


async def boom_node(state: RunState, deps: Deps) -> NodeOutcome:
    raise RuntimeError("the node exploded")


async def test_step_is_appended_with_the_measured_numbers(deps):
    state = make_state(worker_id="worker-test")
    new = await record_step("plan", ok_node, state, deps)

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


async def test_totals_accumulate(deps):
    state = make_state(worker_id="worker-test")
    state = await record_step("plan", ok_node, state, deps)
    state = await record_step("plan", ok_node, state, deps)

    assert state.tokens_in == 200
    assert state.tokens_out == 40
    assert [s.seq for s in state.steps] == [1, 2]


async def test_state_is_saved_after_every_node(deps):
    state = make_state(worker_id="worker-test")
    new = await record_step("plan", ok_node, state, deps)
    assert await deps.store.load(new.run_id) == new


async def test_lease_is_refreshed_after_a_node(deps):
    state = make_state(worker_id="worker-test")
    await deps.store.acquire_lease(state.run_id, "worker-test")
    await deps.store.save(state)
    await record_step("plan", ok_node, state, deps)
    assert await deps.store.lease_holder(state.run_id) == "worker-test"


async def test_a_node_exception_becomes_an_error_step_not_a_crash(deps):
    state = make_state(worker_id="worker-test")
    new = await record_step("research", boom_node, state, deps)

    assert new.steps[0].status == "error"
    assert "the node exploded" in new.steps[0].error
    # the state is otherwise untouched, so the router can still decide what to do
    assert new.plan == []


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

    new = await record_step("research", node, make_state(), deps)
    assert new.steps[0].tool_calls[0].tool == "tavily.search"
    assert new.steps[0].tool_calls[0].credits == 1


def test_digest_ignores_the_trace():
    """Same inputs must give the same input_digest however long the trace already is."""
    from ra.schemas import StepRecord

    state = make_state()
    step = StepRecord(
        seq=1,
        node="plan",
        started_at=state.created_at,
        duration_ms=1,
        status="ok",
        input_digest="a" * 12,
        output_digest="b" * 12,
    )
    assert digest(state) == digest(state.model_copy(update={"steps": [step]}))


def test_digest_changes_when_the_state_changes():
    state = make_state()
    assert digest(state) != digest(state.model_copy(update={"plan": make_plan(1)}))


@pytest.mark.parametrize("field", ["plan", "findings", "report"])
def test_digest_is_stable_across_repeated_calls(field):
    state = make_state()
    assert digest(state) == digest(state)
