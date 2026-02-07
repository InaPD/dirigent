"""The planner. It must cap what it accepts, and never hand back a run with no plan."""

import pytest

from ra.deps import Deps
from ra.llm import LLMError
from ra.nodes.plan import PlanDraft, plan
from ra.routing import route
from ra.schemas import Budgets
from tests.factories import make_state
from tests.fakes import FakeLLM


def build(deps: Deps, *responses) -> Deps:
    return Deps(store=deps.store, settings=deps.settings, llm=FakeLLM({PlanDraft: list(responses)}))


async def test_a_plan_becomes_numbered_sub_questions(deps):
    d = build(deps, PlanDraft(sub_questions=["What is it?", "Who builds it?"]))

    outcome = await plan(make_state(), d)

    assert [sq.id for sq in outcome.state.plan] == ["sq_01", "sq_02"]
    assert outcome.state.plan[0].text == "What is it?"
    assert all(sq.status == "pending" for sq in outcome.state.plan)
    assert all(sq.passes == 0 for sq in outcome.state.plan)
    assert outcome.status == "ok"
    assert outcome.model == "claude-haiku-4-5"


async def test_over_delivery_is_truncated_to_the_cap(deps):
    d = build(deps, PlanDraft(sub_questions=[f"Question {i}?" for i in range(10)]))

    outcome = await plan(make_state(budgets=Budgets(max_subquestions=3)), d)

    assert len(outcome.state.plan) == 3
    assert "over the cap by 7" in outcome.note


async def test_blank_sub_questions_are_discarded(deps):
    d = build(deps, PlanDraft(sub_questions=["  ", "Real question?", ""]))

    outcome = await plan(make_state(), d)

    assert [sq.text for sq in outcome.state.plan] == ["Real question?"]


async def test_the_prompt_carries_the_question_and_the_cap(deps):
    d = build(deps, PlanDraft(sub_questions=["a?"]))
    state = make_state(budgets=Budgets(max_subquestions=3))

    await plan(state, d)

    prompt = d.llm.prompts_for(PlanDraft)[0]
    assert state.question in prompt
    assert "3" in prompt


@pytest.mark.parametrize(
    "response",
    [PlanDraft(sub_questions=[]), PlanDraft(sub_questions=["   "])],
    ids=["empty", "all blank"],
)
async def test_a_run_that_cannot_be_planned_fails_rather_than_looping(deps, response):
    """The router sends a plan-less run back to the planner, so an empty plan must end it."""
    d = build(deps, response)

    outcome = await plan(make_state(), d)

    assert outcome.state.status == "failed"
    assert outcome.state.finished_at is not None
    assert "no sub-questions" in outcome.state.error
    assert outcome.status == "error"
    assert route(outcome.state) == "__end__"


async def test_a_model_failure_ends_the_run_with_a_readable_error(deps):
    d = build(deps, LLMError("refusal", "the model declined (cyber)"))

    outcome = await plan(make_state(), d)

    assert outcome.state.status == "failed"
    assert "planner failed" in outcome.state.error
    assert "refusal" in outcome.state.error
    assert route(outcome.state) == "__end__"


async def test_the_planner_reports_what_it_spent_even_when_it_fails(deps):
    d = build(deps, PlanDraft(sub_questions=[]))

    outcome = await plan(make_state(), d)

    assert outcome.usage.input_tokens == 100
    assert outcome.model == "claude-haiku-4-5"
