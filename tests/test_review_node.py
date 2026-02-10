"""The reviewer. It may reopen a sub-question, but only within the run's revision budget."""

import pytest

from ra.deps import Deps
from ra.llm import LLMError
from ra.nodes.review import MAX_PASSES, Review, Verdict, review
from ra.routing import route
from ra.schemas import Budgets, SubQuestion
from tests.factories import make_finding, make_state
from tests.fakes import FakeLLM


def verdicts(*pairs: tuple[str, str], reason: str = "because") -> Review:
    return Review(
        verdicts=[Verdict(sub_question_id=sq, verdict=v, reason=reason) for sq, v in pairs]
    )


def build(deps: Deps, *responses) -> Deps:
    return Deps(store=deps.store, settings=deps.settings, llm=FakeLLM({Review: list(responses)}))


def a_state(**overrides):
    base = {
        "plan": [
            SubQuestion(id="sq_01", text="first", status="answered", passes=1),
            SubQuestion(id="sq_02", text="second", status="answered", passes=1),
        ],
        "findings": [make_finding(sq="sq_01"), make_finding(sq="sq_02", fid="f_bbb222")],
    }
    return make_state(**(base | overrides))


async def test_a_clean_review_marks_the_plan_reviewed(deps):
    d = build(deps, verdicts(("sq_01", "answered"), ("sq_02", "answered")))

    outcome = await review(a_state(), d)

    assert outcome.state.reviewed is True
    assert [sq.status for sq in outcome.state.plan] == ["answered", "answered"]
    assert outcome.state.revisions_used == 0
    assert route(outcome.state) == "write"


async def test_the_reasons_reach_the_trace(deps):
    """A mediocre reviewer is forgivable as long as you can see what it thought."""
    d = build(deps, verdicts(("sq_01", "answered"), ("sq_02", "unanswerable"), reason="too thin"))

    outcome = await review(a_state(), d)

    assert "sq_01 answered" in outcome.note
    assert "sq_02 unanswerable" in outcome.note
    assert "too thin" in outcome.note


async def test_the_prompt_shows_each_question_with_its_findings(deps):
    d = build(deps, verdicts(("sq_01", "answered")))

    await review(a_state(), d)

    prompt = d.llm.prompts_for(Review)[0]
    assert "sq_01: first" in prompt
    assert "f_aaa111" in prompt
    assert "[full]" in prompt


async def test_a_question_with_no_findings_is_shown_as_such(deps):
    d = build(deps, verdicts(("sq_01", "unanswerable")))

    await review(a_state(findings=[]), d)

    assert "(no findings)" in d.llm.prompts_for(Review)[0]


# -- reopening -----------------------------------------------------------------


async def test_a_reopened_question_goes_back_to_the_researcher(deps):
    d = build(deps, verdicts(("sq_01", "needs_one_more_pass"), ("sq_02", "answered")))

    outcome = await review(a_state(), d)

    assert outcome.state.plan[0].status == "needs_one_more_pass"
    assert outcome.state.revisions_used == 1
    assert route(outcome.state) == "research"
    assert "reopened 1" in outcome.note


async def test_the_revision_budget_is_spent_only_once(deps):
    """max_revisions is 1, so the second review cannot reopen anything."""
    d = build(deps, verdicts(("sq_01", "needs_one_more_pass")))
    state = a_state(revisions_used=1, budgets=Budgets(max_revisions=1))

    outcome = await review(state, d)

    assert outcome.state.plan[0].status == "answered"  # it has findings, so it settles
    assert outcome.state.revisions_used == 1
    assert route(outcome.state) == "write"


async def test_a_question_cannot_be_researched_more_than_twice(deps):
    """A second guard, independent of the run-wide revision count."""
    d = build(deps, verdicts(("sq_01", "needs_one_more_pass")))
    state = a_state(
        plan=[SubQuestion(id="sq_01", text="first", status="answered", passes=MAX_PASSES)],
        budgets=Budgets(max_revisions=5),
    )

    outcome = await review(state, d)

    assert outcome.state.plan[0].status == "answered"


async def test_a_reopened_question_with_nothing_found_becomes_unanswerable(deps):
    d = build(deps, verdicts(("sq_01", "needs_one_more_pass")))
    state = a_state(findings=[], revisions_used=1)

    outcome = await review(state, d)

    assert outcome.state.plan[0].status == "unanswerable"


async def test_the_reopen_verdict_is_recorded_even_when_it_is_refused(deps):
    d = build(deps, verdicts(("sq_01", "needs_one_more_pass"), reason="thin evidence"))

    outcome = await review(a_state(revisions_used=1), d)

    assert "needs_one_more_pass" in outcome.note
    assert "thin evidence" in outcome.note


# -- honesty -------------------------------------------------------------------


async def test_a_question_with_no_findings_cannot_be_called_answered(deps):
    """Whatever the reviewer says, an empty sub-question is not an answered one."""
    d = build(deps, verdicts(("sq_01", "answered")))

    outcome = await review(a_state(findings=[]), d)

    assert outcome.state.plan[0].status == "unanswerable"


async def test_a_question_the_reviewer_ignored_is_left_alone(deps):
    d = build(deps, verdicts(("sq_01", "unanswerable")))

    outcome = await review(a_state(), d)

    assert outcome.state.plan[1].status == "answered"


# -- failure -------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [LLMError("refusal", "declined"), RuntimeError("reviewer exploded")],
    ids=["model error", "anything else"],
)
async def test_a_broken_reviewer_does_not_block_the_writer(deps, failure):
    d = build(deps, failure)

    outcome = await review(a_state(), d)

    assert outcome.status == "error"
    assert "reviewer failed" in outcome.error
    # the plan is marked reviewed anyway, so the run reaches the writer
    assert outcome.state.reviewed is True
    assert route(outcome.state) == "write"


async def test_a_failed_review_changes_the_state_so_the_run_cannot_stall(deps):
    """The stall guard fails a node that errors without progressing. This one progresses."""
    from ra.trace import digest

    d = build(deps, RuntimeError("boom"))
    state = a_state()

    outcome = await review(state, d)

    assert digest(outcome.state) != digest(state)


async def test_an_empty_review_leaves_the_plan_untouched(deps):
    d = build(deps, Review(verdicts=[]))

    outcome = await review(a_state(), d)

    assert [sq.status for sq in outcome.state.plan] == ["answered", "answered"]
    assert outcome.state.reviewed is True
    assert outcome.note is None
