"""The router decides everything, including what resume does. Keep this table exhaustive."""

import pytest
from langgraph.graph import END

from ra.routing import next_open_subquestion, route
from ra.schemas import SubQuestion
from tests.factories import make_plan, make_report, make_state


def answered(n: int) -> list[SubQuestion]:
    return make_plan(n, status="answered")


ROUTER_CASES = [
    ("terminal: done", {"status": "done"}, END),
    ("terminal: failed", {"status": "failed"}, END),
    ("terminal: budget exceeded", {"status": "budget_exceeded"}, END),
    ("report already written", {"report": make_report()}, END),
    ("queued, no plan", {"status": "queued"}, "plan"),
    ("running, no plan", {}, "plan"),
    ("plan with a pending sub-question", {"plan": make_plan(2)}, "research"),
    (
        "plan with a reopened sub-question",
        {"plan": make_plan(1, status="needs_one_more_pass"), "reviewed": True},
        "research",
    ),
    (
        "one answered, one pending",
        {"plan": [*answered(1), SubQuestion(id="sq_02", text="t")]},
        "research",
    ),
    ("all answered, not reviewed", {"plan": answered(2)}, "review"),
    ("all answered, reviewed", {"plan": answered(2), "reviewed": True}, "write"),
    (
        "all unanswerable, not reviewed",
        {"plan": make_plan(2, status="unanswerable")},
        "review",
    ),
    (
        "all unanswerable, reviewed",
        {"plan": make_plan(2, status="unanswerable"), "reviewed": True},
        "write",
    ),
]


@pytest.mark.parametrize(
    "overrides, expected",
    [(o, e) for _, o, e in ROUTER_CASES],
    ids=[name for name, _, _ in ROUTER_CASES],
)
def test_route(overrides, expected):
    assert route(make_state(**overrides)) == expected


def test_terminal_status_beats_an_open_sub_question():
    """Resume must not restart a run that already finished."""
    state = make_state(status="done", plan=make_plan(2))
    assert route(state) == END


def test_next_open_sub_question_is_the_first_in_order():
    plan = [
        SubQuestion(id="sq_01", text="a", status="answered"),
        SubQuestion(id="sq_02", text="b", status="pending"),
        SubQuestion(id="sq_03", text="c", status="pending"),
    ]
    assert next_open_subquestion(make_state(plan=plan)).id == "sq_02"


def test_next_open_sub_question_is_none_when_settled():
    plan = make_plan(2, status="answered") + make_plan(1, status="unanswerable")
    assert next_open_subquestion(make_state(plan=plan)) is None


def test_route_is_pure():
    """route() must not touch the state. Resume depends on it being a plain read."""
    state = make_state(plan=make_plan(2))
    before = state.model_dump_json()
    route(state)
    assert state.model_dump_json() == before
