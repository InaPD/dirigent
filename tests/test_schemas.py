"""Every model must survive a JSON round trip, because that is how state is stored."""

import pytest

from ra.schemas import (
    Budgets,
    Finding,
    Report,
    RunState,
    StepRecord,
    SubQuestion,
    ToolCall,
    Usage,
)
from tests.factories import make_finding, make_plan, make_report, make_state

# A full run document with defaults must stay small. If this fails, something started
# storing extracted page content on the state.
MAX_EMPTY_STATE_BYTES = 2048


def test_empty_state_is_small():
    state = RunState(run_id="run_x", question="q", status="queued")
    assert len(state.model_dump_json()) < MAX_EMPTY_STATE_BYTES


@pytest.mark.parametrize(
    "model, instance",
    [
        (SubQuestion, SubQuestion(id="sq_01", text="t")),
        (Finding, make_finding()),
        (ToolCall, ToolCall(tool="tavily.search", args_digest="abc", status="ok", duration_ms=5)),
        (Usage, Usage(input_tokens=10, output_tokens=2)),
        (Budgets, Budgets()),
        (Report, make_report()),
    ],
)
def test_round_trip(model, instance):
    assert model.model_validate_json(instance.model_dump_json()) == instance


def test_run_state_round_trip_with_everything():
    state = make_state(
        plan=make_plan(2),
        findings=[make_finding()],
        report=make_report(),
        report_markdown="# Title\n",
        steps=[
            StepRecord(
                seq=1,
                node="plan",
                started_at=make_state().created_at,
                duration_ms=12,
                status="ok",
                input_digest="a" * 12,
                output_digest="b" * 12,
            )
        ],
    )
    assert RunState.model_validate_json(state.model_dump_json()) == state


def test_is_terminal():
    assert make_state(status="done").is_terminal
    assert make_state(status="failed").is_terminal
    assert make_state(status="budget_exceeded").is_terminal
    assert not make_state(status="running").is_terminal
    assert not make_state(status="queued").is_terminal


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "file:///etc/passwd", "data:text/html,<script>", "ftp://x.test/a"],
)
def test_a_finding_cannot_carry_a_non_http_source(url):
    """A source url becomes a link in the rendered report, so the scheme is not negotiable."""
    with pytest.raises(ValueError, match="http"):
        make_finding(source_url=url)


@pytest.mark.parametrize("url", ["http://a.test/x", "https://b.test/y?q=1"])
def test_http_sources_are_accepted(url):
    assert make_finding(source_url=url).source_url == url


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_searches_per_sq", 100_000),
        ("max_tavily_credits", 10**9),
        ("max_total_tokens", 10**12),
        ("max_subquestions", 5_000),
        ("max_extracts_per_sq", 0),
        ("max_revisions", -1),
        ("max_wall_clock_s", 10**6),
    ],
)
def test_a_budget_beyond_the_ceiling_is_refused(field, value):
    """POST /research takes these from the request body and they start paid work."""
    with pytest.raises(ValueError):
        Budgets(**{field: value})


def test_the_defaults_are_inside_their_own_ceilings():
    assert Budgets() == Budgets.model_validate(Budgets().model_dump())
