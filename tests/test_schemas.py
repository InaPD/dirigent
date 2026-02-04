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
