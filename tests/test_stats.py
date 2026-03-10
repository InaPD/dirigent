"""Aggregation across runs. Pure arithmetic, so no Redis and no server here."""

from datetime import timedelta

import pytest

from ra.clock import now
from ra.schemas import StepRecord
from ra.stats import QUESTION_PREVIEW, summarise, summarise_run
from tests.factories import make_finding, make_report, make_state

T0 = now()


def step(node: str, *, status="ok", ms=100, cost=0.0, seq=1) -> StepRecord:
    return StepRecord(
        seq=seq,
        node=node,
        started_at=T0,
        duration_ms=ms,
        status=status,
        cost_usd=cost,
        input_digest="a" * 12,
        output_digest="b" * 12,
    )


def a_run(**overrides):
    base = {"created_at": T0, "status": "done"}
    return make_state(**(base | overrides))


# -- the window is what everything is a total over -----------------------------


def test_an_empty_window_totals_nothing():
    overview = summarise([])

    assert overview.window.runs == 0
    assert overview.spend.cost_usd == 0.0
    assert overview.nodes == []
    assert overview.window.oldest is None


def test_the_window_reports_what_was_actually_aggregated():
    older = a_run(created_at=T0 - timedelta(hours=5))
    newer = a_run(created_at=T0)

    overview = summarise([older, newer], since=T0 - timedelta(days=7), truncated=True)

    assert overview.window.runs == 2
    assert overview.window.oldest == older.created_at
    assert overview.window.newest == newer.created_at
    assert overview.window.truncated is True


# -- spend ---------------------------------------------------------------------


def test_spend_adds_up_across_runs():
    runs = [
        a_run(cost_usd=0.012, tokens_in=100, tokens_out=20, tavily_credits=3),
        a_run(cost_usd=0.008, tokens_in=200, tokens_out=40, tavily_credits=5),
    ]

    spend = summarise(runs).spend

    assert spend.cost_usd == pytest.approx(0.02)
    assert spend.tokens_in == 300
    assert spend.tokens_out == 60
    assert spend.tavily_credits == 8


def test_cost_does_not_drift_over_many_runs():
    """Floats accumulate. The total is rounded to the cent fraction costs are quoted in."""
    runs = [a_run(cost_usd=0.001) for _ in range(1000)]

    assert summarise(runs).spend.cost_usd == pytest.approx(1.0)


# -- what went wrong, and where ------------------------------------------------


def test_runs_are_counted_by_status():
    runs = [a_run(status="done"), a_run(status="done"), a_run(status="failed")]

    assert summarise(runs).by_status == {"done": 2, "failed": 1}


def test_the_noisiest_node_comes_first():
    """The question this answers is where things go wrong, so failures sort to the top."""
    runs = [
        a_run(steps=[step("research", status="error"), step("write", seq=2)]),
        a_run(steps=[step("research", status="error"), step("plan", seq=2)]),
        a_run(steps=[step("plan"), step("plan", seq=2), step("plan", seq=3)]),
    ]

    nodes = summarise(runs).nodes

    assert nodes[0].node == "research"
    assert nodes[0].error == 2
    assert nodes[0].ok == 0


def test_a_node_row_counts_every_status():
    runs = [
        a_run(
            steps=[
                step("research"),
                step("research", status="error", seq=2),
                step("research", status="skipped", seq=3),
                step("budget", status="budget_exceeded", seq=4),
            ]
        )
    ]

    by_node = {n.node: n for n in summarise(runs).nodes}

    assert by_node["research"].ok == 1
    assert by_node["research"].error == 1
    assert by_node["research"].skipped == 1
    assert by_node["research"].calls == 3
    assert by_node["budget"].budget_exceeded == 1


def test_a_node_row_carries_its_time_and_cost():
    runs = [
        a_run(steps=[step("write", ms=500, cost=0.01), step("write", ms=1500, cost=0.02, seq=2)])
    ]

    write = next(n for n in summarise(runs).nodes if n.node == "write")

    assert write.total_ms == 2000
    assert write.slowest_ms == 1500
    assert write.cost_usd == pytest.approx(0.03)


def test_a_node_counts_runs_not_calls():
    """A node that runs three times in one run has still only touched one run."""
    runs = [a_run(steps=[step("research"), step("research", seq=2), step("research", seq=3)])]

    research = summarise(runs).nodes[0]

    assert research.runs == 1
    assert research.calls == 3


# -- were the sources any good -------------------------------------------------


def test_findings_are_counted_by_quality():
    """The third question: how much of what gets cited was a real page."""
    runs = [
        a_run(
            findings=[
                make_finding(fid="f_aaa111"),
                make_finding(fid="f_bbb222", extraction_status="snippet_only"),
                make_finding(fid="f_ccc333", extraction_status="snippet_only"),
            ]
        )
    ]

    overview = summarise(runs)

    assert overview.findings_by_quality == {"snippet_only": 2, "full": 1}
    assert overview.findings == 3


def test_reports_written_is_not_the_same_as_runs():
    runs = [a_run(report=make_report()), a_run(status="failed")]

    overview = summarise(runs)

    assert overview.window.runs == 2
    assert overview.reports_written == 1


# -- one line per run ----------------------------------------------------------


def test_a_run_summary_carries_what_you_need_to_pick_one():
    state = a_run(
        status="done",
        cost_usd=0.03,
        findings=[make_finding()],
        report=make_report(),
        steps=[step("plan")],
        started_at=T0,
        finished_at=T0 + timedelta(seconds=42.5),
    )

    summary = summarise_run(state)

    assert summary.run_id == state.run_id
    assert summary.status == "done"
    assert summary.cost_usd == pytest.approx(0.03)
    assert summary.findings == 1
    assert summary.steps == 1
    assert summary.has_report is True
    assert summary.duration_s == pytest.approx(42.5)


def test_a_run_that_never_finished_has_no_duration():
    assert summarise_run(a_run(started_at=T0, finished_at=None)).duration_s is None
    assert summarise_run(a_run(started_at=None, finished_at=None)).duration_s is None


def test_a_long_question_is_shortened():
    summary = summarise_run(a_run(question="w" * 300))

    assert len(summary.question) == QUESTION_PREVIEW
    assert summary.question.endswith("…")


def test_a_short_question_is_left_alone():
    state = a_run(question="What is durable execution?")

    assert summarise_run(state).question == state.question


def test_a_failed_run_carries_its_error():
    assert summarise_run(a_run(status="failed", error="it broke")).error == "it broke"
