"""Caps must provably stop a run, and a stopped run must still produce what it can."""

from datetime import timedelta

import pytest

import ra.nodes.canned as canned
from ra.budgets import apply_budget_stop, check_budgets
from ra.graph import RECURSION_LIMIT, build_graph
from ra.schemas import Budgets, RunState
from tests.factories import make_finding, make_plan, make_state


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


async def run_to_completion(deps, state: RunState) -> RunState:
    graph = build_graph(deps)
    await deps.store.save(state)
    out = await graph.ainvoke(state.model_dump(), config={"recursion_limit": RECURSION_LIMIT})
    return RunState.model_validate(out)


# -- check_budgets ---------------------------------------------------------------


def test_a_fresh_run_is_within_budget():
    assert check_budgets(make_state()).ok


def test_a_tiny_cap_does_not_trip_on_a_run_that_has_spent_nothing():
    """Caps measure spend, not intent. Zero spent is under every cap."""
    budgets = Budgets(max_total_tokens=1, writer_reserve_tokens=0)
    assert check_budgets(make_state(budgets=budgets), next_node="plan").ok


def test_the_writer_reserve_stops_the_run_early():
    """Spend is under the cap, but not by enough to afford the write pass."""
    budgets = Budgets(max_total_tokens=100_000, writer_reserve_tokens=20_000)
    state = make_state(budgets=budgets, tokens_in=85_000, tokens_out=0)

    assert check_budgets(state, next_node="research").exceeded == ["max_total_tokens"]


def test_the_reserve_is_released_for_the_writer_itself():
    """The same state is allowed through when the next node is the writer."""
    budgets = Budgets(max_total_tokens=100_000, writer_reserve_tokens=20_000)
    state = make_state(budgets=budgets, tokens_in=85_000, tokens_out=0)

    assert check_budgets(state, next_node="write").ok


def test_the_writer_is_still_stopped_by_the_hard_token_cap():
    budgets = Budgets(max_total_tokens=100_000, writer_reserve_tokens=20_000)
    state = make_state(budgets=budgets, tokens_in=90_000, tokens_out=20_000)

    assert check_budgets(state, next_node="write").exceeded == ["max_total_tokens"]


def test_tokens_count_both_directions():
    budgets = Budgets(max_total_tokens=1_000, writer_reserve_tokens=0)
    state = make_state(budgets=budgets, tokens_in=600, tokens_out=500)

    assert not check_budgets(state, next_node="write").ok


def test_the_wall_clock_cap_trips():
    state = make_state(budgets=Budgets(max_wall_clock_s=300))
    state = state.model_copy(update={"started_at": state.created_at})
    later = state.created_at + timedelta(seconds=301)

    assert check_budgets(state, later).exceeded == ["max_wall_clock_s"]


def test_the_wall_clock_is_not_checked_before_the_run_starts():
    state = make_state(started_at=None, budgets=Budgets(max_wall_clock_s=0))
    assert check_budgets(state).ok


def test_the_credit_cap_trips_on_reaching_it_not_passing_it():
    budgets = Budgets(max_tavily_credits=20)
    assert check_budgets(make_state(budgets=budgets, tavily_credits=19)).ok
    assert not check_budgets(make_state(budgets=budgets, tavily_credits=20)).ok


def test_several_caps_can_trip_at_once():
    state = make_state(
        budgets=Budgets(max_total_tokens=10, writer_reserve_tokens=0, max_tavily_credits=1),
        tokens_in=100,
        tavily_credits=5,
    )
    verdict = check_budgets(state, next_node="write")

    assert verdict.exceeded == ["max_total_tokens", "max_tavily_credits"]
    assert verdict.reason == "max_total_tokens, max_tavily_credits"


# -- apply_budget_stop -----------------------------------------------------------


def test_the_stop_is_written_into_the_trace():
    state = make_state(findings=[make_finding()])
    verdict = check_budgets(
        make_state(budgets=Budgets(max_total_tokens=1, writer_reserve_tokens=0), tokens_in=5)
    )

    stopped = apply_budget_stop(state, verdict)
    step = stopped.steps[-1]

    assert stopped.budget_stopped is True
    assert step.node == "budget"
    assert step.status == "budget_exceeded"
    assert step.error == "max_total_tokens"
    # it has findings, so the writer still gets a turn: the run is not over and must not
    # look over to a client polling on status
    assert stopped.status == "running"
    assert stopped.finished_at is None


def test_a_stop_with_nothing_to_write_finishes_the_run():
    verdict = check_budgets(make_state(budgets=Budgets(max_tavily_credits=0)))
    stopped = apply_budget_stop(make_state(findings=[]), verdict)

    assert stopped.budget_stopped is True
    assert stopped.status == "budget_exceeded"
    assert stopped.finished_at is not None


# -- through the graph -----------------------------------------------------------


@pytest.mark.parametrize(
    "budgets, spent, cap",
    [
        (
            Budgets(max_total_tokens=1, writer_reserve_tokens=0),
            {"tokens_in": 5},
            "max_total_tokens",
        ),
        (Budgets(max_wall_clock_s=0), {}, "max_wall_clock_s"),
        (Budgets(max_tavily_credits=0), {}, "max_tavily_credits"),
    ],
    ids=["tokens", "wall clock", "credits"],
)
async def test_an_already_exhausted_cap_stops_the_run_before_it_plans(deps, budgets, spent, cap):
    state = make_state(budgets=budgets, **spent)
    state = state.model_copy(update={"started_at": state.created_at})

    final = await run_to_completion(deps, state)

    assert final.status == "budget_exceeded"
    assert final.finished_at is not None
    assert [s.node for s in final.steps] == ["budget"]
    assert cap in final.steps[0].error
    # nothing was researched, so there is nothing to report
    assert final.report is None


async def test_a_run_stopped_mid_flight_still_gets_a_partial_report(deps):
    """The point of the writer reserve: a tripped cap produces a report, not an exception."""
    state = make_state(
        budgets=Budgets(max_tavily_credits=1),
        plan=make_plan(2, status="answered"),
        findings=[make_finding(sq="sq_01"), make_finding(sq="sq_02", fid="f_bbb222")],
        reviewed=True,
        tavily_credits=5,
    )

    final = await run_to_completion(deps, state)

    assert final.status == "budget_exceeded"
    assert final.report is not None
    assert final.report_markdown.startswith("# ")
    assert [s.node for s in final.steps] == ["budget", "write"]
    assert final.finished_at is not None


async def test_the_budget_step_names_every_cap_that_tripped(deps):
    state = make_state(
        budgets=Budgets(max_total_tokens=1, writer_reserve_tokens=0, max_tavily_credits=1),
        tokens_in=50,
        tavily_credits=9,
        findings=[make_finding()],
    )

    final = await run_to_completion(deps, state)
    budget_step = next(s for s in final.steps if s.node == "budget")

    assert "max_total_tokens" in budget_step.error
    assert "max_tavily_credits" in budget_step.error


async def test_the_run_is_not_stopped_twice(deps):
    """Once marked, the router leaves it alone. One budget step, not one per hop."""
    state = make_state(
        budgets=Budgets(max_tavily_credits=0),
        plan=make_plan(1, status="answered"),
        findings=[make_finding()],
        reviewed=True,
    )

    final = await run_to_completion(deps, state)

    assert [s.node for s in final.steps].count("budget") == 1


async def test_a_run_within_budget_is_untouched(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))

    assert final.status == "done"
    assert all(s.node != "budget" for s in final.steps)


async def test_the_budget_stop_refreshes_the_lease(deps):
    """The router does real work, so it must hold the lease open like any other node."""
    state = make_state(
        budgets=Budgets(max_tavily_credits=0),
        worker_id="worker-test",
        findings=[make_finding()],
        plan=make_plan(1, status="answered"),
        reviewed=True,
    )
    await deps.store.acquire_lease(state.run_id, "worker-test")

    final = await run_to_completion(deps, state)

    assert final.status == "budget_exceeded"
    assert await deps.store.lease_holder(state.run_id) == "worker-test"


async def test_status_stays_running_until_the_writer_has_finished(deps):
    """A client polling on status must never see a terminal run with no report behind it."""
    state = make_state(
        budgets=Budgets(max_tavily_credits=1),
        plan=make_plan(1, status="answered"),
        findings=[make_finding()],
        reviewed=True,
        tavily_credits=5,
    )

    verdict = check_budgets(state, next_node="write")
    mid_flight = apply_budget_stop(state, verdict)

    assert mid_flight.status == "running"
    assert mid_flight.report is None

    final = await run_to_completion(deps, state)

    assert final.status == "budget_exceeded"
    assert final.report is not None
    assert final.finished_at is not None
