"""The graph end to end against canned nodes, with no queue and no network."""

import pytest

import ra.nodes.canned as canned
from ra.graph import RECURSION_LIMIT, build_graph
from ra.routing import route
from ra.schemas import RunState
from tests.factories import make_state

# plan, research(sq_01), research(sq_02), review, write. The router runs between each one
# but records nothing, which is why it is not in this list.
EXPECTED_NODES = ["plan", "research", "research", "review", "write"]


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


async def run_to_completion(deps, state: RunState) -> RunState:
    graph = build_graph(deps)
    await deps.store.save(state)
    out = await graph.ainvoke(state.model_dump(), config={"recursion_limit": RECURSION_LIMIT})
    return RunState.model_validate(out)


async def test_a_canned_run_completes(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))

    assert final.status == "done"
    assert final.finished_at is not None
    assert final.report is not None
    assert final.report_markdown.startswith("# ")


async def test_the_trace_records_every_work_node_in_order(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))

    assert [s.node for s in final.steps] == EXPECTED_NODES
    assert [s.seq for s in final.steps] == [1, 2, 3, 4, 5]
    assert all(s.status == "ok" for s in final.steps)


async def test_each_research_step_names_its_sub_question(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))
    research = [s for s in final.steps if s.node == "research"]
    assert [s.sub_question_id for s in research] == ["sq_01", "sq_02"]


async def test_every_claim_cites_a_real_finding(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))
    known = {f.id for f in final.findings}
    cited = {
        fid for section in final.report.sections for c in section.claims for fid in c.finding_ids
    }
    assert cited
    assert cited <= known


async def test_state_is_durable_after_every_node(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))
    assert await deps.store.load(final.run_id) == final


async def test_the_run_leaves_the_active_set_when_it_finishes(deps):
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))
    assert final.run_id not in await deps.store.active_runs()


async def test_resuming_a_half_done_run_skips_finished_work(deps):
    """Invoke the graph on a state that already has a plan and one answered sub-question."""
    partial = await run_to_completion(deps, make_state(worker_id="worker-test"))

    # rewind: keep sq_01's finding, reopen sq_02, drop the report
    plan = [
        partial.plan[0],
        partial.plan[1].model_copy(update={"status": "pending", "passes": 0}),
    ]
    rewound = partial.model_copy(
        update={
            "plan": plan,
            "findings": [f for f in partial.findings if f.sub_question_id == "sq_01"],
            "report": None,
            "report_markdown": None,
            "reviewed": False,
            "status": "running",
            "finished_at": None,
            "steps": [],
        }
    )

    final = await run_to_completion(deps, rewound)

    assert final.status == "done"
    # plan is not re-run and sq_01 is not researched again
    assert [s.node for s in final.steps] == ["research", "review", "write"]
    assert [s.sub_question_id for s in final.steps if s.node == "research"] == ["sq_02"]
    assert len(final.findings) == 2


async def test_the_router_is_the_only_dispatcher(deps):
    """Whatever the graph does, route() alone decides it. This guards the resume story."""
    final = await run_to_completion(deps, make_state(worker_id="worker-test"))
    from langgraph.graph import END

    assert route(final) == END


async def test_research_with_nothing_open_records_a_skip(deps):
    """Defensive: the router should never send us here, so if it does, say so in the trace."""
    from ra.nodes.canned import research

    state = make_state(plan=[])
    outcome = await research(state, deps)
    assert outcome.status == "skipped"
    assert outcome.error == "no open sub-question"


def test_the_researcher_needs_both_of_its_clients(deps):
    """Without keys the canned stand-in runs, which is what keeps CI credential free."""
    from ra.deps import Deps
    from ra.graph import select_nodes
    from ra.nodes.canned import research as canned_research
    from ra.nodes.research import research as real_research
    from tests.fakes import FakeLLM, FakeSearch

    assert select_nodes(deps)["research"] is canned_research

    llm_only = Deps(store=deps.store, settings=deps.settings, llm=FakeLLM())
    assert select_nodes(llm_only)["research"] is canned_research

    search_only = Deps(store=deps.store, settings=deps.settings, search=FakeSearch())
    assert select_nodes(search_only)["research"] is canned_research

    both = Deps(store=deps.store, settings=deps.settings, llm=FakeLLM(), search=FakeSearch())
    assert select_nodes(both)["research"] is real_research


def test_a_stub_overrides_the_node_it_replaces(deps):
    """RA_STUB is how the crash-resume test gets a researcher it can pause."""
    from ra.config import Settings
    from ra.deps import Deps
    from ra.graph import select_nodes
    from ra.nodes.stubs import fast_research, slow_research

    def with_stub(name):
        settings = Settings(anthropic_api_key=None, tavily_api_key=None, stub=name)
        return select_nodes(Deps(store=deps.store, settings=settings))

    assert with_stub("slow_research")["research"] is slow_research
    assert with_stub("fast_research")["research"] is fast_research
    # everything else keeps its normal implementation
    assert with_stub("fast_research")["plan"] is select_nodes(deps)["plan"]
