"""Phase 3's exit criterion, run through the real graph.

A three sub-question plan goes in, real Finding records with provenance come out. The only
fakes are the two network clients; the node, the router, the tracing and the store are real.

The live variant of the same thing is at the bottom, skipped unless keys are present.
"""

import os

import pytest

import ra.nodes.canned as canned
from ra.deps import Deps
from ra.graph import RECURSION_LIMIT, build_graph
from ra.ids import sq_id
from ra.nodes.research import FindingDraft, FindingDrafts
from ra.nodes.review import Review
from ra.nodes.write import ReportDraft
from ra.schemas import Budgets, RunState, SubQuestion
from tests.fakes import (
    FakeLLM,
    FakeSearch,
    citing_writer,
    extract_batch,
    reviewing,
    search_outcome,
)

PLAN = [
    "What does durable execution mean for an agent framework?",
    "Which systems implement it today?",
    "What are the open problems?",
]
URLS = ["https://one.test", "https://two.test", "https://three.test"]


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


@pytest.fixture
def researching_deps(deps) -> Deps:
    """Every search returns the same three sources; the model cites one per sub-question."""
    llm = FakeLLM(
        {
            FindingDrafts: [
                FindingDrafts(
                    findings=[
                        FindingDraft(
                            claim=f"Something true about {url}.",
                            source_url=url,
                            snippet="quoted from the page",
                        )
                    ]
                )
                for url in URLS
            ],
            Review: [reviewing()],
            ReportDraft: [citing_writer(title="Durable agent execution")],
        }
    )
    search = FakeSearch(
        searches=[search_outcome(*URLS)],
        batches=[extract_batch(dict.fromkeys(URLS, "full"))],
    )
    return Deps(store=deps.store, settings=deps.settings, llm=llm, search=search)


async def run_graph(deps: Deps, state: RunState) -> RunState:
    await deps.store.save(state)
    out = await build_graph(deps).ainvoke(
        state.model_dump(), config={"recursion_limit": RECURSION_LIMIT}
    )
    return RunState.model_validate(out)


@pytest.fixture
def planned_state():
    return RunState(
        run_id="run_phase3",
        question="What is the current state of durable agent execution?",
        status="running",
        worker_id="worker-test",
        budgets=Budgets(max_subquestions=3, max_extracts_per_sq=3),
        plan=[SubQuestion(id=sq_id(i), text=t) for i, t in enumerate(PLAN, start=1)],
    )


async def test_a_three_question_plan_produces_findings_with_provenance(
    researching_deps, planned_state
):
    final = await run_graph(researching_deps, planned_state)

    assert final.status == "done"
    assert len(final.findings) == 3
    assert {f.sub_question_id for f in final.findings} == {"sq_01", "sq_02", "sq_03"}
    for finding in final.findings:
        assert finding.source_url in URLS
        assert finding.source_title
        assert finding.snippet
        assert finding.extraction_status == "full"
        assert finding.retrieved_at is not None


async def test_the_researcher_runs_once_per_sub_question(researching_deps, planned_state):
    final = await run_graph(researching_deps, planned_state)
    research_steps = [s for s in final.steps if s.node == "research"]

    assert [s.sub_question_id for s in research_steps] == ["sq_01", "sq_02", "sq_03"]
    assert all(s.status == "ok" for s in research_steps)


async def test_the_trace_records_the_search_calls_and_their_credits(
    researching_deps, planned_state
):
    final = await run_graph(researching_deps, planned_state)
    research_steps = [s for s in final.steps if s.node == "research"]

    tools = {c.tool for s in research_steps for c in s.tool_calls}
    assert tools == {"tavily.search", "tavily.extract"}
    assert final.tavily_credits == sum(c.credits for s in research_steps for c in s.tool_calls)
    assert final.tavily_credits > 0


async def test_the_trace_prices_the_model_calls(researching_deps, planned_state):
    final = await run_graph(researching_deps, planned_state)
    research_steps = [s for s in final.steps if s.node == "research"]

    assert all(s.model == "claude-sonnet-5" for s in research_steps)
    assert all(s.tokens_in > 0 for s in research_steps)
    assert final.cost_usd > 0
    assert final.cost_usd == pytest.approx(sum(s.cost_usd for s in final.steps))


async def test_every_citation_in_the_report_resolves_to_a_finding(researching_deps, planned_state):
    final = await run_graph(researching_deps, planned_state)

    known = {f.id for f in final.findings}
    cited = {
        fid for section in final.report.sections for c in section.claims for fid in c.finding_ids
    }
    assert cited
    assert cited <= known


async def test_the_credit_cap_stops_the_researcher_mid_plan(researching_deps, planned_state):
    """Each sub-question costs credits, so a low cap has to bite before the plan is done."""
    state = planned_state.model_copy(
        update={"budgets": planned_state.budgets.model_copy(update={"max_tavily_credits": 3})}
    )

    final = await run_graph(researching_deps, state)

    assert final.status == "budget_exceeded"
    assert final.report is not None  # the writer reserve still bought a report
    assert len(final.findings) < 3


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="needs real ANTHROPIC_API_KEY and TAVILY_API_KEY",
)
async def test_live_three_question_plan(deps, planned_state, capsys):
    """The real thing. Run it by hand with `pytest -m live -s` and read the findings."""
    from ra.config import Settings
    from ra.llm import LLM
    from ra.search import Tavily

    settings = Settings()
    search = Tavily(settings.require_tavily_key(), deps.store)
    live = Deps(
        store=deps.store,
        settings=settings,
        llm=LLM.from_api_key(settings.require_anthropic_key()),
        search=search,
    )
    try:
        final = await run_graph(live, planned_state)
    finally:
        await search.aclose()

    assert final.findings, "no findings came back from a real search"
    with capsys.disabled():
        print(f"\ncost ${final.cost_usd:.4f}, {final.tavily_credits} credits")
        for f in final.findings:
            print(f"  [{f.extraction_status}] {f.claim[:90]}")
            print(f"      {f.source_url}")
