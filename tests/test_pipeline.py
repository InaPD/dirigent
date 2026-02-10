"""Phase 4's exit criterion: a question goes in, a cited report comes out.

Every node is the real one. Only the two network clients are faked, so the planner, the
researcher, the writer, the router, the budgets, the tracing and the store are all doing
their actual jobs. The live variant at the bottom does the same with real keys.
"""

import os

import pytest

import ra.nodes.canned as canned
from ra.deps import Deps
from ra.graph import RECURSION_LIMIT, build_graph
from ra.nodes.plan import PlanDraft
from ra.nodes.research import FindingDrafts
from ra.nodes.review import Review
from ra.nodes.write import ReportDraft
from ra.schemas import Budgets, RunState
from tests.fakes import (
    FakeLLM,
    FakeSearch,
    citing_researcher,
    citing_writer,
    extract_batch,
    reviewing,
    search_outcome,
)

QUESTION = "What is the current state of durable agent execution?"
# A realistic search: different sub-questions surface different sources.
URLS_BY_SQ = [["https://one.test"], ["https://two.test"]]
URLS = [u for group in URLS_BY_SQ for u in group]
SUB_QUESTIONS = ["What does durable execution mean?", "Which systems implement it?"]


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(canned, "NODE_DELAY_S", 0)


def a_run(**overrides) -> RunState:
    base = {
        "run_id": "run_pipeline01",
        "question": QUESTION,
        "status": "running",
        "worker_id": "worker-test",
        "budgets": Budgets(max_subquestions=2, max_extracts_per_sq=2),
    }
    return RunState(**(base | overrides))


def pipeline_deps(deps, *, writer=None) -> Deps:
    llm = FakeLLM(
        {
            PlanDraft: [PlanDraft(sub_questions=SUB_QUESTIONS)],
            FindingDrafts: [citing_researcher()],
            Review: [reviewing()],
            ReportDraft: [writer or citing_writer(title="Durable agent execution")],
        }
    )
    search = FakeSearch(
        searches=[search_outcome(*group) for group in URLS_BY_SQ],
        batches=[extract_batch(dict.fromkeys(group, "full")) for group in URLS_BY_SQ],
    )
    return Deps(store=deps.store, settings=deps.settings, llm=llm, search=search)


async def run_graph(deps: Deps, state: RunState) -> RunState:
    await deps.store.save(state)
    out = await build_graph(deps).ainvoke(
        state.model_dump(), config={"recursion_limit": RECURSION_LIMIT}
    )
    return RunState.model_validate(out)


async def test_a_question_becomes_a_cited_report(deps):
    final = await run_graph(pipeline_deps(deps), a_run())

    assert final.status == "done"
    assert final.finished_at is not None
    assert final.error is None
    assert final.report is not None
    assert final.report_markdown.startswith("# Durable agent execution\n")
    assert "## Sources" in final.report_markdown


async def test_every_node_ran_in_order(deps):
    final = await run_graph(pipeline_deps(deps), a_run())

    assert [s.node for s in final.steps] == [
        "plan",
        "research",
        "research",
        "review",
        "write",
    ]
    assert all(s.status == "ok" for s in final.steps)


async def test_the_plan_came_from_the_planner(deps):
    final = await run_graph(pipeline_deps(deps), a_run())

    assert [sq.text for sq in final.plan] == SUB_QUESTIONS
    assert all(sq.status == "answered" for sq in final.plan)


async def test_every_citation_resolves_to_a_real_finding(deps):
    """The project's headline promise, checked on a whole run."""
    final = await run_graph(pipeline_deps(deps), a_run())

    known = {f.id for f in final.findings}
    cited = {
        fid for section in final.report.sections for c in section.claims for fid in c.finding_ids
    }

    assert cited
    assert cited <= known


async def test_every_footnote_in_the_markdown_has_a_source(deps):
    final = await run_graph(pipeline_deps(deps), a_run())
    body, sources = final.report_markdown.split("## Sources")

    for n in range(1, len(final.findings) + 1):
        if f"[{n}]" in body:
            assert f"[{n}] " in sources


async def test_the_run_is_priced_and_metered(deps):
    final = await run_graph(pipeline_deps(deps), a_run())

    assert final.cost_usd > 0
    assert final.cost_usd == pytest.approx(sum(s.cost_usd for s in final.steps))
    assert final.tokens_in == sum(s.tokens_in for s in final.steps)
    assert final.tavily_credits > 0


async def test_an_uncorrectable_citation_fails_the_run_with_no_report(deps):
    """Two invented ids in a row, and the run fails rather than publishing bad provenance."""
    d = pipeline_deps(deps, writer=citing_writer(invent="f_ffffff"))

    final = await run_graph(d, a_run())

    assert final.status == "failed"
    assert final.report is None
    assert final.report_markdown is None
    assert "citation validation failed" in final.error
    assert [s.node for s in final.steps][-1] == "write"


async def test_a_run_that_cannot_be_planned_stops_at_the_planner(deps):
    d = pipeline_deps(deps)
    d.llm.responses[PlanDraft] = [PlanDraft(sub_questions=[])]

    final = await run_graph(d, a_run())

    assert final.status == "failed"
    assert [s.node for s in final.steps] == ["plan"]
    assert "no sub-questions" in final.error


@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="needs real ANTHROPIC_API_KEY and TAVILY_API_KEY",
)
async def test_live_question_to_cited_report(deps, capsys):
    """The real thing. Run with `pytest -m live -s` and read the report it prints."""
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
        final = await run_graph(live, a_run(budgets=Budgets(max_subquestions=3)))
    finally:
        await search.aclose()

    assert final.status == "done", final.error
    assert final.report_markdown

    known = {f.id for f in final.findings}
    cited = {
        fid for section in final.report.sections for c in section.claims for fid in c.finding_ids
    }
    assert cited <= known, "a citation did not resolve to a finding"

    with capsys.disabled():
        print(f"\ncost ${final.cost_usd:.4f}, {final.tavily_credits} credits")
        print(final.report_markdown)


async def test_a_sub_question_whose_sources_were_all_used_is_unanswerable(deps):
    """URL dedupe spans the whole run, so a second question offering nothing new says so."""
    d = pipeline_deps(deps)
    d.search.searches = [search_outcome(*URLS_BY_SQ[0])]  # every query returns the same source

    final = await run_graph(d, a_run())

    assert [sq.status for sq in final.plan] == ["answered", "unanswerable"]
    assert len({f.source_url for f in final.findings}) == len(final.findings)
    assert final.status == "done"  # one answered question is still a report
