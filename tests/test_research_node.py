"""The researcher. Provenance must be real, and nothing may take the run down with it."""

import pytest

from ra.deps import Deps
from ra.llm import LLMError
from ra.nodes.research import FindingDraft, FindingDrafts, Rephrased, research
from ra.schemas import Budgets
from tests.factories import make_finding, make_plan, make_state
from tests.fakes import FakeLLM, FakeSearch, extract_batch, search_outcome

A = "https://a.test"
B = "https://b.test"
C = "https://c.test"


def drafts(*pairs: tuple[str, str]) -> FindingDrafts:
    return FindingDrafts(
        findings=[
            FindingDraft(claim=claim, source_url=url, snippet="quoted from the source")
            for claim, url in pairs
        ]
    )


def build(deps_base: Deps, *, llm=None, searches=None, batches=None) -> Deps:
    return Deps(
        store=deps_base.store,
        settings=deps_base.settings,
        llm=llm or FakeLLM({FindingDrafts: [drafts(("A claim.", A))]}),
        search=FakeSearch(searches=searches, batches=batches),
    )


def a_state(**overrides):
    base = {"plan": make_plan(2), "budgets": Budgets(max_extracts_per_sq=5)}
    return make_state(**(base | overrides))


# -- the happy path ------------------------------------------------------------


async def test_findings_carry_their_provenance(deps):
    d = build(
        deps,
        searches=[search_outcome(A, B)],
        batches=[extract_batch({A: "full", B: "full"})],
    )

    outcome = await research(a_state(), d)
    findings = outcome.state.findings

    assert outcome.status == "ok"
    assert len(findings) == 1
    assert findings[0].source_url == A
    assert findings[0].sub_question_id == "sq_01"
    assert findings[0].extraction_status == "full"
    assert findings[0].source_title == f"Title {A}"
    assert findings[0].id.startswith("f_")
    assert findings[0].retrieved_at is not None


async def test_the_sub_question_is_answered_and_its_pass_counted(deps):
    d = build(deps, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)
    sq = outcome.state.plan[0]

    assert sq.status == "answered"
    assert sq.passes == 1
    assert outcome.state.plan[1].status == "pending"  # the router picks it up next
    assert outcome.sub_question_id == "sq_01"


async def test_a_new_finding_reopens_the_review(deps):
    d = build(deps, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(reviewed=True), d)

    assert outcome.state.reviewed is False


async def test_credits_are_added_to_the_run(deps):
    d = build(
        deps,
        searches=[search_outcome(A, credits=2)],
        batches=[extract_batch({A: "full"}, credits=3)],
    )

    outcome = await research(a_state(tavily_credits=4), d)

    assert outcome.state.tavily_credits == 9
    assert sum(c.credits for c in outcome.tool_calls) == 5


async def test_the_step_reports_its_token_usage(deps):
    d = build(deps, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)

    assert outcome.usage.input_tokens == 100
    assert outcome.model == "claude-sonnet-5"


# -- provenance is not negotiable ----------------------------------------------


async def test_an_invented_url_is_dropped(deps):
    """The model citing a source it was never shown does not get to become provenance."""
    llm = FakeLLM({FindingDrafts: [drafts(("Real.", A), ("Invented.", "https://made-up.test"))]})
    d = build(deps, llm=llm, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)

    assert [f.source_url for f in outcome.state.findings] == [A]
    assert "dropped:1" in outcome.note


async def test_every_finding_points_at_a_source_that_was_read(deps):
    llm = FakeLLM({FindingDrafts: [drafts(("One.", A), ("Two.", B), ("Three.", C))]})
    d = build(
        deps,
        llm=llm,
        searches=[search_outcome(A, B)],
        batches=[extract_batch({A: "full", B: "full"})],
    )

    outcome = await research(a_state(), d)

    assert {f.source_url for f in outcome.state.findings} <= {A, B}


async def test_a_url_already_cited_is_not_researched_again(deps):
    d = build(
        deps,
        searches=[search_outcome(A, B)],
        batches=[extract_batch({B: "full"})],
    )
    state = a_state(findings=[make_finding(sq="sq_01", fid="f_old111", source_url=A)])

    await research(state, d)

    assert d.search.extracted == [[B]]


async def test_a_weaker_source_is_labelled_not_discarded(deps):
    """A JS-heavy page still contributes; the writer is told to prefer full ones."""
    llm = FakeLLM({FindingDrafts: [drafts(("From a snippet.", A))]})
    d = build(
        deps,
        llm=llm,
        searches=[search_outcome(A)],
        batches=[extract_batch({A: "paywalled"})],
    )

    outcome = await research(a_state(), d)

    assert outcome.state.findings[0].extraction_status == "snippet_only"


async def test_the_snippet_is_capped(deps):
    llm = FakeLLM(
        {
            FindingDrafts: [
                FindingDrafts(findings=[FindingDraft(claim="c", source_url=A, snippet="y" * 5000)])
            ]
        }
    )
    d = build(deps, llm=llm, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)

    assert len(outcome.state.findings[0].snippet) == 400


# -- failure never takes the run down ------------------------------------------


async def test_no_search_results_makes_the_sub_question_unanswerable(deps):
    d = build(deps, searches=[search_outcome(status="ok")])

    outcome = await research(a_state(), d)

    assert outcome.state.plan[0].status == "unanswerable"
    assert outcome.state.plan[0].passes == 1
    assert outcome.state.findings == []
    assert outcome.status == "skipped"


@pytest.mark.parametrize("failure", ["timeout", "rate_limited", "error"])
async def test_a_failed_search_is_recorded_and_the_run_continues(deps, failure):
    d = build(deps, searches=[search_outcome(status=failure)])

    outcome = await research(a_state(), d)

    assert outcome.state.plan[0].status == "unanswerable"
    assert failure in outcome.note
    assert outcome.state.status == "running"


async def test_nothing_extractable_is_recorded_and_the_run_continues(deps):
    d = build(
        deps,
        searches=[search_outcome(A, snippet="")],
        batches=[extract_batch({A: "timeout"})],
    )

    outcome = await research(a_state(), d)

    assert outcome.state.findings == []
    assert outcome.state.plan[0].status == "unanswerable"
    assert "nothing extractable" in outcome.error


async def test_a_model_failure_is_an_error_step_not_a_crash(deps):
    llm = FakeLLM({FindingDrafts: [LLMError("max_tokens", "truncated")]})
    d = build(deps, llm=llm, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)

    assert outcome.status == "error"
    assert "max_tokens" in outcome.error
    assert outcome.state.plan[0].status == "unanswerable"
    assert outcome.state.plan[0].passes == 1  # the attempt still counts


async def test_the_node_refuses_to_run_with_nothing_open(deps):
    d = build(deps)

    outcome = await research(make_state(plan=make_plan(1, status="answered")), d)

    assert outcome.status == "skipped"
    assert outcome.error == "no open sub-question"


# -- caps ----------------------------------------------------------------------


async def test_a_thin_result_set_earns_one_reworded_query(deps):
    llm = FakeLLM(
        {
            Rephrased: [Rephrased(query="a better query")],
            FindingDrafts: [drafts(("A claim.", A))],
        }
    )
    d = build(
        deps,
        llm=llm,
        searches=[search_outcome(status="ok"), search_outcome(A, B, C)],
        batches=[extract_batch({A: "full", B: "full", C: "full"})],
    )

    outcome = await research(a_state(budgets=Budgets(max_searches_per_sq=3)), d)

    assert d.search.queries[1] == "a better query"
    assert "reworded the query" in outcome.note


async def test_searches_never_exceed_the_per_question_cap(deps):
    llm = FakeLLM(
        {Rephrased: [Rephrased(query="another")], FindingDrafts: [drafts(("A claim.", A))]}
    )
    d = build(deps, llm=llm, searches=[search_outcome(status="ok")])

    await research(a_state(budgets=Budgets(max_searches_per_sq=2)), d)

    assert len(d.search.queries) == 2


async def test_extractions_never_exceed_the_per_question_cap(deps):
    d = build(
        deps,
        searches=[search_outcome(A, B, C)],
        batches=[extract_batch({A: "full"})],
    )

    await research(a_state(budgets=Budgets(max_extracts_per_sq=2)), d)

    assert len(d.search.extracted[0]) == 2


# -- atomicity -----------------------------------------------------------------


async def test_the_sub_question_status_and_its_findings_land_together(deps):
    """What crash-resume depends on: either the whole sub-question landed, or none of it."""
    d = build(deps, searches=[search_outcome(A)], batches=[extract_batch({A: "full"})])

    outcome = await research(a_state(), d)
    answered = [sq for sq in outcome.state.plan if sq.status == "answered"]

    assert len(answered) == 1
    assert {f.sub_question_id for f in outcome.state.findings} == {answered[0].id}


async def test_an_oversized_page_is_truncated_and_the_trace_says_so(deps):
    """Long pages are cut before prompting. That is recorded, because it is not a failure."""
    llm = FakeLLM({FindingDrafts: [drafts(("A claim.", A))]})
    d = build(
        deps,
        llm=llm,
        searches=[search_outcome(A)],
        batches=[extract_batch({A: "full"}, content="z" * 20_000)],
    )

    outcome = await research(a_state(), d)

    assert "truncated:1" in outcome.note
    assert len(llm.prompts_for(FindingDrafts)[0]) < 20_000


async def test_an_empty_reworded_query_ends_the_search(deps):
    llm = FakeLLM({Rephrased: [Rephrased(query="   ")], FindingDrafts: [drafts(("c", A))]})
    d = build(deps, llm=llm, searches=[search_outcome(status="ok")])

    await research(a_state(budgets=Budgets(max_searches_per_sq=3)), d)

    assert len(d.search.queries) == 1


async def test_a_transport_failure_keeps_the_credits_already_spent(deps):
    """The tracing wrapper rebuilds a failed node from the state as it was before the node.

    So a raw SDK exception escaping this node would take the Tavily spend with it, and the
    run's own accounting would under-report what it actually cost.
    """

    class TransportFailure(FakeLLM):
        async def structured(self, **kwargs):
            raise ConnectionError("the connection dropped")

    d = build(
        deps,
        llm=TransportFailure(),
        searches=[search_outcome(A, credits=2)],
        batches=[extract_batch({A: "full"}, credits=3)],
    )

    outcome = await research(a_state(tavily_credits=1), d)

    assert outcome.status == "error"
    assert "ConnectionError" in outcome.error
    assert outcome.state.tavily_credits == 6  # 1 already on the run, plus 2 and 3 just spent
    assert sum(c.credits for c in outcome.tool_calls) == 5
    assert outcome.state.plan[0].status == "unanswerable"


async def test_the_search_loop_stops_when_the_run_is_out_of_credits(deps):
    """check_budgets only runs between nodes, so the loop inside one watches its own spend."""
    llm = FakeLLM({Rephrased: [Rephrased(query="another go")], FindingDrafts: [drafts(("c", A))]})
    d = build(
        deps,
        llm=llm,
        searches=[search_outcome(status="ok", credits=10)],  # empty results, so it would retry
    )
    state = a_state(budgets=Budgets(max_searches_per_sq=5, max_tavily_credits=12), tavily_credits=0)

    outcome = await research(state, d)

    assert len(d.search.queries) == 2  # the second search blew the cap, so no third
    assert "out of budget" in outcome.note
