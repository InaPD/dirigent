"""The writer. A citation that does not resolve must never reach a reader."""

import pytest

from ra.deps import Deps
from ra.llm import LLMError
from ra.nodes.write import (
    ClaimDraft,
    ReportDraft,
    SectionDraft,
    validate_citations,
    write,
)
from tests.factories import make_finding, make_plan, make_state
from tests.fakes import FakeLLM, citing_writer

F1 = "f_aaa111"
F2 = "f_bbb222"


def draft(*claims: tuple[str, list[str]], title: str = "A Report") -> ReportDraft:
    return ReportDraft(
        title=title,
        sections=[
            SectionDraft(
                heading="Findings",
                claims=[ClaimDraft(text=t, finding_ids=ids) for t, ids in claims],
            )
        ],
    )


def build(deps: Deps, *responses) -> Deps:
    return Deps(
        store=deps.store, settings=deps.settings, llm=FakeLLM({ReportDraft: list(responses)})
    )


def a_state(**overrides):
    base = {
        "plan": make_plan(2, status="answered"),
        "findings": [make_finding(sq="sq_01", fid=F1), make_finding(sq="sq_02", fid=F2)],
        "reviewed": True,
    }
    return make_state(**(base | overrides))


# -- validate_citations --------------------------------------------------------


def test_a_clean_draft_has_no_complaints():
    findings = [make_finding(fid=F1)]
    assert validate_citations(draft(("A claim.", [F1])), findings) == []


def test_an_unknown_id_is_named():
    findings = [make_finding(fid=F1)]
    problems = validate_citations(draft(("A claim.", ["f_ffffff"])), findings)

    assert len(problems) == 1
    assert "f_ffffff" in problems[0]


def test_a_claim_citing_nothing_is_a_problem():
    findings = [make_finding(fid=F1)]
    problems = validate_citations(draft(("Unsupported.", [])), findings)

    assert "cite nothing" in problems[0]


def test_every_unknown_id_is_reported_once_and_sorted():
    findings = [make_finding(fid=F1)]
    bad = draft(("One.", ["f_zzzzzz", "f_yyyyyy"]), ("Two.", ["f_zzzzzz"]))

    problems = validate_citations(bad, findings)

    assert problems[0].count("f_zzzzzz") == 1
    assert problems[0].index("f_yyyyyy") < problems[0].index("f_zzzzzz")


# -- the writer ----------------------------------------------------------------


async def test_a_valid_report_is_rendered_and_the_run_finishes(deps):
    d = build(deps, draft(("Something true.", [F1, F2]), title="Durable agents"))

    outcome = await write(a_state(), d)

    assert outcome.status == "ok"
    assert outcome.state.status == "done"
    assert outcome.state.finished_at is not None
    assert outcome.state.report.title == "Durable agents"
    assert outcome.state.report_markdown.startswith("# Durable agents\n")
    assert "## Sources" in outcome.state.report_markdown


async def test_the_prompt_lists_findings_by_id_with_their_quality(deps):
    d = build(deps, citing_writer())
    state = a_state(
        findings=[make_finding(fid=F1, extraction_status="snippet_only")],
    )

    await write(state, d)

    prompt = d.llm.prompts_for(ReportDraft)[0]
    assert F1 in prompt
    assert "snippet_only" in prompt
    # the model cites ids, never URLs, which is what makes validation set membership
    assert "Cite by finding id only" in prompt


async def test_an_invented_id_earns_one_retry(deps):
    d = build(
        deps,
        draft(("Invented.", ["f_ffffff"])),
        draft(("Corrected.", [F1])),
    )

    outcome = await write(a_state(), d)

    assert outcome.status == "ok"
    assert outcome.state.status == "done"
    assert len(d.llm.calls) == 2
    assert outcome.note == "citations fixed on retry"


async def test_the_retry_names_the_offending_ids(deps):
    d = build(deps, draft(("Invented.", ["f_ffffff"])), draft(("Corrected.", [F1])))

    await write(a_state(), d)

    second = d.llm.prompts_for(ReportDraft)[1]
    assert "rejected" in second
    assert "f_ffffff" in second


async def test_a_second_failure_fails_the_run_rather_than_publishing(deps):
    """No report is better than a report whose citations do not resolve."""
    d = build(deps, draft(("Invented.", ["f_ffffff"])))

    outcome = await write(a_state(), d)

    assert outcome.status == "error"
    assert outcome.state.status == "failed"
    assert outcome.state.report is None
    assert outcome.state.report_markdown is None
    assert "citation validation failed" in outcome.state.error
    assert "f_ffffff" in outcome.state.error
    assert len(d.llm.calls) == 2


async def test_a_model_failure_fails_the_run(deps):
    d = build(deps, LLMError("max_tokens", "answer truncated"))

    outcome = await write(a_state(), d)

    assert outcome.state.status == "failed"
    assert "writer failed" in outcome.state.error
    assert outcome.state.report is None


async def test_a_run_with_no_findings_says_so(deps):
    d = build(deps, citing_writer())

    outcome = await write(a_state(findings=[]), d)

    assert outcome.state.status == "failed"
    assert "no findings" in outcome.state.error
    assert d.llm.calls == []  # nothing to write means nothing to spend


async def test_both_attempts_are_billed_to_the_step(deps):
    d = build(deps, draft(("Invented.", ["f_ffffff"])), draft(("Corrected.", [F1])))

    outcome = await write(a_state(), d)

    assert outcome.usage.input_tokens == 200
    assert outcome.usage.output_tokens == 40


# -- a run that was cut short --------------------------------------------------


async def test_a_cut_short_run_keeps_saying_so(deps):
    d = build(deps, draft(("Partial.", [F1])))

    outcome = await write(a_state(budget_stopped=True), d)

    assert outcome.state.status == "budget_exceeded"
    assert outcome.state.report is not None
    assert outcome.state.report_markdown.startswith("# A Report (partial)")


async def test_the_writer_is_told_the_research_was_cut_short(deps):
    d = build(deps, draft(("Partial.", [F1])))

    await write(a_state(budget_stopped=True), d)

    assert "cut short by a budget cap" in d.llm.prompts_for(ReportDraft)[0]


async def test_a_complete_run_is_not_labelled_partial(deps):
    d = build(deps, draft(("Complete.", [F1])))

    outcome = await write(a_state(), d)

    assert "(partial)" not in outcome.state.report_markdown


# -- rendering happens last ----------------------------------------------------


@pytest.mark.parametrize("bad_ids", [["f_ffffff"], []], ids=["unknown id", "no id"])
async def test_nothing_is_rendered_until_the_citations_validate(deps, bad_ids):
    d = build(deps, draft(("Bad.", bad_ids)))

    outcome = await write(a_state(), d)

    assert outcome.state.report_markdown is None
