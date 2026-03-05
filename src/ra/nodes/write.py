"""The writer: findings in, a cited report out.

The model cites finding ids, never URLs. That turns citation checking into set membership,
which is a thing a computer can be certain about, and it means a claim can only point at
something the researcher actually read.

A report is rendered only after its citations validate. An invented id earns one retry with
the offending ids named. A second failure fails the run, loudly, in the trace.
"""

import logging

from pydantic import BaseModel, Field

from ra.clock import now
from ra.deps import Deps
from ra.errors import safe_detail
from ra.llm import add_usage
from ra.render import render_markdown
from ra.schemas import Claim, Finding, NodeOutcome, Report, RunState, Section

log = logging.getLogger("ra.nodes.write")

MAX_SECTIONS = 6
MIN_SECTIONS = 3

WRITER_SYSTEM = (
    "You write a short research report from a list of findings.\n"
    "Every claim must cite at least one finding id from the list you are given, and you may "
    "not use any id that is not on it.\n"
    "Prefer findings marked full over findings marked snippet_only.\n"
    "Do not introduce facts that no finding supports."
)

CUT_SHORT_NOTE = (
    "Research was cut short by a budget cap, so the findings are incomplete. "
    "Say so in a short closing note."
)


class ClaimDraft(BaseModel):
    text: str
    finding_ids: list[str] = Field(default_factory=list)


class SectionDraft(BaseModel):
    heading: str
    claims: list[ClaimDraft] = Field(default_factory=list)


class ReportDraft(BaseModel):
    title: str
    sections: list[SectionDraft] = Field(default_factory=list)


def validate_citations(draft: ReportDraft, findings: list[Finding]) -> list[str]:
    """Complaints about the draft's citations. Empty means it is safe to render."""
    known = {f.id for f in findings}
    unknown: list[str] = []
    uncited = 0

    for section in draft.sections:
        for claim in section.claims:
            if not claim.finding_ids:
                uncited += 1
                continue
            unknown += [fid for fid in claim.finding_ids if fid not in known]

    problems = []
    if unknown:
        problems.append(f"unknown finding ids: {', '.join(sorted(set(unknown)))}")
    if uncited:
        problems.append(f"{uncited} claim(s) cite nothing")
    return problems


async def write(state: RunState, deps: Deps) -> NodeOutcome:
    if not state.findings:
        return _failed(state, "nothing to report: the run produced no findings")

    prompt = _prompt(state)
    usage = None
    model = deps.settings.models.writer
    problems: list[str] = []

    for attempt in (1, 2):
        try:
            result = await deps.llm.structured(
                model=model, system=WRITER_SYSTEM, user=prompt, schema=ReportDraft
            )
        except Exception as exc:
            # Broad for the same reason as the researcher: a transport failure is not an
            # LLMError, and the usage already spent on a first attempt must be kept.
            return _failed(state, f"writer failed: {safe_detail(exc)}", usage=usage, model=model)

        usage = add_usage(usage, result.usage)
        problems = validate_citations(result.parsed, state.findings)
        if not problems:
            return _rendered(state, result.parsed, usage=usage, model=result.model, attempt=attempt)

        log.warning("run %s citation problems on attempt %d: %s", state.run_id, attempt, problems)
        prompt = f"{_prompt(state)}\n\nYour previous attempt was rejected. {'. '.join(problems)}."

    return _failed(
        state,
        f"citation validation failed: {'; '.join(problems)}",
        usage=usage,
        model=model,
    )


def _prompt(state: RunState) -> str:
    lines = [f"Question: {state.question}", "", "Findings:"]
    for f in state.findings:
        title = f.source_title or f.source_url
        lines.append(f"{f.id} [{f.extraction_status}] {f.claim} (source: {title})")
    lines += [
        "",
        f"Write {MIN_SECTIONS} to {MAX_SECTIONS} sections. Cite by finding id only.",
    ]
    if state.budget_stopped:
        lines.append(CUT_SHORT_NOTE)
    return "\n".join(lines)


def _rendered(state: RunState, draft: ReportDraft, *, usage, model, attempt: int) -> NodeOutcome:
    report = Report(
        title=draft.title.strip() or state.question,
        sections=[
            Section(
                heading=s.heading.strip(),
                claims=[Claim(text=c.text.strip(), finding_ids=c.finding_ids) for c in s.claims],
            )
            for s in draft.sections
        ],
        generated_at=now(),
    )
    return NodeOutcome(
        state=state.model_copy(
            update={
                "report": report,
                "report_markdown": render_markdown(
                    report, state.findings, partial=state.budget_stopped
                ),
                # A run cut short keeps saying so. It gets a report, not a clean bill of health.
                "status": "budget_exceeded" if state.budget_stopped else "done",
                "finished_at": now(),
            }
        ),
        model=model,
        usage=usage,
        note="citations fixed on retry" if attempt > 1 else None,
    )


def _failed(state: RunState, error: str, *, usage=None, model=None) -> NodeOutcome:
    """No report is better than a report whose citations do not resolve."""
    return NodeOutcome(
        state=state.model_copy(update={"status": "failed", "error": error, "finished_at": now()}),
        status="error",
        model=model if usage else None,
        usage=usage,
        error=error,
    )
