"""The researcher: one sub-question per invocation, chosen by the router.

Everything this node touches can fail, and none of it raises. A search that returns nothing,
a page behind a paywall, a rate limit: each becomes a status the supervisor can read.

The node saves once, at the end, with the sub-question's new status and its findings in the
same write. That is the atomicity crash-resume depends on. Either the whole sub-question
landed or none of it did, so a worker that dies mid-node leaves no half-answered question.
"""

import logging

from pydantic import BaseModel, Field

from ra.clock import now
from ra.deps import Deps
from ra.ids import new_finding_id
from ra.llm import LLMError, add_usage
from ra.routing import next_open_subquestion
from ra.schemas import Finding, NodeOutcome, RunState, ToolCall
from ra.search import Hit, resolve_extraction

log = logging.getLogger("ra.nodes.research")

# Enough of a page for a claim to be supported, small enough to keep the prompt honest.
MAX_CONTENT_CHARS = 6_000
MAX_SNIPPET_CHARS = 400
MAX_FINDINGS_PER_SQ = 5

# Below this many results, one reformulated query is worth a credit.
THIN_RESULTS = 3

EXTRACTOR_SYSTEM = (
    "You turn source material into findings that answer one specific question.\n"
    "Every claim must be supported by the text of the source you cite, and the snippet you "
    "return must be quoted from that source.\n"
    "Ignore sources that do not help. Returning nothing is better than returning a claim "
    "the source does not support."
)

REPHRASER_SYSTEM = (
    "You rewrite a failed web search query. Return one alternative query that uses "
    "different wording and is more likely to match indexed pages. No commentary."
)


class FindingDraft(BaseModel):
    claim: str
    source_url: str
    snippet: str = ""


class FindingDrafts(BaseModel):
    findings: list[FindingDraft] = Field(default_factory=list)


class Rephrased(BaseModel):
    query: str


async def research(state: RunState, deps: Deps) -> NodeOutcome:
    sq = next_open_subquestion(state)
    if sq is None:
        return NodeOutcome(state=state, status="skipped", error="no open sub-question")

    budgets = state.budgets
    tool_calls: list[ToolCall] = []
    notes: list[str] = []

    hits, search_usage = await _gather_hits(sq.text, state, deps, tool_calls, notes)
    seen_urls = {f.source_url for f in state.findings}
    fresh = [h for h in hits if h.url not in seen_urls][: budgets.max_extracts_per_sq]

    if not fresh:
        return _settled(
            state,
            sq,
            findings=[],
            tool_calls=tool_calls,
            usage=search_usage,
            model=deps.settings.models.researcher,
            status="skipped" if not hits else "ok",
            note="; ".join(notes) or None,
            error="no new sources" if hits else "no search results",
        )

    batch = await deps.search.extract([h.url for h in fresh])
    tool_calls.extend(batch.tool_calls)
    extracted = batch.by_url()

    sources = [(hit, *resolve_extraction(extracted.get(hit.url), hit)) for hit in fresh]
    usable = [(hit, status, content) for hit, status, content in sources if content]
    if not usable:
        return _settled(
            state,
            sq,
            findings=[],
            tool_calls=tool_calls,
            usage=search_usage,
            model=deps.settings.models.researcher,
            status="ok",
            note="; ".join(notes) or None,
            error=f"nothing extractable from {len(sources)} sources",
        )

    truncated = sum(1 for _, _, content in usable if len(content) > MAX_CONTENT_CHARS)
    if truncated:
        notes.append(f"truncated:{truncated}")

    try:
        result = await deps.llm.structured(
            model=deps.settings.models.researcher,
            system=EXTRACTOR_SYSTEM,
            user=_extraction_prompt(sq.text, usable),
            schema=FindingDrafts,
        )
    except LLMError as exc:
        return _settled(
            state,
            sq,
            findings=[],
            tool_calls=tool_calls,
            usage=search_usage,
            model=deps.settings.models.researcher,
            status="error",
            note="; ".join(notes) or None,
            error=str(exc),
        )

    quality = {hit.url: (status, hit) for hit, status, _ in usable}
    findings, dropped = _to_findings(result.parsed.findings, quality, sq.id)
    if dropped:
        notes.append(f"dropped:{dropped}")

    return _settled(
        state,
        sq,
        findings=findings,
        tool_calls=tool_calls,
        usage=add_usage(search_usage, result.usage),
        model=result.model,
        status="ok",
        note="; ".join(notes) or None,
    )


async def _gather_hits(question, state, deps, tool_calls, notes):
    """Search, and rewrite the query once if the first attempt comes back thin."""
    usage = None
    hits: list[Hit] = []
    searches_left = state.budgets.max_searches_per_sq
    query = question

    while searches_left > 0:
        outcome = await deps.search.search(query, max_results=state.budgets.max_extracts_per_sq)
        searches_left -= 1
        tool_calls.append(outcome.tool_call)
        hits = _merge_hits(hits, outcome.hits)

        if outcome.status not in ("ok", "empty"):
            notes.append(f"search {outcome.status}")
            break
        if len(hits) >= THIN_RESULTS or searches_left <= 0:
            break

        try:
            rewrite = await deps.llm.structured(
                model=deps.settings.models.researcher,
                system=REPHRASER_SYSTEM,
                user=f"This query returned {len(hits)} results: {query}",
                schema=Rephrased,
            )
        except LLMError:
            break
        usage = add_usage(usage, rewrite.usage)
        query = rewrite.parsed.query.strip()
        notes.append("reworded the query")
        if not query:
            break

    return hits, usage


def _merge_hits(existing: list[Hit], new: list[Hit]) -> list[Hit]:
    seen = {h.url for h in existing}
    return [*existing, *(h for h in new if h.url not in seen)]


def _extraction_prompt(question: str, usable: list[tuple[Hit, str, str]]) -> str:
    blocks = [f"Question: {question}", ""]
    for hit, status, content in usable:
        blocks += [
            f"SOURCE {hit.url}",
            f"title: {hit.title or 'unknown'}",
            f"quality: {status}",
            content[:MAX_CONTENT_CHARS],
            "",
        ]
    blocks.append(
        f"Return at most {MAX_FINDINGS_PER_SQ} findings. Use only the source URLs listed "
        f"above, copied exactly. Keep each snippet under {MAX_SNIPPET_CHARS} characters."
    )
    return "\n".join(blocks)


def _to_findings(drafts, quality: dict, sub_question_id: str) -> tuple[list[Finding], int]:
    """Keep the drafts that cite a source we actually read. Count the rest."""
    findings: list[Finding] = []
    dropped = 0

    for draft in drafts[:MAX_FINDINGS_PER_SQ]:
        entry = quality.get(draft.source_url)
        if entry is None:
            # The model invented a URL. It does not get to become provenance.
            dropped += 1
            continue
        status, hit = entry
        findings.append(
            Finding(
                id=new_finding_id(),
                sub_question_id=sub_question_id,
                claim=draft.claim.strip(),
                source_url=draft.source_url,
                source_title=hit.title,
                snippet=draft.snippet.strip()[:MAX_SNIPPET_CHARS],
                retrieved_at=now(),
                extraction_status=status,
            )
        )
    return findings, dropped


def _settled(state, sq, *, findings, tool_calls, usage, model, status, note=None, error=None):
    """One save: the sub-question's new status and its findings, together."""
    answered = sq.model_copy(
        update={
            "status": "answered" if findings else "unanswerable",
            "passes": sq.passes + 1,
        }
    )
    credits = sum(call.credits for call in tool_calls)
    new_state = state.model_copy(
        update={
            "plan": [answered if item.id == sq.id else item for item in state.plan],
            "findings": [*state.findings, *findings],
            "tavily_credits": state.tavily_credits + credits,
            "reviewed": False,
        }
    )
    return NodeOutcome(
        state=new_state,
        status=status,
        model=model if usage else None,
        usage=usage,
        tool_calls=tool_calls,
        sub_question_id=sq.id,
        note=note,
        error=error,
    )
