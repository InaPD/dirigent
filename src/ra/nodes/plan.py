"""The planner: one question in, a handful of searchable sub-questions out.

The prompt asks for exactly what the researcher needs and nothing else. No persona, no
instruction to be world class. The model is being asked to split a question, and the useful
constraints are that each part can be searched on its own and that the parts do not overlap.
"""

import logging

from pydantic import BaseModel, Field

from ra.clock import now
from ra.deps import Deps
from ra.errors import safe_detail
from ra.ids import sq_id
from ra.llm import LLMError
from ra.schemas import NodeOutcome, RunState, SubQuestion

log = logging.getLogger("ra.nodes.plan")

PLANNER_SYSTEM = (
    "You split a research question into sub-questions.\n"
    "Each sub-question must be answerable on its own from a web search, and must not "
    "overlap with the others.\n"
    "Prefer fewer, broader sub-questions over many narrow ones. Return the sub-questions "
    "only, with no numbering and no commentary."
)


class PlanDraft(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)


async def plan(state: RunState, deps: Deps) -> NodeOutcome:
    limit = state.budgets.max_subquestions

    try:
        result = await deps.llm.structured(
            model=deps.settings.models.planner,
            system=PLANNER_SYSTEM,
            user=(f"Question: {state.question}\n\nReturn between 2 and {limit} sub-questions."),
            schema=PlanDraft,
        )
    except LLMError as exc:
        return _unplannable(state, f"planner failed: {safe_detail(exc)}")
    except Exception as exc:
        # Deliberately broad. Without a plan the router sends the run straight back here, so
        # anything at all that stops a plan being made has to stop the run too.
        return _unplannable(state, f"planner failed: {safe_detail(exc)}")

    texts = [t.strip() for t in result.parsed.sub_questions if t and t.strip()][:limit]
    if not texts:
        return _unplannable(
            state,
            "planner returned no sub-questions",
            model=result.model,
            usage=result.usage,
        )

    plan = [SubQuestion(id=sq_id(i), text=text) for i, text in enumerate(texts, start=1)]
    dropped = len(result.parsed.sub_questions) - len(texts)

    return NodeOutcome(
        state=state.model_copy(update={"plan": plan}),
        model=result.model,
        usage=result.usage,
        note=f"over the cap by {dropped}" if dropped > 0 else None,
    )


def _unplannable(state: RunState, error: str, *, model=None, usage=None) -> NodeOutcome:
    """No plan means no run.

    The run is failed here rather than passed along, because the router sends a run with no
    plan straight back to the planner. Anything else is a loop.
    """
    log.warning("run %s cannot be planned: %s", state.run_id, error)
    return NodeOutcome(
        state=state.model_copy(update={"status": "failed", "error": error, "finished_at": now()}),
        status="error",
        model=model,
        usage=usage,
        error=error,
    )
