"""Builders for test states. Keeps the test bodies about the thing being tested."""

from ra.clock import now
from ra.ids import sq_id
from ra.schemas import (
    Claim,
    Finding,
    Report,
    RunState,
    Section,
    SubQuestion,
)


def make_state(**overrides) -> RunState:
    base = {
        "run_id": "run_test000001",
        "question": "What is the current state of durable agent execution?",
        "status": "running",
    }
    return RunState(**(base | overrides))


def make_plan(n: int = 2, status: str = "pending") -> list[SubQuestion]:
    return [
        SubQuestion(id=sq_id(i), text=f"sub-question {i}", status=status) for i in range(1, n + 1)
    ]


def make_finding(sq: str = "sq_01", fid: str = "f_aaa111", **overrides) -> Finding:
    base = {
        "id": fid,
        "sub_question_id": sq,
        "claim": "A claim.",
        "source_url": f"https://example.invalid/{fid}",
        "source_title": "Example",
        "snippet": "snippet",
        "retrieved_at": now(),
        "extraction_status": "full",
    }
    return Finding(**(base | overrides))


def make_report(title: str = "Title") -> Report:
    return Report(
        title=title,
        sections=[Section(heading="H", claims=[Claim(text="A claim.", finding_ids=["f_aaa111"])])],
        generated_at=now(),
    )
