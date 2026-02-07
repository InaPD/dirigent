"""Rendering. Phase 6 replays a recorded run and asserts byte identity, so this must be
deterministic: same report and findings in, same bytes out, every time and every machine.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ra.render import render_markdown
from ra.schemas import Claim, Finding, Report, Section

GOLDEN = Path(__file__).parent / "golden" / "report_small.md"
FIXED_TIME = datetime(2026, 9, 14, 10, 30, tzinfo=UTC)


def finding(fid: str, url: str, *, title: str | None = "A Source", status="full") -> Finding:
    return Finding(
        id=fid,
        sub_question_id="sq_01",
        claim="claim text",
        source_url=url,
        source_title=title,
        snippet="snippet",
        retrieved_at=FIXED_TIME,
        extraction_status=status,
    )


@pytest.fixture
def report() -> Report:
    return Report(
        title="Durable agent execution",
        generated_at=FIXED_TIME,
        sections=[
            Section(
                heading="State of the art",
                claims=[
                    Claim(text="Two sources agree.", finding_ids=["f_aaa111", "f_bbb222"]),
                    Claim(text="One source says more.", finding_ids=["f_bbb222"]),
                ],
            ),
            Section(
                heading="Open problems",
                claims=[Claim(text="Resume is the hard part.", finding_ids=["f_ccc333"])],
            ),
        ],
    )


@pytest.fixture
def findings() -> list[Finding]:
    return [
        finding("f_aaa111", "https://one.test"),
        finding("f_bbb222", "https://two.test", title=None, status="snippet_only"),
        finding("f_ccc333", "https://three.test", title="Third Source"),
    ]


def test_matches_the_golden_file(report, findings):
    assert render_markdown(report, findings) == GOLDEN.read_text()


def test_rendering_is_deterministic(report, findings):
    assert render_markdown(report, findings) == render_markdown(report, findings)


def test_footnotes_are_numbered_by_first_appearance(report, findings):
    out = render_markdown(report, findings)

    assert "Two sources agree. [1][2]" in out
    assert "One source says more. [2]" in out
    assert "Resume is the hard part. [3]" in out


def test_every_footnote_has_a_source_line(report, findings):
    out = render_markdown(report, findings)
    sources = out.split("## Sources")[1]

    for n, url in enumerate(["https://one.test", "https://two.test", "https://three.test"], 1):
        assert f"[{n}] " in sources
        assert url in sources


def test_a_source_line_carries_its_provenance(report, findings):
    out = render_markdown(report, findings)

    assert "[1] A Source - https://one.test (retrieved 2026-09-14, full)" in out
    assert "[2] untitled - https://two.test (retrieved 2026-09-14, snippet_only)" in out


def test_an_uncited_finding_is_not_listed(report, findings):
    extra = [*findings, finding("f_ddd444", "https://unused.test")]

    out = render_markdown(report, extra)

    assert "https://unused.test" not in out
    assert "[4]" not in out


def test_a_partial_run_is_labelled_in_the_title(report, findings):
    out = render_markdown(report, findings, partial=True)

    assert out.startswith("# Durable agent execution (partial)\n")


def test_a_report_with_no_citations_has_no_sources_section():
    report = Report(
        title="Empty",
        generated_at=FIXED_TIME,
        sections=[Section(heading="H", claims=[Claim(text="Nothing.", finding_ids=[])])],
    )

    out = render_markdown(report, [])

    assert "## Sources" not in out
    assert out == "# Empty\n\n## H\n\nNothing.\n"


def test_output_ends_with_exactly_one_newline(report, findings):
    out = render_markdown(report, findings)

    assert out.endswith("\n")
    assert not out.endswith("\n\n")


def test_a_claim_is_not_left_with_a_trailing_space():
    """A claim whose ids were all unknown would otherwise render with a dangling space."""
    report = Report(
        title="T",
        generated_at=FIXED_TIME,
        sections=[Section(heading="H", claims=[Claim(text="Orphan.", finding_ids=["f_gone11"])])],
    )

    out = render_markdown(report, [])

    assert "Orphan.\n" in out
    assert "Orphan. \n" not in out
