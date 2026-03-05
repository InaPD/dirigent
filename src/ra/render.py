"""Report to markdown.

A pure function with no clock, no randomness and no set iteration, because Phase 6 replays
a recorded run and asserts the markdown comes out byte for byte identical. Keep it boring.

Rendering happens only after citations have been validated. Nothing here checks anything;
by the time a report reaches this module every finding id in it is known to exist.

The output is markdown, and it contains text a model wrote after reading web pages it was
sent to. Treat it as untrusted. Anything that turns it into HTML must escape or sanitise it
first, because most markdown renderers pass inline HTML straight through. `demo/trace.html`
escapes before rendering, and so should anything else. Escaping here instead would corrupt
legitimate content and would make the report markdown wrong rather than safe.
"""

from ra.schemas import Finding, Report

PARTIAL_SUFFIX = " (partial)"
DATE_FORMAT = "%Y-%m-%d"


def render_markdown(report: Report, findings: list[Finding], *, partial: bool = False) -> str:
    """Markdown with [n] footnotes, numbered by first appearance and listed at the end."""
    by_id = {f.id: f for f in findings}
    order: list[str] = []  # finding ids, in the order a reader meets them

    def number(finding_id: str) -> int:
        if finding_id not in order:
            order.append(finding_id)
        return order.index(finding_id) + 1

    lines = [f"# {report.title}{PARTIAL_SUFFIX if partial else ''}", ""]

    for section in report.sections:
        lines += [f"## {section.heading}", ""]
        for claim in section.claims:
            marks = "".join(f"[{number(fid)}]" for fid in claim.finding_ids if fid in by_id)
            lines += [f"{claim.text} {marks}".strip(), ""]

    if order:
        lines += ["## Sources", ""]
        for position, finding_id in enumerate(order, start=1):
            lines.append(f"[{position}] {_source_line(by_id[finding_id])}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _source_line(finding: Finding) -> str:
    title = finding.source_title or "untitled"
    retrieved = finding.retrieved_at.strftime(DATE_FORMAT)
    return f"{title} - {finding.source_url} (retrieved {retrieved}, {finding.extraction_status})"
