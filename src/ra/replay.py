"""Replay a recorded run offline.

This is the demo. A reader cannot hit your service, but they can clone the repo and watch a
real run play back from its own trace, with no keys and no network.

Replay re-renders the report from the recorded findings rather than printing the stored
markdown. If the renderer has drifted, the bytes differ and the test says so, which makes
every fixture a golden file for `render.py`.

This module must not import anything that talks to the network or to Redis, so that
`make demo` works on a clean clone with nothing but uv installed. There is a test for that.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TextIO

from ra.render import render_markdown
from ra.schemas import RunState

DEFAULT_SPEED = 20.0
HEADER_WIDTH = 78


def load_fixture(path: str | Path) -> RunState:
    return RunState.model_validate_json(Path(path).read_text())


def dump_fixture(state: RunState) -> str:
    """Stable JSON, so a re-recorded fixture produces a readable diff."""
    return json.dumps(json.loads(state.model_dump_json()), indent=2, sort_keys=True) + "\n"


def replay(state: RunState, *, speed: float = DEFAULT_SPEED, out: TextIO | None = None) -> str:
    """Print the trace at a fraction of its original pace, then the report.

    Returns the re-rendered markdown, which is what the test compares.

    `out` is resolved when the function runs, not when the module is imported, so that
    anything redirecting stdout actually sees the output.
    """
    out = sys.stdout if out is None else out
    _write_header(state, out)
    for step in state.steps:
        if speed > 0 and step.duration_ms:
            time.sleep(step.duration_ms / speed / 1000)
        out.write(_step_line(step) + "\n")
        out.flush()

    out.write("\n" + _rule("report") + "\n\n")
    markdown = (
        render_markdown(state.report, state.findings, partial=state.budget_stopped)
        if state.report
        else ""
    )
    out.write(markdown or "(this run produced no report)\n")
    if state.error:
        out.write(f"\nrun ended with: {state.error}\n")
    return markdown


def _rule(label: str) -> str:
    return f"-- {label} " + "-" * max(0, HEADER_WIDTH - len(label) - 4)


def _write_header(state: RunState, out: TextIO) -> None:
    out.write(_rule("run") + "\n")
    out.write(f"{state.question}\n\n")
    out.write(
        f"{state.run_id}  status={state.status}  "
        f"cost=${state.cost_usd:.4f}  tokens={state.tokens_in}/{state.tokens_out}  "
        f"credits={state.tavily_credits}\n\n"
    )
    out.write(_rule("trace") + "\n")
    out.write(
        f"{'seq':>3}  {'node':<9} {'sq':<6} {'model':<17} "
        f"{'in':>6} {'out':>6} {'cost':>8} {'ms':>6}  status\n"
    )


def _step_line(step) -> str:
    line = (
        f"{step.seq:>3}  {step.node:<9} {step.sub_question_id or '-':<6} "
        f"{step.model or '-':<17} {step.tokens_in:>6} {step.tokens_out:>6} "
        f"{step.cost_usd:>8.5f} {step.duration_ms:>6}  {step.status}"
    )
    detail = step.error or step.note
    return f"{line}\n     {detail}" if detail else line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ra.replay",
        description="Replay a recorded run from its trace. No keys, no network.",
    )
    parser.add_argument("fixture", help="path to a fixture under fixtures/runs/")
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"how much faster than real time (default {DEFAULT_SPEED:g}, 0 for instant)",
    )
    args = parser.parse_args(argv)

    try:
        state = load_fixture(args.fixture)
    except FileNotFoundError:
        print(f"no such fixture: {args.fixture}", file=sys.stderr)
        return 1

    replay(state, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
