"""Replay. The demo has to work on a clean clone with no keys, and it has to stay honest.

Every fixture is also a golden file for render.py: replay re-renders the report from the
recorded findings and the test compares it to the markdown that was stored at the time. Any
drift in the renderer breaks this, which is the point.
"""

import io
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from ra.clock import now
from ra.replay import DEFAULT_SPEED, dump_fixture, load_fixture, main, replay
from ra.schemas import RunState

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "runs"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

# Anything here means replay has grown a dependency that needs a network, a key or a server.
FORBIDDEN_IMPORTS = {
    "redis",
    "anthropic",
    "arq",
    "httpx",
    "httpx2",
    "fastapi",
    "langgraph",
    "uvicorn",
}


@pytest.fixture
def no_network(monkeypatch):
    """Any socket at all is a failure, not just an HTTP client."""

    def forbidden(*args, **kwargs):
        raise AssertionError("replay tried to open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_there_is_at_least_one_fixture():
    """`make demo` on a clean clone needs something to replay."""
    assert FIXTURES, f"no fixtures in {FIXTURE_DIR}"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_loads(path):
    state = load_fixture(path)

    assert state.run_id
    assert state.steps
    assert state.finished_at is not None


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_replay_reproduces_the_recorded_report_byte_for_byte(path, no_network):
    """This is what makes every fixture a golden file for the renderer."""
    state = load_fixture(path)
    out = io.StringIO()

    markdown = replay(state, speed=0, out=out)

    assert markdown == state.report_markdown


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_every_citation_in_a_fixture_resolves(path):
    """The project's headline promise, checked against what was actually recorded."""
    state = load_fixture(path)
    known = {f.id for f in state.findings}

    cited = {
        fid
        for section in (state.report.sections if state.report else [])
        for claim in section.claims
        for fid in claim.finding_ids
    }
    assert cited <= known


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_stays_small(path):
    """A large fixture means page content is being stored on the run state somewhere."""
    assert path.stat().st_size < 300_000, f"{path.name} is too big; check the snippet cap"


def test_the_trace_and_the_report_are_both_printed(no_network):
    state = load_fixture(FIXTURES[0])
    out = io.StringIO()

    replay(state, speed=0, out=out)
    printed = out.getvalue()

    assert state.question in printed
    assert state.run_id in printed
    for step in state.steps:
        assert step.node in printed
    assert "-- report" in printed
    assert state.report_markdown in printed


def test_notes_and_errors_are_shown(no_network):
    state = load_fixture(FIXTURES[0])
    noted = state.model_copy(
        update={"steps": [state.steps[0].model_copy(update={"note": "a visible note"})]}
    )
    out = io.StringIO()

    replay(noted, speed=0, out=out)

    assert "a visible note" in out.getvalue()


def test_a_run_with_no_report_says_so(no_network):
    state = load_fixture(FIXTURES[0]).model_copy(
        update={"report": None, "report_markdown": None, "error": "it went wrong"}
    )
    out = io.StringIO()

    markdown = replay(state, speed=0, out=out)

    assert markdown == ""
    assert "no report" in out.getvalue()
    assert "it went wrong" in out.getvalue()


def test_speed_scales_the_pace(monkeypatch, no_network):
    slept: list[float] = []
    monkeypatch.setattr("ra.replay.time.sleep", slept.append)
    state = load_fixture(FIXTURES[0])

    replay(state, speed=10, out=io.StringIO())
    fast = sum(slept)
    slept.clear()
    replay(state, speed=5, out=io.StringIO())
    slow = sum(slept)

    assert slow == pytest.approx(fast * 2)
    recorded_s = sum(s.duration_ms for s in state.steps) / 1000
    assert fast == pytest.approx(recorded_s / 10)


def test_speed_zero_never_sleeps(monkeypatch, no_network):
    slept: list[float] = []
    monkeypatch.setattr("ra.replay.time.sleep", slept.append)

    replay(load_fixture(FIXTURES[0]), speed=0, out=io.StringIO())

    assert slept == []


def test_the_default_speed_keeps_the_demo_under_thirty_seconds():
    """The demo has to be watchable, not a five minute wait."""
    for path in FIXTURES:
        state = load_fixture(path)
        wall_clock_s = sum(s.duration_ms for s in state.steps) / 1000 / DEFAULT_SPEED
        assert wall_clock_s < 30, f"{path.name} replays in {wall_clock_s:.0f}s"


# -- the command line ----------------------------------------------------------


def test_the_cli_replays_a_fixture(capsys):
    assert main([str(FIXTURES[0]), "--speed", "0"]) == 0
    assert "-- report" in capsys.readouterr().out


def test_the_cli_reports_a_missing_fixture(capsys):
    assert main(["fixtures/runs/not-here.json"]) == 1
    assert "no such fixture" in capsys.readouterr().err


# -- isolation -----------------------------------------------------------------


def test_replay_imports_nothing_that_needs_a_key_or_a_server():
    """`make demo` must work on a clean clone with only uv installed.

    Run in a subprocess, because by now this process has imported half the project.
    """
    code = (
        "import ra.replay, sys, json;"
        f"print(json.dumps(sorted({FORBIDDEN_IMPORTS!r} & set(sys.modules))))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_the_demo_command_runs_as_a_module():
    """What `make demo` actually invokes."""
    result = subprocess.run(
        [sys.executable, "-m", "ra.replay", str(FIXTURES[0]), "--speed", "0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "-- trace" in result.stdout


# -- fixture format ------------------------------------------------------------


def test_fixtures_are_written_stably_so_diffs_are_readable():
    state = load_fixture(FIXTURES[0])

    dumped = dump_fixture(state)

    assert dumped.endswith("\n")
    assert json.loads(dumped) == json.loads(state.model_dump_json())
    keys = list(json.loads(dumped))
    assert keys == sorted(keys)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_committed_fixture_matches_how_it_would_be_written_today(path):
    """Guards against a hand-edited fixture, which would break the byte-identity test."""
    assert path.read_text() == dump_fixture(load_fixture(path))


def test_a_round_trip_through_a_fixture_preserves_everything(tmp_path):
    state = load_fixture(FIXTURES[0]).model_copy(update={"finished_at": now()})
    path = tmp_path / "round-trip.json"

    path.write_text(dump_fixture(state))

    assert load_fixture(path) == state


def test_loading_a_missing_fixture_raises():
    with pytest.raises(FileNotFoundError):
        load_fixture("fixtures/runs/definitely-not-here.json")


def test_a_fixture_is_not_a_partial_state():
    """A fixture has to carry everything replay needs, not just the report."""
    state = load_fixture(FIXTURES[0])

    assert state.findings or state.error
    assert all(s.input_digest and s.output_digest for s in state.steps)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_carries_no_credentials(path):
    """Fixtures are committed, so they must never have picked up a key."""
    raw = path.read_text()

    assert "sk-ant" not in raw
    assert "tvly-" not in raw


def test_replay_does_not_mutate_the_state(no_network):
    state = load_fixture(FIXTURES[0])
    before = state.model_dump_json()

    replay(state, speed=0, out=io.StringIO())

    assert state.model_dump_json() == before


def test_a_fixture_with_no_steps_still_replays(no_network):
    state = RunState(
        run_id="run_empty", question="q" * 20, status="failed", error="nothing happened"
    )

    replay(state, speed=0, out=io.StringIO())


# -- the static trace viewer ---------------------------------------------------

TRACE_HTML = Path(__file__).resolve().parents[1] / "demo" / "trace.html"


def test_the_trace_viewer_points_at_a_fixture_that_exists():
    """Cheap guard: the default path in the viewer must not drift away from the fixtures."""
    html = TRACE_HTML.read_text()
    default = html.split('|| "')[1].split('"')[0]

    assert (TRACE_HTML.parents[1] / default).is_file(), f"trace.html defaults to {default}"


def test_the_trace_viewer_escapes_before_rendering():
    """Fixtures are data. The viewer must not be a way to inject markup into the page."""
    html = TRACE_HTML.read_text()

    assert "const esc =" in html
    assert "esc(src)" in html  # markdown escapes first, then adds its own tags
