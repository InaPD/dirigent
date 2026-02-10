"""Record a real run to a fixture.

    make record Q="What is the current state of durable agent execution?"

Posts the question to a running API, waits for the run to finish, then writes the whole
RunState to fixtures/runs/. That file is what `make demo` replays, and what the test in
tests/test_replay.py pins render.py against.

Needs the stack up (`make up`) with real keys in .env. The run document is read straight
from Redis, because the HTTP API deliberately does not expose findings.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

import httpx

from ra.config import get_settings
from ra.replay import dump_fixture
from ra.schemas import RunState
from ra.store import RunStore, make_redis

DEFAULT_API = "http://localhost:8080"
FIXTURE_DIR = Path("fixtures/runs")
POLL_S = 2.0
DEFAULT_TIMEOUT_S = 600


def slugify(question: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    return slug[:limit].rstrip("-") or "run"


async def record(question: str, *, api_url: str, timeout_s: int) -> RunState:
    """Run the question through the real stack and return the finished run document."""
    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as http:
        response = await http.post("/research", json={"question": question})
        response.raise_for_status()
        run_id = response.json()["data"]["run_id"]
        print(f"queued {run_id}", flush=True)

        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            body = (await http.get(f"/research/{run_id}")).json()["data"]
            if body["finished_at"]:
                print(f"finished as {body['status']} for ${body['cost_usd']:.4f}", flush=True)
                break
            print(f"  {body['status']}...", flush=True)
            await asyncio.sleep(POLL_S)
        else:
            raise TimeoutError(f"run {run_id} did not finish within {timeout_s}s")

    redis = make_redis(get_settings().redis_url)
    try:
        state = await RunStore(redis).load(run_id)
    finally:
        await redis.aclose()

    if state is None:
        raise RuntimeError(f"run {run_id} finished but its document is gone")
    return state


def save(state: RunState, question: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(question)}.json"
    path.write_text(dump_fixture(state))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(state.steps)} steps)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="record_run.py")
    parser.add_argument("question")
    parser.add_argument("--api-url", default=DEFAULT_API)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    if not args.question.strip():
        print("give me a question to research", file=sys.stderr)
        return 1

    try:
        state = asyncio.run(record(args.question, api_url=args.api_url, timeout_s=args.timeout))
    except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
        print(f"recording failed: {exc}", file=sys.stderr)
        return 1

    save(state, args.question, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
