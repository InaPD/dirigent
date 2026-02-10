"""Kill a worker mid-run. A new worker finishes the run without redoing completed work.

This is the project's central claim, so it is a real test: two real worker processes, a real
Redis, a real queue, and a real SIGKILL. Nothing is mocked except the researcher, which is
stubbed so the test has an exact moment to kill at.

Synchronisation is through Redis, never sleeps. The stub pushes to a gate key when it starts
the second sub-question and blocks; the test blocks on that same key. When it returns, the
run is provably half finished, which is the only moment the kill means anything.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import pytest

from ra.clock import now
from ra.ids import sq_id
from ra.nodes.stubs import GATE_REACHED
from ra.schemas import RunState, SubQuestion
from ra.store import RunStore
from ra.worker import enqueue_run
from tests.conftest import TEST_REDIS_URL

pytestmark = pytest.mark.redis

LEASE_TTL_S = 2
SWEEP_SECONDS = 1
GATE_WAIT_S = 30
FINISH_WAIT_S = 45
POLL_S = 0.1


def start_worker(stub: str, worker_id: str) -> subprocess.Popen:
    """A real arq worker in its own process, so SIGKILL means what it says."""
    env = {
        **os.environ,
        "REDIS_URL": TEST_REDIS_URL,
        "ANTHROPIC_API_KEY": "",
        "TAVILY_API_KEY": "",
        "RA_STUB": stub,
        "RA_WORKER_ID": worker_id,
        "RA_LEASE_TTL_S": str(LEASE_TTL_S),
        "RA_SWEEP_SECONDS": str(SWEEP_SECONDS),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "arq", "ra.worker.WorkerSettings"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.kill()
    process.wait(timeout=10)


@pytest.fixture
def workers():
    started: list[subprocess.Popen] = []
    yield started
    for process in started:
        stop(process)


async def wait_for_gate(store: RunStore) -> str:
    """Block until the stub says it has started the second sub-question."""
    reached = await store.client.blpop(GATE_REACHED, GATE_WAIT_S)
    assert reached is not None, (
        f"the worker never reached the gate within {GATE_WAIT_S}s. "
        "It probably failed to start; check the worker output."
    )
    return reached[1]


async def wait_until_finished(store: RunStore, run_id: str) -> RunState:
    deadline = time.monotonic() + FINISH_WAIT_S
    while time.monotonic() < deadline:
        state = await store.load(run_id)
        if state is not None and state.finished_at is not None:
            return state
        await asyncio.sleep(POLL_S)
    pytest.fail(f"run {run_id} did not finish within {FINISH_WAIT_S}s")


async def seed_run(store: RunStore) -> RunState:
    """A run that already has a plan, so only the researcher's progress is under test."""
    state = RunState(
        run_id="run_killtest01",
        question="What survives a worker being killed mid-run?",
        status="queued",
        created_at=now(),
        plan=[
            SubQuestion(id=sq_id(1), text="the first sub-question"),
            SubQuestion(id=sq_id(2), text="the second sub-question"),
        ],
    )
    await store.save(state)
    return state


async def test_resume_after_kill(real_store: RunStore, workers):
    from arq import create_pool
    from arq.connections import RedisSettings

    state = await seed_run(real_store)
    pool = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
    try:
        await enqueue_run(pool, state)
    finally:
        await pool.aclose()

    # -- worker A gets as far as the second sub-question, then waits at the gate
    worker_a = start_worker("slow_research", "worker-a")
    workers.append(worker_a)

    blocked_on = await wait_for_gate(real_store)
    assert blocked_on == "sq_02"

    half_done = await real_store.load(state.run_id)
    assert half_done.status == "running"
    assert half_done.plan[0].status == "answered"
    assert half_done.plan[1].status == "pending"
    assert len(half_done.findings) == 1
    first_finding = half_done.findings[0]
    assert await real_store.lease_holder(state.run_id) == "worker-a"

    # -- kill it where it stands
    os.kill(worker_a.pid, signal.SIGKILL)
    worker_a.wait(timeout=10)

    still_running = await real_store.load(state.run_id)
    assert still_running.status == "running"  # nobody got to mark it otherwise
    assert still_running.finished_at is None

    # -- worker B notices the dead lease and finishes the run
    worker_b = start_worker("fast_research", "worker-b")
    workers.append(worker_b)

    final = await wait_until_finished(real_store, state.run_id)

    assert final.status == "done", final.error
    assert final.report_markdown

    # the sub-question that completed before the kill was not researched again
    research_steps = [s for s in final.steps if s.node == "research"]
    by_question = [s.sub_question_id for s in research_steps]
    assert by_question.count("sq_01") == 1, (
        f"sq_01 was researched {by_question.count('sq_01')} times"
    )
    assert by_question.count("sq_02") == 1

    # and its finding survived the crash untouched
    assert [f.id for f in final.findings if f.sub_question_id == "sq_01"] == [first_finding.id]
    assert len(final.findings) == 2

    # the sweeper re-enqueued it exactly once, and left nothing behind
    assert final.attempt == 1
    assert final.worker_id == "worker-b"
    assert await real_store.active_runs() == set()
    assert await real_store.lease_holder(state.run_id) is None


async def test_a_live_run_is_not_swept_out_from_under_its_worker(real_store: RunStore, workers):
    """The heartbeat exists so a slow node is not mistaken for a dead worker.

    The stub blocks for far longer than the lease TTL. If the heartbeat were not refreshing,
    the sweeper would re-enqueue this run and a second worker would redo sq_01.
    """
    state = await seed_run(real_store)
    from arq import create_pool
    from arq.connections import RedisSettings

    pool = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL))
    try:
        await enqueue_run(pool, state)
    finally:
        await pool.aclose()

    worker = start_worker("slow_research", "worker-a")
    workers.append(worker)
    await wait_for_gate(real_store)

    # sit at the gate for several lease lifetimes
    await asyncio.sleep(LEASE_TTL_S * 3)

    held = await real_store.load(state.run_id)
    assert await real_store.lease_holder(state.run_id) == "worker-a"
    assert held.attempt == 0, "the sweeper re-enqueued a run that was alive"
    assert [s.sub_question_id for s in held.steps if s.node == "research"] == ["sq_01"]
