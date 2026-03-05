"""The arq worker. This is where the graph actually runs.

The API only enqueues and reads. A FastAPI BackgroundTask would die with the API process,
and surviving exactly that is the point of the project.
"""

import asyncio
import contextlib
import logging

from arq import create_pool, cron
from arq.connections import ArqRedis, RedisSettings

from ra.clock import now
from ra.config import get_settings
from ra.deps import Deps
from ra.errors import safe_detail
from ra.graph import RECURSION_LIMIT, build_graph
from ra.llm import LLM
from ra.schemas import RunState
from ra.search import Tavily
from ra.store import RunStore, make_redis

log = logging.getLogger("ra.worker")


def job_id(state: RunState) -> str:
    """One job per attempt.

    arq refuses a duplicate job id while that job is queued or running, which makes enqueue
    idempotent. The attempt suffix lets the sweeper re-enqueue after a crash without
    colliding with the stale job record the dead worker left behind.
    """
    return f"{state.run_id}:{state.attempt}"


async def enqueue_run(pool: ArqRedis, state: RunState) -> None:
    await pool.enqueue_job("run_graph", state.run_id, _job_id=job_id(state))


async def _hold_lease(store: RunStore, run_id: str, worker_id: str) -> None:
    """Keep the lease alive while a slow node runs.

    Nodes refresh the lease when they finish, but a node can legitimately outlive the TTL:
    a batch of extractions plus a Sonnet call is easily more than a minute. Without this the
    sweeper would re-enqueue a run that is perfectly healthy, just slow.
    """
    interval = max(1.0, store.lease_ttl_s / 3)
    while True:
        await asyncio.sleep(interval)
        if not await store.refresh_lease(run_id, worker_id):
            log.warning("run %s lost its lease while running", run_id)
            return


async def run_graph(ctx: dict, run_id: str) -> None:
    """Take the lease, run the graph to completion, release the lease."""
    store: RunStore = ctx["store"]
    worker_id: str = ctx["worker_id"]

    if not await store.acquire_lease(run_id, worker_id):
        log.info("run %s already leased, skipping", run_id)
        return

    heartbeat: asyncio.Task | None = None
    try:
        state = await store.load(run_id)
        if state is None:
            log.warning("run %s has no state document", run_id)
            return
        if state.is_terminal:
            log.info("run %s already %s", run_id, state.status)
            return

        state = state.model_copy(
            update={
                "status": "running",
                "started_at": state.started_at or now(),
                "worker_id": worker_id,
            }
        )
        await store.save(state)

        # Race the work against the lease. Whichever finishes first decides what happens.
        #
        # The heartbeat only finishes early when a refresh fails, which means another worker
        # now owns this run. Carrying on would be worse than stopping: save() is an
        # unconditional write, so a worker that stalled past its TTL and then woke up would
        # overwrite whatever its replacement had already done, interleaving steps and
        # double-counting spend. So losing the lease cancels the work.
        heartbeat = asyncio.create_task(_hold_lease(store, run_id, worker_id))
        work = asyncio.create_task(
            ctx["graph"].ainvoke(state.model_dump(), config={"recursion_limit": RECURSION_LIMIT})
        )
        done, _ = await asyncio.wait({heartbeat, work}, return_when=asyncio.FIRST_COMPLETED)

        if work not in done:
            log.warning("run %s was taken over by another worker, abandoning it", run_id)
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
            return

        failure = work.exception()
        if failure is not None:
            # A node failure is recorded as a step and never reaches here. This is a
            # graph-level failure: recursion limit, state validation, a broken Redis.
            log.error("run %s failed: %s", run_id, safe_detail(failure))
            latest = await store.load(run_id) or state
            await store.save(
                latest.model_copy(
                    update={
                        "status": "failed",
                        "error": safe_detail(failure),
                        "finished_at": now(),
                    }
                )
            )
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        # Compare-and-act, so this is a no-op if the lease is already someone else's.
        await store.release_lease(run_id, worker_id)


async def sweep(ctx: dict) -> None:
    """Re-enqueue runs whose worker died.

    A lease outlives its worker by at most its TTL. A run that is still marked running with
    no lease behind it has nobody working on it, so it goes back on the queue. The router
    picks it up from wherever it got to, because the next step is derived from the state.
    """
    store: RunStore = ctx["store"]
    pool: ArqRedis = ctx["pool"]

    for run_id in await store.active_runs():
        state = await store.load(run_id)
        if state is None or state.status != "running":
            await store.forget_active(run_id)
            continue
        if await store.lease_holder(run_id) is not None:
            continue

        bumped = state.model_copy(update={"attempt": state.attempt + 1})
        await store.save(bumped)
        await enqueue_run(pool, bumped)
        log.warning("re-enqueued orphaned run %s as attempt %d", run_id, bumped.attempt)


def sweep_schedule(interval_s: int) -> set[int]:
    """Which seconds of each minute the sweeper runs on."""
    interval = max(1, min(interval_s, 60))
    return set(range(0, 60, interval))


def build_deps(settings, store: RunStore) -> Deps:
    """Assemble what the nodes need.

    The model client is only built when a key is configured. Without one the canned nodes
    still run, which is what keeps the test suite and CI free of credentials.
    """
    llm = LLM.from_api_key(settings.require_anthropic_key()) if settings.anthropic_api_key else None
    if llm is None:
        log.warning("no ANTHROPIC_API_KEY, model-backed nodes will not run")

    search = (
        Tavily(settings.require_tavily_key(), store) if settings.tavily_api_key and store else None
    )
    if search is None:
        log.warning("no TAVILY_API_KEY, the researcher will fall back to canned findings")

    return Deps(store=store, settings=settings, llm=llm, search=search)


async def on_startup(ctx: dict) -> None:
    settings = get_settings()
    redis = make_redis(settings.redis_url)
    store = RunStore(redis, lease_ttl_s=settings.lease_ttl_s)
    ctx["settings"] = settings
    ctx["redis"] = redis
    ctx["store"] = store
    ctx["worker_id"] = settings.worker_id
    ctx["deps"] = build_deps(settings, store)
    ctx["graph"] = build_graph(ctx["deps"])
    ctx["pool"] = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    log.info("worker %s ready", settings.worker_id)


async def on_shutdown(ctx: dict) -> None:
    deps = ctx.get("deps")
    if deps is not None:
        for client in (deps.search, deps.llm):
            if client is not None:
                await client.aclose()
    for key in ("redis", "pool"):
        client = ctx.get(key)
        if client is not None:
            await client.aclose()


class WorkerSettings:
    """Entry point: arq ra.worker.WorkerSettings"""

    functions = [run_graph]
    cron_jobs = [
        cron(
            sweep,
            second=sweep_schedule(get_settings().sweeper_interval_s),
            run_at_startup=True,
            max_tries=1,
        )
    ]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 4
    job_timeout = 360  # max_wall_clock_s plus a minute
    keep_result = 3600
    # arq reads this class's __dict__ and passes the values straight to Worker(), so this
    # has to be an instance, not a method.
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
