"""The arq worker. This is where the graph actually runs.

The API only enqueues and reads. A FastAPI BackgroundTask would die with the API process,
and surviving exactly that is the point of the project.
"""

import logging

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from ra.clock import now
from ra.config import get_settings
from ra.deps import Deps
from ra.graph import RECURSION_LIMIT, build_graph
from ra.schemas import RunState
from ra.store import RunStore, make_redis

log = logging.getLogger("ra.worker")

ERROR_LEN = 500


def job_id(state: RunState) -> str:
    """One job per attempt.

    arq refuses a duplicate job id while that job is queued or running, which makes enqueue
    idempotent. The attempt suffix lets the sweeper re-enqueue after a crash without
    colliding with the stale job record the dead worker left behind.
    """
    return f"{state.run_id}:{state.attempt}"


async def enqueue_run(pool: ArqRedis, state: RunState) -> None:
    await pool.enqueue_job("run_graph", state.run_id, _job_id=job_id(state))


async def run_graph(ctx: dict, run_id: str) -> None:
    """Take the lease, run the graph to completion, release the lease."""
    store: RunStore = ctx["store"]
    worker_id: str = ctx["worker_id"]

    if not await store.acquire_lease(run_id, worker_id):
        log.info("run %s already leased, skipping", run_id)
        return

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

        try:
            await ctx["graph"].ainvoke(
                state.model_dump(), config={"recursion_limit": RECURSION_LIMIT}
            )
        except Exception as exc:
            # A node failure is recorded as a step and never reaches here. This is a
            # graph-level failure: recursion limit, state validation, a broken Redis.
            log.exception("run %s failed", run_id)
            latest = await store.load(run_id) or state
            await store.save(
                latest.model_copy(
                    update={
                        "status": "failed",
                        "error": repr(exc)[:ERROR_LEN],
                        "finished_at": now(),
                    }
                )
            )
    finally:
        await store.release_lease(run_id, worker_id)


async def on_startup(ctx: dict) -> None:
    settings = get_settings()
    redis = make_redis(settings.redis_url)
    store = RunStore(redis, lease_ttl_s=settings.lease_ttl_s)
    ctx["settings"] = settings
    ctx["redis"] = redis
    ctx["store"] = store
    ctx["worker_id"] = settings.worker_id
    ctx["graph"] = build_graph(Deps(store=store, settings=settings))
    ctx["pool"] = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    log.info("worker %s ready", settings.worker_id)


async def on_shutdown(ctx: dict) -> None:
    for key in ("redis", "pool"):
        client = ctx.get(key)
        if client is not None:
            await client.aclose()


class WorkerSettings:
    """Entry point: arq ra.worker.WorkerSettings"""

    functions = [run_graph]
    cron_jobs: list = []  # the sweeper lands here in Phase 5
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 4
    job_timeout = 360  # max_wall_clock_s plus a minute
    keep_result = 3600
    # arq reads this class's __dict__ and passes the values straight to Worker(), so this
    # has to be an instance, not a method.
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
