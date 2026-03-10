"""Redis access: the run document, the worker lease, the active set, and the search cache.

Everything the rest of the system knows about persistence lives here.

The lease is the only subtle part. refresh and release must compare the holder before
acting, otherwise a worker that stalled past its TTL could refresh or delete a lease that
the replacement worker now holds. Both are Lua so the compare and the act are one step.
"""

from collections.abc import Sequence
from datetime import datetime

from redis.asyncio import Redis

from ra.schemas import RunState

# KEYS[1] = lease key, ARGV[1] = worker id, ARGV[2] = ttl seconds
_REFRESH_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
else
  return 0
end
"""

# KEYS[1] = lease key, ARGV[1] = worker id
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
else
  return 0
end
"""

ACTIVE_SET = "runs:active"
RUN_INDEX = "runs:index"

# The index is for looking back over recent runs, not for keeping them forever. Trimming it
# bounds the one structure here that would otherwise grow for the life of the deployment.
MAX_INDEXED_RUNS = 10_000

DEFAULT_CACHE_TTL_S = 7 * 24 * 60 * 60


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


def lease_key(run_id: str) -> str:
    return f"run:{run_id}:lease"


class RunStore:
    """All Redis reads and writes for a run. Construct one per process."""

    def __init__(self, redis: Redis, *, lease_ttl_s: int = 60) -> None:
        self._r = redis
        self.lease_ttl_s = lease_ttl_s
        self._refresh = redis.register_script(_REFRESH_LUA)
        self._release = redis.register_script(_RELEASE_LUA)

    @property
    def client(self) -> Redis:
        """The underlying client, for the few things that need Redis and not a run document."""
        return self._r

    # -- run document -------------------------------------------------------------

    async def save(self, state: RunState) -> None:
        """Write the whole run document, and keep the two indexes in step with it.

        runs:active drives the sweeper. runs:index is ordered by creation time and is what
        lets anything ask about runs as a set rather than one at a time. Re-adding the same
        id with the same score on every save is idempotent and costs nothing.
        """
        pipe = self._r.pipeline()
        pipe.set(run_key(state.run_id), state.model_dump_json())
        if state.status == "running":
            pipe.sadd(ACTIVE_SET, state.run_id)
        else:
            pipe.srem(ACTIVE_SET, state.run_id)
        pipe.zadd(RUN_INDEX, {state.run_id: state.created_at.timestamp()})
        pipe.zremrangebyrank(RUN_INDEX, 0, -(MAX_INDEXED_RUNS + 1))
        await pipe.execute()

    async def recent_run_ids(self, *, since: datetime | None = None, limit: int = 50) -> list[str]:
        """Ids of the most recently created runs, newest first."""
        return await self._r.zrevrangebyscore(
            RUN_INDEX,
            max="+inf",
            min=since.timestamp() if since else "-inf",
            start=0,
            num=max(0, limit),
        )

    async def load_many(self, run_ids: Sequence[str]) -> list[RunState]:
        """Load several runs in one round trip. Ids that have expired are skipped."""
        if not run_ids:
            return []
        raw = await self._r.mget([run_key(run_id) for run_id in run_ids])
        return [RunState.model_validate_json(item) for item in raw if item is not None]

    async def indexed_run_count(self) -> int:
        return await self._r.zcard(RUN_INDEX)

    async def load(self, run_id: str) -> RunState | None:
        """Return the run document, or None if there is no such run."""
        raw = await self._r.get(run_key(run_id))
        if raw is None:
            return None
        return RunState.model_validate_json(raw)

    async def exists(self, run_id: str) -> bool:
        return bool(await self._r.exists(run_key(run_id)))

    # -- lease --------------------------------------------------------------------

    async def acquire_lease(self, run_id: str, worker_id: str) -> bool:
        """Take the lease if it is free. False means another worker holds it."""
        got = await self._r.set(lease_key(run_id), worker_id, nx=True, ex=self.lease_ttl_s)
        return bool(got)

    async def refresh_lease(self, run_id: str, worker_id: str) -> bool:
        """Extend the lease, but only if this worker still holds it."""
        result = await self._refresh(keys=[lease_key(run_id)], args=[worker_id, self.lease_ttl_s])
        return bool(result)

    async def release_lease(self, run_id: str, worker_id: str) -> bool:
        """Drop the lease, but only if this worker still holds it."""
        result = await self._release(keys=[lease_key(run_id)], args=[worker_id])
        return bool(result)

    async def lease_holder(self, run_id: str) -> str | None:
        return await self._r.get(lease_key(run_id))

    # -- active set ---------------------------------------------------------------

    async def active_runs(self) -> set[str]:
        return set(await self._r.smembers(ACTIVE_SET))

    async def forget_active(self, run_id: str) -> None:
        await self._r.srem(ACTIVE_SET, run_id)

    # -- search cache (used from Phase 3) -----------------------------------------

    async def cache_get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def cache_set(self, key: str, value: str, ttl_s: int = DEFAULT_CACHE_TTL_S) -> None:
        await self._r.set(key, value, ex=ttl_s)


def make_redis(url: str) -> Redis:
    """The one place a Redis client is built. decode_responses keeps callers off bytes."""
    return Redis.from_url(url, decode_responses=True)
