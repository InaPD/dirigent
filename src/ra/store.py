"""Redis access: the run document, the worker lease, the active set, and the search cache.

Everything the rest of the system knows about persistence lives here.

The lease is the only subtle part. refresh and release must compare the holder before
acting, otherwise a worker that stalled past its TTL could refresh or delete a lease that
the replacement worker now holds. Both are Lua so the compare and the act are one step.
"""

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

    # -- run document -------------------------------------------------------------

    async def save(self, state: RunState) -> None:
        """Write the whole run document and keep runs:active in step with its status."""
        pipe = self._r.pipeline()
        pipe.set(run_key(state.run_id), state.model_dump_json())
        if state.status == "running":
            pipe.sadd(ACTIVE_SET, state.run_id)
        else:
            pipe.srem(ACTIVE_SET, state.run_id)
        await pipe.execute()

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
