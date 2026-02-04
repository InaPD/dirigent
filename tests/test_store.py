"""Store behaviour, with the lease compare-and-act rules as the main event."""

import asyncio

import pytest

from ra.store import ACTIVE_SET, RunStore, lease_key
from tests.factories import make_state


async def test_save_and_load_round_trip(store: RunStore):
    state = make_state(plan=[], status="queued")
    await store.save(state)
    assert await store.load(state.run_id) == state


async def test_load_missing_run_returns_none(store: RunStore):
    assert await store.load("run_nope") is None
    assert await store.exists("run_nope") is False


async def test_active_set_tracks_status(store: RunStore, fake_redis):
    state = make_state(status="running")
    await store.save(state)
    assert await store.active_runs() == {state.run_id}

    await store.save(state.model_copy(update={"status": "done"}))
    assert await store.active_runs() == set()


async def test_acquire_lease_is_exclusive(store: RunStore):
    assert await store.acquire_lease("run_1", "worker-a") is True
    assert await store.acquire_lease("run_1", "worker-b") is False
    assert await store.lease_holder("run_1") == "worker-a"


async def test_refresh_by_holder_extends_ttl(store: RunStore, fake_redis):
    await store.acquire_lease("run_1", "worker-a")
    await asyncio.sleep(0.05)
    assert await store.refresh_lease("run_1", "worker-a") is True
    assert await fake_redis.ttl(lease_key("run_1")) > 0


async def test_refresh_by_non_holder_is_refused(store: RunStore, fake_redis):
    await store.acquire_lease("run_1", "worker-a")
    before = await fake_redis.ttl(lease_key("run_1"))
    assert await store.refresh_lease("run_1", "worker-b") is False
    assert await fake_redis.ttl(lease_key("run_1")) <= before
    assert await store.lease_holder("run_1") == "worker-a"


async def test_release_by_non_holder_is_a_no_op(store: RunStore):
    await store.acquire_lease("run_1", "worker-a")
    assert await store.release_lease("run_1", "worker-b") is False
    assert await store.lease_holder("run_1") == "worker-a"


async def test_release_by_holder_frees_the_lease(store: RunStore):
    await store.acquire_lease("run_1", "worker-a")
    assert await store.release_lease("run_1", "worker-a") is True
    assert await store.lease_holder("run_1") is None
    assert await store.acquire_lease("run_1", "worker-b") is True


async def test_refresh_missing_lease_returns_false(store: RunStore):
    assert await store.refresh_lease("run_absent", "worker-a") is False


async def test_lease_expires(store: RunStore, fake_redis):
    await store.acquire_lease("run_1", "worker-a")
    await fake_redis.expire(lease_key("run_1"), 0)
    assert await store.lease_holder("run_1") is None


async def test_forget_active(store: RunStore, fake_redis):
    await fake_redis.sadd(ACTIVE_SET, "run_ghost")
    await store.forget_active("run_ghost")
    assert await store.active_runs() == set()


async def test_cache_get_set(store: RunStore):
    assert await store.cache_get("cache:search:x") is None
    await store.cache_set("cache:search:x", "payload", ttl_s=60)
    assert await store.cache_get("cache:search:x") == "payload"


@pytest.mark.redis
async def test_lease_rules_hold_on_real_redis(real_store: RunStore):
    assert await real_store.acquire_lease("run_r", "worker-a") is True
    assert await real_store.refresh_lease("run_r", "worker-b") is False
    assert await real_store.release_lease("run_r", "worker-b") is False
    assert await real_store.refresh_lease("run_r", "worker-a") is True
    assert await real_store.release_lease("run_r", "worker-a") is True
