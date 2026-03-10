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


# -- the run index -------------------------------------------------------------


async def test_saving_a_run_indexes_it(store: RunStore):
    state = make_state(status="queued")
    await store.save(state)

    assert await store.recent_run_ids() == [state.run_id]
    assert await store.indexed_run_count() == 1


async def test_the_index_is_ordered_newest_first(store: RunStore):
    from datetime import timedelta

    from ra.clock import now

    base = now()
    for i in range(3):
        await store.save(
            make_state(run_id=f"run_{i}", created_at=base - timedelta(hours=i), status="done")
        )

    assert await store.recent_run_ids() == ["run_0", "run_1", "run_2"]


async def test_the_index_can_be_limited_and_windowed(store: RunStore):
    from datetime import timedelta

    from ra.clock import now

    base = now()
    for i in range(5):
        await store.save(
            make_state(run_id=f"run_{i}", created_at=base - timedelta(hours=i), status="done")
        )

    assert len(await store.recent_run_ids(limit=2)) == 2
    assert len(await store.recent_run_ids(since=base - timedelta(hours=2))) == 3


async def test_re_saving_a_run_does_not_duplicate_it(store: RunStore):
    """save() runs after every node, so the index must be idempotent."""
    state = make_state(status="running")
    for _ in range(5):
        await store.save(state)

    assert await store.indexed_run_count() == 1


async def test_the_index_is_trimmed_so_it_cannot_grow_forever(store: RunStore, monkeypatch):
    import ra.store as mod

    monkeypatch.setattr(mod, "MAX_INDEXED_RUNS", 3)
    from datetime import timedelta

    from ra.clock import now

    base = now()
    for i in range(6):
        await store.save(
            make_state(run_id=f"run_{i}", created_at=base + timedelta(hours=i), status="done")
        )

    assert await store.indexed_run_count() == 3
    # the newest survive
    assert await store.recent_run_ids() == ["run_5", "run_4", "run_3"]


async def test_load_many_skips_ids_whose_document_is_gone(store: RunStore):
    kept = make_state(run_id="run_kept", status="done")
    await store.save(kept)
    await store.save(make_state(run_id="run_gone", status="done"))
    await store.client.delete("run:run_gone")

    loaded = await store.load_many(["run_gone", "run_kept"])

    assert [s.run_id for s in loaded] == ["run_kept"]


async def test_load_many_of_nothing_makes_no_request(store: RunStore):
    assert await store.load_many([]) == []
