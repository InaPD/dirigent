import os

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from ra.config import Settings
from ra.deps import Deps
from ra.store import RunStore, make_redis

# Tests use db 1 so a local app running on db 0 is never touched.
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture
def settings() -> Settings:
    """Test settings, explicitly keyless.

    The keys are pinned to None so the suite behaves the same whether or not the developer
    has a .env on disk. Anything that needs a key builds its own Settings.
    """
    return Settings(
        redis_url=TEST_REDIS_URL,
        lease_ttl_s=2,
        worker_id="worker-test",
        anthropic_api_key=None,
        tavily_api_key=None,
    )


@pytest_asyncio.fixture
async def fake_redis():
    r = FakeRedis(decode_responses=True)
    try:
        yield r
    finally:
        await r.aclose()


@pytest_asyncio.fixture
async def store(fake_redis) -> RunStore:
    return RunStore(fake_redis, lease_ttl_s=2)


@pytest.fixture
def deps(store, settings) -> Deps:
    return Deps(store=store, settings=settings)


@pytest_asyncio.fixture
async def real_redis():
    """A real Redis on db 1, flushed before use. Skips the test when none is running."""
    r = make_redis(TEST_REDIS_URL)
    try:
        await r.ping()
    except Exception:
        await r.aclose()
        pytest.skip(f"no Redis at {TEST_REDIS_URL}")
    await r.flushdb()
    try:
        yield r
    finally:
        await r.flushdb()
        await r.aclose()


@pytest_asyncio.fixture
async def real_store(real_redis) -> RunStore:
    return RunStore(real_redis, lease_ttl_s=2)
