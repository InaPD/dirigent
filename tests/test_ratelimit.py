"""The bucket itself, so the HTTP test does not have to care about timing."""

from ra.ratelimit import TokenBucket


def test_bucket_allows_up_to_capacity():
    bucket = TokenBucket(capacity=3)
    assert [bucket.allow("a") for _ in range(4)] == [True, True, True, False]


def test_buckets_are_per_client():
    bucket = TokenBucket(capacity=1)
    assert bucket.allow("a") is True
    assert bucket.allow("b") is True
    assert bucket.allow("a") is False


def test_bucket_refills_over_time(monkeypatch):
    import ra.ratelimit as mod

    clock = {"t": 0.0}
    monkeypatch.setattr(mod, "monotonic", lambda: clock["t"])

    bucket = TokenBucket(capacity=60, window_s=60.0)  # one token per second
    for _ in range(60):
        assert bucket.allow("a") is True
    assert bucket.allow("a") is False

    clock["t"] = 2.0
    assert bucket.allow("a") is True
    assert bucket.allow("a") is True
    assert bucket.allow("a") is False
