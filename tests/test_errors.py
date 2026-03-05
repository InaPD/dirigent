"""Error text is stored in the run document, which is readable over HTTP."""

import pytest

from ra.errors import redact, safe_detail


@pytest.mark.parametrize(
    "text, must_not_contain",
    [
        ("connecting to redis://admin:hunter2@cache:6379", "hunter2"),
        ("bad key sk-ant-api03-AbC123_xyz", "AbC123_xyz"),
        ("tavily rejected tvly-dev-9f8e7d6c", "9f8e7d6c"),
        ("postgres://user:s3cret@db/app failed", "s3cret"),
    ],
)
def test_credentials_are_scrubbed(text, must_not_contain):
    assert must_not_contain not in redact(text)


def test_the_useful_part_survives():
    out = redact("connecting to redis://admin:hunter2@cache.internal:6379 timed out")

    assert "cache.internal:6379" in out
    assert "timed out" in out
    assert "redis://" in out


def test_ordinary_text_is_untouched():
    text = "the planner returned no sub-questions"
    assert redact(text) == text


def test_safe_detail_keeps_the_exception_type():
    assert safe_detail(TimeoutError("too slow")) == "TimeoutError: too slow"


def test_an_exception_with_no_message_still_says_something():
    assert safe_detail(ValueError()) == "ValueError: ValueError"


def test_safe_detail_redacts():
    detail = safe_detail(ConnectionError("redis://bob:pw123@host"))

    assert "pw123" not in detail
    assert detail.startswith("ConnectionError:")


def test_safe_detail_is_bounded():
    assert len(safe_detail(RuntimeError("x" * 5000))) <= 500


def test_a_key_in_an_exception_never_reaches_the_run_document():
    """The concrete case: a provider rejecting a key and echoing it back."""
    exc = RuntimeError("401 unauthorized for sk-ant-api03-LIVEKEYMATERIAL")

    stored = safe_detail(exc)

    assert "LIVEKEYMATERIAL" not in stored
    assert "401 unauthorized" in stored
