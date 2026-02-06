"""Tavily. Every failure must become a status, and nothing may raise at the caller."""

import json

import httpx
import pytest
import respx

from ra.search import (
    MIN_FULL_CONTENT_CHARS,
    TAVILY_BASE,
    ExtractOutcome,
    Hit,
    Tavily,
    resolve_extraction,
)

SEARCH_URL = f"{TAVILY_BASE}/search"
EXTRACT_URL = f"{TAVILY_BASE}/extract"

LONG_BODY = "x" * (MIN_FULL_CONTENT_CHARS + 50)


@pytest.fixture
def tavily(store) -> Tavily:
    return Tavily("tvly-test", store)


def search_body(urls: list[str], *, credits: int | None = 1) -> dict:
    body = {
        "query": "q",
        "results": [
            {"title": f"Title {u}", "url": u, "content": f"snippet for {u}", "score": 0.9}
            for u in urls
        ],
        "response_time": 0.4,
    }
    if credits is not None:
        body["usage"] = {"credits": credits}
    return body


# -- search --------------------------------------------------------------------


@respx.mock
async def test_a_search_returns_hits_and_counts_credits(tavily: Tavily):
    respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=search_body(["https://a.test", "https://b.test"]))
    )

    outcome = await tavily.search("durable agents")

    assert outcome.status == "ok"
    assert [h.url for h in outcome.hits] == ["https://a.test", "https://b.test"]
    assert outcome.hits[0].content == "snippet for https://a.test"
    assert outcome.credits == 1
    assert outcome.tool_call.tool == "tavily.search"
    assert outcome.tool_call.credits == 1


@respx.mock
async def test_credits_are_read_from_the_response_not_assumed(tavily: Tavily):
    respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=search_body(["https://a.test"], credits=7))
    )

    assert (await tavily.search("q")).credits == 7


@respx.mock
async def test_credits_fall_back_to_the_documented_rate(tavily: Tavily):
    respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=search_body(["https://a.test"], credits=None))
    )

    assert (await tavily.search("q", depth="advanced")).credits == 2


@respx.mock
async def test_the_request_uses_basic_depth_and_a_bearer_token(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=search_body(["https://a.test"]))
    )

    await tavily.search("durable agents", max_results=4)

    request = route.calls.last.request
    sent = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer tvly-test"
    assert sent["search_depth"] == "basic"  # advanced costs double
    assert sent["max_results"] == 4
    assert sent["include_usage"] is True


@respx.mock
async def test_empty_results_are_a_status_not_an_error(tavily: Tavily):
    respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=search_body([])))

    outcome = await tavily.search("nothing matches this")

    assert outcome.status == "empty"
    assert outcome.hits == []


@respx.mock
async def test_a_rate_limit_is_honoured_once_then_succeeds(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json=search_body(["https://a.test"])),
        ]
    )

    outcome = await tavily.search("q")

    assert outcome.status == "ok"
    assert route.call_count == 2


@respx.mock
async def test_a_second_rate_limit_gives_up(tavily: Tavily):
    respx.post(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(429, headers={"retry-after": "0"}),
        ]
    )

    outcome = await tavily.search("q")

    assert outcome.status == "rate_limited"
    assert outcome.hits == []


@respx.mock
async def test_an_unreasonable_retry_after_is_not_waited_out(tavily: Tavily):
    """A node cannot afford to sleep for a minute inside a five minute run."""
    route = respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "600"})
    )

    outcome = await tavily.search("q")

    assert outcome.status == "error"
    assert route.call_count == 1


@respx.mock
async def test_a_timeout_is_a_status(tavily: Tavily):
    respx.post(SEARCH_URL).mock(side_effect=httpx.ReadTimeout("too slow"))

    outcome = await tavily.search("q")

    assert outcome.status == "timeout"
    assert outcome.tool_call.status == "timeout"


@respx.mock
@pytest.mark.parametrize("code", [400, 401, 403, 500, 503])
async def test_any_http_failure_is_a_status(tavily: Tavily, code):
    respx.post(SEARCH_URL).mock(return_value=httpx.Response(code, json={"detail": "no"}))

    outcome = await tavily.search("q")

    assert outcome.status == "error"
    assert str(code) in outcome.error


@respx.mock
async def test_a_connection_failure_is_a_status(tavily: Tavily):
    respx.post(SEARCH_URL).mock(side_effect=httpx.ConnectError("no route"))

    assert (await tavily.search("q")).status == "error"


@respx.mock
async def test_malformed_json_is_a_status(tavily: Tavily):
    respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, content=b"not json"))

    assert (await tavily.search("q")).status == "error"


@respx.mock
async def test_a_repeated_search_is_served_from_cache_and_costs_nothing(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=search_body(["https://a.test"]))
    )

    first = await tavily.search("durable agents")
    second = await tavily.search("durable agents")

    assert route.call_count == 1
    assert first.cached is False and first.credits == 1
    assert second.cached is True and second.credits == 0
    assert [h.url for h in second.hits] == [h.url for h in first.hits]


@respx.mock
async def test_a_failed_search_is_not_cached(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=search_body(["https://a.test"])),
        ]
    )

    assert (await tavily.search("q")).status == "error"
    assert (await tavily.search("q")).status == "ok"
    assert route.call_count == 2


# -- extract -------------------------------------------------------------------


@respx.mock
async def test_extraction_splits_into_full_and_paywalled(tavily: Tavily):
    respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://full.test", "raw_content": LONG_BODY},
                    {"url": "https://thin.test", "raw_content": "Subscribe to read"},
                ],
                "failed_results": [],
                "usage": {"credits": 1},
            },
        )
    )

    batch = await tavily.extract(["https://full.test", "https://thin.test"])
    by_url = batch.by_url()

    assert by_url["https://full.test"].status == "full"
    assert by_url["https://thin.test"].status == "paywalled"
    assert batch.credits == 1


@respx.mock
async def test_a_failed_result_becomes_an_error_status(tavily: Tavily):
    respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [],
                "failed_results": [{"url": "https://bad.test", "error": "forbidden"}],
            },
        )
    )

    batch = await tavily.extract(["https://bad.test"])

    assert batch.by_url()["https://bad.test"].status == "error"
    assert "forbidden" in batch.by_url()["https://bad.test"].error


@respx.mock
async def test_a_url_the_api_ignored_is_still_accounted_for(tavily: Tavily):
    respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(200, json={"results": [], "failed_results": []})
    )

    batch = await tavily.extract(["https://ghost.test"])

    assert batch.by_url()["https://ghost.test"].status == "error"


@respx.mock
async def test_a_whole_batch_timeout_marks_every_url(tavily: Tavily):
    respx.post(EXTRACT_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    batch = await tavily.extract(["https://a.test", "https://b.test"])

    assert {r.status for r in batch.results} == {"timeout"}
    assert batch.tool_calls[0].status == "timeout"


@respx.mock
async def test_extraction_is_cached_per_url(tavily: Tavily):
    route = respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"url": "https://a.test", "raw_content": LONG_BODY}],
                "failed_results": [],
                "usage": {"credits": 1},
            },
        )
    )

    first = await tavily.extract(["https://a.test"])
    second = await tavily.extract(["https://a.test"])

    assert route.call_count == 1
    assert first.credits == 1
    assert second.credits == 0
    assert second.by_url()["https://a.test"].status == "full"


@respx.mock
async def test_only_the_uncached_urls_are_requested(tavily: Tavily):
    route = respx.post(EXTRACT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [{"url": "https://a.test", "raw_content": LONG_BODY}],
                    "failed_results": [],
                },
            ),
            httpx.Response(
                200,
                json={
                    "results": [{"url": "https://b.test", "raw_content": LONG_BODY}],
                    "failed_results": [],
                },
            ),
        ]
    )

    await tavily.extract(["https://a.test"])
    await tavily.extract(["https://a.test", "https://b.test"])

    assert json.loads(route.calls[1].request.content)["urls"] == ["https://b.test"]


@respx.mock
async def test_duplicate_urls_are_asked_for_once(tavily: Tavily):
    route = respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"url": "https://a.test", "raw_content": LONG_BODY}],
                "failed_results": [],
            },
        )
    )

    await tavily.extract(["https://a.test", "https://a.test"])

    assert json.loads(route.calls.last.request.content)["urls"] == ["https://a.test"]


async def test_an_empty_url_list_makes_no_request(tavily: Tavily):
    batch = await tavily.extract([])

    assert batch.results == []
    assert batch.tool_calls == []


# -- resolve_extraction --------------------------------------------------------


def hit(content: str = "") -> Hit:
    return Hit(url="https://a.test", title="T", content=content)


def test_a_full_extraction_is_used_as_is():
    extract = ExtractOutcome(url="https://a.test", status="full", content=LONG_BODY)
    assert resolve_extraction(extract, hit("snippet")) == ("full", LONG_BODY)


@pytest.mark.parametrize("failure", ["paywalled", "timeout", "error"])
def test_a_failed_extraction_falls_back_to_the_search_snippet(failure):
    """A JS-heavy page is not a dead end. The snippet is a legitimate, weaker source."""
    extract = ExtractOutcome(url="https://a.test", status=failure)
    status, content = resolve_extraction(extract, hit("the snippet"))

    assert status == "snippet_only"
    assert content == "the snippet"


@pytest.mark.parametrize("failure", ["paywalled", "timeout", "error"])
def test_a_failure_with_no_snippet_keeps_its_reason(failure):
    extract = ExtractOutcome(url="https://a.test", status=failure)
    assert resolve_extraction(extract, hit("")) == (failure, "")


def test_a_url_with_no_extraction_at_all():
    assert resolve_extraction(None, hit("")) == ("error", "")
    assert resolve_extraction(None, hit("snippet")) == ("snippet_only", "snippet")


@respx.mock
async def test_a_rate_limit_with_no_header_still_retries_once(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=search_body(["https://a.test"])),
        ]
    )

    outcome = await tavily.search("q")

    assert outcome.status == "ok"
    assert route.call_count == 2


@respx.mock
async def test_a_nonsense_retry_after_is_not_retried(tavily: Tavily):
    route = respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    assert (await tavily.search("q")).status == "error"
    assert route.call_count == 1


@respx.mock
async def test_results_without_a_url_are_ignored(tavily: Tavily):
    """A malformed entry must not become a source with no provenance."""
    respx.post(SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"title": "no url here", "content": "text"}, {"url": "https://a.test"}]
            },
        )
    )

    outcome = await tavily.search("q")

    assert [h.url for h in outcome.hits] == ["https://a.test"]


@respx.mock
async def test_extract_entries_without_a_url_are_ignored(tavily: Tavily):
    respx.post(EXTRACT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"raw_content": LONG_BODY},
                    {"url": "https://a.test", "raw_content": LONG_BODY},
                ],
                "failed_results": [{"error": "no url"}],
            },
        )
    )

    batch = await tavily.extract(["https://a.test"])

    assert [r.url for r in batch.results] == ["https://a.test"]


async def test_the_client_can_be_closed(store):
    tavily = Tavily("tvly-test", store)
    await tavily.aclose()
