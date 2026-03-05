"""Tavily search and extraction, with every failure turned into a status.

Nothing in here raises at the caller. A 429, a timeout, a paywall and an empty result set
all come back as a status on a record, because the supervisor is supposed to see statuses,
never stack traces.

Request and response shapes verified against the Tavily API reference on 14 Sep 2026.
Credits are read from the response's own usage block when the API returns one, and only
computed from the documented rates when it does not.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ra.schemas import ExtractionStatus, ToolCall
from ra.store import RunStore

log = logging.getLogger("ra.search")

TAVILY_BASE = "https://api.tavily.com"
SEARCH_PATH = "/search"
EXTRACT_PATH = "/extract"

DEFAULT_DEPTH = "basic"
DEFAULT_MAX_RESULTS = 5
EXTRACT_BATCH_MAX = 20
REQUEST_TIMEOUT_S = 30.0

# Below this, a 200 response is a cookie wall or a teaser, not an article.
MIN_FULL_CONTENT_CHARS = 200

# A 429 is honoured once. Anything longer than this is not worth a node's time.
RETRY_AFTER_CAP_S = 10.0

# Documented rates, used only when the response carries no usage block.
CREDITS_PER_SEARCH = {"basic": 1, "fast": 1, "ultra-fast": 1, "advanced": 2}
EXTRACT_URLS_PER_CREDIT = 5

CACHE_TTL_S = 7 * 24 * 60 * 60
# A timeout or an empty result set is a fact about one moment, not about the web. Caching
# one for a week turns a blip into a week of the same wrong answer.
FAILURE_CACHE_TTL_S = 10 * 60

SearchDepth = Literal["basic", "advanced"]
CallStatus = Literal["ok", "empty", "rate_limited", "timeout", "error"]


def _digest(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()


class Hit(BaseModel):
    """One search result, before anything has been extracted from it."""

    url: str
    title: str | None = None
    content: str = ""  # the snippet Tavily returns with the result
    score: float | None = None


class SearchOutcome(BaseModel):
    status: CallStatus
    hits: list[Hit] = Field(default_factory=list)
    credits: int = 0
    cached: bool = False
    error: str | None = None
    tool_call: ToolCall


class ExtractOutcome(BaseModel):
    """What came back for one URL. Status describes the request, not the final quality."""

    url: str
    status: ExtractionStatus
    content: str = ""
    error: str | None = None


class ExtractBatch(BaseModel):
    results: list[ExtractOutcome] = Field(default_factory=list)
    credits: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def by_url(self) -> dict[str, ExtractOutcome]:
        return {r.url: r for r in self.results}


def resolve_extraction(extract: ExtractOutcome | None, hit: Hit) -> tuple[ExtractionStatus, str]:
    """Decide what content a finding may cite, and how good it is.

    The status describes the content we ended up holding, which is what the writer needs in
    order to prefer one source over another. Why the extraction fell short is recorded on
    the ToolCall, not here.
    """
    if extract is not None and extract.status == "full":
        return "full", extract.content

    snippet = (hit.content or "").strip()
    if snippet:
        return "snippet_only", snippet

    if extract is None:
        return "error", ""
    return extract.status, ""


class Tavily:
    """Search and extraction, cached in Redis and metered in credits."""

    def __init__(self, api_key: str, store: RunStore, client: httpx.AsyncClient | None = None):
        self._key = api_key
        self._store = store
        self._client = client or httpx.AsyncClient(base_url=TAVILY_BASE, timeout=REQUEST_TIMEOUT_S)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- HTTP ---------------------------------------------------------------------

    async def _post(self, path: str, payload: dict) -> tuple[CallStatus, dict | None, str | None]:
        """One request, with a single retry on 429. Never raises."""
        headers = {"Authorization": f"Bearer {self._key}"}
        for attempt in (1, 2):
            try:
                response = await self._client.post(path, json=payload, headers=headers)
            except httpx.TimeoutException:
                return "timeout", None, "request timed out"
            except httpx.HTTPError as exc:
                return "error", None, type(exc).__name__

            if response.status_code == 429 and attempt == 1:
                delay = _retry_after_seconds(response.headers.get("retry-after"))
                if delay is None:
                    return "error", None, "rate limited"
                log.warning("tavily rate limited, waiting %.1fs", delay)
                await asyncio.sleep(delay)
                continue
            if response.status_code == 429:
                return "rate_limited", None, "rate limited"
            if response.status_code >= 400:
                return "error", None, f"http {response.status_code}"

            try:
                return "ok", response.json(), None
            except ValueError:
                return "error", None, "malformed json"

        return "error", None, "rate limited"

    # -- search -------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        depth: SearchDepth = DEFAULT_DEPTH,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> SearchOutcome:
        """Basic depth by default. Advanced costs double and rarely earns it here."""
        key = f"cache:search:{_digest(query, depth, max_results)}"
        cached = await self._store.cache_get(key)
        if cached is not None:
            body = json.loads(cached)
            hits = _hits_from(body)
            return SearchOutcome(
                status="ok" if hits else "empty",
                hits=hits,
                credits=0,
                cached=True,
                tool_call=ToolCall(
                    tool="tavily.search",
                    args_digest=key[-12:],
                    status="cached",
                    duration_ms=0,
                    credits=0,
                ),
            )

        payload = {
            "query": query,
            "search_depth": depth,
            "max_results": max_results,
            "include_usage": True,
        }
        started = asyncio.get_running_loop().time()
        status, body, error = await self._post(SEARCH_PATH, payload)
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        credits = 0
        hits: list[Hit] = []
        if status == "ok" and body is not None:
            credits = _credits_from(body, default=CREDITS_PER_SEARCH.get(depth, 1))
            hits = _hits_from(body)
            if not hits:
                status = "empty"
            await self._store.cache_set(
                key,
                json.dumps(body),
                ttl_s=CACHE_TTL_S if hits else FAILURE_CACHE_TTL_S,
            )

        return SearchOutcome(
            status=status,
            hits=hits,
            credits=credits,
            cached=False,
            error=error,
            tool_call=ToolCall(
                tool="tavily.search",
                args_digest=key[-12:],
                status=status,
                duration_ms=duration_ms,
                credits=credits,
            ),
        )

    # -- extract ------------------------------------------------------------------

    async def extract(self, urls: list[str]) -> ExtractBatch:
        """Fetch page bodies, serving what is already cached and asking for the rest."""
        batch = ExtractBatch()
        wanted = list(dict.fromkeys(urls))[:EXTRACT_BATCH_MAX]
        missing: list[str] = []

        for url in wanted:
            cached = await self._store.cache_get(f"cache:extract:{_digest(url)}")
            if cached is None:
                missing.append(url)
                continue
            payload = json.loads(cached)
            batch.results.append(
                ExtractOutcome(
                    url=url,
                    status=payload["status"],
                    content=payload.get("content", ""),
                )
            )
        if batch.results:
            batch.tool_calls.append(
                ToolCall(
                    tool="tavily.extract",
                    args_digest=_digest(*[r.url for r in batch.results])[:12],
                    status="cached",
                    duration_ms=0,
                    credits=0,
                )
            )
        if not missing:
            return batch

        started = asyncio.get_running_loop().time()
        status, body, error = await self._post(
            EXTRACT_PATH, {"urls": missing, "extract_depth": "basic", "format": "markdown"}
        )
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        credits = 0
        if status == "ok" and body is not None:
            credits = _credits_from(
                body, default=max(1, -(-len(missing) // EXTRACT_URLS_PER_CREDIT))
            )
            fetched = await self._store_extracted(body, missing)
            batch.results.extend(fetched)
        else:
            # The whole request failed, so every URL in it carries that failure.
            failure: ExtractionStatus = "timeout" if status == "timeout" else "error"
            batch.results.extend(
                ExtractOutcome(url=url, status=failure, error=error) for url in missing
            )

        batch.credits += credits
        batch.tool_calls.append(
            ToolCall(
                tool="tavily.extract",
                args_digest=_digest(*missing)[:12],
                status=status,
                duration_ms=duration_ms,
                credits=credits,
            )
        )
        return batch

    async def _store_extracted(self, body: dict, asked_for: list[str]) -> list[ExtractOutcome]:
        outcomes: list[ExtractOutcome] = []
        seen: set[str] = set()

        for item in body.get("results") or []:
            url = item.get("url")
            if not url:
                continue
            seen.add(url)
            content = (item.get("raw_content") or "").strip()
            status: ExtractionStatus = (
                "full" if len(content) >= MIN_FULL_CONTENT_CHARS else "paywalled"
            )
            outcomes.append(ExtractOutcome(url=url, status=status, content=content))

        for item in body.get("failed_results") or []:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            outcomes.append(
                ExtractOutcome(url=url, status="error", error=str(item.get("error"))[:200])
            )

        # A URL the API mentioned in neither list simply did not come back.
        for url in asked_for:
            if url not in seen:
                outcomes.append(ExtractOutcome(url=url, status="error", error="no result"))

        for outcome in outcomes:
            await self._store.cache_set(
                f"cache:extract:{_digest(outcome.url)}",
                json.dumps({"status": outcome.status, "content": outcome.content}),
                ttl_s=CACHE_TTL_S if outcome.status == "full" else FAILURE_CACHE_TTL_S,
            )
        return outcomes


def _retry_after_seconds(header: str | None) -> float | None:
    """Honour Retry-After, but never wait longer than a node can afford."""
    if header is None:
        return 1.0
    try:
        delay = float(header)
    except ValueError:
        return None
    if delay > RETRY_AFTER_CAP_S:
        return None
    return max(0.0, delay)


def _credits_from(body: dict, *, default: int) -> int:
    usage = body.get("usage") or {}
    reported = usage.get("credits")
    return int(reported) if isinstance(reported, int | float) else default


def _hits_from(body: dict) -> list[Hit]:
    hits: list[Hit] = []
    for item in body.get("results") or []:
        url = item.get("url")
        if not url:
            continue
        hits.append(
            Hit(
                url=url,
                title=item.get("title"),
                content=item.get("content") or "",
                score=item.get("score"),
            )
        )
    return hits
