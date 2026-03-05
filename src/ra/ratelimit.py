"""A small per-client token bucket.

In-process and per-worker, so it is a courtesy limit, not a guarantee. A real deployment
moves this to the gateway; the README says so. It is here because an unauthenticated
endpoint that starts paid model work should not be trivially floodable.
"""

import json
from collections import defaultdict
from time import monotonic

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ra.envelope import fail

WINDOW_S = 60.0
EXEMPT_PATHS = frozenset({"/healthz"})


class TokenBucket:
    """One bucket per client, with the idle ones thrown away.

    Without the eviction this grows for the life of the process: one entry per distinct
    client address ever seen, which anyone rotating source addresses can turn into a slow
    memory leak. A bucket that has been full for a whole window is indistinguishable from a
    fresh one, so forgetting it costs nothing.
    """

    def __init__(self, capacity: int, window_s: float = WINDOW_S) -> None:
        self.capacity = capacity
        self.window_s = window_s
        self.refill_per_s = capacity / window_s
        self._tokens: dict[str, float] = defaultdict(lambda: float(capacity))
        self._last: dict[str, float] = {}
        self._swept_at = monotonic()

    def allow(self, key: str) -> bool:
        t = monotonic()
        if t - self._swept_at > self.window_s:
            self._forget_idle(t)

        last = self._last.get(key, t)
        self._tokens[key] = min(self.capacity, self._tokens[key] + (t - last) * self.refill_per_s)
        self._last[key] = t
        if self._tokens[key] < 1.0:
            return False
        self._tokens[key] -= 1.0
        return True

    def _forget_idle(self, t: float) -> None:
        """Drop every bucket that has had time to refill completely."""
        self._swept_at = t
        idle = [key for key, seen in self._last.items() if t - seen > self.window_s]
        for key in idle:
            self._last.pop(key, None)
            self._tokens.pop(key, None)

    def tracked(self) -> int:
        return len(self._last)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, per_minute: int) -> None:
        super().__init__(app)
        self.bucket = TokenBucket(per_minute)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        if not self.bucket.allow(client):
            return JSONResponse(status_code=429, content=fail("rate limit exceeded"))

        return await call_next(request)


class BodySizeLimitMiddleware:
    """Refuse an oversized request body, counting the bytes that actually arrive.

    A Content-Length header is whatever the client says it is. A chunked request carries no
    Content-Length at all. So this buffers the body as it comes in and gives up the moment
    it passes the limit, rather than trusting a number in a header.

    Plain ASGI rather than BaseHTTPMiddleware, because the body has to be replayed to the
    application after being counted, and BaseHTTPMiddleware gives no way to do that.
    """

    METHODS_WITH_BODIES = frozenset({"POST", "PUT", "PATCH"})

    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in self.METHODS_WITH_BODIES:
            return await self.app(scope, receive, send)

        buffered: list[dict] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered.append(message)
                break
            total += len(message.get("body", b"") or b"")
            if total > self.max_bytes:
                return await _too_large(send)
            buffered.append(message)
            if not message.get("more_body"):
                break

        replay = iter(buffered)

        async def replayed():
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replayed, send)


async def _too_large(send) -> None:
    body = json.dumps(fail("request body too large")).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
