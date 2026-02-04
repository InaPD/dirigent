"""A small per-client token bucket.

In-process and per-worker, so it is a courtesy limit, not a guarantee. A real deployment
moves this to the gateway; the README says so. It is here because an unauthenticated
endpoint that starts paid model work should not be trivially floodable.
"""

from collections import defaultdict
from time import monotonic

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ra.envelope import fail

WINDOW_S = 60.0
EXEMPT_PATHS = frozenset({"/healthz"})


class TokenBucket:
    def __init__(self, capacity: int, window_s: float = WINDOW_S) -> None:
        self.capacity = capacity
        self.refill_per_s = capacity / window_s
        self._tokens: dict[str, float] = defaultdict(lambda: float(capacity))
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        t = monotonic()
        last = self._last.get(key, t)
        self._tokens[key] = min(self.capacity, self._tokens[key] + (t - last) * self.refill_per_s)
        self._last[key] = t
        if self._tokens[key] < 1.0:
            return False
        self._tokens[key] -= 1.0
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, per_minute: int, max_body_bytes: int) -> None:
        super().__init__(app)
        self.bucket = TokenBucket(per_minute)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            return JSONResponse(status_code=413, content=fail("request body too large"))

        client = request.client.host if request.client else "unknown"
        if not self.bucket.allow(client):
            return JSONResponse(status_code=429, content=fail("rate limit exceeded"))

        return await call_next(request)
