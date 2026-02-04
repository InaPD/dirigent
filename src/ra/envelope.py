"""One response shape for every route, success or failure.

    {"ok": true,  "data": {...}, "error": null}
    {"ok": false, "data": null,  "error": "human readable reason"}

Error strings are written for a caller, never copied from an exception, so no internal
detail leaves the process.
"""

from typing import Any

from pydantic import BaseModel


class Envelope(BaseModel):
    ok: bool
    data: Any | None = None
    error: str | None = None


def ok(data: Any) -> dict:
    return Envelope(ok=True, data=data).model_dump()


def fail(error: str) -> dict:
    return Envelope(ok=False, data=None, error=error).model_dump()
