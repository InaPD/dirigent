"""Turning an exception into something safe to store.

The trace is meant to be detailed, and an error with no detail in it is worth very little
when you are trying to understand a failed run. But the run document is readable over HTTP,
and an exception from a dependency can carry a connection string or a key in its text.

So: keep the type and the message, and redact the two shapes that carry secrets.
"""

import re

ERROR_LEN = 500

# scheme://user:password@host -> scheme://***@host
_CREDENTIALS_IN_URL = re.compile(r"(?P<scheme>\w+://)[^/\s:@]+:[^/\s@]+@")
# Anything that looks like one of our provider keys, wherever it turned up.
_API_KEY = re.compile(r"\b(sk-ant-|tvly-)[A-Za-z0-9_\-]+")


def redact(text: str) -> str:
    text = _CREDENTIALS_IN_URL.sub(r"\g<scheme>***@", text)
    return _API_KEY.sub(r"\1***", text)


def safe_detail(exc: BaseException, *, limit: int = ERROR_LEN) -> str:
    """A one-line description of a failure, fit to be read by whoever can read the run."""
    message = str(exc).strip() or exc.__class__.__name__
    return redact(f"{exc.__class__.__name__}: {message}")[:limit]
