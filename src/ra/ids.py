"""Short identifiers. They appear in traces and in prompts, so keep them small."""

import secrets


def new_run_id() -> str:
    """run_ plus 12 hex characters."""
    return "run_" + secrets.token_hex(6)


def new_finding_id() -> str:
    """f_ plus 6 hex characters."""
    return "f_" + secrets.token_hex(3)


def sq_id(n: int) -> str:
    """Sub-question id, one-based: sq_01, sq_02, ..."""
    return f"sq_{n:02d}"
