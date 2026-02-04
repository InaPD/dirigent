"""Single source of the current time, so tests can freeze it in one place."""

from datetime import UTC, datetime


def now() -> datetime:
    """Timezone-aware current time. Always use this, never datetime.now()."""
    return datetime.now(UTC)
