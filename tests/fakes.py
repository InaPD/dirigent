"""Stand-ins for things Phase 1 does not have yet, or does not want in a unit test."""


class FakePool:
    """Records enqueue_job calls instead of talking to arq."""

    def __init__(self) -> None:
        self.jobs: list[tuple[tuple, dict]] = []

    async def enqueue_job(self, *args, **kwargs):
        self.jobs.append((args, kwargs))
        return None

    async def aclose(self) -> None:
        return None
