"""Stand-ins for the queue, the model and the search API."""

from ra.llm import LLMError, LLMResult
from ra.schemas import ToolCall, Usage
from ra.search import ExtractBatch, ExtractOutcome, Hit, SearchOutcome

FAKE_USAGE = Usage(input_tokens=100, output_tokens=20)


class FakePool:
    """Records enqueue_job calls instead of talking to arq."""

    def __init__(self) -> None:
        self.jobs: list[tuple[tuple, dict]] = []

    async def enqueue_job(self, *args, **kwargs):
        self.jobs.append((args, kwargs))
        return None

    async def aclose(self) -> None:
        return None


class FakeLLM:
    """Answers by schema. A queued Exception is raised instead of returned.

    The last queued item for a schema repeats, so a test only queues what it cares about.
    """

    def __init__(self, responses: dict | None = None, usage: Usage | None = None) -> None:
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.usage = usage or FAKE_USAGE
        self.calls: list[dict] = []

    async def structured(self, *, model, system, user, schema, max_tokens=4096):
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        queue = self.responses.get(schema)
        if not queue:
            raise LLMError("unparsed", f"no fake response queued for {schema.__name__}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return LLMResult[schema](parsed=item, usage=self.usage, model=model, latency_ms=1)

    def prompts_for(self, schema) -> list[str]:
        return [c["user"] for c in self.calls if c["schema"] is schema]


def hits(*urls: str, snippet: str = "a snippet") -> list[Hit]:
    return [Hit(url=u, title=f"Title {u}", content=snippet) for u in urls]


def search_outcome(*urls: str, status: str = "ok", credits: int = 1, snippet="a snippet"):
    found = hits(*urls, snippet=snippet)
    return SearchOutcome(
        status=status if found else ("empty" if status == "ok" else status),
        hits=found,
        credits=credits,
        tool_call=ToolCall(
            tool="tavily.search",
            args_digest="d" * 12,
            status=status,
            duration_ms=5,
            credits=credits,
        ),
    )


def extract_batch(statuses: dict[str, str], *, content: str = "x" * 400, credits: int = 1):
    return ExtractBatch(
        results=[
            ExtractOutcome(url=u, status=s, content=content if s == "full" else "")
            for u, s in statuses.items()
        ],
        credits=credits,
        tool_calls=[
            ToolCall(
                tool="tavily.extract",
                args_digest="e" * 12,
                status="ok",
                duration_ms=9,
                credits=credits,
            )
        ],
    )


class FakeSearch:
    """Serves queued search outcomes and extract batches, recording what was asked."""

    def __init__(self, searches=None, batches=None) -> None:
        self.searches = list(searches or [])
        self.batches = list(batches or [])
        self.queries: list[str] = []
        self.extracted: list[list[str]] = []

    async def search(self, query, *, depth="basic", max_results=5):
        self.queries.append(query)
        if not self.searches:
            return search_outcome(status="ok")
        return self.searches.pop(0) if len(self.searches) > 1 else self.searches[0]

    async def extract(self, urls):
        self.extracted.append(list(urls))
        if not self.batches:
            return ExtractBatch()
        return self.batches.pop(0) if len(self.batches) > 1 else self.batches[0]

    async def aclose(self):
        return None
