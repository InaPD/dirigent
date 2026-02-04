"""The Anthropic client wrapper.

Two rules live here.

Token counts always come from the response's own usage block. The moment anything is
estimated, every cost in every trace stops being trustworthy, and the trace is the point of
this project.

Every call is a structured output call. A node asks for a Pydantic schema and gets a
validated instance or an LLMError. No node ever parses free text.
"""

import logging
from time import perf_counter
from typing import TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from ra.schemas import Usage

log = logging.getLogger("ra.llm")

T = TypeVar("T", bound=BaseModel)

# Well under the 300s default wall clock, so a hung call cannot eat a whole run.
REQUEST_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 4096


class LLMError(RuntimeError):
    """A call that produced no usable answer. The node records it and the run continues."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class LLMResult[TModel: BaseModel](BaseModel):
    parsed: TModel
    usage: Usage
    model: str
    latency_ms: int


def usage_from_response(raw) -> Usage:
    """Copy the reported counts. Missing cache fields mean zero, never a guess."""
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
    )


def add_usage(*usages: Usage | None) -> Usage:
    """Sum the usage of several calls made inside one node, for one StepRecord."""
    total = Usage()
    for u in usages:
        if u is None:
            continue
        total = Usage(
            input_tokens=total.input_tokens + u.input_tokens,
            output_tokens=total.output_tokens + u.output_tokens,
            cache_creation_input_tokens=total.cache_creation_input_tokens
            + u.cache_creation_input_tokens,
            cache_read_input_tokens=total.cache_read_input_tokens + u.cache_read_input_tokens,
        )
    return total


class LLM:
    """Thin async wrapper. Stateless: it never touches RunState."""

    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    @classmethod
    def from_api_key(cls, api_key: str) -> "LLM":
        # max_retries is the SDK default of 2, which covers 429 and 5xx. Do not add another
        # retry layer on top; two stacked layers turn one slow call into a very slow one.
        return cls(AsyncAnthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S))

    async def structured[TModel: BaseModel](
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[TModel],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResult[TModel]:
        """One structured-output call. Raises LLMError when there is no usable answer.

        Thinking is left at the model default on purpose. Sonnet 5 runs adaptive thinking
        when the parameter is omitted and Haiku 4.5 runs without it, which is what these
        steps want. temperature is not passed; Sonnet 5 rejects it.
        """
        t0 = perf_counter()
        response = await self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        latency_ms = int((perf_counter() - t0) * 1000)
        usage = usage_from_response(response.usage)

        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            detail = getattr(getattr(response, "stop_details", None), "category", "unspecified")
            raise LLMError("refusal", f"the model declined ({detail})")
        if stop == "max_tokens":
            raise LLMError("max_tokens", f"answer truncated at {max_tokens} output tokens")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("unparsed", f"no {schema.__name__} in the response")

        log.debug("%s %s in %dms", model, schema.__name__, latency_ms)
        return LLMResult[schema](parsed=parsed, usage=usage, model=model, latency_ms=latency_ms)
