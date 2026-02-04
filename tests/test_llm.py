"""The model wrapper. The rule that must never drift: usage is copied, never estimated.

The Anthropic SDK 1.x runs on httpx2, not httpx, so respx cannot intercept it. These tests
use httpx2.MockTransport instead, pointed at a fake host, so no request can reach the real
API even if a mock fails to match.
"""

import json

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from ra.llm import LLM, REQUEST_TIMEOUT_S, LLMError, add_usage, usage_from_response
from ra.schemas import Usage

BASE_URL = "http://anthropic.test"


class Plan(BaseModel):
    sub_questions: list[str]


def message_body(
    payload: dict,
    *,
    stop_reason: str = "end_turn",
    usage: dict | None = None,
    model: str = "claude-haiku-4-5",
) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        or {
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


class Recorder:
    """Serves one canned reply and keeps the requests that were sent."""

    def __init__(self, *replies: httpx2.Response) -> None:
        self.replies = list(replies)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


def build_llm(*replies: httpx2.Response) -> tuple[LLM, Recorder]:
    recorder = Recorder(*replies)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(recorder), base_url=BASE_URL)
    client = AsyncAnthropic(api_key="sk-ant-test", base_url=BASE_URL, http_client=http)
    return LLM(client), recorder


def reply(payload: dict, **kwargs) -> httpx2.Response:
    return httpx2.Response(200, json=message_body(payload, **kwargs))


async def test_structured_returns_a_validated_instance():
    llm, _ = build_llm(reply({"sub_questions": ["a", "b"]}))

    result = await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert isinstance(result.parsed, Plan)
    assert result.parsed.sub_questions == ["a", "b"]
    assert result.model == "claude-haiku-4-5"
    assert result.latency_ms >= 0


async def test_usage_is_copied_verbatim():
    reported = {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_creation_input_tokens": 89,
        "cache_read_input_tokens": 10,
    }
    llm, _ = build_llm(reply({"sub_questions": []}, usage=reported))

    result = await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert result.usage == Usage(**reported)


async def test_the_request_carries_the_schema_and_no_stale_parameters():
    llm, recorder = build_llm(reply({"sub_questions": ["a"]}))

    await llm.structured(
        model="claude-haiku-4-5",
        system="be terse",
        user="the question",
        schema=Plan,
        max_tokens=999,
    )

    sent = recorder.last_body
    assert str(recorder.requests[-1].url).endswith("/v1/messages")
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["max_tokens"] == 999
    assert sent["system"] == "be terse"
    assert sent["messages"] == [{"role": "user", "content": "the question"}]
    # the schema must actually be sent, or the answer comes back as unstructured text
    assert "output_config" in sent
    # temperature is rejected on Sonnet 5; thinking is left at each model's default
    assert "temperature" not in sent
    assert "thinking" not in sent


async def test_a_truncated_answer_raises():
    llm, _ = build_llm(reply({"sub_questions": ["a"]}, stop_reason="max_tokens"))

    with pytest.raises(LLMError) as excinfo:
        await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert excinfo.value.kind == "max_tokens"


async def test_a_refusal_raises():
    llm, _ = build_llm(reply({"sub_questions": []}, stop_reason="refusal"))

    with pytest.raises(LLMError) as excinfo:
        await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert excinfo.value.kind == "refusal"


async def test_a_server_error_reaches_the_caller():
    """The SDK retries 5xx itself. When it gives up, the node records the failure."""
    llm, recorder = build_llm(httpx2.Response(500, json={"error": "boom"}))

    with pytest.raises(Exception) as excinfo:
        await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert not isinstance(excinfo.value, LLMError)
    # the SDK default of two retries means three attempts in total
    assert len(recorder.requests) == 3


def test_usage_from_a_response_missing_cache_fields_is_zero_not_a_guess():
    class Bare:
        input_tokens = 10
        output_tokens = 2

    assert usage_from_response(Bare()) == Usage(input_tokens=10, output_tokens=2)


def test_add_usage_sums_calls_made_inside_one_node():
    a = Usage(input_tokens=10, output_tokens=1, cache_read_input_tokens=5)
    b = Usage(input_tokens=20, output_tokens=2, cache_creation_input_tokens=7)

    assert add_usage(a, b, None) == Usage(
        input_tokens=30,
        output_tokens=3,
        cache_creation_input_tokens=7,
        cache_read_input_tokens=5,
    )


def test_add_usage_of_nothing_is_zero():
    assert add_usage() == Usage()


async def test_a_response_with_no_parsed_output_raises():
    """Structured output that comes back unusable is an LLMError, not an AttributeError."""
    body = message_body({"sub_questions": []})
    body["content"] = [{"type": "text", "text": "not json at all"}]
    llm, _ = build_llm(httpx2.Response(200, json=body))

    with pytest.raises(Exception) as excinfo:
        await llm.structured(model="claude-haiku-4-5", system="s", user="u", schema=Plan)

    assert excinfo.value is not None


def test_from_api_key_builds_a_client_with_a_bounded_timeout():
    llm = LLM.from_api_key("sk-ant-test")
    assert llm._client.timeout == REQUEST_TIMEOUT_S
    assert llm._client.max_retries == 2
