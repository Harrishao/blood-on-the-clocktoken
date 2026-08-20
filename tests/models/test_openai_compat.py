from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from clocktower.models.openai_compat import ModelCallError, OpenAICompatibleAdapter

from fakes import ScriptedSSETransport, delta, sample_request, tool_delta


@pytest.fixture
def mock_transport() -> ScriptedSSETransport:
    return ScriptedSSETransport()


def adapter(transport: ScriptedSSETransport) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(transport)


async def collect(adapter_: OpenAICompatibleAdapter, request):
    return [segment async for segment in adapter_.stream(request)]


async def test_reasoning_tool_and_content_order_is_preserved(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        delta(reasoning_content="Think A"),
        delta(tool_calls=[tool_delta("update_notebook", '{"notes":"x"}')]),
        delta(reasoning_content="Think B"),
        delta(content="Public text"),
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.source_field, segment.text) for segment in segments] == [
        ("reasoning", "reasoning_content", "Think A"),
        ("tool_call", "tool_calls", '{"notes":"x"}'),
        ("reasoning", "reasoning_content", "Think B"),
        ("final_message", "content", "Public text"),
    ]


async def test_configured_reasoning_fields_are_discovered_in_provider_order(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(thinking="second", reasoning_content="first")])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.source_field, segment.text) for segment in segments] == [
        ("reasoning_content", "first"),
        ("thinking", "second"),
    ]


async def test_contiguous_text_and_tool_argument_fragments_are_stitched(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        delta(content="Public "),
        delta(content="text"),
        delta(tool_calls=[tool_delta("speak_public", '{"text":"hel', index=0)]),
        delta(tool_calls=[tool_delta("speak_public", 'lo"}', index=0)]),
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.text) for segment in segments] == [
        ("final_message", "Public text"),
        ("tool_call", '{"text":"hello"}'),
    ]


async def test_multiple_tool_call_indexes_do_not_merge_into_each_other(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        delta(tool_calls=[tool_delta("first", "A", index=0), tool_delta("second", "B", index=1)]),
        delta(tool_calls=[tool_delta("first", "C", index=0)]),
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [segment.text for segment in segments] == ["A", "B", "C"]
    assert [segment.index for segment in segments] == [0, 1, 2]


async def test_usage_finish_reason_and_provider_metadata_are_exposed_without_headers(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        {"id": "chatcmpl-1", "model": "scripted-model", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
    ])

    segments = await collect(adapter(mock_transport), sample_request(api_key="super-secret"))

    assert [(segment.kind, segment.source_field, segment.text) for segment in segments] == [
        ("provider_metadata", "id", '"chatcmpl-1"'),
        ("provider_metadata", "model", '"scripted-model"'),
        ("provider_metadata", "finish_reason", '"tool_calls"'),
        ("provider_metadata", "usage", '{"completion_tokens":5,"prompt_tokens":3}'),
    ]
    assert all("super-secret" not in segment.text for segment in segments)
    request = mock_transport.requests[0]
    assert request.url == "https://provider.example/v1/chat/completions"
    assert request.body["stream"] is True


async def test_done_marker_ends_stream_without_an_output_segment(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(content="done"), "data: [DONE]", delta(content="ignored")])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.text) for segment in segments] == [("final_message", "done")]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ReadTimeout("Bearer super-secret"), "Model request timed out"),
        (httpx.HTTPStatusError("Authorization: Bearer super-secret", request=httpx.Request("POST", "https://provider.example"), response=httpx.Response(401)), "Model provider returned HTTP status 401"),
    ],
)
async def test_transport_errors_are_categorized_and_redacted(mock_transport: ScriptedSSETransport, error: Exception, message: str):
    mock_transport.script([error])

    with pytest.raises(ModelCallError, match=f"^{message}$") as caught:
        await collect(adapter(mock_transport), sample_request(api_key="super-secret"))

    assert "super-secret" not in str(caught.value)


async def test_interruption_flushes_the_buffered_segment_as_incomplete_then_raises(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(reasoning_content="partial"), ConnectionError("Bearer super-secret")])
    received = []

    with pytest.raises(ModelCallError, match="^Model stream interrupted$"):
        async for segment in adapter(mock_transport).stream(sample_request(api_key="super-secret")):
            received.append(segment)

    assert [(segment.kind, segment.text, segment.incomplete) for segment in received] == [
        ("reasoning", "partial", True),
    ]


async def test_nonstandard_tool_results_are_preserved_with_their_source_field(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(tool_result={"status": "ok"})])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.source_field, segment.text) for segment in segments] == [
        ("tool_result", "tool_result", '{"status":"ok"}'),
    ]
