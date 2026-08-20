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
        "data: [DONE]",
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.source_field, segment.text) for segment in segments] == [
        ("reasoning", "reasoning_content", "Think A"),
        ("tool_call", "tool_calls", '{"notes":"x"}'),
        ("reasoning", "reasoning_content", "Think B"),
        ("final_message", "content", "Public text"),
    ]


async def test_configured_reasoning_fields_are_discovered_in_provider_order(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(thinking="second", reasoning_content="first"), "data: [DONE]"])

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
        "data: [DONE]",
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
        "data: [DONE]",
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [segment.text for segment in segments] == ["A", "B", "C"]
    assert [segment.index for segment in segments] == [0, 1, 2]


async def test_usage_finish_reason_and_provider_metadata_are_exposed_without_headers(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        {"id": "chatcmpl-1", "model": "scripted-model", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
        "data: [DONE]",
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
    mock_transport.script([delta(tool_result={"status": "ok"}), "data: [DONE]"])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.source_field, segment.text) for segment in segments] == [
        ("tool_result", "tool_result", '{"status":"ok"}'),
    ]


async def test_tool_identity_and_name_fragments_survive_parallel_tool_indexes(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        delta(tool_calls=[{
            "index": 0,
            "id": "tool-notebook",
            "type": "function",
            "function": {"name": "update_"},
        }]),
        delta(tool_calls=[{
            "index": 1,
            "id": "tool-public",
            "type": "function",
            "function": {"name": "speak_public", "arguments": '{"text":"hi"}'},
        }]),
        delta(tool_calls=[{
            "index": 0,
            "function": {"name": "notebook", "arguments": '{"notes":"x"}'},
        }]),
        "data: [DONE]",
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    tool_segments = [segment for segment in segments if segment.kind == "tool_call"]
    assert [(segment.text, segment.tool_index, segment.tool_call_id, segment.tool_type, segment.tool_name) for segment in tool_segments] == [
        ("", 0, "tool-notebook", "function", "update_"),
        ('{"text":"hi"}', 1, "tool-public", "function", "speak_public"),
        ('{"notes":"x"}', 0, "tool-notebook", "function", "update_notebook"),
    ]


async def test_unknown_and_nested_sensitive_metadata_never_reach_segments(mock_transport: ScriptedSSETransport):
    mock_transport.script([
        {
            "id": "chatcmpl-safe",
            "request_headers": {"Authorization": "Bearer super-secret"},
            "provider_debug": {"api_key": "super-secret"},
            "usage": {"prompt_tokens": 3, "nested": {"TOKEN": "super-secret"}},
            "choices": [],
        },
        "data: [DONE]",
    ])

    segments = await collect(adapter(mock_transport), sample_request(api_key="super-secret"))

    assert [(segment.source_field, segment.text) for segment in segments] == [
        ("id", '"chatcmpl-safe"'),
        ("usage", '{"nested":{},"prompt_tokens":3}'),
    ]
    assert all("super-secret" not in segment.text for segment in segments)
    assert all(segment.source_field not in {"request_headers", "provider_debug"} for segment in segments)


async def test_repeated_standard_envelopes_do_not_split_contiguous_content_or_emit_null_usage(mock_transport: ScriptedSSETransport):
    envelope = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "scripted-model"}
    mock_transport.script([
        {**envelope, "choices": [{"delta": {"content": "Hel"}}]},
        {**envelope, "choices": [{"delta": {"content": "lo"}}], "usage": None},
        "data: [DONE]",
    ])

    segments = await collect(adapter(mock_transport), sample_request())

    assert [(segment.kind, segment.text) for segment in segments if segment.kind == "final_message"] == [
        ("final_message", "Hello"),
    ]
    assert all(segment.source_field != "usage" for segment in segments)


async def test_eof_without_done_flushes_partial_text_then_raises_interruption(mock_transport: ScriptedSSETransport):
    mock_transport.script([delta(content="partial")])
    received = []

    with pytest.raises(ModelCallError, match="^Model stream interrupted$"):
        async for segment in adapter(mock_transport).stream(sample_request()):
            received.append(segment)

    assert [(segment.text, segment.incomplete) for segment in received] == [("partial", True)]


async def test_empty_sse_without_done_is_an_interruption(mock_transport: ScriptedSSETransport):
    mock_transport.script([])

    with pytest.raises(ModelCallError, match="^Model stream interrupted$"):
        await collect(adapter(mock_transport), sample_request())


async def test_httpx_read_error_is_an_interruption_and_is_redacted(mock_transport: ScriptedSSETransport):
    mock_transport.script([httpx.ReadError("Bearer super-secret")])

    with pytest.raises(ModelCallError, match="^Model stream interrupted$") as caught:
        await collect(adapter(mock_transport), sample_request(api_key="super-secret"))

    assert "super-secret" not in str(caught.value)
