"""Streaming adapter for OpenAI-compatible Chat Completions providers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from clocktower.models.protocol import ModelCallError, ModelRequest, ModelSegment, SegmentKind


@dataclass(frozen=True, slots=True)
class SSERequest:
    """Internal HTTP request data for an injectable SSE transport."""

    url: str
    body: dict[str, Any]
    headers: dict[str, str]
    timeout_seconds: float


class SSETransport(Protocol):
    def stream(self, request: SSERequest) -> AsyncIterator[object]:
        """Yield decoded SSE data values or already-decoded JSON mappings."""


class HttpxSSETransport:
    """A small direct HTTPX implementation; no OpenAI SDK is involved."""

    async def stream(self, request: SSERequest) -> AsyncIterator[object]:
        timeout = httpx.Timeout(request.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", request.url, json=request.body, headers=request.headers) as response:
                if response.is_error:
                    raise httpx.HTTPStatusError(
                        "Provider returned an error response",
                        request=response.request,
                        response=response,
                    )
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        yield line[5:].strip()


@dataclass(slots=True)
class _PendingSegment:
    kind: SegmentKind
    source_field: str
    identity: object
    text: str


class OpenAICompatibleAdapter:
    """Parse standard Chat Completions deltas plus configured provider extensions."""

    def __init__(self, transport: SSETransport | None = None) -> None:
        self._transport = transport or HttpxSSETransport()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelSegment]:
        pending: _PendingSegment | None = None
        segment_index = 0

        def flush(*, incomplete: bool = False) -> ModelSegment | None:
            nonlocal pending, segment_index
            if pending is None:
                return None
            segment = ModelSegment(
                call_id=request.call_id,
                index=segment_index,
                kind=pending.kind,
                source_field=pending.source_field,
                text=pending.text,
                incomplete=incomplete,
            )
            segment_index += 1
            pending = None
            return segment

        try:
            async for raw_chunk in self._transport.stream(self._build_transport_request(request)):
                payload = self._decode_chunk(raw_chunk)
                if payload is None:
                    continue
                if payload is _DONE:
                    break
                if "error" in payload:
                    raise _ProviderPayloadError()

                for kind, source_field, identity, text, mergeable in self._chunk_parts(payload, request):
                    if mergeable and pending is not None and (
                        pending.kind,
                        pending.source_field,
                        pending.identity,
                    ) == (kind, source_field, identity):
                        pending.text += text
                        continue
                    flushed = flush()
                    if flushed is not None:
                        yield flushed
                    if mergeable:
                        pending = _PendingSegment(kind, source_field, identity, text)
                    else:
                        yield ModelSegment(
                            call_id=request.call_id,
                            index=segment_index,
                            kind=kind,
                            source_field=source_field,
                            text=text,
                        )
                        segment_index += 1
        except Exception as error:
            flushed = flush(incomplete=True)
            if flushed is not None:
                yield flushed
            raise self._safe_error(error) from None

        flushed = flush()
        if flushed is not None:
            yield flushed

    @staticmethod
    def _build_transport_request(request: ModelRequest) -> SSERequest:
        body: dict[str, Any] = {
            "model": request.model.name,
            "messages": [dict(message) for message in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            body["tools"] = [dict(tool) for tool in request.tools]
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if request.model.api_key:
            headers["Authorization"] = f"Bearer {request.model.api_key}"
        return SSERequest(
            url=f"{request.model.base_url.rstrip('/')}/chat/completions",
            body=body,
            headers=headers,
            timeout_seconds=request.timeout_seconds,
        )

    @staticmethod
    def _decode_chunk(raw_chunk: object) -> Mapping[str, Any] | object | None:
        if isinstance(raw_chunk, Mapping):
            return raw_chunk
        if isinstance(raw_chunk, bytes):
            raw_chunk = raw_chunk.decode("utf-8")
        if not isinstance(raw_chunk, str):
            raise _ProviderPayloadError()
        data = raw_chunk.strip()
        if not data or data.startswith(":") or data.startswith("event:") or data.startswith("retry:"):
            return None
        if data.startswith("data:"):
            data = data[5:].strip()
        if data == "[DONE]":
            return _DONE
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as error:
            raise _ProviderPayloadError() from error
        if not isinstance(decoded, dict):
            raise _ProviderPayloadError()
        return decoded

    @classmethod
    def _chunk_parts(
        cls,
        payload: Mapping[str, Any],
        request: ModelRequest,
    ) -> list[tuple[SegmentKind, str, object, str, bool]]:
        parts: list[tuple[SegmentKind, str, object, str, bool]] = []
        for field, value in payload.items():
            if field not in {"choices", "error"}:
                parts.append(("provider_metadata", field, field, cls._json_text(value), False))

        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise _ProviderPayloadError()
        for choice_index, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                raise _ProviderPayloadError()
            delta = choice.get("delta", {})
            if not isinstance(delta, Mapping):
                raise _ProviderPayloadError()
            for reasoning_field in request.model.reasoning_fields:
                text = delta.get(reasoning_field)
                if isinstance(text, str) and text:
                    parts.append(("reasoning", reasoning_field, reasoning_field, text, True))
            cls._append_tool_calls(parts, delta.get("tool_calls"), choice_index)
            cls._append_tool_results(parts, delta)
            content = delta.get("content")
            if isinstance(content, str) and content:
                parts.append(("final_message", "content", "content", content, True))
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                parts.append(("provider_metadata", "finish_reason", ("finish_reason", choice_index), cls._json_text(finish_reason), False))
        return parts

    @staticmethod
    def _append_tool_calls(
        parts: list[tuple[SegmentKind, str, object, str, bool]],
        tool_calls: object,
        choice_index: int,
    ) -> None:
        if tool_calls is None:
            return
        if not isinstance(tool_calls, list):
            raise _ProviderPayloadError()
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, Mapping):
                raise _ProviderPayloadError()
            tool_index = tool_call.get("index", position)
            function = tool_call.get("function", {})
            if not isinstance(function, Mapping):
                raise _ProviderPayloadError()
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments:
                parts.append(("tool_call", "tool_calls", (choice_index, tool_index), arguments, True))

    @staticmethod
    def _append_tool_results(
        parts: list[tuple[SegmentKind, str, object, str, bool]],
        delta: Mapping[str, Any],
    ) -> None:
        for field in ("tool_result", "tool_results"):
            if field in delta:
                parts.append(("tool_result", field, field, OpenAICompatibleAdapter._json_text(delta[field]), False))

    @staticmethod
    def _json_text(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _safe_error(error: Exception) -> ModelCallError:
        if isinstance(error, httpx.TimeoutException):
            return ModelCallError("Model request timed out")
        if isinstance(error, httpx.HTTPStatusError):
            return ModelCallError(f"Model provider returned HTTP status {error.response.status_code}")
        if isinstance(error, (ConnectionError, OSError)):
            return ModelCallError("Model stream interrupted")
        return ModelCallError("Model provider stream failed")


class _ProviderPayloadError(Exception):
    """Internal marker for an invalid provider payload, never exposed verbatim."""


_DONE = object()


__all__ = ["HttpxSSETransport", "ModelCallError", "OpenAICompatibleAdapter", "SSERequest", "SSETransport"]
