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
    tool_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_type: str | None = None


@dataclass(frozen=True, slots=True)
class _SegmentPart:
    kind: SegmentKind
    source_field: str
    identity: object
    text: str
    mergeable: bool
    tool_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_type: str | None = None


@dataclass(slots=True)
class _ToolCallState:
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_type: str | None = None


class OpenAICompatibleAdapter:
    """Parse standard Chat Completions deltas plus configured provider extensions."""

    _METADATA_FIELDS = frozenset({"id", "object", "created", "model", "system_fingerprint", "service_tier", "usage"})
    _SENSITIVE_KEY_PARTS = frozenset({"authorization", "api_key", "apikey", "headers", "header", "cookie", "cookies", "token", "secret"})

    def __init__(self, transport: SSETransport | None = None) -> None:
        self._transport = transport or HttpxSSETransport()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelSegment]:
        pending: _PendingSegment | None = None
        segment_index = 0
        saw_done = False
        tool_states: dict[tuple[int, int], _ToolCallState] = {}
        metadata_queue: list[tuple[str, str]] = []
        seen_metadata: set[tuple[str, str]] = set()

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
                tool_index=pending.tool_index,
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                tool_type=pending.tool_type,
            )
            segment_index += 1
            pending = None
            return segment

        def flush_metadata() -> list[ModelSegment]:
            nonlocal segment_index
            segments = [
                ModelSegment(
                    call_id=request.call_id,
                    index=segment_index + offset,
                    kind="provider_metadata",
                    source_field=source_field,
                    text=text,
                )
                for offset, (source_field, text) in enumerate(metadata_queue)
            ]
            segment_index += len(segments)
            metadata_queue.clear()
            return segments

        try:
            async for raw_chunk in self._transport.stream(self._build_transport_request(request)):
                payload = self._decode_chunk(raw_chunk)
                if payload is None:
                    continue
                if payload is _DONE:
                    saw_done = True
                    break
                if "error" in payload:
                    raise _ProviderPayloadError()

                semantic_parts, metadata_parts = self._chunk_parts(payload, request, tool_states)
                for metadata in metadata_parts:
                    key = (metadata.source_field, metadata.text)
                    if key not in seen_metadata:
                        seen_metadata.add(key)
                        metadata_queue.append(key)
                for part in semantic_parts:
                    if part.mergeable and pending is not None and (
                        pending.kind,
                        pending.source_field,
                        pending.identity,
                    ) == (part.kind, part.source_field, part.identity):
                        pending.text += part.text
                        pending.tool_call_id = part.tool_call_id
                        pending.tool_name = part.tool_name
                        pending.tool_type = part.tool_type
                        continue
                    flushed = flush()
                    if flushed is not None:
                        yield flushed
                    if part.mergeable:
                        pending = _PendingSegment(
                            kind=part.kind,
                            source_field=part.source_field,
                            identity=part.identity,
                            text=part.text,
                            tool_index=part.tool_index,
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            tool_type=part.tool_type,
                        )
                    else:
                        yield ModelSegment(
                            call_id=request.call_id,
                            index=segment_index,
                            kind=part.kind,
                            source_field=part.source_field,
                            text=part.text,
                            tool_index=part.tool_index,
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            tool_type=part.tool_type,
                        )
                        segment_index += 1
        except Exception as error:
            flushed = flush(incomplete=True)
            if flushed is not None:
                yield flushed
            for metadata in flush_metadata():
                yield metadata
            raise self._safe_error(error) from None

        if not saw_done:
            flushed = flush(incomplete=True)
            if flushed is not None:
                yield flushed
            for metadata in flush_metadata():
                yield metadata
            raise ModelCallError("Model stream interrupted")

        flushed = flush()
        if flushed is not None:
            yield flushed
        for metadata in flush_metadata():
            yield metadata

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
        tool_states: dict[tuple[int, int], _ToolCallState],
    ) -> tuple[list[_SegmentPart], list[_SegmentPart]]:
        semantic_parts: list[_SegmentPart] = []
        metadata_parts: list[_SegmentPart] = []
        for field, value in payload.items():
            if field in cls._METADATA_FIELDS:
                text = cls._safe_metadata_text(value)
                if text is not None:
                    metadata_parts.append(_SegmentPart("provider_metadata", field, field, text, False))

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
                    semantic_parts.append(_SegmentPart("reasoning", reasoning_field, reasoning_field, text, True))
            cls._append_tool_calls(semantic_parts, delta.get("tool_calls"), choice_index, tool_states)
            cls._append_tool_results(semantic_parts, delta)
            content = delta.get("content")
            if isinstance(content, str) and content:
                semantic_parts.append(_SegmentPart("final_message", "content", "content", content, True))
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                text = cls._safe_metadata_text(finish_reason)
                if text is not None:
                    metadata_parts.append(_SegmentPart("provider_metadata", "finish_reason", ("finish_reason", choice_index), text, False))
        return semantic_parts, metadata_parts

    @staticmethod
    def _append_tool_calls(
        parts: list[_SegmentPart],
        tool_calls: object,
        choice_index: int,
        tool_states: dict[tuple[int, int], _ToolCallState],
    ) -> None:
        if tool_calls is None:
            return
        if not isinstance(tool_calls, list):
            raise _ProviderPayloadError()
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, Mapping):
                raise _ProviderPayloadError()
            tool_index = tool_call.get("index", position)
            if not isinstance(tool_index, int) or isinstance(tool_index, bool):
                raise _ProviderPayloadError()
            state = tool_states.setdefault((choice_index, tool_index), _ToolCallState())
            if isinstance(tool_call.get("id"), str):
                state.tool_call_id = OpenAICompatibleAdapter._merge_identity_fragment(state.tool_call_id, tool_call["id"])
            if isinstance(tool_call.get("type"), str):
                state.tool_type = OpenAICompatibleAdapter._merge_identity_fragment(state.tool_type, tool_call["type"])
            function = tool_call.get("function")
            arguments = ""
            if function is not None:
                if not isinstance(function, Mapping):
                    raise _ProviderPayloadError()
                if isinstance(function.get("name"), str):
                    state.tool_name = OpenAICompatibleAdapter._merge_identity_fragment(state.tool_name, function["name"])
                if isinstance(function.get("arguments"), str):
                    arguments = function["arguments"]
            parts.append(_SegmentPart(
                kind="tool_call",
                source_field="tool_calls",
                identity=(choice_index, tool_index),
                text=arguments,
                mergeable=True,
                tool_index=tool_index,
                tool_call_id=state.tool_call_id,
                tool_name=state.tool_name,
                tool_type=state.tool_type,
            ))

    @staticmethod
    def _append_tool_results(parts: list[_SegmentPart], delta: Mapping[str, Any]) -> None:
        for field in ("tool_result", "tool_results"):
            if field in delta:
                text = OpenAICompatibleAdapter._safe_metadata_text(delta[field])
                if text is not None:
                    parts.append(_SegmentPart("tool_result", field, field, text, False))

    @classmethod
    def _safe_metadata_text(cls, value: object) -> str | None:
        if value is None:
            return None
        safe_value = cls._scrub_sensitive_metadata(value)
        try:
            return json.dumps(safe_value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _scrub_sensitive_metadata(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): cls._scrub_sensitive_metadata(nested)
                for key, nested in value.items()
                if isinstance(key, str) and not cls._is_sensitive_key(key)
            }
        if isinstance(value, list):
            return [cls._scrub_sensitive_metadata(item) for item in value]
        return value

    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        normalized = "".join(character for character in key.lower() if character.isalnum())
        key_parts = {part for part in key.lower().replace("-", "_").split("_") if part}
        return (
            normalized in cls._SENSITIVE_KEY_PARTS
            or any(normalized.startswith(part) or normalized.endswith(part) for part in cls._SENSITIVE_KEY_PARTS)
            or bool(key_parts & cls._SENSITIVE_KEY_PARTS)
        )

    @staticmethod
    def _merge_identity_fragment(current: str | None, incoming: str) -> str:
        if not current:
            return incoming
        if incoming == current or current.startswith(incoming):
            return current
        if incoming.startswith(current):
            return incoming
        return current + incoming

    @staticmethod
    def _safe_error(error: Exception) -> ModelCallError:
        if isinstance(error, httpx.TimeoutException):
            return ModelCallError("Model request timed out")
        if isinstance(error, httpx.HTTPStatusError):
            return ModelCallError(f"Model provider returned HTTP status {error.response.status_code}")
        if isinstance(error, (httpx.NetworkError, ConnectionError, OSError)):
            return ModelCallError("Model stream interrupted")
        return ModelCallError("Model provider stream failed")


class _ProviderPayloadError(Exception):
    """Internal marker for an invalid provider payload, never exposed verbatim."""


_DONE = object()


__all__ = ["HttpxSSETransport", "ModelCallError", "OpenAICompatibleAdapter", "SSERequest", "SSETransport"]
