from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from clocktower.config import ResolvedModel


class ScriptedSSETransport:
    """A socket-free transport that yields one scripted provider chunk at a time."""

    def __init__(self, chunks: Iterable[object] = ()) -> None:
        self.chunks = list(chunks)
        self.requests: list[object] = []

    def script(self, chunks: Iterable[object]) -> None:
        self.chunks = list(chunks)

    async def stream(self, request: object) -> AsyncIterator[object]:
        self.requests.append(request)
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def delta(**fields: Any) -> dict[str, Any]:
    return {"choices": [{"delta": fields}]}


def tool_delta(name: str, arguments: str, *, index: int = 0, call_id: str | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name, "arguments": arguments}
    tool_call: dict[str, Any] = {"index": index, "type": "function", "function": function}
    if call_id is not None:
        tool_call["id"] = call_id
    return tool_call


def sample_request(*, api_key: str | None = "not-a-real-key", reasoning_fields: tuple[str, ...] = ("reasoning_content", "thinking")):
    from clocktower.models.protocol import ModelRequest

    return ModelRequest(
        call_id="call-42",
        model=ResolvedModel(
            provider="scripted",
            name="scripted-model",
            base_url="https://provider.example/v1",
            api_key_env="SCRIPTED_KEY",
            api_key=api_key,
            reasoning_fields=reasoning_fields,
            source="models.global",
        ),
        messages=({"role": "user", "content": "continue the game"},),
    )
