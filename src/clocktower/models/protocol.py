"""Provider-neutral model-call contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from clocktower.config import ResolvedModel


SegmentKind = Literal["reasoning", "tool_call", "tool_result", "final_message", "provider_metadata"]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """All provider-neutral inputs for one Chat Completions call."""

    call_id: str
    model: ResolvedModel
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class ModelSegment:
    """One ordered, raw provider output segment."""

    call_id: str
    index: int
    kind: SegmentKind
    source_field: str
    text: str
    incomplete: bool = False


class ModelCallError(RuntimeError):
    """A safe, provider-independent model-call failure."""


class ModelAdapter(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelSegment]:
        """Yield raw ordered segments for one model call."""
