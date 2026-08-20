from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AudienceKind = Literal["public", "players", "player", "observer"]
SegmentKind = Literal[
    "reasoning",
    "tool_call",
    "tool_result",
    "final_message",
    "provider_metadata",
]


class Audience(BaseModel):
    """The players authorized to receive an event."""

    kind: AudienceKind
    player_ids: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def validate_recipients(self) -> Audience:
        if self.kind == "players" and not self.player_ids:
            raise ValueError("players audience requires at least one recipient")
        if self.kind == "player" and len(self.player_ids) != 1:
            raise ValueError("player audience requires exactly one recipient")
        if self.kind in {"public", "observer"} and self.player_ids:
            raise ValueError(f"{self.kind} audience cannot name player recipients")
        return self

    @classmethod
    def public(cls) -> Audience:
        return cls(kind="public")

    @classmethod
    def players(cls, player_ids: set[str] | frozenset[str]) -> Audience:
        return cls(kind="players", player_ids=frozenset(player_ids))

    @classmethod
    def player(cls, player_id: str) -> Audience:
        return cls(kind="player", player_ids=frozenset({player_id}))

    @classmethod
    def observer(cls) -> Audience:
        return cls(kind="observer")


class ModelOutputSegment(BaseModel):
    """An unmodified, ordered segment emitted by a model provider."""

    call_id: str
    player_id: str
    call_purpose: str
    segment_index: int
    kind: SegmentKind
    source_field: str
    text: str
    incomplete: bool = False
    tool_index: int | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_call_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_type: str | None = Field(default=None, exclude_if=lambda value: value is None)


class EventRecord(BaseModel):
    """A sequenced fact that can be stored and delivered to an audience."""

    seq: int = 0
    time: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    phase: str = ""
    type: str
    actor: str | None = None
    audience: Audience
    payload: dict[str, Any]

    def visible_to(self, player_id: str) -> bool:
        if self.audience.kind == "observer":
            return False
        return self.audience.kind == "public" or player_id in self.audience.player_ids
