"""Authorization-first prompt projection for one AI player."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from clocktower.agents.tools import ToolDefinition, tools_for
from clocktower.domain.events import EventRecord
from clocktower.domain.state import GameState, Notebook


class PlayerEventVisibility(StrEnum):
    """Minimum audience class required before an event may enter a prompt."""

    PUBLIC = "public"
    PLAYER_PRIVATE = "player_private"
    OBSERVER = "observer"


_PUBLIC_EVENT_TYPES = frozenset(
    {
        "chat.public_message",
        "chat.private_ended",
        "chat.private_started",
        "claim.public",
        "day.ended",
        "execution.none",
        "execution.resolved",
        "game.ended",
        "night.deaths_announced",
        "nomination.closed",
        "nomination.opened",
        "player.public_message",
        "player.yielded",
        "vote.cast",
        "vote.resolved",
    }
)

_PLAYER_PRIVATE_EVENT_TYPES = frozenset(
    {
        "ability.choice_requested",
        "chat.private_invitation",
        "chat.private_message",
        "chat.private_response",
        "evil.info_received",
        "information.received",
        "notebook.updated",
        "role.assigned",
        "role.change_notified",
        "role.changed_private",
        "tool.error",
    }
)

_OBSERVER_EVENT_TYPES = frozenset(
    {
        "ability.no_effect",
        "butler.master_set",
        "checkpoint",
        "death.failed",
        "death.prevented",
        "death.redirected",
        "effect.suppressed",
        "game.header",
        "model.output_segment",
        "poison.applied",
        "protection.applied",
        "role.transformed",
        "setup.completed",
        "storyteller.decision",
        "vote.rule_resolved",
    }
)


class PlayerContext(BaseModel):
    """The complete and deliberately narrow knowledge supplied to one player."""

    model_config = ConfigDict(frozen=True)

    identity: str
    alignment: str
    ability_text: str
    notebook: Notebook
    events: tuple[EventRecord, ...]
    tools: tuple[ToolDefinition, ...]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    def tool_schemas(self) -> tuple[dict[str, object], ...]:
        return tuple(tool.as_openai_tool() for tool in self.tools)


def _event_visibility(event: EventRecord) -> PlayerEventVisibility | None:
    if event.type in _PUBLIC_EVENT_TYPES:
        return PlayerEventVisibility.PUBLIC
    if event.type in _PLAYER_PRIVATE_EVENT_TYPES:
        return PlayerEventVisibility.PLAYER_PRIVATE
    if event.type in _OBSERVER_EVENT_TYPES or event.type.startswith("storyteller."):
        return PlayerEventVisibility.OBSERVER
    if event.type == "player.died":
        return (
            PlayerEventVisibility.PUBLIC
            if event.phase.startswith("day")
            else PlayerEventVisibility.OBSERVER
        )
    if event.type == "ability.used":
        return (
            PlayerEventVisibility.PUBLIC
            if event.payload.get("ability") == "slayer"
            else PlayerEventVisibility.OBSERVER
        )
    return None


def _allowed_player_event(event: EventRecord, player_id: str) -> bool:
    visibility = _event_visibility(event)
    if visibility is PlayerEventVisibility.PUBLIC:
        return event.audience.kind == "public"
    if visibility is PlayerEventVisibility.PLAYER_PRIVATE:
        return (
            event.audience.kind in {"player", "players"}
            and player_id in event.audience.player_ids
        )
    return False


def is_safe_public_event(event: EventRecord) -> bool:
    """Return true only for an explicitly classified public fact with public audience."""

    return _event_visibility(event) is PlayerEventVisibility.PUBLIC and event.audience.kind == "public"


def project_context(
    player_id: str,
    state: GameState,
    events: Sequence[EventRecord],
) -> PlayerContext:
    """Filter authorization before copying or serializing any event payload."""

    if player_id not in state.players:
        raise ValueError(f"unknown player: {player_id}")

    authorized = tuple(event for event in events if event.visible_to(player_id))
    visible = tuple(
        event.model_copy(deep=True)
        for event in authorized
        if _allowed_player_event(event, player_id)
    )
    player = state.players[player_id]
    return PlayerContext(
        identity=player.perceived_identity,
        alignment=player.known_alignment,
        ability_text=player.perceived_ability_text,
        notebook=player.notebook.model_copy(deep=True),
        events=visible,
        tools=tools_for(state.phase, player),
    )


__all__ = ["PlayerContext", "PlayerEventVisibility", "is_safe_public_event", "project_context"]
