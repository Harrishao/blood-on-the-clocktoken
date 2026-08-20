"""Authorization-first prompt projection for one AI player."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from clocktower.agents.tools import ToolDefinition, tools_for
from clocktower.domain.events import EventRecord
from clocktower.domain.state import GameState, Notebook


_OBSERVER_ONLY_EVENT_TYPES = frozenset(
    {
        "checkpoint",
        "game.header",
        "model.output_segment",
        "setup.completed",
        "storyteller.decision",
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


def _allowed_player_event(event: EventRecord) -> bool:
    return (
        event.type not in _OBSERVER_ONLY_EVENT_TYPES
        and not event.type.startswith("storyteller.")
    )


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
        if _allowed_player_event(event)
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


__all__ = ["PlayerContext", "project_context"]
