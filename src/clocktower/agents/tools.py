"""Phase-scoped player tool schemas and canonical intent parsing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clocktower.domain.actions import (
    CastVote,
    LeavePrivateChat,
    Nominate,
    PlayerAction,
    RequestPrivateChat,
    RespondPrivateChat,
    SpeakPrivate,
    SpeakPublic,
    UpdateNotebook,
    UseAbility,
    YieldAction,
)
from clocktower.domain.state import PlayerState


class ToolDefinition(BaseModel):
    """A provider-neutral function tool exposed to a player model."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object_schema(
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_TOOLS = {
    "speak_public": ToolDefinition(
        name="speak_public",
        description="Propose one public message to the active scene.",
        parameters=_object_schema({"text": {"type": "string"}}, ("text",)),
    ),
    "request_private_chat": ToolDefinition(
        name="request_private_chat",
        description="Request a private conversation with one player.",
        parameters=_object_schema(
            {"target_player": {"type": "string"}}, ("target_player",)
        ),
    ),
    "respond_private_chat": ToolDefinition(
        name="respond_private_chat",
        description="Accept or reject the pending private-chat invitation.",
        parameters=_object_schema(
            {
                "request_id": {"type": "string"},
                "accept": {"type": "boolean"},
            },
            ("request_id", "accept"),
        ),
    ),
    "speak_private": ToolDefinition(
        name="speak_private",
        description="Propose one message inside the active private chat.",
        parameters=_object_schema(
            {"chat_id": {"type": "string"}, "text": {"type": "string"}},
            ("chat_id", "text"),
        ),
    ),
    "leave_private_chat": ToolDefinition(
        name="leave_private_chat",
        description="Leave the active private chat.",
        parameters=_object_schema({"chat_id": {"type": "string"}}, ("chat_id",)),
    ),
    "nominate": ToolDefinition(
        name="nominate",
        description="Nominate one player and provide a public accusation.",
        parameters=_object_schema(
            {
                "target_player": {"type": "string"},
                "accusation": {"type": "string"},
            },
            ("target_player", "accusation"),
        ),
    ),
    "cast_vote": ToolDefinition(
        name="cast_vote",
        description="Cast the requested vote in the active nomination.",
        parameters=_object_schema(
            {
                "nomination_id": {"type": "string"},
                "vote": {"type": "boolean"},
            },
            ("nomination_id", "vote"),
        ),
    ),
    "use_ability": ToolDefinition(
        name="use_ability",
        description="Submit the requested role ability choice.",
        parameters=_object_schema(
            {
                "action": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
            },
            ("action", "targets"),
        ),
    ),
    "update_notebook": ToolDefinition(
        name="update_notebook",
        description="Replace your private notebook with text and structured scheduling attention.",
        parameters=_object_schema(
            {
                "notebook": {
                    "type": "object",
                    "properties": {
                        "notes": {"type": "string"},
                        "attention": {
                            "type": "object",
                            "properties": {
                                "players": {"type": "array", "items": {"type": "string"}},
                                "pending_actions": {"type": "array", "items": {"type": "string"}},
                                "watch_triggers": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["players", "pending_actions", "watch_triggers"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["notes", "attention"],
                    "additionalProperties": False,
                }
            },
            ("notebook",),
        ),
    ),
    "yield_action": ToolDefinition(
        name="yield_action",
        description="End this action opportunity without another game action.",
        parameters=_object_schema({"reason": {"type": "string"}}, ("reason",)),
    ),
}


_PHASE_TOOLS: dict[str, tuple[str, ...]] = {
    "day.discussion": (
        "speak_public",
        "request_private_chat",
        "nominate",
        "update_notebook",
        "yield_action",
    ),
    "day.private_invite": (
        "respond_private_chat",
        "update_notebook",
        "yield_action",
    ),
    "day.private": (
        "speak_private",
        "leave_private_chat",
        "update_notebook",
        "yield_action",
    ),
    "day.nomination_response": (
        "speak_public",
        "update_notebook",
        "yield_action",
    ),
    "day.voting": ("cast_vote", "update_notebook", "yield_action"),
    "night": ("use_ability", "update_notebook", "yield_action"),
}


def tools_for(phase: str, player: PlayerState) -> tuple[ToolDefinition, ...]:
    """Return only tools explicitly authorized for this phase and player state."""

    names = list(_PHASE_TOOLS.get(phase, ()))
    if not player.alive and "nominate" in names:
        names.remove("nominate")
    if not player.alive and not player.dead_vote_available and "cast_vote" in names:
        names.remove("cast_vote")
    return tuple(_TOOLS[name] for name in names)


class ToolIntentError(ValueError):
    """A model tool call is unknown, malformed, or unauthorized in this scene."""


def _exact_arguments(
    arguments: dict[str, Any],
    *,
    required: frozenset[str],
) -> None:
    keys = frozenset(arguments)
    if keys != required:
        missing = sorted(required - keys)
        unexpected = sorted(keys - required)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ToolIntentError("invalid arguments: " + ", ".join(details))


def _validate_json_schema_types(name: str, arguments: dict[str, Any]) -> None:
    properties = _TOOLS[name].parameters["properties"]
    for key, value in arguments.items():
        expected = properties[key]["type"]
        valid = (
            (expected == "string" and isinstance(value, str))
            or (expected == "boolean" and type(value) is bool)
            or (expected == "object" and isinstance(value, dict))
            or (
                expected == "array"
                and isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            )
        )
        if not valid:
            raise ToolIntentError(f"invalid type for {key}: expected {expected}")


def parse_tool_intent(
    name: str,
    arguments: dict[str, Any],
    *,
    player_id: str,
    allowed_tools: frozenset[str],
) -> PlayerAction:
    """Convert one authorized tool call into a canonical, self-scoped action."""

    if name not in _TOOLS:
        raise ToolIntentError(f"unknown tool: {name}")
    if name not in allowed_tools:
        raise ToolIntentError(f"tool is not available in this phase: {name}")

    constructors: dict[str, tuple[frozenset[str], Any]] = {
        "speak_public": (frozenset({"text"}), lambda data: SpeakPublic(actor=player_id, **data)),
        "request_private_chat": (
            frozenset({"target_player"}),
            lambda data: RequestPrivateChat(actor=player_id, **data),
        ),
        "respond_private_chat": (
            frozenset({"request_id", "accept"}),
            lambda data: RespondPrivateChat(actor=player_id, **data),
        ),
        "speak_private": (
            frozenset({"chat_id", "text"}),
            lambda data: SpeakPrivate(actor=player_id, **data),
        ),
        "leave_private_chat": (
            frozenset({"chat_id"}),
            lambda data: LeavePrivateChat(actor=player_id, **data),
        ),
        "nominate": (
            frozenset({"target_player", "accusation"}),
            lambda data: Nominate(
                actor=player_id,
                target=data["target_player"],
                accusation=data["accusation"],
            ),
        ),
        "cast_vote": (
            frozenset({"nomination_id", "vote"}),
            lambda data: CastVote(actor=player_id, **data),
        ),
        "use_ability": (
            frozenset({"action", "targets"}),
            lambda data: UseAbility(
                actor=player_id,
                action=data["action"],
                targets=tuple(data["targets"]),
            ),
        ),
        "update_notebook": (
            frozenset({"notebook"}),
            lambda data: UpdateNotebook(actor=player_id, **data),
        ),
        "yield_action": (
            frozenset({"reason"}),
            lambda data: YieldAction(actor=player_id, **data),
        ),
    }
    required, constructor = constructors[name]
    _exact_arguments(arguments, required=required)
    _validate_json_schema_types(name, arguments)
    try:
        return constructor(arguments)
    except (KeyError, TypeError, ValidationError) as error:
        raise ToolIntentError(f"invalid arguments for {name}") from error


__all__ = ["ToolDefinition", "ToolIntentError", "parse_tool_intent", "tools_for"]
