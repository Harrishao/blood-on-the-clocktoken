"""Canonical structured intents submitted by a player agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class IllegalAction(ValueError):
    """An action cannot be performed under the current game rules."""


class PlayerAction(BaseModel):
    """Base contract for every player-submitted intent."""

    actor: str


class SpeakPublic(PlayerAction):
    kind: Literal["speak_public"] = "speak_public"
    text: str


class RequestPrivateChat(PlayerAction):
    kind: Literal["request_private_chat"] = "request_private_chat"
    target_player: str


class RespondPrivateChat(PlayerAction):
    kind: Literal["respond_private_chat"] = "respond_private_chat"
    request_id: str
    accept: bool


class SpeakPrivate(PlayerAction):
    kind: Literal["speak_private"] = "speak_private"
    chat_id: str
    text: str


class LeavePrivateChat(PlayerAction):
    kind: Literal["leave_private_chat"] = "leave_private_chat"
    chat_id: str


class Nominate(PlayerAction):
    kind: Literal["nominate"] = "nominate"
    target: str
    accusation: str


class CastVote(PlayerAction):
    kind: Literal["cast_vote"] = "cast_vote"
    nomination_id: str
    vote: bool


class UseAbility(PlayerAction):
    kind: Literal["use_ability"] = "use_ability"
    action: str
    targets: tuple[str, ...] = ()


class UpdateNotebook(PlayerAction):
    kind: Literal["update_notebook"] = "update_notebook"
    patch: str


class YieldAction(PlayerAction):
    kind: Literal["yield_action"] = "yield_action"
    reason: str
