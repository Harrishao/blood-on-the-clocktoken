"""Shared, read-only contracts for Trouble Brewing role handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from clocktower.domain.state import GameState, PlayerState


@dataclass(frozen=True, slots=True)
class AbilityContext:
    """The actor and immutable game reference needed to describe an ability.

    Role handlers do not mutate ``GameState``.  A later RuleEngine can select a
    legal result and apply the returned effect atomically.
    """

    state: GameState
    actor_id: str

    @classmethod
    def from_state(cls, state: GameState, actor_id: str) -> AbilityContext:
        if actor_id not in state.players:
            raise ValueError(f"unknown ability actor: {actor_id}")
        return cls(state=state, actor_id=actor_id)

    @property
    def actor(self) -> PlayerState:
        return self.state.players[self.actor_id]

    @property
    def is_misinformed(self) -> bool:
        """Whether the actor may receive a legal false information result."""

        return self.actor.role == "drunk" or "poisoned" in self.actor.reminders


@dataclass(frozen=True, slots=True)
class AbilityChoice:
    """A player intent that can be validated and resolved by the RuleEngine."""

    actor_id: str
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuleEffect:
    """A proposed, non-mutating rule result for the future RuleEngine."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    """One player-visible legal information result.

    ``truthful`` tells the Storyteller policy which results remain available
    when an information role is drunk or poisoned; handlers never choose a lie.
    ``grimoire`` is only populated for Spy and is explicitly recipient-scoped.
    """

    kind: str
    number: int | None = None
    yes: bool | None = None
    character: str | None = None
    player_ids: tuple[str, ...] = ()
    options: tuple[Observation, ...] = ()
    private_to: str | None = None
    grimoire: dict[str, Any] | None = None
    truthful: bool = True


class RoleHandler(Protocol):
    """Common handler shape without granting handlers mutation authority."""

    role: str
    first_night_order: int | None
    other_night_order: int | None

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]: ...

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]: ...
