"""Legal Trouble Brewing registration alternatives.

Registration is deliberately a set of possible facts, not a Storyteller
decision.  Consumers must select one of these alternatives through policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clocktower.rules.roles.base import AbilityContext
from clocktower.rules.setup import ROLE_CATEGORIES


class RegistrationQuery(StrEnum):
    ALIGNMENT = "alignment"
    CHARACTER = "character"


@dataclass(frozen=True, slots=True)
class Registration:
    player_id: str
    alignment: str
    character: str
    category: str
    truthful: bool


def registrations_for(
    player_id: str, query: RegistrationQuery, context: AbilityContext
) -> tuple[Registration, ...]:
    """Return every registration that a named player may legally provide.

    Query selects the information axis used by the caller while each returned
    value remains a complete registration for auditing and later policy choice.
    """

    try:
        player = context.state.players[player_id]
    except KeyError as error:
        raise ValueError(f"unknown registered player: {player_id}") from error

    true = Registration(
        player_id=player_id,
        alignment=player.alignment,
        character=player.role,
        category=_category_for(player.role),
        truthful=True,
    )
    alternatives: list[Registration] = [true]
    if player.role == "recluse":
        if query is RegistrationQuery.ALIGNMENT:
            alternatives.append(
                Registration(player_id, "evil", player.role, _category_for(player.role), False)
            )
        else:
            alternatives.extend(
                Registration(player_id, "evil", role, category, False)
                for role, category in ROLE_CATEGORIES.items()
                if category in {"minion", "demon"}
            )
    elif player.role == "spy":
        if query is RegistrationQuery.ALIGNMENT:
            alternatives.append(
                Registration(player_id, "good", player.role, _category_for(player.role), False)
            )
        else:
            alternatives.extend(
                Registration(player_id, "good", role, category, False)
                for role, category in ROLE_CATEGORIES.items()
                if category in {"townsfolk", "outsider"}
            )
    return _deduplicate(alternatives)


def _category_for(role: str) -> str:
    return ROLE_CATEGORIES.get(role, "unknown")


def _deduplicate(registrations: list[Registration]) -> tuple[Registration, ...]:
    unique: dict[tuple[str, str, str, str], Registration] = {}
    for registration in registrations:
        unique.setdefault(
            (
                registration.player_id,
                registration.alignment,
                registration.character,
                registration.category,
            ),
            registration,
        )
    return tuple(unique.values())
