"""Deterministic Trouble Brewing setup validation and seat assignment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


SETUP_COUNTS: dict[int, tuple[int, int, int, int]] = {
    5: (3, 0, 1, 1),
    6: (3, 1, 1, 1),
    7: (5, 0, 1, 1),
    8: (5, 1, 1, 1),
    9: (5, 2, 1, 1),
    10: (7, 0, 2, 1),
    11: (7, 1, 2, 1),
    12: (7, 2, 2, 1),
    13: (9, 0, 3, 1),
    14: (9, 1, 3, 1),
    15: (9, 2, 3, 1),
}

ROLE_CATEGORIES: dict[str, str] = {
    "washerwoman": "townsfolk",
    "librarian": "townsfolk",
    "investigator": "townsfolk",
    "chef": "townsfolk",
    "empath": "townsfolk",
    "fortune_teller": "townsfolk",
    "undertaker": "townsfolk",
    "monk": "townsfolk",
    "ravenkeeper": "townsfolk",
    "virgin": "townsfolk",
    "slayer": "townsfolk",
    "soldier": "townsfolk",
    "mayor": "townsfolk",
    "recluse": "outsider",
    "drunk": "outsider",
    "saint": "outsider",
    "butler": "outsider",
    "poisoner": "minion",
    "spy": "minion",
    "scarlet_woman": "minion",
    "baron": "minion",
    "imp": "demon",
}

_CATEGORY_ORDER = ("townsfolk", "outsider", "minion", "demon")


@dataclass(frozen=True, slots=True)
class SeatAssignment:
    """One role's deterministic position in the circle."""

    seat: int
    role: str


@dataclass(frozen=True, slots=True)
class SetupResult:
    """A validated role list in seeded seat order, without mutating game state."""

    player_count: int
    category_counts: tuple[int, int, int, int]
    roles_by_seat: tuple[str, ...]

    @property
    def seats(self) -> tuple[SeatAssignment, ...]:
        return tuple(SeatAssignment(seat, role) for seat, role in enumerate(self.roles_by_seat))


def setup_counts(player_count: int) -> tuple[int, int, int, int]:
    """Return the official Townsfolk/Outsider/Minion/Demon counts."""

    try:
        return SETUP_COUNTS[player_count]
    except KeyError as error:
        raise ValueError("Trouble Brewing requires 5 to 15 players") from error


def build_setup(player_count: int, selected_roles: Sequence[str], seed: int) -> SetupResult:
    """Validate a Storyteller-selected role set and put it into seeded seats."""

    base_counts = setup_counts(player_count)
    roles = tuple(selected_roles)
    if len(roles) != player_count:
        raise ValueError(f"selected_roles must contain exactly {player_count} roles")
    if len(set(roles)) != len(roles):
        raise ValueError("selected_roles cannot contain duplicate roles")

    unknown_roles = sorted(set(roles).difference(ROLE_CATEGORIES))
    if unknown_roles:
        raise ValueError(f"unknown Trouble Brewing roles: {', '.join(unknown_roles)}")

    counts = tuple(sum(ROLE_CATEGORIES[role] == category for role in roles) for category in _CATEGORY_ORDER)
    expected_counts = _baron_adjusted_counts(base_counts, roles)
    if counts != expected_counts:
        raise ValueError(
            "selected_roles category counts do not match official setup: "
            f"expected {expected_counts}, got {counts}"
        )

    roles_by_seat = list(roles)
    random.Random(seed).shuffle(roles_by_seat)
    return SetupResult(
        player_count=player_count,
        category_counts=counts,
        roles_by_seat=tuple(roles_by_seat),
    )


def _baron_adjusted_counts(
    base_counts: tuple[int, int, int, int], roles: tuple[str, ...]
) -> tuple[int, int, int, int]:
    townsfolk, outsiders, minions, demons = base_counts
    if "baron" in roles:
        return townsfolk - 2, outsiders + 2, minions, demons
    return base_counts
