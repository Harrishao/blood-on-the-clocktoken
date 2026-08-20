from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field


class AttentionState(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    NEEDS_ATTENTION = "needs_attention"


class Notebook(BaseModel):
    notes: str = ""
    attention: AttentionState = AttentionState.IDLE


class PlayerState(BaseModel):
    player_id: str
    seat: int
    role: str
    perceived_identity: str
    alignment: str
    known_alignment: str
    alive: bool = True
    dead_vote_available: bool = True
    notebook: Notebook = Field(default_factory=Notebook)
    perceived_ability_text: str = ""
    reminders: set[str] = Field(default_factory=set)


class RoleState(BaseModel):
    """Mutable role-specific state; role handlers extend this contract later."""

    fortune_teller_red_herring: str | None = None


class GameState(BaseModel):
    """The in-process state for one game, deliberately separate from history."""

    ROLE_ALIGNMENTS: ClassVar[dict[str, str]] = {
        "washerwoman": "good",
        "librarian": "good",
        "investigator": "good",
        "chef": "good",
        "empath": "good",
        "fortune_teller": "good",
        "undertaker": "good",
        "monk": "good",
        "ravenkeeper": "good",
        "virgin": "good",
        "slayer": "good",
        "soldier": "good",
        "mayor": "good",
        "recluse": "good",
        "drunk": "good",
        "saint": "good",
        "butler": "good",
        "poisoner": "evil",
        "spy": "evil",
        "scarlet_woman": "evil",
        "baron": "evil",
        "imp": "evil",
        "townsfolk": "good",
        "outsider": "good",
        "minion": "evil",
        "demon": "evil",
    }
    TEST_PLAYER_IDS: ClassVar[tuple[str, ...]] = ("alice", "bob", "carol", "david", "eve")

    seed: int
    players: dict[str, PlayerState]
    day: int = 1
    phase: str = "setup"
    active_scene: str | None = None
    alive_count: int
    role_state: RoleState = Field(default_factory=RoleState)
    stopped: bool = False

    @classmethod
    def from_assignments(
        cls,
        assignments: dict[str, str],
        *,
        dead: set[str] | None = None,
        alive_count: int | None = None,
        seed: int,
    ) -> GameState:
        dead_players = dead or set()
        players = {
            player_id: PlayerState(
                player_id=player_id,
                seat=seat,
                role=role,
                perceived_identity=role,
                alignment=cls.ROLE_ALIGNMENTS.get(role, "good"),
                known_alignment=cls.ROLE_ALIGNMENTS.get(role, "good"),
                alive=player_id not in dead_players,
                dead_vote_available=player_id in dead_players,
            )
            for seat, (player_id, role) in enumerate(assignments.items())
        }
        current_alive_count = sum(player.alive for player in players.values())
        return cls(
            seed=seed,
            players=players,
            alive_count=current_alive_count if alive_count is None else alive_count,
        )

    @classmethod
    def from_alignments(cls, alignments: list[str], *, seed: int) -> GameState:
        player_ids = list(cls.TEST_PLAYER_IDS[: len(alignments)])
        if len(player_ids) < len(alignments):
            player_ids.extend(f"player-{seat + 1}" for seat in range(len(player_ids), len(alignments)))
        players = {
            player_id: PlayerState(
                player_id=player_id,
                seat=seat,
                role="unknown",
                perceived_identity="unknown",
                alignment=alignment,
                known_alignment=alignment,
            )
            for seat, (player_id, alignment) in enumerate(zip(player_ids, alignments, strict=True))
        }
        return cls(seed=seed, players=players, alive_count=len(players))
