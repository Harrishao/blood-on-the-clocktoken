from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState


def sample_game_state(
    *,
    roles: dict[str, str] | None = None,
    dead: set[str] | None = None,
    alive_count: int | None = None,
) -> GameState:
    assigned = roles or {
        "alice": "washerwoman",
        "bob": "chef",
        "carol": "recluse",
        "david": "poisoner",
        "eve": "imp",
    }
    return GameState.from_assignments(assigned, dead=dead or set(), alive_count=alive_count, seed=17)


def sample_voting_state(*, dead: set[str] | None = None) -> GameState:
    return sample_game_state(dead=dead)


def game_with_roles(**roles: str) -> GameState:
    return sample_game_state(roles=roles)


def game_with_seats(alignments: list[str]) -> GameState:
    return GameState.from_alignments(alignments, seed=17)


def public_claim(*, actor: str, mentions: set[str]) -> EventRecord:
    return EventRecord(
        phase="day.discussion",
        type="claim.public",
        actor=actor,
        audience=Audience(kind="public"),
        payload={"mentions": sorted(mentions)},
    )


def private_message(participants: set[str]) -> EventRecord:
    return EventRecord(
        phase="day.private",
        type="chat.private_message",
        actor=None,
        audience=Audience(kind="players", player_ids=frozenset(participants)),
        payload={},
    )
