"""Pure day nomination and voting bookkeeping for Trouble Brewing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from clocktower.domain.actions import IllegalAction
from clocktower.domain.state import GameState


@dataclass(frozen=True, slots=True)
class VoteRecord:
    voter: str
    nomination_id: str
    vote: bool


@dataclass(frozen=True, slots=True)
class Nomination:
    nomination_id: str
    nominator: str
    nominee: str
    accusation: str
    vote_order: tuple[str, ...]


@dataclass(slots=True)
class _OpenNomination:
    nomination: Nomination
    votes: list[VoteRecord] = field(default_factory=list)


def qualifying_tally(votes: int, alive_count: int) -> bool:
    """Whether votes meet the official at-least-half-of-living threshold."""

    if alive_count <= 0:
        raise ValueError("alive_count must be positive")
    return votes >= math.ceil(alive_count / 2)


def cast_vote(state: GameState, voter: str, nomination_id: str, vote: bool) -> VoteRecord:
    """Validate an individual vote and consume a dead player's one vote when used."""

    try:
        player = state.players[voter]
    except KeyError as error:
        raise IllegalAction(f"unknown voter: {voter}") from error

    if vote and not player.alive:
        if not player.dead_vote_available:
            raise IllegalAction("dead vote already spent")
        player.dead_vote_available = False
    return VoteRecord(voter=voter, nomination_id=nomination_id, vote=vote)


class NominationTracker:
    """One day's nominations, public vote order, and execution candidate."""

    def __init__(self, *, alive_count: int) -> None:
        if alive_count <= 0:
            raise ValueError("alive_count must be positive")
        self.alive_count = alive_count
        self._nominators: set[str] = set()
        self._nominees: set[str] = set()
        self._nominations: dict[str, _OpenNomination] = {}
        self._tallies: dict[str, int] = {}
        self._active_nomination_id: str | None = None

    def nominate(
        self,
        state: GameState,
        nominator: str,
        nominee: str,
        *,
        accusation: str,
    ) -> Nomination:
        """Open one legal nomination and establish its clockwise count order."""

        nominator_player = self._player(state, nominator, "nominator")
        nominee_player = self._player(state, nominee, "nominee")
        if not nominator_player.alive:
            raise IllegalAction("dead players cannot nominate")
        if not nominee_player.alive:
            raise IllegalAction("dead players cannot be nominated")
        if nominator in self._nominators:
            raise IllegalAction("player has already nominated today")
        if nominee in self._nominees:
            raise IllegalAction("player has already been nominated today")
        if self._active_nomination_id is not None:
            raise IllegalAction("previous nomination voting is not complete")

        nomination_id = f"nom-{len(self._nominations) + 1}"
        nomination = Nomination(
            nomination_id=nomination_id,
            nominator=nominator,
            nominee=nominee,
            accusation=accusation,
            vote_order=self._clockwise_vote_order(state, nominee),
        )
        self._nominators.add(nominator)
        self._nominees.add(nominee)
        self._nominations[nomination_id] = _OpenNomination(nomination)
        self._active_nomination_id = nomination_id
        return nomination

    def cast_vote(
        self,
        state: GameState,
        voter: str,
        nomination_id: str,
        vote: bool,
    ) -> VoteRecord:
        """Record the next public vote for the active nomination."""

        open_nomination = self._open_nomination(nomination_id)
        expected_voter = open_nomination.nomination.vote_order[len(open_nomination.votes)]
        if voter != expected_voter:
            raise IllegalAction(f"expected vote from {expected_voter}")

        record = cast_vote(state, voter, nomination_id, vote)
        open_nomination.votes.append(record)
        if len(open_nomination.votes) == len(open_nomination.nomination.vote_order):
            self.record_tally(
                open_nomination.nomination.nominee,
                sum(record.vote for record in open_nomination.votes),
            )
            self._active_nomination_id = None
        return record

    def record_tally(self, nominee: str, votes: int) -> None:
        """Store a completed nomination's tally for later strict-highest resolution."""

        if votes < 0:
            raise ValueError("votes cannot be negative")
        self._tallies[nominee] = votes

    def tally_for(self, nomination_id: str) -> int:
        """Return the current tally for a recorded nomination."""

        open_nomination = self._open_nomination_or_complete(nomination_id)
        return sum(record.vote for record in open_nomination.votes)

    def resolve_execution(self) -> str | None:
        """Return the lone qualifying highest nominee, or no execution on a tie."""

        qualifying = {
            nominee: votes
            for nominee, votes in self._tallies.items()
            if qualifying_tally(votes, self.alive_count)
        }
        if not qualifying:
            return None
        highest_votes = max(qualifying.values())
        highest = [nominee for nominee, votes in qualifying.items() if votes == highest_votes]
        return highest[0] if len(highest) == 1 else None

    def _open_nomination(self, nomination_id: str) -> _OpenNomination:
        if nomination_id != self._active_nomination_id:
            raise IllegalAction("nomination is not open for voting")
        return self._open_nomination_or_complete(nomination_id)

    def _open_nomination_or_complete(self, nomination_id: str) -> _OpenNomination:
        try:
            return self._nominations[nomination_id]
        except KeyError as error:
            raise IllegalAction(f"unknown nomination: {nomination_id}") from error

    @staticmethod
    def _player(state: GameState, player_id: str, label: str):
        try:
            return state.players[player_id]
        except KeyError as error:
            raise IllegalAction(f"unknown {label}: {player_id}") from error

    @staticmethod
    def _clockwise_vote_order(state: GameState, nominee: str) -> tuple[str, ...]:
        seated_players = sorted(state.players.values(), key=lambda player: player.seat)
        nominee_index = next(
            index for index, player in enumerate(seated_players) if player.player_id == nominee
        )
        clockwise = seated_players[nominee_index + 1 :] + seated_players[: nominee_index + 1]
        return tuple(player.player_id for player in clockwise)
