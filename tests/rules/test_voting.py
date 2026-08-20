import pytest

from clocktower.domain.actions import IllegalAction
from clocktower.domain.state import GameState
from clocktower.rules.voting import NominationTracker, cast_vote, qualifying_tally
from tests.builders import sample_voting_state


def seven_player_state() -> GameState:
    return GameState.from_alignments(["good"] * 7, seed=17)


def test_tie_clears_execution_candidate():
    """Keeping the earlier candidate after an equal qualifying tally is illegal."""

    tracker = NominationTracker(alive_count=7)
    tracker.record_tally("alice", 4)
    tracker.record_tally("bob", 4)

    assert tracker.resolve_execution() is None


@pytest.mark.parametrize(
    ("votes", "alive_count", "expected"),
    [(2, 5, False), (3, 5, True), (3, 6, True), (3, 7, False), (4, 7, True)],
)
def test_qualification_uses_ceiling_of_half_the_living_players(votes, alive_count, expected):
    """Rounding down would let a minority execute in an odd-sized living town."""

    assert qualifying_tally(votes, alive_count) is expected


def test_dead_vote_returns_consumption_intent_without_mutating_game_state():
    """Mutating the token here would bypass the rule engine's atomic boundary."""

    state = sample_voting_state(dead={"bob"})

    vote = cast_vote(state, "bob", "nom-1", True)

    assert vote.consumes_dead_vote is True
    assert state.players["bob"].dead_vote_available is True

    state.players["bob"].dead_vote_available = False
    with pytest.raises(IllegalAction, match="dead vote already spent"):
        cast_vote(state, "bob", "nom-2", True)


def test_only_alive_players_can_nominate_once_and_each_nominee_is_limited_once_per_day():
    """Removing either nomination guard permits a second daily nomination."""

    state = sample_voting_state(dead={"eve"})
    tracker = NominationTracker(alive_count=state.alive_count)

    tracker.nominate(state, "alice", "bob", accusation="first")

    with pytest.raises(IllegalAction, match="already nominated"):
        tracker.nominate(state, "alice", "carol", accusation="again")
    with pytest.raises(IllegalAction, match="already been nominated"):
        tracker.nominate(state, "carol", "bob", accusation="again")
    with pytest.raises(IllegalAction, match="dead players cannot nominate"):
        tracker.nominate(state, "eve", "david", accusation="dead")


def test_alive_player_can_nominate_dead_player():
    """Rejecting a dead nominee would remove a legal Trouble Brewing nomination."""

    state = sample_voting_state(dead={"eve"})
    tracker = NominationTracker(alive_count=state.alive_count)

    nomination = tracker.nominate(state, "david", "eve", accusation="legal target")

    assert nomination.nominee == "eve"


def test_votes_follow_clockwise_seat_order_and_nominee_votes_last():
    """Accepting an early nominee vote would break the public counting order."""

    state = sample_voting_state()
    tracker = NominationTracker(alive_count=state.alive_count)
    nomination = tracker.nominate(state, "alice", "bob", accusation="test")

    assert nomination.vote_order == ("carol", "david", "eve", "alice", "bob")
    with pytest.raises(IllegalAction, match="expected vote from carol"):
        tracker.cast_vote(state, "bob", nomination.nomination_id, True)

    for voter in nomination.vote_order[:-1]:
        tracker.cast_vote(state, voter, nomination.nomination_id, False)
    tracker.cast_vote(state, "bob", nomination.nomination_id, True)

    assert tracker.tally_for(nomination.nomination_id) == 1


def test_alive_player_can_vote_in_more_than_one_nomination():
    """Spending an alive vote per day would incorrectly block later nominations."""

    state = sample_voting_state()
    tracker = NominationTracker(alive_count=state.alive_count)
    first = tracker.nominate(state, "alice", "bob", accusation="first")
    for voter in first.vote_order:
        tracker.cast_vote(state, voter, first.nomination_id, voter == "alice")

    second = tracker.nominate(state, "carol", "david", accusation="second")
    for voter in second.vote_order:
        tracker.cast_vote(state, voter, second.nomination_id, voter == "alice")

    assert tracker.tally_for(first.nomination_id) == tracker.tally_for(second.nomination_id) == 1
