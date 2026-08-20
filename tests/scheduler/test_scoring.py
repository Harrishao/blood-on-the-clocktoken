from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import NotebookAttention
from clocktower.scheduler.scoring import ScoreContext, choose_candidate, score_candidates
from tests.builders import public_claim, sample_game_state


def test_mentioned_player_scores_above_unrelated_player_with_auditable_features():
    """Removing the public-mention contribution would lose a direct discussion reply."""

    state = sample_game_state()
    state.players["bob"].notebook.attention = NotebookAttention(watch_triggers=["claim.public"])

    scores = score_candidates(
        public_claim(actor="alice", mentions={"bob"}),
        state,
        ScoreContext(),
    )

    by_id = {score.player_id: score for score in scores}
    assert by_id["bob"].base_total > by_id["carol"].base_total
    assert {feature.name for feature in by_id["bob"].features} == {
        "direct_target",
        "mentioned",
        "trigger",
        "pending_action",
        "fairness",
        "recent_speaker",
        "repeat_risk",
        "budget_pressure",
    }
    mentioned = next(feature for feature in by_id["bob"].features if feature.name == "mentioned")
    assert mentioned.contribution == 25
    assert mentioned.reason == "named in public event"


def test_base_ranking_has_a_stable_player_id_tie_break():
    """A hash- or container-order tie break would make an identical scene non-replayable."""

    state = sample_game_state()
    scores = score_candidates(
        public_claim(actor="alice", mentions=set()),
        state,
        ScoreContext(),
    )

    assert [score.player_id for score in scores] == ["alice", "bob", "carol", "david", "eve"]


def test_seeded_weighted_choice_is_repeatable_without_global_rng_state():
    """Replacing the local seeded generator with random.choice would break replay determinism."""

    state = sample_game_state()
    scores = score_candidates(
        public_claim(actor="alice", mentions={"bob"}),
        state,
        ScoreContext(),
    )

    first = choose_candidate(scores, seed_state=991)
    second = choose_candidate(scores, seed_state=991)

    assert first == second
    assert first in {score.player_id for score in scores if score.total > 0}


def test_cooldown_repeat_risk_and_budget_pressure_keep_separate_reasons():
    """Collapsing these penalties would make a recent repeat speaker indistinguishable in audit."""

    state = sample_game_state()
    scores = score_candidates(
        public_claim(actor="alice", mentions=set()),
        state,
        ScoreContext(action_counts={"bob": 1}, last_speaker="bob", per_player_action_limit=2),
    )

    bob = next(score for score in scores if score.player_id == "bob")
    by_feature = {feature.name: feature for feature in bob.features}
    assert by_feature["recent_speaker"].contribution == -20
    assert by_feature["repeat_risk"].contribution == -25
    assert by_feature["budget_pressure"].contribution == -15
    assert all(feature.reason != "not applicable" for feature in by_feature.values() if feature.contribution)


def test_structured_attention_matches_only_the_current_public_event():
    """Applying nonempty attention to every event would create permanent hidden scheduling bias."""

    state = sample_game_state()
    state.players["bob"].notebook.attention = NotebookAttention(
        players=["alice"],
        watch_triggers=["claim.public"],
        pending_actions=["nominate"],
    )
    claim = public_claim(actor="alice", mentions={"carol"})
    unrelated = EventRecord(
        phase="day.discussion",
        type="player.public_message",
        actor="david",
        audience=Audience.public(),
        payload={"mentions": ["carol"]},
    )

    matched = {score.player_id: score for score in score_candidates(claim, state)}["bob"]
    unmatched = {score.player_id: score for score in score_candidates(unrelated, state)}["bob"]
    matched_features = {feature.name: feature for feature in matched.features}
    unmatched_features = {feature.name: feature for feature in unmatched.features}

    assert matched_features["trigger"].contribution == 20
    assert "alice" in matched_features["trigger"].reason
    assert matched_features["pending_action"].contribution == 15
    assert "nominate" in matched_features["pending_action"].reason
    assert unmatched_features["trigger"].contribution == 0
    assert unmatched_features["pending_action"].contribution == 15


def test_pending_action_uses_discussion_tool_availability_without_an_event_action_key():
    """Production public events do not need an artificial action_key to make a pending nomination relevant."""

    state = sample_game_state()
    state.players["bob"].notebook.attention = NotebookAttention(pending_actions=["nominate"])
    score = {score.player_id: score for score in score_candidates(public_claim(actor="alice", mentions=set()), state)}["bob"]

    pending = next(feature for feature in score.features if feature.name == "pending_action")
    assert pending.contribution == 15
    assert "nominate" in pending.reason

    state.players["bob"].notebook.attention = NotebookAttention(pending_actions=["use_ability"])
    invalid = {score.player_id: score for score in score_candidates(public_claim(actor="alice", mentions=set()), state)}["bob"]
    assert next(feature for feature in invalid.features if feature.name == "pending_action").contribution == 0


def test_dead_player_with_an_agent_remains_a_discussion_candidate():
    """Filtering by alive status removes a dead player's permitted public speech opportunity."""

    state = sample_game_state(dead={"bob"})
    scores = score_candidates(public_claim(actor="alice", mentions={"bob"}), state)

    assert "bob" in {score.player_id for score in scores}
