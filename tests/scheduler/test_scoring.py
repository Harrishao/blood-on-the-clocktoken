from clocktower.domain.state import AttentionState
from clocktower.scheduler.scoring import ScoreContext, choose_candidate, score_candidates
from tests.builders import public_claim, sample_game_state


def test_mentioned_player_scores_above_unrelated_player_with_auditable_features():
    """Removing the public-mention contribution would lose a direct discussion reply."""

    state = sample_game_state()
    state.players["bob"].notebook.attention = AttentionState.NEEDS_ATTENTION

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
