"""Deterministic, explainable candidate scoring for public discussion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Mapping

from clocktower.domain.events import EventRecord
from clocktower.domain.state import AttentionState, GameState


WEIGHTS = {
    "direct_target": 40,
    "mentioned": 25,
    "trigger": 20,
    "pending_action": 15,
    "fairness": 10,
    "recent_speaker": -20,
    "repeat_risk": -25,
    "budget_pressure": -15,
}


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One score input retained with a human-auditable reason."""

    name: str
    contribution: int
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """A base ranking plus its bounded short-call adjustment."""

    player_id: str
    features: tuple[FeatureContribution, ...]
    base_total: int
    probe_adjustment: int = 0

    @property
    def total(self) -> int:
        return self.base_total + self.probe_adjustment

    def with_probe_adjustment(self, adjustment: int) -> CandidateScore:
        return replace(self, probe_adjustment=max(-15, min(15, int(adjustment))))


@dataclass(frozen=True, slots=True)
class ScoreContext:
    """Public scheduler metadata; deliberately excludes notebook text and model reasoning."""

    action_counts: Mapping[str, int] = field(default_factory=dict)
    last_speaker: str | None = None
    per_player_action_limit: int = 2


def score_candidates(
    event: EventRecord,
    state: GameState,
    context: ScoreContext | None = None,
) -> list[CandidateScore]:
    """Score eligible living players from a public event and structured metadata only."""

    if event.audience.kind != "public" or not event.phase.startswith("day.discussion"):
        return []

    context = context or ScoreContext()
    action_counts = context.action_counts
    eligible = [
        player
        for player in state.players.values()
        if player.alive and action_counts.get(player.player_id, 0) < context.per_player_action_limit
    ]
    if not eligible:
        return []

    target_ids = _direct_target_ids(event)
    mentioned_ids = _string_set(event.payload.get("mentions"))
    minimum_actions = min(action_counts.get(player.player_id, 0) for player in eligible)
    scores = [
        _score_player(
            player_id=player.player_id,
            attention=player.notebook.attention,
            target_ids=target_ids,
            mentioned_ids=mentioned_ids,
            action_count=action_counts.get(player.player_id, 0),
            minimum_actions=minimum_actions,
            last_speaker=context.last_speaker,
            per_player_action_limit=context.per_player_action_limit,
        )
        for player in eligible
    ]
    return sorted(scores, key=lambda score: (-score.base_total, score.player_id))


def choose_candidate(scores: list[CandidateScore], seed_state: int | str | bytes) -> str | None:
    """Choose proportionally with a local deterministic RNG, never process-global state."""

    eligible = sorted(
        (score for score in scores if score.total > 0),
        key=lambda score: score.player_id,
    )
    if not eligible:
        return None

    total_weight = sum(score.total for score in eligible)
    draw = random.Random(seed_state).randrange(total_weight)
    cumulative = 0
    for score in eligible:
        cumulative += score.total
        if draw < cumulative:
            return score.player_id
    raise AssertionError("weighted choice did not select an eligible candidate")


def _score_player(
    *,
    player_id: str,
    attention: AttentionState,
    target_ids: frozenset[str],
    mentioned_ids: frozenset[str],
    action_count: int,
    minimum_actions: int,
    last_speaker: str | None,
    per_player_action_limit: int,
) -> CandidateScore:
    active = {
        "direct_target": player_id in target_ids,
        "mentioned": player_id in mentioned_ids,
        "trigger": attention == AttentionState.NEEDS_ATTENTION,
        "pending_action": attention == AttentionState.ACTIVE,
        "fairness": action_count == minimum_actions,
        "recent_speaker": player_id == last_speaker,
        "repeat_risk": action_count > 0,
        "budget_pressure": action_count == per_player_action_limit - 1,
    }
    reasons = {
        "direct_target": "event actor or structured target",
        "mentioned": "named in public event",
        "trigger": "notebook attention requests a public trigger follow-up",
        "pending_action": "notebook attention marks a pending action",
        "fairness": "fewest completed discussion actions",
        "recent_speaker": "most recent public speaker is cooling down",
        "repeat_risk": "player has already acted in this scene",
        "budget_pressure": "one action remains before the personal scene limit",
    }
    features = tuple(
        FeatureContribution(
            name=name,
            contribution=WEIGHTS[name] if active[name] else 0,
            reason=reasons[name] if active[name] else "not applicable",
        )
        for name in WEIGHTS
    )
    return CandidateScore(
        player_id=player_id,
        features=features,
        base_total=sum(feature.contribution for feature in features),
    )


def _direct_target_ids(event: EventRecord) -> frozenset[str]:
    target_ids = {event.actor} if event.actor is not None else set()
    for key in ("target", "target_player", "nominee", "player_id"):
        value = event.payload.get(key)
        if isinstance(value, str):
            target_ids.add(value)
    return frozenset(target_ids)


def _string_set(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))
