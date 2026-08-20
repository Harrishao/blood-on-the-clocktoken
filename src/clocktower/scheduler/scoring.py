"""Deterministic, explainable candidate scoring for public discussion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Mapping

from clocktower.agents.context import is_safe_public_event
from clocktower.domain.events import EventRecord
from clocktower.domain.state import GameState, NotebookAttention


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
    available_player_ids: frozenset[str] | None = None


def score_candidates(
    event: EventRecord,
    state: GameState,
    context: ScoreContext | None = None,
) -> list[CandidateScore]:
    """Score eligible living players from a public event and structured metadata only."""

    if not is_safe_public_event(event) or not event.phase.startswith("day.discussion"):
        return []

    context = context or ScoreContext()
    action_counts = context.action_counts
    eligible = [
        player
        for player in state.players.values()
        if (
            action_counts.get(player.player_id, 0) < context.per_player_action_limit
            and (context.available_player_ids is None or player.player_id in context.available_player_ids)
        )
    ]
    if not eligible:
        return []

    target_ids = _direct_target_ids(event)
    mentioned_ids = _string_set(event.payload.get("mentions"))
    related_player_ids = target_ids | mentioned_ids
    trigger_keys = frozenset({event.type}) | _explicit_keys(event.payload, "trigger_key", "trigger_keys")
    action_keys = _explicit_keys(event.payload, "action_key", "action_keys")
    minimum_actions = min(action_counts.get(player.player_id, 0) for player in eligible)
    scores = [
        _score_player(
            player_id=player.player_id,
            attention=player.notebook.attention,
            target_ids=target_ids,
            mentioned_ids=mentioned_ids,
            related_player_ids=related_player_ids,
            trigger_keys=trigger_keys,
            action_keys=action_keys,
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
    attention: NotebookAttention,
    target_ids: frozenset[str],
    mentioned_ids: frozenset[str],
    related_player_ids: frozenset[str],
    trigger_keys: frozenset[str],
    action_keys: frozenset[str],
    action_count: int,
    minimum_actions: int,
    last_speaker: str | None,
    per_player_action_limit: int,
) -> CandidateScore:
    matched_attention_players = tuple(sorted(set(attention.players) & set(related_player_ids)))
    matched_watch_triggers = tuple(sorted(set(attention.watch_triggers) & set(trigger_keys)))
    matched_pending_actions = tuple(
        sorted(
            set(attention.pending_actions)
            & {"speak_public", "nominate"}
            & set(action_keys)
        )
    )
    active = {
        "direct_target": player_id in target_ids,
        "mentioned": player_id in mentioned_ids,
        "trigger": bool(matched_attention_players or matched_watch_triggers),
        "pending_action": bool(matched_pending_actions),
        "fairness": action_count == minimum_actions,
        "recent_speaker": player_id == last_speaker,
        "repeat_risk": action_count > 0,
        "budget_pressure": action_count == per_player_action_limit - 1,
    }
    reasons = {
        "direct_target": "event actor or structured target",
        "mentioned": "named in public event",
        "trigger": _trigger_reason(matched_attention_players, matched_watch_triggers),
        "pending_action": _pending_reason(matched_pending_actions),
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


def _explicit_keys(payload: Mapping[str, object], single: str, plural: str) -> frozenset[str]:
    values = set(_string_set(payload.get(plural)))
    value = payload.get(single)
    if isinstance(value, str):
        values.add(value)
    return frozenset(values)


def _trigger_reason(players: tuple[str, ...], triggers: tuple[str, ...]) -> str:
    matches = [f"attention player {value!r}" for value in players]
    matches.extend(f"watch trigger {value!r}" for value in triggers)
    return "matched " + ", ".join(matches) if matches else "not applicable"


def _pending_reason(actions: tuple[str, ...]) -> str:
    if not actions:
        return "not applicable"
    return "pending action matched " + ", ".join(repr(action) for action in actions)
