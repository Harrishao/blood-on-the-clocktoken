"""A bounded public-discussion scheduler with observer-only audit records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from clocktower.agents.player import AgentOutcome, AgentScene, PlayerAgent, ReactionProbe
from clocktower.domain.actions import Nominate, PlayerAction, SpeakPublic, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState
from clocktower.rules.engine import RuleEngine

from .scoring import CandidateScore, ScoreContext, choose_candidate, score_candidates


class _Rules(Protocol):
    def apply_action(self, action: PlayerAction) -> list[EventRecord]: ...


class DiscussionScheduler:
    """Select at most one public action per event-driven step."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], GameState],
        agents: Mapping[str, PlayerAgent],
        rules: _Rules | RuleEngine,
        trigger_event: EventRecord,
        seed: int,
        action_budget: int = 40,
        quiet_windows: int = 3,
        per_player_action_limit: int = 2,
        eligibility_threshold: int = 1,
    ) -> None:
        if action_budget <= 0 or quiet_windows <= 0 or per_player_action_limit <= 0:
            raise ValueError("discussion budgets must be positive")
        self._state_provider = state_provider
        self.agents = dict(agents)
        self.rules = rules
        self.trigger_event = trigger_event
        self.seed = seed
        self.action_budget = action_budget
        self.quiet_windows = quiet_windows
        self.per_player_action_limit = per_player_action_limit
        self.eligibility_threshold = eligibility_threshold
        self.action_count = 0
        self.quiet_count = 0
        self.end_reason: str | None = None
        self.last_speaker: str | None = None
        self.action_counts: dict[str, int] = {}
        self.initial_ranking: tuple[str, ...] = ()
        self.probed_player_ids: tuple[str, ...] = ()
        self.probe_adjustments: dict[str, int] = {}
        self._selection_number = 0

    async def step(self) -> list[EventRecord]:
        """Audit one ranking, probe at most two players, then run at most one normal action."""

        if self.end_reason is not None:
            return []
        if self.action_count >= self.action_budget:
            return [self._end("action_budget")]

        state = self._state_provider()
        if self.trigger_event.audience.kind != "public" or not state.phase.startswith("day.discussion"):
            self._increment_quiet()
            return self._quiet_events([])

        base_scores = score_candidates(
            self.trigger_event,
            state,
            ScoreContext(
                action_counts=self.action_counts,
                last_speaker=self.last_speaker,
                per_player_action_limit=self.per_player_action_limit,
            ),
        )
        self.initial_ranking = tuple(score.player_id for score in base_scores)
        events = [self._ranking_event(base_scores)]
        if not base_scores:
            self._increment_quiet()
            return events + self._quiet_events(base_scores)

        adjusted_scores, probe_events = await self._probe_top_two(base_scores)
        events.extend(probe_events)
        eligible = [
            score
            for score in adjusted_scores
            if score.total >= self.eligibility_threshold
        ]
        selected_id = choose_candidate(
            eligible,
            seed_state=f"{self.seed}:{self._selection_number}:{self.trigger_event.seq}:{self.trigger_event.actor or ''}",
        )
        self._selection_number += 1
        if selected_id is None:
            self._increment_quiet()
            return events + self._quiet_events(adjusted_scores)

        selected_score = next(score for score in adjusted_scores if score.player_id == selected_id)
        events.append(self._selection_event(selected_score))
        agent = self.agents.get(selected_id)
        if agent is None:
            self._increment_quiet()
            events.append(self._observer_event("scheduler.action_rejected", {"player_id": selected_id, "reason": "missing_agent"}))
            return events + self._quiet_events(adjusted_scores)

        outcome = await agent.run_action(
            AgentScene(
                phase="day.discussion",
                purpose="public_discussion",
                allowed_tools=("speak_public", "nominate", "yield_action"),
            )
        )
        return events + self._apply_outcome(selected_id, outcome)

    async def _probe_top_two(
        self,
        base_scores: list[CandidateScore],
    ) -> tuple[list[CandidateScore], list[EventRecord]]:
        selected = base_scores[:2]
        self.probed_player_ids = tuple(score.player_id for score in selected)
        adjustments: dict[str, int] = {}
        audit_events: list[EventRecord] = []
        for score in selected:
            adjustment = 0
            decision = "probe_failed"
            agent = self.agents.get(score.player_id)
            if agent is not None:
                try:
                    probe = await agent.probe(self.trigger_event)
                    adjustment, decision = _bounded_adjustment(probe)
                except Exception:
                    adjustment = 0
                    decision = "probe_failed"
            adjustments[score.player_id] = adjustment
            audit_events.append(
                self._observer_event(
                    "scheduler.probe_adjustment",
                    {
                        "player_id": score.player_id,
                        "decision": decision,
                        "urgency_adjustment": adjustment,
                    },
                )
            )
        self.probe_adjustments = adjustments
        return (
            [score.with_probe_adjustment(adjustments.get(score.player_id, 0)) for score in base_scores],
            audit_events,
        )

    def _apply_outcome(self, selected_id: str, outcome: AgentOutcome) -> list[EventRecord]:
        action = outcome.action
        if action is None or isinstance(action, YieldAction):
            self._increment_quiet()
            return self._quiet_events([])
        if not _is_public_discussion_action(action, selected_id):
            self._increment_quiet()
            return [
                self._observer_event(
                    "scheduler.action_rejected",
                    {"player_id": selected_id, "reason": "non_public_or_wrong_actor"},
                ),
                *self._quiet_events([]),
            ]
        try:
            rule_events = self.rules.apply_action(action)
        except Exception as error:
            self._increment_quiet()
            return [
                self._observer_event(
                    "scheduler.action_rejected",
                    {"player_id": selected_id, "reason": type(error).__name__},
                ),
                *self._quiet_events([]),
            ]

        self.action_count += 1
        self.action_counts[selected_id] = self.action_counts.get(selected_id, 0) + 1
        self.last_speaker = selected_id
        self.quiet_count = 0
        public_events = [event for event in rule_events if event.audience.kind == "public"]
        if public_events:
            self.trigger_event = public_events[-1]
        events = list(rule_events)
        if self.action_count >= self.action_budget:
            events.append(self._end("action_budget"))
        return events

    def _increment_quiet(self) -> None:
        self.quiet_count += 1

    def _quiet_events(self, scores: list[CandidateScore]) -> list[EventRecord]:
        events = [
            self._observer_event(
                "scheduler.quiet_window",
                {"quiet_count": self.quiet_count, "eligible_count": len(scores)},
            )
        ]
        if self.quiet_count >= self.quiet_windows:
            events.append(self._end("quiet"))
        return events

    def _end(self, reason: str) -> EventRecord:
        self.end_reason = reason
        return self._observer_event(
            "scheduler.ended",
            {"reason": reason, "action_count": self.action_count, "quiet_count": self.quiet_count},
        )

    def _ranking_event(self, scores: list[CandidateScore]) -> EventRecord:
        return self._observer_event(
            "scheduler.ranking",
            {
                "trigger_seq": self.trigger_event.seq,
                "candidates": [
                    {
                        "player_id": score.player_id,
                        "base_total": score.base_total,
                        "features": [
                            {
                                "name": feature.name,
                                "contribution": feature.contribution,
                                "reason": feature.reason,
                            }
                            for feature in score.features
                        ],
                    }
                    for score in scores
                ],
            },
        )

    def _selection_event(self, score: CandidateScore) -> EventRecord:
        return self._observer_event(
            "scheduler.selection",
            {
                "player_id": score.player_id,
                "base_total": score.base_total,
                "probe_adjustment": score.probe_adjustment,
                "total": score.total,
                "reason": "seeded_weighted_choice_above_threshold",
            },
        )

    def _observer_event(self, event_type: str, payload: dict[str, Any]) -> EventRecord:
        return EventRecord(
            phase="day.discussion",
            type=event_type,
            audience=Audience.observer(),
            payload=payload,
        )


def _bounded_adjustment(probe: ReactionProbe) -> tuple[int, str]:
    urgency = max(-15, min(15, probe.urgency))
    if probe.decision == "respond":
        return urgency, "respond"
    if probe.decision == "defer":
        return min(0, urgency), "defer"
    return -15, "silent"


def _is_public_discussion_action(action: object, selected_id: str) -> bool:
    return isinstance(action, (SpeakPublic, Nominate)) and action.actor == selected_id
