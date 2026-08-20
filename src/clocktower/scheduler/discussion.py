"""A bounded public-discussion scheduler with observer-only audit records."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from clocktower.agents.context import is_safe_public_event
from clocktower.agents.player import AgentOutcome, AgentScene, PlayerAgent, ReactionProbe
from clocktower.domain.actions import (
    IllegalAction,
    Nominate,
    PlayerAction,
    RequestPrivateChat,
    SpeakPublic,
    YieldAction,
)
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState
from clocktower.rules.engine import RuleEngine
from clocktower.models.protocol import ModelCallError

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
        allow_private_chat_requests: bool = False,
        event_sink: Callable[[Sequence[EventRecord]], Awaitable[object]] | None = None,
        safe_point: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if action_budget <= 0 or quiet_windows <= 0 or per_player_action_limit <= 0:
            raise ValueError("discussion budgets must be positive")
        if eligibility_threshold <= 0:
            raise ValueError("eligibility_threshold must be positive")
        self._state_provider = state_provider
        self.agents = dict(agents)
        self.rules = rules
        self.trigger_event = trigger_event
        self.seed = seed
        self.action_budget = action_budget
        self.quiet_windows = quiet_windows
        self.per_player_action_limit = per_player_action_limit
        self.eligibility_threshold = eligibility_threshold
        self.allow_private_chat_requests = allow_private_chat_requests
        self._event_sink = event_sink
        self._safe_point = safe_point
        self.action_count = 0
        self.quiet_count = 0
        self.end_reason: str | None = None
        self.last_speaker: str | None = None
        self.action_counts: dict[str, int] = {}
        self.initial_ranking: tuple[str, ...] = ()
        self.probed_player_ids: tuple[str, ...] = ()
        self.probe_adjustments: dict[str, int] = {}
        self.pending_private_request: RequestPrivateChat | None = None
        self._selection_number = 0

    async def step(self) -> list[EventRecord]:
        """Audit one ranking, probe at most two players, then run at most one normal action."""

        if self.end_reason is not None:
            return []
        state = self._state_provider()
        if state.stopped:
            events = list(await self._emit((self._stop_event(),)))
            return events
        if self.action_count >= self.action_budget:
            events = list(await self._emit((self._end("action_budget"),)))
            return events

        if not is_safe_public_event(self.trigger_event) or not state.phase.startswith("day.discussion"):
            self._increment_quiet()
            events = self._quiet_events([])
            return list(await self._emit(events))

        base_scores = score_candidates(
            self.trigger_event,
            state,
            ScoreContext(
                action_counts=self.action_counts,
                last_speaker=self.last_speaker,
                per_player_action_limit=self.per_player_action_limit,
                available_player_ids=frozenset(self.agents),
            ),
        )
        self.initial_ranking = tuple(score.player_id for score in base_scores)
        events = list(await self._emit((self._ranking_event(base_scores),)))
        await self._at_safe_point()
        if self._state_provider().stopped:
            stopped = self._stop_event()
            events.extend(await self._emit((stopped,)))
            return events
        if not base_scores:
            self._increment_quiet()
            quiet_events = list(await self._emit(self._quiet_events(base_scores)))
            return events + quiet_events

        adjusted_scores, probe_events = await self._probe_top_two(base_scores)
        events.extend(probe_events)
        if self._state_provider().stopped:
            stopped = self._stop_event()
            events.extend(await self._emit((stopped,)))
            return events
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
            quiet_events = list(await self._emit(self._quiet_events(adjusted_scores)))
            return events + quiet_events

        selected_score = next(score for score in adjusted_scores if score.player_id == selected_id)
        selection_event = self._selection_event(selected_score)
        events.extend(await self._emit((selection_event,)))
        await self._at_safe_point()
        if self._state_provider().stopped:
            stopped = self._stop_event()
            events.extend(await self._emit((stopped,)))
            return events
        agent = self.agents.get(selected_id)
        if agent is None:
            self._increment_quiet()
            rejected = self._observer_event("scheduler.action_rejected", {"player_id": selected_id, "reason": "missing_agent"})
            quiet_events = self._quiet_events(adjusted_scores)
            rejected_events = list(await self._emit((rejected, *quiet_events)))
            return events + rejected_events

        scene = AgentScene(
            phase="day.discussion",
            purpose="public_discussion",
            allowed_tools=(
                "speak_public",
                "nominate",
                *(("request_private_chat",) if self.allow_private_chat_requests else ()),
                "yield_action",
            ),
        )
        try:
            outcome = await agent.run_action(scene)
        except ModelCallError:
            await self._at_safe_point()
            if self._state_provider().stopped:
                stopped = self._stop_event()
                return events + list(await self._emit((stopped,)))
            try:
                outcome = await agent.run_action(scene)
            except ModelCallError:
                self._increment_quiet()
                yielded_events = self.rules.apply_action(
                    YieldAction(actor=selected_id, reason="model_call_failed")
                )
                failed_events = list(await self._emit((
                    self._observer_event(
                        "scheduler.normal_action_failed",
                        {"player_id": selected_id, "reason": "model_call_failed_after_retry"},
                    ),
                    *yielded_events,
                    *self._quiet_events([]),
                )))
                self._update_public_trigger(failed_events)
                await self._at_safe_point()
                return events + failed_events
        outcome_events = await self._apply_with_rule_correction(
            selected_id,
            agent,
            scene,
            outcome,
        )
        await self._at_safe_point()
        return events + outcome_events

    async def _apply_with_rule_correction(
        self,
        selected_id: str,
        agent: PlayerAgent,
        scene: AgentScene,
        outcome: AgentOutcome,
    ) -> list[EventRecord]:
        try:
            outcome_events = self._apply_outcome(selected_id, outcome)
        except IllegalAction as first_error:
            correction = (await self._emit((self._tool_error(selected_id, first_error),)))[0]
            await self._at_safe_point()
            if self._state_provider().stopped:
                stopped = self._stop_event()
                return [correction, *(await self._emit((stopped,)))]

            corrected: AgentOutcome | None = None
            for attempt in range(2):
                try:
                    corrected = await agent.run_action(scene)
                except ModelCallError:
                    if attempt == 0:
                        await self._at_safe_point()
                        if self._state_provider().stopped:
                            stopped = self._stop_event()
                            return [correction, *(await self._emit((stopped,)))]
                        continue
                    failure_events = self._optional_failure_events(
                        selected_id,
                        reason="model_call_failed_after_retry",
                    )
                    committed_failures = list(await self._emit(failure_events))
                    self._update_public_trigger(committed_failures)
                    return [correction, *committed_failures]
                break

            if corrected is None:
                failure_events = self._optional_failure_events(
                    selected_id,
                    reason="missing_correction",
                )
                committed_failures = list(await self._emit(failure_events))
                self._update_public_trigger(committed_failures)
                return [correction, *committed_failures]
            try:
                outcome_events = self._apply_outcome(selected_id, corrected)
            except IllegalAction as second_error:
                self._increment_quiet()
                outcome_events = [
                    self._observer_event(
                        "scheduler.action_rejected",
                        {
                            "player_id": selected_id,
                            "reason": type(second_error).__name__,
                        },
                    ),
                    *self.rules.apply_action(
                        YieldAction(actor=selected_id, reason="illegal_action_after_correction")
                    ),
                    *self._quiet_events([]),
                ]
            committed_outcome = list(await self._emit(outcome_events))
            self._update_public_trigger(committed_outcome)
            return [correction, *committed_outcome]

        committed_outcome = list(await self._emit(outcome_events))
        self._update_public_trigger(committed_outcome)
        return committed_outcome

    def _optional_failure_events(self, player_id: str, *, reason: str) -> list[EventRecord]:
        self._increment_quiet()
        return [
            self._observer_event(
                "scheduler.normal_action_failed",
                {"player_id": player_id, "reason": reason},
            ),
            *self.rules.apply_action(
                YieldAction(actor=player_id, reason="model_call_failed")
            ),
            *self._quiet_events([]),
        ]

    async def _probe_top_two(
        self,
        base_scores: list[CandidateScore],
    ) -> tuple[list[CandidateScore], list[EventRecord]]:
        selected = base_scores[:2]
        self.probed_player_ids = tuple(score.player_id for score in selected)
        adjustments: dict[str, int] = {}
        audit_events: list[EventRecord] = []
        for score in selected:
            if self._state_provider().stopped:
                break
            adjustment = 0
            decision = "probe_failed"
            agent = self.agents.get(score.player_id)
            if agent is not None:
                for attempt in range(2):
                    try:
                        probe = await agent.probe(self.trigger_event)
                    except ModelCallError:
                        decision = "probe_model_call_failed"
                        if attempt == 1:
                            break
                        await self._at_safe_point()
                        if self._state_provider().stopped:
                            break
                        continue
                    if getattr(probe, "fallback", False):
                        decision = "probe_fallback"
                        if attempt == 1:
                            break
                        await self._at_safe_point()
                        if self._state_provider().stopped:
                            break
                        continue
                    adjustment, decision = _bounded_adjustment(probe)
                    break
            adjustments[score.player_id] = adjustment
            committed_audit = await self._emit(
                (
                    self._observer_event(
                    "scheduler.probe_adjustment",
                    {
                        "player_id": score.player_id,
                        "decision": decision,
                        "urgency_adjustment": adjustment,
                    },
                    ),
                )
            )
            audit_events.extend(committed_audit)
            await self._at_safe_point()
        self.probe_adjustments = adjustments
        return (
            [score.with_probe_adjustment(adjustments.get(score.player_id, 0)) for score in base_scores],
            audit_events,
        )

    def _apply_outcome(self, selected_id: str, outcome: AgentOutcome) -> list[EventRecord]:
        if self._state_provider().stopped:
            return [self._stop_event()]
        action = outcome.action
        if action is None or isinstance(action, YieldAction):
            self._increment_quiet()
            return self._quiet_events([])
        if not _is_public_discussion_action(action, selected_id):
            if (
                self.allow_private_chat_requests
                and isinstance(action, RequestPrivateChat)
                and action.actor == selected_id
            ):
                self.pending_private_request = action
                self.action_count += 1
                self.action_counts[selected_id] = self.action_counts.get(selected_id, 0) + 1
                self.quiet_count = 0
                return [
                    self._observer_event(
                        "scheduler.private_chat_requested",
                        {
                            "player_id": selected_id,
                            "target_player": action.target_player,
                        },
                    )
                ]
            self._increment_quiet()
            return [
                self._observer_event(
                    "scheduler.action_rejected",
                    {"player_id": selected_id, "reason": "non_public_or_wrong_actor"},
                ),
                *self._quiet_events([]),
            ]
        rule_events = self.rules.apply_action(action)

        self.action_count += 1
        self.action_counts[selected_id] = self.action_counts.get(selected_id, 0) + 1
        self.last_speaker = selected_id
        self.quiet_count = 0
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
                {
                    "quiet_count": self.quiet_count,
                    "eligible_count": sum(
                        score.total >= self.eligibility_threshold for score in scores
                    ),
                },
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

    def _stop_event(self) -> EventRecord:
        self.end_reason = "stopped"
        return self._observer_event(
            "scheduler.stopped",
            {"action_count": self.action_count, "quiet_count": self.quiet_count},
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

    @staticmethod
    def _tool_error(player_id: str, error: IllegalAction) -> EventRecord:
        return EventRecord(
            phase="day.discussion",
            type="tool.error",
            actor=player_id,
            audience=Audience.player(player_id),
            payload={
                "reason": type(error).__name__,
                "correction_allowed": True,
            },
        )

    async def _emit(self, events: Sequence[EventRecord]) -> tuple[EventRecord, ...]:
        drafts = tuple(events)
        if self._event_sink is None or not drafts:
            return drafts
        result = await self._event_sink(drafts)
        if (
            isinstance(result, Sequence)
            and len(result) == len(drafts)
            and all(isinstance(event, EventRecord) for event in result)
        ):
            return tuple(result)
        return drafts

    def _update_public_trigger(self, events: Sequence[EventRecord]) -> None:
        public_events = [event for event in events if is_safe_public_event(event)]
        if public_events:
            self.trigger_event = public_events[-1]

    async def _at_safe_point(self) -> None:
        if self._safe_point is not None:
            await self._safe_point()


def _bounded_adjustment(probe: ReactionProbe) -> tuple[int, str]:
    urgency = max(-15, min(15, probe.urgency))
    if probe.decision == "respond":
        return urgency, "respond"
    if probe.decision == "defer":
        return min(0, urgency), "defer"
    return -15, "silent"


def _is_public_discussion_action(action: object, selected_id: str) -> bool:
    return isinstance(action, (SpeakPublic, Nominate)) and action.actor == selected_id
