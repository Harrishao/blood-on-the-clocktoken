"""One bounded, authorization-first private conversation between two players."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from clocktower.agents.player import AgentOutcome, AgentScene, PlayerAgent, ReactionProbe
from clocktower.domain.actions import LeavePrivateChat, SpeakPrivate, UpdateNotebook, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import GameState
from clocktower.models.protocol import ModelCallError

from .scoring import CandidateScore, FeatureContribution, choose_candidate


@dataclass
class PrivateChatScene:
    """The sole active private subscene; participant order is invitation order."""

    chat_id: str
    participant_ids: tuple[str, str]
    parent_phase: str
    action_count: int = 0
    quiet_count: int = 0
    last_speaker: str | None = None
    action_counts: dict[str, int] = field(default_factory=dict)
    quiet_player_ids: set[str] = field(default_factory=set)
    transcript: list[EventRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PrivateChatRequest:
    """The result of asking one invitee for private-chat consent."""

    request_id: str
    decision: str
    scene: PrivateChatScene | None
    events: tuple[EventRecord, ...]


class PrivateChatScheduler:
    """Coordinate invitation consent and a strictly two-player private scene."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], GameState],
        agents: Mapping[str, PlayerAgent],
        seed: int,
        action_budget: int = 8,
        quiet_windows: int = 2,
        per_player_action_limit: int = 2,
        eligibility_threshold: int = 1,
        event_sink: Callable[[Sequence[EventRecord]], Awaitable[object]] | None = None,
        safe_point: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if min(action_budget, quiet_windows, per_player_action_limit, eligibility_threshold) <= 0:
            raise ValueError("private-chat budgets must be positive")
        self._state_provider = state_provider
        self.agents = dict(agents)
        self.seed = seed
        self.action_budget = action_budget
        self.quiet_windows = quiet_windows
        self.per_player_action_limit = per_player_action_limit
        self.eligibility_threshold = eligibility_threshold
        self._event_sink = event_sink
        self._safe_point = safe_point
        self._scene: PrivateChatScene | None = None
        self._request_number = 0
        self._selection_number = 0
        self._lock = asyncio.Lock()
        self.end_reason: str | None = None
        self.probed_player_ids: tuple[str, ...] = ()
        self.probe_adjustments: dict[str, int] = {}

    async def request(self, inviter: str, invitee: str) -> PrivateChatRequest:
        """Ask exactly one invitee, using a retried short call with no normal continuation."""

        async with self._lock:
            state = self._state_provider()
            self._validate_request(state, inviter, invitee)
            self._request_number += 1
            request_id = f"private-request-{state.seed}-{self._request_number}"
            invitation = EventRecord(
                phase="day.private_invite",
                type="chat.private_invitation",
                actor=inviter,
                audience=Audience.player(invitee),
                payload={"request_id": request_id, "inviter": inviter},
            )
            await self._emit((invitation,))
            decision = await self._request_decision(invitee, invitation)
            if decision == "accept" and not self._can_accept_after_wait(inviter, invitee):
                decision = "defer"
            response = EventRecord(
                phase="day.private_invite",
                type="chat.private_response",
                actor=invitee,
                audience=Audience.player(invitee),
                payload={"request_id": request_id, "decision": decision},
            )
            await self._emit((response,))
            if decision != "accept":
                await self._at_safe_point()
                return PrivateChatRequest(
                    request_id=request_id,
                    decision=decision,
                    scene=None,
                    events=(invitation, response),
                )

            accepted_state = self._state_provider()
            chat_id = f"private-chat-{accepted_state.seed}-{self._request_number}"
            scene = PrivateChatScene(
                chat_id=chat_id,
                participant_ids=(inviter, invitee),
                parent_phase=accepted_state.phase,
            )
            self._scene = scene
            accepted_state.active_scene = chat_id
            self.end_reason = None
            self.probed_player_ids = ()
            self.probe_adjustments = {}
            await self._at_safe_point()
            return PrivateChatRequest(
                request_id=request_id,
                decision=decision,
                scene=scene,
                events=(invitation, response),
            )

    async def run(self, chat_id: str) -> list[EventRecord]:
        """Run the accepted subscene to one of its bounded termination conditions."""

        async with self._lock:
            scene = self._scene
            if scene is None or scene.chat_id != chat_id:
                raise ValueError("chat_id does not identify the active private scene")
            state = self._state_provider()
            if state.active_scene != chat_id or state.phase != scene.parent_phase:
                self._end(scene, "ownership_lost")
                self._release_reservation(scene)
                ended = self._public_shell("chat.private_ended", scene)
                await self._emit((ended,))
                return [ended]
            state.phase = "day.private"
            events: list[EventRecord] = [self._public_shell("chat.private_started", scene)]
            await self._emit(events)
            try:
                while self.end_reason is None:
                    current_state = self._state_provider()
                    if not self._owns(scene):
                        self._end(scene, "ownership_lost")
                        break
                    if current_state.stopped:
                        stopped = self._observer_event("scheduler.stopped", scene)
                        events.append(stopped)
                        await self._emit((stopped,))
                        self._end(scene, "stopped")
                        break
                    if scene.action_count >= self.action_budget:
                        self._end(scene, "action_budget")
                        break

                    scores = self._score_participants(scene)
                    if not scores:
                        self._end(scene, "quiet" if self._all_quiet(scene) else "per_player_action_limit")
                        break
                    ranking = self._ranking_event(scene, scores)
                    events.append(ranking)
                    await self._emit((ranking,))
                    adjusted, probe_events = await self._probe_top_two(scene, scores)
                    events.extend(probe_events)
                    if not self._owns(scene):
                        self._end(scene, "ownership_lost")
                        break
                    if self._state_provider().stopped:
                        stopped = self._observer_event("scheduler.stopped", scene)
                        events.append(stopped)
                        await self._emit((stopped,))
                        self._end(scene, "stopped")
                        break
                    selected = self._choose(scene, adjusted)
                    if selected is None:
                        self._end(scene, "quiet" if self._all_quiet(scene) else "per_player_action_limit")
                        quiet_events = self._quiet_events(scene, adjusted)
                        events.extend(quiet_events)
                        await self._emit(quiet_events)
                        continue
                    selection = self._selection_event(scene, selected)
                    events.append(selection)
                    await self._emit((selection,))
                    action_events = await self._run_one_action(scene, selected)
                    events.extend(action_events)
                    await self._emit(action_events)
                    await self._at_safe_point()
            finally:
                if self.end_reason is None:
                    self._end(scene, "model_failed")
                self._release_reservation(scene)
                ended = self._public_shell("chat.private_ended", scene)
                events.append(ended)
                await self._emit((ended,))
            return events

    def _validate_request(self, state: GameState, inviter: str, invitee: str) -> None:
        if inviter == invitee:
            raise ValueError("private-chat participants must be different")
        if inviter not in state.players or invitee not in state.players:
            raise ValueError("unknown private-chat participant")
        if state.stopped:
            raise ValueError("private chat cannot start while stopped")
        if state.phase != "day.discussion":
            raise ValueError("private chat is only available during day.discussion")
        if state.active_scene is not None or self._scene is not None:
            raise ValueError("another scene is already active")

    def _can_accept_after_wait(self, inviter: str, invitee: str) -> bool:
        state = self._state_provider()
        return (
            not state.stopped
            and state.phase == "day.discussion"
            and state.active_scene is None
            and self._scene is None
            and inviter in state.players
            and invitee in state.players
            and inviter != invitee
        )

    async def _request_decision(self, invitee: str, invitation: EventRecord) -> str:
        agent = self.agents.get(invitee)
        if agent is None:
            return "defer"
        for attempt in range(2):
            if self._state_provider().stopped:
                return "defer"
            try:
                response = await agent.respond_private_invitation(invitation)  # type: ignore[attr-defined]
            except ModelCallError:
                if self._state_provider().stopped or attempt == 1:
                    return "defer"
                await self._at_safe_point()
                if self._state_provider().stopped:
                    return "defer"
                continue
            decision = getattr(response, "decision", response)
            if decision in {"accept", "reject", "defer"} and not getattr(response, "fallback", False):
                return str(decision)
            if self._state_provider().stopped:
                return "defer"
            if attempt == 1:
                return "defer"
            await self._at_safe_point()
            if self._state_provider().stopped:
                return "defer"
        return "defer"

    def _score_participants(self, scene: PrivateChatScene) -> list[CandidateScore]:
        """Use Task 9's auditable score form without reading notes or public context."""

        scores: list[CandidateScore] = []
        available_ids = tuple(
            player_id
            for player_id in scene.participant_ids
            if player_id not in scene.quiet_player_ids
            and scene.action_counts.get(player_id, 0) < self.per_player_action_limit
        )
        if not available_ids:
            return []
        minimum = min(
            scene.action_counts.get(player_id, 0)
            for player_id in available_ids
        )
        for player_id in available_ids:
            action_count = scene.action_counts.get(player_id, 0)
            active = {
                "private_available": True,
                "fairness": action_count == minimum,
                "recent_speaker": player_id == scene.last_speaker,
                "repeat_risk": action_count > 0,
                "budget_pressure": action_count == self.per_player_action_limit - 1,
            }
            weights = {
                "private_available": 30,
                "fairness": 10,
                "recent_speaker": -5,
                "repeat_risk": -5,
                "budget_pressure": -5,
            }
            features = tuple(
                FeatureContribution(
                    name=name,
                    contribution=weights[name] if enabled else 0,
                    reason=(
                        "fewest completed private actions"
                        if name == "fairness" and enabled
                        else "participant remains eligible for this private scene"
                        if name == "private_available" and enabled
                        else "most recent private speaker is cooling down"
                        if name == "recent_speaker" and enabled
                        else "participant has already acted in this private scene"
                        if name == "repeat_risk" and enabled
                        else "one private action remains before the personal limit"
                        if name == "budget_pressure" and enabled
                        else "not applicable"
                    ),
                )
                for name, enabled in active.items()
            )
            scores.append(
                CandidateScore(
                    player_id=player_id,
                    features=features,
                    base_total=sum(feature.contribution for feature in features),
                )
            )
        return sorted(scores, key=lambda score: (-score.base_total, score.player_id))

    async def _probe_top_two(
        self,
        scene: PrivateChatScene,
        scores: list[CandidateScore],
    ) -> tuple[list[CandidateScore], list[EventRecord]]:
        self.probed_player_ids = tuple(score.player_id for score in scores[:2])
        adjustments: dict[str, int] = {}
        events: list[EventRecord] = []
        trigger = self._private_trigger(scene)
        for score in scores[:2]:
            if self._state_provider().stopped or not self._owns(scene):
                break
            adjustment = 0
            decision = "probe_failed"
            agent = self.agents.get(score.player_id)
            if agent is not None:
                for attempt in range(2):
                    if not self._owns(scene):
                        break
                    try:
                        probe = await agent.probe(trigger)
                    except ModelCallError:
                        decision = "probe_model_call_failed"
                        if attempt == 1 or self._state_provider().stopped or not self._owns(scene):
                            break
                        await self._at_safe_point()
                        if self._state_provider().stopped or not self._owns(scene):
                            break
                        continue
                    if not self._owns(scene):
                        break
                    if getattr(probe, "fallback", False):
                        decision = "probe_fallback"
                        if attempt == 1 or self._state_provider().stopped or not self._owns(scene):
                            break
                        await self._at_safe_point()
                        if self._state_provider().stopped or not self._owns(scene):
                            break
                        continue
                    adjustment, decision = self._bounded_adjustment(probe)
                    break
            if not self._owns(scene):
                break
            if decision == "silent":
                self._mark_quiet(scene, score.player_id)
            adjustments[score.player_id] = adjustment
            events.append(
                self._observer_event(
                    "scheduler.probe_adjustment",
                    scene,
                    {"player_id": score.player_id, "decision": decision, "urgency_adjustment": adjustment},
                )
            )
            await self._emit((events[-1],))
            await self._at_safe_point()
        self.probe_adjustments = adjustments
        return ([score.with_probe_adjustment(adjustments.get(score.player_id, 0)) for score in scores], events)

    async def _run_one_action(self, scene: PrivateChatScene, player_id: str) -> list[EventRecord]:
        agent = self.agents.get(player_id)
        if agent is None:
            self._mark_quiet(scene, player_id)
            return [self._observer_event("scheduler.action_rejected", scene, {"player_id": player_id, "reason": "missing_agent"}), *self._quiet_events(scene, [])]
        agent_scene = AgentScene(
            phase="day.private",
            purpose="private_chat",
            allowed_tools=("speak_private", "leave_private_chat", "update_notebook", "yield_action"),
            private_context_only=True,
            context_events=tuple(scene.transcript),
            details={"chat_id": scene.chat_id, "participants": list(scene.participant_ids)},
        )
        if not self._owns(scene):
            self._end(scene, "ownership_lost")
            return []
        if self._state_provider().stopped:
            self._end(scene, "stopped")
            return [self._observer_event("scheduler.stopped", scene)]
        try:
            outcome = await agent.run_action(agent_scene)
        except ModelCallError:
            if not self._owns(scene):
                self._end(scene, "ownership_lost")
                return []
            if self._state_provider().stopped:
                self._end(scene, "stopped")
                return [self._observer_event("scheduler.stopped", scene)]
            await self._at_safe_point()
            if not self._owns(scene):
                self._end(scene, "ownership_lost")
                return []
            if self._state_provider().stopped:
                self._end(scene, "stopped")
                return [self._observer_event("scheduler.stopped", scene)]
            try:
                outcome = await agent.run_action(agent_scene)
            except ModelCallError:
                if not self._owns(scene):
                    self._end(scene, "ownership_lost")
                    return []
                self._mark_quiet(scene, player_id)
                return [
                    self._observer_event(
                        "scheduler.normal_action_failed",
                        scene,
                        {"player_id": player_id, "reason": "yield_after_retry"},
                    ),
                    *self._quiet_events(scene, []),
                ]
        if not self._owns(scene):
            self._end(scene, "ownership_lost")
            return []
        return self._apply_outcome(scene, player_id, outcome)

    def _apply_outcome(self, scene: PrivateChatScene, player_id: str, outcome: AgentOutcome) -> list[EventRecord]:
        if not self._owns(scene):
            self._end(scene, "ownership_lost")
            return []
        if self._state_provider().stopped:
            self._end(scene, "stopped")
            return [self._observer_event("scheduler.stopped", scene)]
        action = outcome.action
        if action is None:
            self._mark_quiet(scene, player_id)
            return self._quiet_events(scene, [])
        if getattr(action, "actor", None) != player_id:
            self._mark_quiet(scene, player_id)
            return [
                self._observer_event("scheduler.action_rejected", scene, {"player_id": player_id, "reason": "wrong_actor"}),
                *self._quiet_events(scene, []),
            ]
        if isinstance(action, UpdateNotebook):
            self._mark_quiet(scene, player_id)
            return [
                self._observer_event("scheduler.action_rejected", scene, {"player_id": player_id, "reason": "outward_notebook"}),
                *self._quiet_events(scene, []),
            ]
        if isinstance(action, YieldAction):
            self._mark_quiet(scene, player_id)
            return self._quiet_events(scene, [])
        if isinstance(action, LeavePrivateChat) and action.actor == player_id and action.chat_id == scene.chat_id:
            self._end(scene, "left")
            return [self._observer_event("scheduler.private_left", scene, {"player_id": player_id})]
        if isinstance(action, SpeakPrivate) and action.actor == player_id and action.chat_id == scene.chat_id:
            scene.action_count += 1
            scene.action_counts[player_id] = scene.action_counts.get(player_id, 0) + 1
            scene.last_speaker = player_id
            events = [
                EventRecord(
                    phase="day.private",
                    type="chat.private_message",
                    actor=player_id,
                    audience=Audience.players(set(scene.participant_ids)),
                    payload={"chat_id": scene.chat_id, "text": action.text},
                )
            ]
            scene.transcript.append(events[0])
            scene.quiet_player_ids.clear()
            scene.quiet_count = 0
            if scene.action_count >= self.action_budget:
                self._end(scene, "action_budget")
            return events
        self._mark_quiet(scene, player_id)
        return [
            self._observer_event("scheduler.action_rejected", scene, {"player_id": player_id, "reason": "non_private_or_wrong_actor"}),
            *self._quiet_events(scene, []),
        ]

    def _choose(self, scene: PrivateChatScene, scores: list[CandidateScore]) -> str | None:
        eligible = [
            score
            for score in scores
            if score.total >= self.eligibility_threshold
            and score.player_id not in scene.quiet_player_ids
            and scene.action_counts.get(score.player_id, 0) < self.per_player_action_limit
        ]
        selected = choose_candidate(
            eligible,
            seed_state=f"{self.seed}:private:{self._selection_number}:{scene.chat_id}",
        )
        self._selection_number += 1
        return selected

    def _mark_quiet(self, scene: PrivateChatScene, player_id: str) -> None:
        scene.quiet_player_ids.add(player_id)
        scene.quiet_count = len(scene.quiet_player_ids)
        if self._all_quiet(scene):
            self._end(scene, "quiet")

    @staticmethod
    def _all_quiet(scene: PrivateChatScene) -> bool:
        return set(scene.participant_ids) <= scene.quiet_player_ids

    def _quiet_events(self, scene: PrivateChatScene, scores: list[CandidateScore]) -> list[EventRecord]:
        return [
            self._observer_event(
                "scheduler.quiet_window",
                scene,
                {
                    "quiet_count": scene.quiet_count,
                    "eligible_count": sum(score.total >= self.eligibility_threshold for score in scores),
                },
            )
        ]

    def _ranking_event(
        self,
        scene: PrivateChatScene,
        scores: list[CandidateScore],
    ) -> EventRecord:
        return self._observer_event(
            "scheduler.ranking",
            scene,
            {
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
                ]
            },
        )

    def _selection_event(self, scene: PrivateChatScene, player_id: str) -> EventRecord:
        return self._observer_event(
            "scheduler.selection",
            scene,
            {"player_id": player_id, "reason": "seeded_private_participant_choice"},
        )

    def _end(self, scene: PrivateChatScene, reason: str) -> None:
        self.end_reason = reason

    def _release_reservation(self, scene: PrivateChatScene) -> None:
        state = self._state_provider()
        if state.active_scene == scene.chat_id:
            state.active_scene = None
            if state.phase == "day.private":
                state.phase = scene.parent_phase
        if self._scene is scene:
            self._scene = None

    @staticmethod
    def _bounded_adjustment(probe: ReactionProbe) -> tuple[int, str]:
        urgency = max(-15, min(15, probe.urgency))
        if probe.decision == "respond":
            return urgency, "respond"
        if probe.decision == "defer":
            return min(0, urgency), "defer"
        return -15, "silent"

    @staticmethod
    def _private_trigger(scene: PrivateChatScene) -> EventRecord:
        if scene.transcript:
            return scene.transcript[-1].model_copy(deep=True)
        return EventRecord(
            phase="day.private",
            type="chat.private_message",
            audience=Audience.players(set(scene.participant_ids)),
            payload={"chat_id": scene.chat_id},
        )

    def _owns(self, scene: PrivateChatScene) -> bool:
        state = self._state_provider()
        return self._scene is scene and state.active_scene == scene.chat_id and state.phase == "day.private"

    @staticmethod
    def _public_shell(event_type: str, scene: PrivateChatScene) -> EventRecord:
        return EventRecord(
            phase="day.discussion",
            type=event_type,
            audience=Audience.public(),
            payload={"chat_id": scene.chat_id, "participants": list(scene.participant_ids)},
        )

    @staticmethod
    def _observer_event(
        event_type: str,
        scene: PrivateChatScene,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        base = {"chat_id": scene.chat_id}
        if payload:
            base.update(payload)
        return EventRecord(
            phase="day.private",
            type=event_type,
            audience=Audience.observer(),
            payload=base,
        )

    async def _emit(self, events: Sequence[EventRecord]) -> None:
        if self._event_sink is not None and events:
            await self._event_sink(tuple(events))

    async def _at_safe_point(self) -> None:
        if self._safe_point is not None:
            await self._safe_point()


__all__ = ["PrivateChatRequest", "PrivateChatScene", "PrivateChatScheduler"]
