"""Single-run autonomous Trouble Brewing lifecycle and runtime controls."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from clocktower.agents.player import AgentOutcome, AgentScene, PlayerAgent, ReactionProbe
from clocktower.config import GameConfig
from clocktower.domain.actions import (
    CastVote,
    IllegalAction,
    Nominate,
    PlayerAction,
    RequestPrivateChat,
    SpeakPublic,
    UseAbility,
    YieldAction,
)
from clocktower.domain.events import Audience, EventRecord
from clocktower.history import HistoryWriteError, HistoryWriter
from clocktower.models.protocol import ModelCallError
from clocktower.rules.engine import RuleEngine
from clocktower.scheduler.discussion import DiscussionScheduler
from clocktower.scheduler.private_chat import PrivateChatScheduler


RuntimeState = Literal["ready", "running", "stopped", "ended"]


class RuntimeStatus(BaseModel):
    """Small read-only projection consumed by the later HTTP API."""

    model_config = ConfigDict(frozen=True)

    state: RuntimeState
    reason: str | None
    phase: str
    day: int
    winner: str | None
    history_path: str


class GameOrchestrator:
    """Own the only run loop while rules remain the only true-state authority."""

    def __init__(
        self,
        *,
        rules: RuleEngine,
        agents: Mapping[str, PlayerAgent | Any],
        history: HistoryWriter,
        game_config: GameConfig,
        reload_model_config: Callable[[], object | Awaitable[object]] | None = None,
    ) -> None:
        if tuple(agents) != tuple(rules.state.players):
            raise ValueError("agents must match the production player order")
        if tuple(game_config.player_ids) != tuple(rules.state.players):
            raise ValueError("game configuration must match the active game")
        if game_config.seed != rules.state.seed:
            raise ValueError("game configuration seed must match the active game")
        self.rules = rules
        self.agents = dict(agents)
        self.history = history
        self.game_config = game_config
        self._reload_model_config = reload_model_config or (lambda: None)
        self._runtime_state: RuntimeState = "ready"
        self._reason: str | None = None
        self._stop_requested = False
        self._run_started = False
        self._control_lock = asyncio.Lock()
        self._stopped_event = asyncio.Event()
        self._continue_event = asyncio.Event()
        self._continue_pending = False
        self._private_scheduler = PrivateChatScheduler(
            state_provider=lambda: self.rules.state,
            agents=self.agents,
            seed=self.rules.state.seed,
            action_budget=game_config.private_chat_action_budget,
            quiet_windows=game_config.private_chat_quiet_windows,
            event_sink=self._commit_events,
            safe_point=self._safe_point,
        )

    async def run(self) -> None:
        """Run exactly one setup/night/day loop to the RuleEngine's winner."""

        if self._run_started:
            raise RuntimeError("this single-game orchestrator has already been run")
        self._run_started = True
        self._runtime_state = "running"
        await self._commit_events(self.rules.events)
        await self._safe_point()

        while self.rules.check_winner() is None:
            phase = self.rules.state.phase
            if phase == "night":
                await self._run_night()
            elif phase.startswith("day"):
                await self._run_day()
            else:
                raise RuntimeError(f"unsupported game phase: {phase}")
            if self.rules.check_winner() is None:
                await self._safe_point()

        self.rules.state.stopped = False
        self._runtime_state = "ended"
        self._reason = self.rules.state.role_state.winner_reason

    async def request_stop(self) -> None:
        """Idempotently request the next safe scheduling boundary."""

        async with self._control_lock:
            if self._runtime_state in {"stopped", "ended"}:
                return
            self._stop_requested = True

    async def continue_game(self) -> None:
        """Reload only model connection configuration, then resume the same run."""

        async with self._control_lock:
            if self._runtime_state != "stopped" or self._continue_pending:
                return
            self._continue_pending = True
        try:
            reloaded = self._reload_model_config()
            if inspect.isawaitable(reloaded):
                await reloaded
        except BaseException:
            async with self._control_lock:
                self._continue_pending = False
            raise
        async with self._control_lock:
            if self._runtime_state != "stopped":
                self._continue_pending = False
                return
            self._continue_event.set()

    def status(self) -> RuntimeStatus:
        """Return only the runtime fields needed by the control API."""

        state = self.rules.state
        return RuntimeStatus(
            state=self._runtime_state,
            reason=self._reason,
            phase=state.phase,
            day=state.day,
            winner=self.rules.check_winner(),
            history_path=str(self.history.path),
        )

    async def wait_until_stopped(self) -> None:
        """Wait until a requested or policy stop has reached a safe boundary."""

        await self._stopped_event.wait()

    async def _run_night(self) -> None:
        while self.rules.state.phase == "night" and self.rules.check_winner() is None:
            await self._safe_point()
            role_state = self.rules.state.role_state
            if role_state.pending_night_role is None:
                events = self.rules.advance_night_step()
                await self._commit_events(events)
                continue

            actor_id = role_state.pending_night_actor_id
            if actor_id is None:
                raise RuntimeError("pending night ability has no actor")
            details = self._pending_night_details(actor_id, role_state.pending_night_role)
            scene = AgentScene(
                phase="night",
                purpose="night_ability",
                required=True,
                allowed_tools=("use_ability", "update_notebook"),
                details=details,
            )
            await self._required_rule_action(
                actor_id,
                scene,
                lambda action: (
                    isinstance(action, UseAbility)
                    and action.actor == actor_id
                    and action.action == details["ability"]
                ),
            )

    async def _run_day(self) -> None:
        if self.rules.state.phase != "day.discussion":
            raise RuntimeError(f"day started in unsupported phase: {self.rules.state.phase}")
        day_started = EventRecord(
            phase="day.discussion",
            type="day.started",
            audience=Audience.public(),
            payload={"day": self.rules.state.day},
        )
        committed_start = await self._commit_events((day_started,))
        scheduler = DiscussionScheduler(
            state_provider=lambda: self.rules.state,
            agents=self.agents,
            rules=self.rules,
            trigger_event=committed_start[0],
            seed=self.rules.state.seed + self.rules.state.day,
            action_budget=self.game_config.discussion_action_budget,
            quiet_windows=self.game_config.discussion_quiet_windows,
            allow_private_chat_requests=True,
            event_sink=self._commit_events,
            safe_point=self._safe_point,
        )

        while (
            self.rules.state.phase.startswith("day")
            and self.rules.check_winner() is None
            and scheduler.end_reason is None
        ):
            await self._safe_point()
            try:
                events = await scheduler.step()
            except HistoryWriteError:
                await self._pause_until_continue("history_write_failed")
                continue
            private_request = scheduler.pending_private_request
            scheduler.pending_private_request = None
            if private_request is not None:
                private_events = await self._run_private_chat(private_request)
                public_shells = [event for event in private_events if event.audience.kind == "public"]
                if public_shells:
                    scheduler.trigger_event = public_shells[-1]
            if self.rules.state.phase == "day.nomination":
                resumed = await self._run_nomination()
                if resumed is not None:
                    scheduler.trigger_event = resumed
            if self.rules.state.phase == "night" or self.rules.check_winner() is not None:
                return
            if not events and scheduler.end_reason is None:
                raise RuntimeError("discussion scheduler made no progress")

        if self.rules.state.phase != "day.discussion" or self.rules.check_winner() is not None:
            return
        await self._run_final_nomination_probes()
        if self.rules.state.phase != "day.discussion" or self.rules.check_winner() is not None:
            return
        await self._safe_point()
        await self._commit_events(self.rules.end_day())

    async def _run_private_chat(self, action: RequestPrivateChat) -> list[EventRecord]:
        while True:
            try:
                request = await self._private_scheduler.request(
                    action.actor,
                    action.target_player,
                )
            except HistoryWriteError:
                await self._pause_until_continue("history_write_failed")
                continue
            except ValueError as error:
                await self._commit_events(
                    (
                        self._tool_error(
                            action.actor,
                            phase="day.discussion",
                            reason=type(error).__name__,
                        ),
                    )
                )
                return []
            break
        if request.scene is None:
            return list(request.events)
        try:
            return await self._private_scheduler.run(request.scene.chat_id)
        except HistoryWriteError:
            await self._pause_until_continue("history_write_failed")
            return []

    async def _run_nomination(self) -> EventRecord | None:
        opened = self._latest_rule_event("nomination.opened")
        if opened is None:
            raise RuntimeError("nomination phase has no opened nomination")
        if self.rules.state.phase != "day.nomination" or self.rules.check_winner() is not None:
            return None

        nominee = str(opened.payload["nominee"])
        response_scene = AgentScene(
            phase="day.nomination_response",
            purpose="nomination_response",
            required=True,
            allowed_tools=("speak_public", "update_notebook"),
            details={"nomination_id": opened.payload["nomination_id"]},
        )
        await self._required_rule_action(
            nominee,
            response_scene,
            lambda action: isinstance(action, SpeakPublic) and action.actor == nominee,
        )
        if self.rules.check_winner() is not None or self.rules.state.phase != "day.nomination":
            return None

        nomination_id = str(opened.payload["nomination_id"])
        for voter_id in tuple(self.rules.current_vote_order):
            vote_scene = AgentScene(
                phase="day.voting",
                purpose="vote",
                required=True,
                allowed_tools=("cast_vote", "update_notebook"),
                details={"nomination_id": nomination_id},
            )
            await self._required_rule_action(
                voter_id,
                vote_scene,
                lambda action, voter_id=voter_id: (
                    isinstance(action, CastVote)
                    and action.actor == voter_id
                    and action.nomination_id == nomination_id
                ),
            )
            if self.rules.check_winner() is not None:
                return None

        if self.rules.state.phase != "day.discussion":
            return None
        resumed = EventRecord(
            phase="day.discussion",
            type="day.discussion_resumed",
            audience=Audience.public(),
            payload={"day": self.rules.state.day, "after": nomination_id},
        )
        return (await self._commit_events((resumed,)))[0]

    async def _run_final_nomination_probes(self) -> None:
        probed: set[str] = set()
        while self.rules.state.phase == "day.discussion" and self.rules.check_winner() is None:
            nominators, nominees = self._day_nomination_sets()
            eligible = [
                player.player_id
                for player in sorted(self.rules.state.players.values(), key=lambda item: item.seat)
                if player.alive
                and player.player_id not in nominators
                and player.player_id not in probed
            ]
            if not eligible:
                return
            player_id = eligible[0]
            probed.add(player_id)
            trigger = EventRecord(
                phase="day.discussion",
                type="day.final_nomination_probe",
                audience=Audience.public(),
                payload={"day": self.rules.state.day, "player_id": player_id},
            )
            committed_trigger = (await self._commit_events((trigger,)))[0]
            probe = await self._short_probe(player_id, committed_trigger)
            await self._commit_events(
                (
                    EventRecord(
                        phase="day.discussion",
                        type="scheduler.final_probe_result",
                        actor=player_id,
                        audience=Audience.observer(),
                        payload={
                            "player_id": player_id,
                            "decision": probe.decision,
                            "action_type": probe.action_type,
                            "urgency": probe.urgency,
                            "fallback": probe.fallback,
                        },
                    ),
                )
            )
            await self._safe_point()
            if probe.decision != "respond" or probe.action_type != "nominate":
                continue
            legal_targets = tuple(
                candidate.player_id
                for candidate in sorted(
                    self.rules.state.players.values(), key=lambda item: item.seat
                )
                if candidate.player_id not in nominees
            )
            scene = AgentScene(
                phase="day.discussion",
                purpose="final_nomination",
                allowed_tools=("nominate", "yield_action", "update_notebook"),
                details={"legal_targets": legal_targets},
            )
            nominated = await self._optional_nomination_action(player_id, scene)
            if nominated and self.rules.state.phase == "day.nomination":
                await self._run_nomination()

    async def _short_probe(self, player_id: str, event: EventRecord) -> ReactionProbe:
        agent = self.agents.get(player_id)
        if agent is None:
            return ReactionProbe.fallback_silent()
        for attempt in range(2):
            await self._safe_point()
            probe: ReactionProbe | None = None
            while True:
                try:
                    probe = await agent.probe(event)
                except HistoryWriteError:
                    await self._pause_until_continue("history_write_failed")
                    continue
                except ModelCallError:
                    probe = None
                break
            if probe is None:
                if attempt == 0:
                    continue
                return ReactionProbe.fallback_silent()
            if getattr(probe, "fallback", False) and attempt == 0:
                continue
            return probe
        return ReactionProbe.fallback_silent()

    async def _optional_nomination_action(self, player_id: str, scene: AgentScene) -> bool:
        agent = self.agents.get(player_id)
        call_failures = 0
        correction_used = False
        while True:
            await self._safe_point()
            if agent is None:
                break
            try:
                outcome = await agent.run_action(scene)
            except HistoryWriteError:
                await self._pause_until_continue("history_write_failed")
                continue
            except ModelCallError:
                call_failures += 1
                if call_failures < 2:
                    continue
                break
            call_failures = 0
            action = outcome.action
            if isinstance(action, YieldAction) or action is None:
                await self._commit_events(
                    self.rules.apply_action(
                        action
                        if isinstance(action, YieldAction)
                        else YieldAction(actor=player_id, reason="optional_action_failed")
                    )
                )
                return False
            if not isinstance(action, Nominate) or action.actor != player_id:
                error: Exception = IllegalAction("final action is not a nomination")
            else:
                try:
                    events = self.rules.apply_action(action)
                except IllegalAction as caught:
                    error = caught
                else:
                    await self._commit_events(events)
                    await self._safe_point()
                    return True
            if not correction_used:
                correction_used = True
                await self._commit_events(
                    (self._tool_error(player_id, phase="day.discussion", reason=type(error).__name__),)
                )
                continue
            break
        await self._commit_events(
            self.rules.apply_action(
                YieldAction(actor=player_id, reason="optional_model_call_failed")
            )
        )
        return False

    async def _required_rule_action(
        self,
        player_id: str,
        scene: AgentScene,
        accepts: Callable[[PlayerAction | None], bool],
    ) -> None:
        while True:
            agent = self.agents.get(player_id)
            call_failures = 0
            correction_used = False
            while True:
                await self._safe_point()
                outcome: AgentOutcome | None = None
                if agent is None:
                    call_failure = True
                else:
                    while True:
                        try:
                            outcome = await agent.run_action(scene)
                        except HistoryWriteError:
                            await self._pause_until_continue("history_write_failed")
                            continue
                        except ModelCallError:
                            call_failure = True
                        else:
                            call_failure = (
                                outcome.status == "required_action_failed"
                                or outcome.action is None
                            )
                        break

                if call_failure:
                    call_failures += 1
                    if call_failures < 2:
                        continue
                    break

                call_failures = 0
                action = outcome.action if outcome is not None else None
                failure: IllegalAction | None = None
                if not accepts(action):
                    failure = IllegalAction("required action was not completed")
                elif action is not None:
                    try:
                        events = self.rules.apply_action(action)
                    except IllegalAction as error:
                        failure = error
                    else:
                        await self._commit_events(events)
                        await self._safe_point()
                        return

                if failure is not None and not correction_used:
                    correction_used = True
                    await self._commit_events(
                        (
                            self._tool_error(
                                player_id,
                                phase=scene.phase or self.rules.state.phase,
                                reason=type(failure).__name__,
                            ),
                        )
                    )
                    continue
                break

            await self._commit_events(
                (
                    EventRecord(
                        phase=scene.phase or self.rules.state.phase,
                        type="orchestrator.required_action_failed",
                        actor=player_id,
                        audience=Audience.observer(),
                        payload={"player_id": player_id, "purpose": scene.purpose},
                    ),
                )
            )
            await self._pause_until_continue("required_model_call_failed")

    def _pending_night_details(self, actor_id: str, role: str) -> dict[str, Any]:
        for event in reversed(self.rules.events):
            if (
                event.type == "ability.choice_requested"
                and event.payload.get("actor_id") == actor_id
                and event.payload.get("role") == role
            ):
                return {
                    "ability": role,
                    "legal_targets": event.payload["legal_targets"],
                }
        raise RuntimeError("pending night ability has no legal target record")

    def _day_nomination_sets(self) -> tuple[set[str], set[str]]:
        prefix = f"nom-day-{self.rules.state.day}-"
        opened = [
            event
            for event in self.rules.events
            if event.type == "nomination.opened"
            and str(event.payload.get("nomination_id", "")).startswith(prefix)
        ]
        return (
            {str(event.payload["nominator"]) for event in opened},
            {str(event.payload["nominee"]) for event in opened},
        )

    def _latest_rule_event(self, event_type: str) -> EventRecord | None:
        return next(
            (event for event in reversed(self.rules.events) if event.type == event_type),
            None,
        )

    async def _commit_events(
        self,
        events: Sequence[EventRecord],
    ) -> tuple[EventRecord, ...]:
        committed: list[EventRecord] = []
        for event in events:
            recovering = False
            while True:
                try:
                    record = await self.history.append(event)
                except HistoryWriteError:
                    recovering = True
                    await self._wait_for_continue("history_write_failed")
                    continue
                committed.append(record)
                break
            if recovering:
                await self._publish_reload_and_resume()
        return tuple(committed)

    async def _safe_point(self) -> None:
        if self.rules.check_winner() is not None:
            return
        if self._stop_requested:
            await self._pause_until_continue("stop_requested")
        elif self.rules.state.stopped:
            await self._pause_until_continue(self._reason or "stopped")

    async def _pause_until_continue(self, reason: str) -> None:
        await self._wait_for_continue(reason)
        await self._publish_reload_and_resume()

    async def _wait_for_continue(self, reason: str) -> None:
        async with self._control_lock:
            self.rules.state.stopped = True
            self._runtime_state = "stopped"
            self._reason = reason
            self._continue_pending = False
            self._continue_event.clear()
            self._stopped_event.set()
        await self._continue_event.wait()

    async def _publish_reload_and_resume(self) -> None:
        while True:
            reload_event = EventRecord(
                phase=self.rules.state.phase,
                type="model_config_reloaded",
                audience=Audience.observer(),
                payload={"source": "continue"},
            )
            try:
                await self.history.append(reload_event)
            except HistoryWriteError:
                await self._wait_for_continue("history_write_failed")
                continue
            async with self._control_lock:
                self._continue_pending = False
                self._continue_event.clear()
                self._stop_requested = False
                self.rules.state.stopped = False
                self._runtime_state = "running"
                self._reason = None
                self._stopped_event.clear()
            return

    @staticmethod
    def _tool_error(player_id: str, *, phase: str, reason: str) -> EventRecord:
        return EventRecord(
            phase=phase,
            type="tool.error",
            actor=player_id,
            audience=Audience.player(player_id),
            payload={"reason": reason, "correction_allowed": True},
        )


__all__ = ["GameOrchestrator", "RuntimeStatus"]
