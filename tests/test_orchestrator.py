from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from clocktower.agents.player import AgentOutcome, AgentScene, ReactionProbe
from clocktower.config import GameConfig
from clocktower.domain.actions import (
    CastVote,
    Nominate,
    RequestPrivateChat,
    SpeakPrivate,
    SpeakPublic,
    UseAbility,
    YieldAction,
)
from clocktower.event_stream import EventStream
from clocktower.history import HistoryWriteError, HistoryWriter
from clocktower.models.protocol import ModelCallError
from clocktower.orchestrator import GameOrchestrator
from clocktower.rules.engine import RuleEngine


@dataclass
class NightAgent:
    player_id: str
    started: asyncio.Event | None = None
    release: asyncio.Event | None = None
    failures: list[Exception] = field(default_factory=list)
    scenes: list[object] = field(default_factory=list)
    probes: list[object] = field(default_factory=list)

    async def probe(self, event):
        self.probes.append(event)
        return ReactionProbe(decision="silent", urgency=0, action_type="yield")

    async def respond_private_invitation(self, invitation):
        return "reject"

    async def run_action(self, scene):
        self.scenes.append(scene)
        if self.failures:
            raise self.failures.pop(0)
        if scene.purpose == "night_ability":
            if self.started is not None:
                self.started.set()
            if self.release is not None:
                await self.release.wait()
            targets = tuple(scene.details["legal_targets"][0])
            return AgentOutcome(
                action=UseAbility(
                    actor=self.player_id,
                    action=scene.details["ability"],
                    targets=targets,
                ),
                round_trips=1,
            )
        if scene.purpose == "nomination_response":
            return AgentOutcome(
                action=SpeakPublic(actor=self.player_id, text="I respond."),
                round_trips=1,
            )
        if scene.purpose == "vote":
            return AgentOutcome(
                action=CastVote(
                    actor=self.player_id,
                    nomination_id=scene.details["nomination_id"],
                    vote=True,
                ),
                round_trips=1,
            )
        return AgentOutcome(
            action=YieldAction(actor=self.player_id, reason="quiet"),
            status="yielded",
            round_trips=1,
        )


class ToggleHistoryWriter(HistoryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_writes = True

    async def append(self, event):
        if self.fail_writes:
            raise HistoryWriteError("disk unavailable")
        return await super().append(event)


def make_orchestrator(
    tmp_path,
    agents,
    *,
    history_type=HistoryWriter,
    reload_model_config=None,
    discussion_action_budget=4,
    private_chat_action_budget=2,
):
    rules = RuleEngine.start_game(tuple(agents), seed=17)
    history = history_type(tmp_path / "game.jsonl", EventStream())
    orchestrator = GameOrchestrator(
        rules=rules,
        agents=agents,
        history=history,
        game_config=GameConfig(
            seed=17,
            player_ids=tuple(agents),
            history_directory=tmp_path,
            discussion_action_budget=discussion_action_budget,
            discussion_quiet_windows=1,
            private_chat_action_budget=private_chat_action_budget,
            private_chat_quiet_windows=1,
        ),
        reload_model_config=reload_model_config,
    )
    return orchestrator, rules, history


async def test_stop_waits_for_atomic_model_rule_and_history_commit_before_blocking_next_step(tmp_path):
    """Checking stop inside the model await would discard its valid rule action."""

    started = asyncio.Event()
    release = asyncio.Event()
    agents = {
        player_id: NightAgent(
            player_id,
            started=started if player_id == "david" else None,
            release=release if player_id == "david" else None,
        )
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    orchestrator, rules, history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await orchestrator.request_stop()
        await orchestrator.request_stop()
        release.set()
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)

        assert orchestrator.status().state == "stopped"
        assert orchestrator.status().reason == "stop_requested"
        assert rules.state.stopped is True
        assert rules.state.role_state.pending_night_role is None
        assert rules.state.role_state.night_step_index == 3
        assert any(event.type == "poison.applied" for event in history.stream.after(0))
        assert not any(
            event.type == "storyteller.decision"
            and str(event.payload.get("request_key", "")).startswith("information:spy")
            for event in history.stream.after(0)
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_required_model_failure_retries_once_then_stops_without_fallback(tmp_path):
    """A third call or a synthesized target would violate the required-choice policy."""

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"].failures = [ModelCallError("first"), ModelCallError("second")]
    orchestrator, rules, history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)

        assert orchestrator.status().reason == "required_model_call_failed"
        assert len(agents["david"].scenes) == 2
        assert rules.state.role_state.pending_night_role == "poisoner"
        assert rules.state.role_state.pending_night_actor_id == "david"
        assert rules.state.role_state.poisoned_player_id is None
        assert not any(event.type == "poison.applied" for event in history.stream.after(0))
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_required_agent_illegal_exhaustion_stops_without_second_run_action(tmp_path):
    """PlayerAgent already spent its one correction, so the orchestrator cannot grant another."""

    class ExhaustedIllegalAgent(NightAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            return AgentOutcome(
                action=None,
                status="required_action_failed",
                round_trips=2,
                illegal_corrections=1,
            )

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"] = ExhaustedIllegalAgent("david")
    orchestrator, rules, _history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)

        assert orchestrator.status().reason == "required_illegal_action_failed"
        assert len(agents["david"].scenes) == 1
        assert rules.state.role_state.pending_night_role == "poisoner"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_required_missing_action_has_its_own_single_retry_and_reason(tmp_path):
    """A valid provider response without a required action is neither provider failure nor fresh correction."""

    class MissingRequiredActionAgent(NightAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            return AgentOutcome(
                action=None,
                status="required_action_failed",
                round_trips=1,
                illegal_corrections=0,
            )

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"] = MissingRequiredActionAgent("david")
    orchestrator, rules, _history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)

        assert orchestrator.status().reason == "required_action_failed"
        assert len(agents["david"].scenes) == 2
        assert rules.state.role_state.pending_night_role == "poisoner"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_required_rule_correction_is_independent_of_a_prior_model_retry(tmp_path):
    """A recovered provider call must not consume the one rule-legality correction allowance."""

    correction_started = asyncio.Event()
    correction_release = asyncio.Event()

    class ModelThenIllegalThenCorrectedAgent(NightAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            if len(self.scenes) == 1:
                raise ModelCallError("temporary")
            if len(self.scenes) == 2:
                return AgentOutcome(
                    action=UseAbility(
                        actor=self.player_id,
                        action="imp",
                        targets=tuple(scene.details["legal_targets"][0]),
                    ),
                    round_trips=1,
                )
            correction_started.set()
            await correction_release.wait()
            return AgentOutcome(
                action=UseAbility(
                    actor=self.player_id,
                    action=scene.details["ability"],
                    targets=tuple(scene.details["legal_targets"][0]),
                ),
                round_trips=1,
            )

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"] = ModelThenIllegalThenCorrectedAgent("david")
    orchestrator, rules, history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(correction_started.wait(), timeout=1)
        assert orchestrator.status().state == "running"
        assert len(agents["david"].scenes) == 3
        errors = [event for event in history.stream.after(0) if event.type == "tool.error"]
        assert len(errors) == 1
        assert errors[0].audience.kind == "player"

        await orchestrator.request_stop()
        correction_release.set()
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert rules.state.role_state.pending_night_role is None
    finally:
        correction_release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_final_nomination_rule_correction_is_independent_of_model_retry(tmp_path):
    """The hard-budget nomination chance keeps its correction after one provider retry."""

    class ModelThenIllegalThenNominateAgent(NightAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            if scene.purpose != "final_nomination":
                return await NightAgent.run_action(self, scene)
            final_calls = sum(item.purpose == "final_nomination" for item in self.scenes)
            if final_calls == 1:
                raise ModelCallError("temporary")
            if final_calls == 2:
                return AgentOutcome(
                    action=Nominate(actor="bob", target="eve", accusation="wrong actor"),
                    round_trips=1,
                )
            return AgentOutcome(
                action=Nominate(actor=self.player_id, target="eve", accusation="corrected"),
                round_trips=1,
            )

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["alice"] = ModelThenIllegalThenNominateAgent("alice")
    orchestrator, rules, history = make_orchestrator(tmp_path, agents)
    await orchestrator._commit_events(rules.events)
    await orchestrator._run_night()

    nominated = await orchestrator._optional_nomination_action(
        "alice",
        AgentScene(
            phase="day.discussion",
            purpose="final_nomination",
            allowed_tools=("nominate", "yield_action", "update_notebook"),
        ),
    )

    assert nominated is True
    assert rules.state.phase == "day.nomination"
    assert sum(item.purpose == "final_nomination" for item in agents["alice"].scenes) == 3
    assert sum(event.type == "tool.error" for event in history.stream.after(0)) == 1


async def test_history_failure_stops_before_rules_then_continue_retries_history_and_reloads_models(tmp_path):
    """Losing the first audit record must not start any rule or model step."""

    started = asyncio.Event()
    release = asyncio.Event()
    agents = {
        player_id: NightAgent(
            player_id,
            started=started if player_id == "david" else None,
            release=release if player_id == "david" else None,
        )
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    reloads: list[str] = []

    def reload_models():
        reloads.append("reloaded")

    orchestrator, rules, history = make_orchestrator(
        tmp_path,
        agents,
        history_type=ToggleHistoryWriter,
        reload_model_config=reload_models,
    )
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert orchestrator.status().reason == "history_write_failed"
        assert all(not agent.scenes and not agent.probes for agent in agents.values())
        assert rules.state.role_state.night_step_index == 0

        history.fail_writes = False
        before = rules.state.model_dump(exclude={"stopped"})
        await orchestrator.continue_game()
        assert rules.state.model_dump(exclude={"stopped"}) == before
        await asyncio.wait_for(started.wait(), timeout=1)

        assert reloads == ["reloaded"]
        records = history.stream.after(0)
        assert records[0].type == "game.header"
        reload_event = next(event for event in records if event.type == "model_config_reloaded")
        assert reload_event.audience.kind == "observer"
        assert orchestrator.status().history_path == str(history.path)
    finally:
        release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_unknown_required_call_exception_is_not_swallowed_or_relabelled(tmp_path):
    """Catching RuntimeError as a provider failure would hide programming defects."""

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"].failures = [RuntimeError("programming defect")]
    orchestrator, _rules, _history = make_orchestrator(tmp_path, agents)

    with pytest.raises(RuntimeError, match="programming defect"):
        await asyncio.wait_for(orchestrator.run(), timeout=1)


async def test_agent_history_failure_stops_and_continue_retries_the_same_required_choice(tmp_path):
    """Treating an agent-side history failure as a provider retry would run while unaudited."""

    retry_started = asyncio.Event()
    retry_release = asyncio.Event()

    class HistoryThenBlockingAgent(NightAgent):
        async def run_action(self, scene):
            if len(self.scenes) == 0:
                self.scenes.append(scene)
                raise HistoryWriteError("agent segment was not durable")
            self.started = retry_started
            self.release = retry_release
            return await super().run_action(scene)

    agents = {
        player_id: NightAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    agents["david"] = HistoryThenBlockingAgent("david")
    orchestrator, rules, _history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert orchestrator.status().reason == "history_write_failed"
        assert rules.state.role_state.pending_night_role == "poisoner"
        assert len(agents["david"].scenes) == 1

        await orchestrator.continue_game()
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        assert len(agents["david"].scenes) == 2
        await orchestrator.request_stop()
        retry_release.set()
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert rules.state.role_state.pending_night_role is None
        assert rules.state.role_state.poisoned_player_id is not None
    finally:
        retry_release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_discussion_history_failure_stops_then_retries_without_starting_a_normal_call(tmp_path):
    """A failed probe audit must stop the owner loop instead of escaping or continuing unaudited."""

    retry_started = asyncio.Event()
    retry_release = asyncio.Event()
    failed_once = False

    class HistoryThenBlockingProbeAgent(NightAgent):
        async def probe(self, event):
            nonlocal failed_once
            self.probes.append(event)
            if event.type == "day.started" and not failed_once:
                failed_once = True
                raise HistoryWriteError("discussion segment was not durable")
            if event.type == "day.started":
                retry_started.set()
                await retry_release.wait()
            return ReactionProbe(decision="silent", urgency=0, action_type="yield")

    agents = {
        player_id: HistoryThenBlockingProbeAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    orchestrator, rules, _history = make_orchestrator(tmp_path, agents)
    task = asyncio.create_task(orchestrator.run())
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert orchestrator.status().reason == "history_write_failed"
        assert rules.state.phase == "day.discussion"
        assert all(not agent.scenes or agent.scenes[-1].purpose == "night_ability" for agent in agents.values())

        await orchestrator.continue_game()
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        await orchestrator.request_stop()
        retry_release.set()
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        assert orchestrator.status().reason == "stop_requested"
        assert sum(len(agent.probes) for agent in agents.values()) == 2
    finally:
        retry_release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_private_history_failure_continue_retries_same_action_in_same_scene(tmp_path):
    """Continue resumes the failed private opportunity instead of ending the scene and returning empty."""

    retry_started = asyncio.Event()
    retry_release = asyncio.Event()

    class PrivateHistoryThenBlockingAgent(NightAgent):
        private_calls = 0

        async def respond_private_invitation(self, invitation):
            return "accept"

        async def probe(self, event):
            self.probes.append(event)
            if event.phase == "day.private":
                return ReactionProbe(decision="respond", urgency=5, action_type="speak")
            return ReactionProbe(decision="silent", urgency=0, action_type="yield")

        async def run_action(self, scene):
            if scene.purpose != "private_chat":
                return await super().run_action(scene)
            self.scenes.append(scene)
            self.private_calls += 1
            if self.private_calls == 1:
                raise HistoryWriteError("private notebook was not durable")
            retry_started.set()
            await retry_release.wait()
            return AgentOutcome(
                action=SpeakPrivate(
                    actor=self.player_id,
                    chat_id=scene.details["chat_id"],
                    text="recovered once",
                ),
                round_trips=1,
            )

    agents = {
        player_id: PrivateHistoryThenBlockingAgent(player_id)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    orchestrator, rules, history = make_orchestrator(tmp_path, agents)
    await orchestrator._commit_events(rules.events)
    await orchestrator._run_night()
    task = asyncio.create_task(
        orchestrator._run_private_chat(
            RequestPrivateChat(actor="alice", target_player="bob")
        )
    )
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        chat_id = rules.state.active_scene
        assert orchestrator.status().reason == "history_write_failed"
        assert rules.state.phase == "day.private"
        assert chat_id is not None
        assert not any(event.type == "chat.private_ended" for event in history.stream.after(0))

        await orchestrator.continue_game()
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        assert rules.state.active_scene == chat_id
        assert sum(agent.private_calls for agent in agents.values()) == 2
        await orchestrator.request_stop()
        retry_release.set()
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        messages = [event for event in history.stream.after(0) if event.type == "chat.private_message"]
        assert len(messages) == 1
        assert messages[0].payload["chat_id"] == chat_id
    finally:
        retry_release.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_concurrent_continue_requests_reload_model_configuration_once(tmp_path):
    """Continue is one control transition even when two API callers race."""

    night_started = asyncio.Event()
    night_release = asyncio.Event()
    agents = {
        player_id: NightAgent(
            player_id,
            started=night_started if player_id == "david" else None,
            release=night_release if player_id == "david" else None,
        )
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    reload_started = asyncio.Event()
    reload_release = asyncio.Event()
    reload_count = 0

    async def reload_models():
        nonlocal reload_count
        reload_count += 1
        reload_started.set()
        await reload_release.wait()

    orchestrator, _rules, history = make_orchestrator(
        tmp_path,
        agents,
        history_type=ToggleHistoryWriter,
        reload_model_config=reload_models,
    )
    run_task = asyncio.create_task(orchestrator.run())
    first_continue = second_continue = None
    try:
        await asyncio.wait_for(orchestrator.wait_until_stopped(), timeout=1)
        history.fail_writes = False
        first_continue = asyncio.create_task(orchestrator.continue_game())
        await asyncio.wait_for(reload_started.wait(), timeout=1)
        second_continue = asyncio.create_task(orchestrator.continue_game())
        await asyncio.sleep(0)

        assert reload_count == 1
        reload_release.set()
        await asyncio.wait_for(
            asyncio.gather(first_continue, second_continue),
            timeout=1,
        )
        await asyncio.wait_for(night_started.wait(), timeout=1)
    finally:
        reload_release.set()
        night_release.set()
        for task in (first_continue, second_continue, run_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(
                *(task for task in (first_continue, second_continue, run_task) if task is not None),
                return_exceptions=True,
            ),
            timeout=1,
        )


@dataclass
class DayBudgetScript:
    private_requested: bool = False
    public_calls: int = 0
    private_calls: int = 0
    final_probe_ids: list[str] = field(default_factory=list)


class DayBudgetAgent(NightAgent):
    def __init__(self, player_id: str, script: DayBudgetScript):
        super().__init__(player_id)
        self.script = script

    async def respond_private_invitation(self, invitation):
        return "accept"

    async def probe(self, event):
        self.probes.append(event)
        if event.type == "day.final_nomination_probe":
            self.script.final_probe_ids.append(self.player_id)
            return ReactionProbe(decision="silent", urgency=0, action_type="yield")
        if event.phase == "day.private":
            return ReactionProbe(decision="respond", urgency=5, action_type="speak")
        if not self.script.private_requested:
            return ReactionProbe(decision="respond", urgency=10, action_type="private_chat")
        return ReactionProbe(decision="silent", urgency=0, action_type="yield")

    async def run_action(self, scene):
        if scene.purpose == "public_discussion":
            self.scenes.append(scene)
            self.script.public_calls += 1
            self.script.private_requested = True
            target = "bob" if self.player_id != "bob" else "alice"
            return AgentOutcome(
                action=RequestPrivateChat(actor=self.player_id, target_player=target),
                round_trips=1,
            )
        if scene.purpose == "private_chat":
            self.scenes.append(scene)
            self.script.private_calls += 1
            return AgentOutcome(
                action=SpeakPrivate(
                    actor=self.player_id,
                    chat_id=scene.details["chat_id"],
                    text=f"private action {self.script.private_calls}",
                ),
                round_trips=1,
            )
        return await super().run_action(scene)


async def test_private_request_that_exhausts_day_budget_goes_directly_to_final_probes(tmp_path):
    """The request itself is an outward action and cannot defer the hard-limit final sweep."""

    script = DayBudgetScript()
    agents = {
        player_id: DayBudgetAgent(player_id, script)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    orchestrator, rules, history = make_orchestrator(
        tmp_path,
        agents,
        discussion_action_budget=1,
        private_chat_action_budget=5,
    )
    await orchestrator._commit_events(rules.events)
    await orchestrator._run_night()

    await orchestrator._run_day()

    records = history.stream.after(0)
    assert script.public_calls == 1
    assert script.private_calls == 0
    assert not any(event.type == "chat.private_invitation" for event in records)
    assert len(script.final_probe_ids) == 5
    assert rules.state.phase == "night"


async def test_private_actions_consume_the_same_global_day_budget(tmp_path):
    """Private actions use only the day owner's remainder and prevent another public action."""

    script = DayBudgetScript()
    agents = {
        player_id: DayBudgetAgent(player_id, script)
        for player_id in ("alice", "bob", "carol", "david", "eve")
    }
    orchestrator, rules, history = make_orchestrator(
        tmp_path,
        agents,
        discussion_action_budget=3,
        private_chat_action_budget=5,
    )
    await orchestrator._commit_events(rules.events)
    await orchestrator._run_night()

    await orchestrator._run_day()

    records = history.stream.after(0)
    assert script.public_calls == 1
    assert script.private_calls == 2
    assert sum(event.type == "chat.private_message" for event in records) == 2
    assert len(script.final_probe_ids) == 5
    assert rules.state.phase == "night"
