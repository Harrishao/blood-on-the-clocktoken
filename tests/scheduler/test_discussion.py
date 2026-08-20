import asyncio
from dataclasses import dataclass, field

from clocktower.agents.player import AgentOutcome, ReactionProbe
from clocktower.domain.actions import IllegalAction, Nominate, RequestPrivateChat, SpeakPublic, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.history import HistoryWriteError
from clocktower.models.protocol import ModelCallError
from clocktower.scheduler.discussion import DiscussionScheduler
from tests.builders import private_message, public_claim, sample_game_state


@dataclass
class ScriptedAgent:
    probe_result: ReactionProbe | Exception = field(
        default_factory=lambda: ReactionProbe(decision="respond", urgency=0, action_type="speak")
    )
    action: object | None = None
    probes: list[str] = field(default_factory=list)
    scenes: list[object] = field(default_factory=list)

    async def probe(self, event):
        self.probes.append(event.actor or "")
        if isinstance(self.probe_result, list):
            result = self.probe_result.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if isinstance(self.probe_result, Exception):
            raise self.probe_result
        return self.probe_result

    async def run_action(self, scene):
        self.scenes.append(scene)
        if isinstance(self.action, list):
            result = self.action.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return AgentOutcome(action=self.action, round_trips=1)


@dataclass
class RecordingRules:
    actions: list[object] = field(default_factory=list)

    def apply_action(self, action):
        self.actions.append(action)
        return [
            EventRecord(
                phase="day.discussion",
                type="player.public_message",
                actor=action.actor,
                audience=Audience.public(),
                payload={"text": action.text},
            )
        ]


def make_scheduler(*, event=None, agents=None, **kwargs):
    state = sample_game_state()
    state.phase = "day.discussion"
    public_event = event or public_claim(actor="alice", mentions={"bob"})
    agents = agents or {
        player_id: ScriptedAgent(action=SpeakPublic(actor=player_id, text=f"{player_id} speaks"))
        for player_id in state.players
    }
    rules = RecordingRules()
    scheduler = DiscussionScheduler(
        state_provider=lambda: state,
        agents=agents,
        rules=rules,
        trigger_event=public_event,
        seed=17,
        **kwargs,
    )
    scheduler.test_state = state
    return scheduler, agents, rules


async def test_only_top_two_candidates_are_probed_and_the_probe_adjustment_is_bounded():
    """Probing every player or accepting unbounded urgency would let short calls control the scene."""

    scheduler, agents, rules = make_scheduler()
    agents["bob"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=-15, action_type="speak")

    events = await scheduler.step()

    assert scheduler.probed_player_ids == scheduler.initial_ranking[:2]
    assert sum(len(agent.probes) for agent in agents.values()) == 2
    assert all(-15 <= adjustment <= 15 for adjustment in scheduler.probe_adjustments.values())
    assert len(rules.actions) == 1
    audit = next(event for event in events if event.type == "scheduler.ranking")
    assert audit.audience.kind == "observer"


async def test_probe_failure_uses_zero_adjustment_and_still_runs_one_normal_action():
    """An unavailable short model must not abort public discussion or alter a base score."""

    scheduler, agents, rules = make_scheduler()
    agents["bob"].probe_result = ModelCallError("short model unavailable")

    events = await scheduler.step()

    assert scheduler.probe_adjustments["bob"] == 0
    assert len(rules.actions) == 1
    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert any(event.type == "scheduler.probe_adjustment" for event in events)


async def test_scheduler_rejects_private_action_and_never_sends_it_to_rules():
    """Forwarding a private-chat request would let public discussion escape its scene boundary."""

    scheduler, agents, rules = make_scheduler()
    for player_id, agent in agents.items():
        agent.action = RequestPrivateChat(actor=player_id, target_player="alice")

    events = await scheduler.step()

    assert rules.actions == []
    assert scheduler.action_count == 0
    assert any(event.type == "scheduler.action_rejected" for event in events)


async def test_orchestrated_discussion_surfaces_private_request_without_sending_it_to_rules():
    """Rejecting an authorized private request would make day/private alternation unreachable."""

    scheduler, agents, rules = make_scheduler(allow_private_chat_requests=True)
    for player_id, agent in agents.items():
        target = "bob" if player_id != "bob" else "alice"
        agent.action = RequestPrivateChat(actor=player_id, target_player=target)

    events = await scheduler.step()

    assert rules.actions == []
    assert scheduler.action_count == 1
    assert scheduler.pending_private_request is not None
    assert scheduler.pending_private_request.actor in agents
    assert any(event.type == "scheduler.private_chat_requested" for event in events)


async def test_orchestrator_hooks_commit_scheduler_facts_before_dependent_calls_and_check_safe_points():
    """Returning every audit event at step end would place rankings after their model calls."""

    markers: list[str] = []

    class OrderedAgent(ScriptedAgent):
        async def probe(self, event):
            markers.append(f"probe:{self.action.actor}")
            return await super().probe(event)

        async def run_action(self, scene):
            markers.append(f"normal:{self.action.actor}")
            return await super().run_action(scene)

    async def event_sink(events):
        markers.extend(f"event:{event.type}" for event in events)

    safe_points = 0

    async def safe_point():
        nonlocal safe_points
        safe_points += 1

    state = sample_game_state()
    state.phase = "day.discussion"
    agents = {
        player_id: OrderedAgent(action=SpeakPublic(actor=player_id, text="ordered"))
        for player_id in state.players
    }
    scheduler = DiscussionScheduler(
        state_provider=lambda: state,
        agents=agents,
        rules=RecordingRules(),
        trigger_event=public_claim(actor="alice", mentions={"bob"}),
        seed=17,
        event_sink=event_sink,
        safe_point=safe_point,
    )

    await scheduler.step()

    ranking = markers.index("event:scheduler.ranking")
    first_probe = next(index for index, marker in enumerate(markers) if marker.startswith("probe:"))
    selection = markers.index("event:scheduler.selection")
    normal = next(index for index, marker in enumerate(markers) if marker.startswith("normal:"))
    public = markers.index("event:player.public_message")
    assert ranking < first_probe < selection < normal < public
    assert safe_points == 5


async def test_committed_public_action_becomes_the_next_nonzero_trigger():
    """The next ranking must cite the durable public event, never its pre-commit seq=0 draft."""

    next_seq = 100

    async def event_sink(events):
        nonlocal next_seq
        committed = tuple(
            event.model_copy(update={"seq": next_seq + index})
            for index, event in enumerate(events)
        )
        next_seq += len(committed)
        return committed

    scheduler, _agents, _rules = make_scheduler(event_sink=event_sink)

    first = await scheduler.step()
    public = next(event for event in first if event.type == "player.public_message")
    assert public.seq > 0
    assert scheduler.trigger_event == public

    second = await scheduler.step()
    ranking = next(event for event in second if event.type == "scheduler.ranking")
    assert ranking.seq > public.seq
    assert ranking.payload["trigger_seq"] == public.seq
    assert all(event.seq > 0 for event in first + second)


async def test_public_action_sink_failure_retries_exact_batch_without_replaying_model_or_rules():
    """A durable-write retry owns the prepared action batch, not a second model/rule opportunity."""

    import pytest

    next_seq = 300
    failed_batch: tuple[EventRecord, ...] | None = None
    action_sink_attempts: list[tuple[EventRecord, ...]] = []

    async def event_sink(events):
        nonlocal next_seq, failed_batch
        batch = tuple(events)
        if any(event.type == "player.public_message" for event in batch):
            action_sink_attempts.append(batch)
            if failed_batch is None:
                failed_batch = batch
                raise HistoryWriteError("public action batch not durable")
        committed = tuple(
            event.model_copy(update={"seq": next_seq + index})
            for index, event in enumerate(batch)
        )
        next_seq += len(committed)
        return committed

    scheduler, agents, rules = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
    )

    with pytest.raises(HistoryWriteError, match="public action batch not durable"):
        await scheduler.step()

    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert len(rules.actions) == 1
    assert scheduler.action_count == 0
    assert scheduler.end_reason is None
    assert scheduler.pending_action_commit is not None
    assert isinstance(scheduler.pending_action_commit.action, SpeakPublic)

    recovered = await scheduler.step()

    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert len(rules.actions) == 1
    assert len(action_sink_attempts) == 2
    assert action_sink_attempts[1] == failed_batch
    assert scheduler.action_count == 1
    assert scheduler.end_reason == "action_budget"
    assert [event.type for event in recovered] == [
        "player.public_message",
        "scheduler.ended",
    ]


async def test_stop_arriving_during_ranking_sink_prevents_any_probe():
    """A committed ranking is a causal boundary before the first short model call."""

    ranking_started = asyncio.Event()
    ranking_release = asyncio.Event()
    stop_requested = False

    async def event_sink(events):
        if any(event.type == "scheduler.ranking" for event in events):
            ranking_started.set()
            await ranking_release.wait()
        return tuple(
            event.model_copy(update={"seq": index + 10})
            for index, event in enumerate(events)
        )

    async def safe_point():
        if stop_requested:
            scheduler.test_state.stopped = True

    scheduler, agents, rules = make_scheduler(
        event_sink=event_sink,
        safe_point=safe_point,
    )
    task = asyncio.create_task(scheduler.step())
    try:
        await asyncio.wait_for(ranking_started.wait(), timeout=1)
        stop_requested = True
        ranking_release.set()
        events = await asyncio.wait_for(task, timeout=1)

        assert all(not agent.probes and not agent.scenes for agent in agents.values())
        assert rules.actions == []
        assert events[-1].type == "scheduler.stopped"
    finally:
        ranking_release.set()
        if not task.done():
            task.cancel()


async def test_stop_arriving_during_selection_sink_prevents_normal_model_call():
    """Selection must be durable and pass the owner safe point before normal inference starts."""

    selection_started = asyncio.Event()
    selection_release = asyncio.Event()
    stop_requested = False

    async def event_sink(events):
        if any(event.type == "scheduler.selection" for event in events):
            selection_started.set()
            await selection_release.wait()
        return tuple(
            event.model_copy(update={"seq": index + 20})
            for index, event in enumerate(events)
        )

    async def safe_point():
        if stop_requested:
            scheduler.test_state.stopped = True

    scheduler, agents, rules = make_scheduler(
        event_sink=event_sink,
        safe_point=safe_point,
    )
    task = asyncio.create_task(scheduler.step())
    try:
        await asyncio.wait_for(selection_started.wait(), timeout=1)
        stop_requested = True
        selection_release.set()
        events = await asyncio.wait_for(task, timeout=1)

        assert sum(len(agent.probes) for agent in agents.values()) == 2
        assert all(not agent.scenes for agent in agents.values())
        assert rules.actions == []
        assert events[-1].type == "scheduler.stopped"
    finally:
        selection_release.set()
        if not task.done():
            task.cancel()


async def test_quiet_windows_and_hard_action_budget_end_the_scene():
    """Without both guards, a no-response scene could spin forever."""

    quiet_scheduler, _agents, _rules = make_scheduler(eligibility_threshold=1000, quiet_windows=2)
    await quiet_scheduler.step()
    await quiet_scheduler.step()

    assert quiet_scheduler.end_reason == "quiet"
    assert quiet_scheduler.quiet_count == 2

    capped_scheduler, _agents, capped_rules = make_scheduler(action_budget=1)
    await capped_scheduler.step()
    await capped_scheduler.step()

    assert capped_scheduler.end_reason == "action_budget"
    assert len(capped_rules.actions) == 1


async def test_private_trigger_is_not_scored_or_sent_to_any_player_agent():
    """Reading a private event at this boundary would expose another scene's information."""

    scheduler, agents, rules = make_scheduler(event=private_message({"alice", "bob"}))

    events = await scheduler.step()

    assert rules.actions == []
    assert all(not agent.probes and not agent.scenes for agent in agents.values())
    assert [event.type for event in events] == ["scheduler.quiet_window"]


async def test_out_of_range_probe_urgency_is_clamped_before_selection():
    """Trusting an adapter-bypassed urgency value would let a short call overpower base ranking."""

    @dataclass(frozen=True)
    class UncheckedProbe:
        decision: str = "respond"
        urgency: int = 999

    scheduler, agents, _rules = make_scheduler()
    agents["bob"].probe_result = UncheckedProbe()  # type: ignore[assignment]

    await scheduler.step()

    assert scheduler.probe_adjustments["bob"] == 15


async def test_probe_fallback_retries_once_then_uses_zero_without_silent_penalty():
    """A parser fallback is not a player's silent decision and must not receive a -15 penalty."""

    scheduler, agents, _rules = make_scheduler()
    agents["bob"].probe_result = [
        ReactionProbe.fallback_silent(),
        ReactionProbe(decision="respond", urgency=4, action_type="speak"),
    ]

    await scheduler.step()

    assert len(agents["bob"].probes) == 2
    assert scheduler.probe_adjustments["bob"] == 4


async def test_history_write_failure_from_probe_is_not_swallowed():
    """Turning durable-history failure into a scoring fallback leaves the game unaudited."""

    scheduler, agents, _rules = make_scheduler()
    agents["bob"].probe_result = HistoryWriteError("disk failed")

    import pytest

    with pytest.raises(HistoryWriteError, match="disk failed"):
        await scheduler.step()


async def test_normal_model_call_retries_once_but_history_failure_propagates():
    """Only ModelCallError is retriable; broad RuntimeError handling would hide persistence loss."""

    scheduler, agents, rules = make_scheduler()
    for player_id, agent in agents.items():
        agent.action = [
            ModelCallError("temporary provider error"),
            AgentOutcome(action=SpeakPublic(actor=player_id, text="retry succeeded"), round_trips=1),
        ]

    await scheduler.step()

    selected = next(player_id for player_id, agent in agents.items() if agent.scenes)
    assert len(agents[selected].scenes) == 2
    assert len(rules.actions) == 1


async def test_optional_normal_failure_retries_once_then_commits_a_yield():
    """Two provider failures must become a visible yield, never an invented public action."""

    @dataclass
    class YieldAwareRules(RecordingRules):
        def apply_action(self, action):
            self.actions.append(action)
            return [
                EventRecord(
                    phase="day.discussion",
                    type="player.yielded",
                    actor=action.actor,
                    audience=Audience.public(),
                    payload={"reason": action.reason},
                )
            ]

    scheduler, agents, _rules = make_scheduler()
    rules = YieldAwareRules()
    scheduler.rules = rules
    for agent in agents.values():
        agent.action = [ModelCallError("first"), ModelCallError("second")]

    events = await scheduler.step()

    selected = next(player_id for player_id, agent in agents.items() if agent.scenes)
    assert len(agents[selected].scenes) == 2
    assert len(rules.actions) == 1
    assert isinstance(rules.actions[0], YieldAction)
    assert any(event.type == "player.yielded" for event in events)


async def test_history_write_failure_from_normal_action_is_not_swallowed():
    """A history failure in the normal call cannot be downgraded to an optional yield."""

    scheduler, agents, _rules = make_scheduler()
    for agent in agents.values():
        agent.action = [HistoryWriteError("normal disk failure")]

    import pytest

    with pytest.raises(HistoryWriteError, match="normal disk failure"):
        await scheduler.step()


async def test_rule_illegal_action_gets_one_private_correction_turn():
    """Rule legality is known only after the model call, so the selected player gets one correction."""

    @dataclass
    class RejectFirstNominationRules:
        actions: list[object] = field(default_factory=list)

        def apply_action(self, action):
            self.actions.append(action)
            if isinstance(action, Nominate):
                raise IllegalAction("nomination is no longer legal")
            return [
                EventRecord(
                    phase="day.discussion",
                    type="player.public_message",
                    actor=action.actor,
                    audience=Audience.public(),
                    payload={"text": action.text},
                )
            ]

    scheduler, agents, _rules = make_scheduler()
    rules = RejectFirstNominationRules()
    scheduler.rules = rules
    for player_id, agent in agents.items():
        target = "bob" if player_id != "bob" else "alice"
        agent.action = [
            AgentOutcome(
                action=Nominate(actor=player_id, target=target, accusation="first"),
                round_trips=1,
            ),
            AgentOutcome(
                action=SpeakPublic(actor=player_id, text="corrected"),
                round_trips=1,
            ),
        ]

    events = await scheduler.step()

    selected_id = next(player_id for player_id, agent in agents.items() if agent.scenes)
    assert len(agents[selected_id].scenes) == 2
    assert len(rules.actions) == 2
    correction = next(event for event in events if event.type == "tool.error")
    assert correction.actor == selected_id
    assert correction.audience == Audience.player(selected_id)
    assert any(event.type == "player.public_message" for event in events)


async def test_second_rule_illegal_action_becomes_visible_optional_yield():
    """The one correction allowance must not become an unbounded rule-rejection loop."""

    @dataclass
    class RejectNominationsRules:
        actions: list[object] = field(default_factory=list)

        def apply_action(self, action):
            self.actions.append(action)
            if isinstance(action, Nominate):
                raise IllegalAction("still illegal")
            return [
                EventRecord(
                    phase="day.discussion",
                    type="player.yielded",
                    actor=action.actor,
                    audience=Audience.public(),
                    payload={"reason": action.reason},
                )
            ]

    scheduler, agents, _rules = make_scheduler()
    rules = RejectNominationsRules()
    scheduler.rules = rules
    for player_id, agent in agents.items():
        target = "bob" if player_id != "bob" else "alice"
        agent.action = [
            AgentOutcome(
                action=Nominate(actor=player_id, target=target, accusation="first"),
                round_trips=1,
            ),
            AgentOutcome(
                action=Nominate(actor=player_id, target=target, accusation="second"),
                round_trips=1,
            ),
        ]

    events = await scheduler.step()

    selected_id = next(player_id for player_id, agent in agents.items() if agent.scenes)
    assert len(agents[selected_id].scenes) == 2
    assert [type(action) for action in rules.actions] == [Nominate, Nominate, YieldAction]
    assert sum(event.type == "tool.error" for event in events) == 1
    assert any(event.type == "scheduler.action_rejected" for event in events)
    assert any(event.type == "player.yielded" for event in events)


async def test_forged_public_checkpoint_never_reaches_probe_or_normal_action():
    """Audience alone must not downgrade observer checkpoint state into a public trigger."""

    forged = EventRecord(
        phase="day.discussion",
        type="checkpoint",
        audience=Audience.public(),
        payload={"secret": "never prompt this"},
    )
    scheduler, agents, rules = make_scheduler(event=forged)

    await scheduler.step()

    assert rules.actions == []
    assert all(not agent.probes and not agent.scenes for agent in agents.values())


def test_scheduler_rejects_nonpositive_eligibility_threshold():
    """A zero threshold would let nonpositive candidates bypass quiet termination."""

    import pytest

    with pytest.raises(ValueError, match="eligibility_threshold"):
        make_scheduler(eligibility_threshold=0)


async def test_stopped_state_ends_at_each_safe_scheduler_boundary():
    """Calling a normal model or applying a rule after stop would cross a requested safe point."""

    scheduler, agents, rules = make_scheduler()
    scheduler.test_state.stopped = True
    initial_events = await scheduler.step()

    assert [event.type for event in initial_events] == ["scheduler.stopped"]
    assert rules.actions == []
    assert all(not agent.probes and not agent.scenes for agent in agents.values())

    class StopDuringNormalAction(ScriptedAgent):
        async def run_action(self, scene):
            outcome = await super().run_action(scene)
            scheduler.test_state.stopped = True
            return outcome

    scheduler, agents, rules = make_scheduler()
    scheduler.agents = {
        player_id: StopDuringNormalAction(action=SpeakPublic(actor=player_id, text="stop"))
        for player_id in agents
    }
    events = await scheduler.step()

    assert events[-1].type == "scheduler.stopped"
    assert rules.actions == []


async def test_stop_during_first_probe_prevents_second_probe_and_normal_action():
    """A stop raised within the first short call must be observed before another player is called."""

    scheduler, agents, rules = make_scheduler()

    class StopDuringProbe(ScriptedAgent):
        async def probe(self, event):
            result = await super().probe(event)
            scheduler.test_state.stopped = True
            return result

    agents["alice"] = StopDuringProbe(action=SpeakPublic(actor="alice", text="must not run"))
    scheduler.agents = agents
    events = await scheduler.step()

    assert scheduler.end_reason == "stopped"
    assert len(agents["alice"].probes) == 1
    assert all(not agent.probes for player_id, agent in agents.items() if player_id != "alice")
    assert all(not agent.scenes for agent in agents.values())
    assert rules.actions == []
    assert events[-1].type == "scheduler.stopped"


async def test_stop_after_first_normal_model_failure_prevents_retry():
    """A retry scheduled before rechecking stop would issue a model request after the safe boundary."""

    scheduler, agents, rules = make_scheduler()

    class StopOnModelFailure(ScriptedAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            scheduler.test_state.stopped = True
            raise ModelCallError("provider failed")

    agents["alice"] = StopOnModelFailure()
    scheduler.agents = agents
    events = await scheduler.step()

    assert scheduler.end_reason == "stopped"
    assert len(agents["alice"].scenes) == 1
    assert rules.actions == []
    assert events[-1].type == "scheduler.stopped"


async def test_stop_request_reaches_safe_point_before_failed_probe_retry():
    """The short-call retry is a new model call and may only start after the owner safe point."""

    stop_requested = False

    async def safe_point():
        if stop_requested:
            scheduler.test_state.stopped = True

    class RequestStopOnProbeFailure(ScriptedAgent):
        async def probe(self, event):
            nonlocal stop_requested
            self.probes.append(event.actor or "")
            stop_requested = True
            raise ModelCallError("short unavailable")

    scheduler, agents, rules = make_scheduler(safe_point=safe_point)
    agents["alice"] = RequestStopOnProbeFailure()
    scheduler.agents = agents

    events = await scheduler.step()

    assert len(agents["alice"].probes) == 1
    assert all(not agent.probes for player_id, agent in agents.items() if player_id != "alice")
    assert rules.actions == []
    assert events[-1].type == "scheduler.stopped"


async def test_unsafe_rule_event_does_not_replace_the_last_safe_trigger():
    """A forged public checkpoint may remain auditable but cannot drive the next discussion step."""

    class UnsafeRules(RecordingRules):
        def apply_action(self, action):
            self.actions.append(action)
            return [
                EventRecord(
                    phase="day.discussion",
                    type="checkpoint",
                    audience=Audience.public(),
                    payload={"observer_secret": "not a trigger"},
                )
            ]

    scheduler, agents, _rules = make_scheduler()
    rules = UnsafeRules()
    scheduler.rules = rules
    original = scheduler.trigger_event

    events = await scheduler.step()

    assert rules.actions
    assert scheduler.trigger_event == original
    assert any(event.type == "checkpoint" for event in events)


async def test_quiet_audit_counts_only_candidates_above_threshold():
    """Reporting raw scores as eligible would mislead observers about why a scene ended quietly."""

    scheduler, _agents, _rules = make_scheduler(eligibility_threshold=1000)
    events = await scheduler.step()

    quiet = next(event for event in events if event.type == "scheduler.quiet_window")
    assert quiet.payload["eligible_count"] == 0
