from dataclasses import dataclass, field

from clocktower.agents.player import AgentOutcome, ReactionProbe
from clocktower.domain.actions import RequestPrivateChat, SpeakPublic
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


async def test_history_write_failure_from_normal_action_is_not_swallowed():
    """A history failure in the normal call cannot be downgraded to an optional yield."""

    scheduler, agents, _rules = make_scheduler()
    for agent in agents.values():
        agent.action = [HistoryWriteError("normal disk failure")]

    import pytest

    with pytest.raises(HistoryWriteError, match="normal disk failure"):
        await scheduler.step()


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


async def test_quiet_audit_counts_only_candidates_above_threshold():
    """Reporting raw scores as eligible would mislead observers about why a scene ended quietly."""

    scheduler, _agents, _rules = make_scheduler(eligibility_threshold=1000)
    events = await scheduler.step()

    quiet = next(event for event in events if event.type == "scheduler.quiet_window")
    assert quiet.payload["eligible_count"] == 0
