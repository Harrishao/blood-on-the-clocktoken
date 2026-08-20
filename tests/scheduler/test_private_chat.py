from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from clocktower.agents.player import AgentOutcome, ReactionProbe
from clocktower.domain.actions import LeavePrivateChat, SpeakPrivate, SpeakPublic, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.history import HistoryWriteError
from clocktower.models.protocol import ModelCallError
from clocktower.scheduler.private_chat import PrivateChatScheduler
from tests.builders import sample_game_state


@dataclass
class ScriptedPrivateAgent:
    invitation: str | Exception = "accept"
    probe_result: ReactionProbe | Exception = field(
        default_factory=lambda: ReactionProbe(decision="respond", urgency=0, action_type="speak")
    )
    actions: list[object | Exception] = field(default_factory=list)
    invitations: list[EventRecord] = field(default_factory=list)
    probes: list[EventRecord] = field(default_factory=list)
    scenes: list[object] = field(default_factory=list)

    async def respond_private_invitation(self, invitation: EventRecord):
        self.invitations.append(invitation)
        if isinstance(self.invitation, Exception):
            raise self.invitation
        return self.invitation

    async def probe(self, event: EventRecord):
        self.probes.append(event)
        if isinstance(self.probe_result, Exception):
            raise self.probe_result
        return self.probe_result

    async def run_action(self, scene):
        self.scenes.append(scene)
        item = self.actions.pop(0) if self.actions else YieldAction(actor="alice", reason="quiet")
        if isinstance(item, Exception):
            raise item
        return AgentOutcome(action=item, round_trips=1)


def make_scheduler(*, invitee_decision: str | Exception = "accept", actions=None, **kwargs):
    state = sample_game_state()
    state.phase = "day.discussion"
    agents = {
        player_id: ScriptedPrivateAgent()
        for player_id in state.players
    }
    agents["bob"].invitation = invitee_decision
    if actions is not None:
        agents["alice"].actions = list(actions)
    scheduler = PrivateChatScheduler(
        state_provider=lambda: state,
        agents=agents,
        seed=17,
        **kwargs,
    )
    return scheduler, state, agents


async def test_request_is_private_to_invitee_and_accept_creates_only_active_two_person_scene():
    """An invitation must not leak to the inviter, spectators, or a public prompt."""

    scheduler, state, agents = make_scheduler()

    result = await scheduler.request("alice", "bob")

    assert result.decision == "accept"
    assert result.scene is not None
    assert result.scene.participant_ids == ("alice", "bob")
    assert state.active_scene == result.scene.chat_id
    invitation = result.events[0]
    assert invitation.type == "chat.private_invitation"
    assert invitation.audience == Audience.player("bob")
    assert agents["bob"].invitations == [invitation]
    assert all(not agent.invitations for player_id, agent in agents.items() if player_id != "bob")
    assert all(event.audience.kind != "public" for event in result.events)


@pytest.mark.parametrize("decision", ["reject", "defer"])
async def test_rejected_or_deferred_invitation_creates_no_active_chat(decision: str):
    """Only acceptance may reserve the single private-scene slot."""

    scheduler, state, _agents = make_scheduler(invitee_decision=decision)

    result = await scheduler.request("alice", "bob")

    assert result.decision == decision
    assert result.scene is None
    assert state.active_scene is None
    assert all(event.audience.kind != "public" for event in result.events)


async def test_invitation_short_call_retries_once_then_defers_without_a_scene():
    """An unavailable or malformed short model cannot create a chat by guessing."""

    scheduler, state, agents = make_scheduler(
        invitee_decision=ModelCallError("short unavailable")
    )

    result = await scheduler.request("alice", "bob")

    assert result.decision == "defer"
    assert result.scene is None
    assert state.active_scene is None
    assert len(agents["bob"].invitations) == 2


async def test_request_rejects_invalid_players_night_and_an_existing_scene():
    """Private conversation is a day-only, single-scene operation."""

    scheduler, state, _agents = make_scheduler()
    with pytest.raises(ValueError, match="different"):
        await scheduler.request("alice", "alice")
    with pytest.raises(ValueError, match="unknown"):
        await scheduler.request("alice", "nobody")

    state.phase = "night"
    with pytest.raises(ValueError, match="day"):
        await scheduler.request("alice", "bob")
    state.phase = "day.discussion"
    state.active_scene = "other-chat"
    with pytest.raises(ValueError, match="active"):
        await scheduler.request("alice", "bob")


async def test_private_messages_are_visible_only_to_two_participants_and_public_shells_have_no_secret_payload():
    """Bodies and reasons never cross the private audience boundary."""

    scheduler, state, agents = make_scheduler(
        actions=[SpeakPrivate(actor="alice", chat_id="pending", text="secret body")],
        action_budget=1,
    )
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].actions = [
        SpeakPrivate(actor="alice", chat_id=result.scene.chat_id, text="secret body")
    ]

    events = await scheduler.run(result.scene.chat_id)

    message = next(event for event in events if event.type == "chat.private_message")
    assert message.audience.player_ids == frozenset({"alice", "bob"})
    assert not message.visible_to("carol")
    shells = [event for event in events if event.audience.kind == "public"]
    assert [event.type for event in shells] == ["chat.private_started", "chat.private_ended"]
    assert all("secret body" not in repr(event.payload) for event in shells)
    assert all(set(event.payload) <= {"chat_id", "participants"} for event in shells)
    assert state.active_scene is None
    assert scheduler.end_reason == "action_budget"


async def test_private_candidates_and_normal_actions_are_limited_to_participants():
    """A third player must not be probed, prompted, or accepted inside the subscene."""

    scheduler, _state, agents = make_scheduler(action_budget=1)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].actions = [
        SpeakPrivate(actor="alice", chat_id=result.scene.chat_id, text="two only")
    ]

    await scheduler.run(result.scene.chat_id)

    assert set(scheduler.probed_player_ids) <= {"alice", "bob"}
    assert all(not agents[player_id].probes and not agents[player_id].scenes for player_id in {"carol", "david", "eve"})
    scene = agents["alice"].scenes[0]
    assert scene.phase == "day.private"
    assert scene.allowed_tools == ("speak_private", "leave_private_chat", "update_notebook", "yield_action")
    assert scene.details == {"chat_id": result.scene.chat_id, "participants": ["alice", "bob"]}
    assert scene.private_context_only is True


async def test_private_chat_ends_after_both_players_have_no_new_information():
    """Two participant silences end a private scene instead of spinning indefinitely."""

    scheduler, state, agents = make_scheduler(quiet_windows=2)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    for player_id in ("alice", "bob"):
        agents[player_id].probe_result = ReactionProbe(decision="silent", urgency=0, action_type="yield")

    await scheduler.run(result.scene.chat_id)

    assert scheduler.end_reason == "quiet"
    assert state.active_scene is None


async def test_leave_stop_and_model_failure_end_and_clear_the_active_scene():
    """Every bounded termination path must release the one-scene reservation."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].actions = [LeavePrivateChat(actor="alice", chat_id=result.scene.chat_id)]
    await scheduler.run(result.scene.chat_id)
    assert scheduler.end_reason == "left"
    assert state.active_scene is None

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].actions = [ModelCallError("normal unavailable"), ModelCallError("still unavailable")]
    await scheduler.run(result.scene.chat_id)
    assert scheduler.end_reason == "model_failed"
    assert state.active_scene is None

    scheduler, state, _agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    state.stopped = True
    await scheduler.run(result.scene.chat_id)
    assert scheduler.end_reason == "stopped"
    assert state.active_scene is None


async def test_run_rejects_wrong_chat_and_private_action_from_nonparticipant():
    """Chat identifiers and actor membership are authoritative scene boundaries."""

    scheduler, _state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    with pytest.raises(ValueError, match="active"):
        await scheduler.run("wrong-chat")
    agents["alice"].actions = [SpeakPublic(actor="alice", text="not private")]

    events = await scheduler.run(result.scene.chat_id)

    assert not any(event.type == "player.public_message" for event in events)
    assert scheduler.end_reason == "quiet"


async def test_history_write_failure_from_private_calls_is_not_swallowed():
    """Persistence failures must reach Task 11 instead of being recast as a quiet chat."""

    scheduler, _state, _agents = make_scheduler(invitee_decision=HistoryWriteError("disk failed"))

    with pytest.raises(HistoryWriteError, match="disk failed"):
        await scheduler.request("alice", "bob")
