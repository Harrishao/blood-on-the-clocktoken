from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from clocktower.agents.player import AgentOutcome, PrivateInvitationResponse, ReactionProbe
from clocktower.domain.actions import LeavePrivateChat, SpeakPrivate, SpeakPublic, UpdateNotebook, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import Notebook
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
        if isinstance(self.invitation, list):
            result = self.invitation.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if isinstance(self.invitation, Exception):
            raise self.invitation
        return self.invitation

    async def probe(self, event: EventRecord):
        self.probes.append(event)
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
    state.phase = "day.nomination_response"
    with pytest.raises(ValueError, match="day.discussion"):
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


async def test_orchestrator_sink_commits_private_transcript_once_at_each_causal_boundary():
    """Batching a private transcript until scene end would reorder it after later model calls."""

    markers: list[str] = []

    async def event_sink(events):
        markers.extend(f"event:{event.type}" for event in events)
        return tuple(events)

    scheduler, _state, agents = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
    )
    original_invitation = agents["bob"].respond_private_invitation

    async def invitation_with_marker(event):
        markers.append("model:invitation")
        return await original_invitation(event)

    agents["bob"].respond_private_invitation = invitation_with_marker
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].actions = [
        SpeakPrivate(actor="alice", chat_id=result.scene.chat_id, text="one committed body")
    ]
    original_action = agents["alice"].run_action

    async def action_with_marker(scene):
        markers.append("model:private")
        return await original_action(scene)

    agents["alice"].run_action = action_with_marker

    returned = await scheduler.run(result.scene.chat_id)

    assert markers.index("event:chat.private_invitation") < markers.index("model:invitation")
    assert markers.index("event:chat.private_started") < markers.index("model:private")
    assert markers.index("model:private") < markers.index("event:chat.private_message")
    assert markers.index("event:chat.private_message") < markers.index("event:chat.private_ended")
    assert markers.count("event:chat.private_message") == 1
    assert sum(event.type == "chat.private_message" for event in returned) == 1


async def test_committed_private_message_replaces_seq_zero_transcript_draft():
    """The next participant receives one durable transcript record, while no-sink mode stays local."""

    next_seq = 200

    async def event_sink(events):
        nonlocal next_seq
        committed = tuple(
            event.model_copy(update={"seq": next_seq + index})
            for index, event in enumerate(events)
        )
        next_seq += len(committed)
        return committed

    scheduler, _state, agents = make_scheduler(
        action_budget=2,
        event_sink=event_sink,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    assert all(event.seq > 0 for event in request.events)
    chat_id = request.scene.chat_id
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["bob"].probe_result = ReactionProbe(decision="respond", urgency=-15, action_type="speak")
    agents["alice"].actions = [
        SpeakPrivate(actor="alice", chat_id=chat_id, text="one durable secret")
    ]
    agents["bob"].actions = [LeavePrivateChat(actor="bob", chat_id=chat_id)]

    returned = await scheduler.run(chat_id)

    assert all(event.seq > 0 for event in returned)
    assert len(request.scene.transcript) == 1
    assert request.scene.transcript[0].seq > 0
    bob_scene = agents["bob"].scenes[0]
    assert bob_scene.context_events == tuple(request.scene.transcript)


async def test_private_action_sink_failure_retries_exact_batch_without_replaying_model():
    """A prepared secret message remains pending until durable and finalizes its counters once."""

    next_seq = 400
    failed_batch: tuple[EventRecord, ...] | None = None
    action_sink_attempts: list[tuple[EventRecord, ...]] = []

    async def event_sink(events):
        nonlocal next_seq, failed_batch
        batch = tuple(events)
        if any(event.type == "chat.private_message" for event in batch):
            action_sink_attempts.append(batch)
            if failed_batch is None:
                failed_batch = batch
                raise HistoryWriteError("private action batch not durable")
        committed = tuple(
            event.model_copy(update={"seq": next_seq + index})
            for index, event in enumerate(batch)
        )
        next_seq += len(committed)
        return committed

    scheduler, state, agents = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} once")
        ]

    with pytest.raises(HistoryWriteError, match="private action batch not durable"):
        await scheduler.run(chat_id)

    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert request.scene.action_count == 0
    assert request.scene.transcript == []
    assert scheduler.end_reason is None
    assert state.active_scene == chat_id
    assert request.scene.pending_action_commit is not None
    assert isinstance(request.scene.pending_action_commit.action, SpeakPrivate)

    recovered = await scheduler.run(chat_id)

    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert len(action_sink_attempts) == 2
    assert action_sink_attempts[1] == failed_batch
    assert request.scene.action_count == 1
    assert len(request.scene.transcript) == 1
    assert request.scene.transcript[0].seq > 0
    assert sum(event.type == "chat.private_message" for event in recovered) == 1
    assert scheduler.end_reason == "action_budget"


async def test_partial_private_action_sink_result_preserves_the_pending_checkpoint():
    """A partial private sink result cannot finalize counters, transcript, or ending."""

    action_attempts = 0

    async def event_sink(events):
        nonlocal action_attempts
        batch = tuple(events)
        if any(event.type == "chat.private_message" for event in batch):
            action_attempts += 1
            return ()
        return batch

    scheduler, _state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} pending")
        ]

    with pytest.raises(RuntimeError, match="event sink returned 0 records for 1 events"):
        await scheduler.run(chat_id)

    assert action_attempts == 1
    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert request.scene.action_count == 0
    assert request.scene.transcript == []
    assert scheduler.end_reason is None
    assert request.scene.pending_action_commit is not None


async def test_response_commit_revalidates_before_creating_an_accepted_scene():
    """A committed accept cannot overwrite phase or ownership changed during its sink await."""

    async def event_sink(events):
        batch = tuple(events)
        if any(event.type == "chat.private_response" for event in batch):
            state.phase = "night"
            state.active_scene = "external-scene"
        return batch

    scheduler, state, _agents = make_scheduler(event_sink=event_sink)

    result = await scheduler.request("alice", "bob")

    assert result.decision == "accept"
    assert result.scene is None
    assert state.phase == "night"
    assert state.active_scene == "external-scene"
    assert scheduler._scene is None
    assert scheduler._pending_request is None


async def test_invitation_commit_revalidates_before_starting_the_invitee_model():
    """Every pending invitation stage rechecks current day ownership before a new call."""

    async def event_sink(events):
        batch = tuple(events)
        if any(event.type == "chat.private_invitation" for event in batch):
            state.phase = "night"
            state.active_scene = "external-scene"
        return batch

    scheduler, state, agents = make_scheduler(event_sink=event_sink)

    result = await scheduler.request("alice", "bob")

    assert result.decision == "defer"
    assert result.scene is None
    assert agents["bob"].invitations == []
    assert state.phase == "night"
    assert state.active_scene == "external-scene"
    assert scheduler._scene is None


async def test_failed_private_action_is_committed_before_external_ownership_loss():
    """An already-executed action stays reachable even if another owner advances the phase."""

    action_attempts: list[tuple[EventRecord, ...]] = []
    emitted: list[EventRecord] = []

    async def event_sink(events):
        batch = tuple(events)
        if any(event.type == "chat.private_message" for event in batch):
            action_attempts.append(batch)
            if len(action_attempts) == 1:
                raise HistoryWriteError("private message not durable")
        emitted.extend(batch)
        return batch

    scheduler, state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} happened")
        ]

    with pytest.raises(HistoryWriteError, match="private message not durable"):
        await scheduler.run(chat_id)

    state.phase = "night"
    state.active_scene = "external-scene"
    recovered = await scheduler.run(chat_id)

    assert len(action_attempts) == 2
    assert action_attempts[1] == action_attempts[0]
    assert sum(len(agent.scenes) for agent in agents.values()) == 1
    assert request.scene.pending_action_commit is None
    assert request.scene.action_count == 1
    assert len(request.scene.transcript) == 1
    assert scheduler.end_reason == "ownership_lost"
    assert [event.type for event in recovered] == [
        "chat.private_message",
        "chat.private_ended",
    ]
    assert state.phase == "night"
    assert state.active_scene == "external-scene"


async def test_private_started_sink_failure_keeps_parent_phase_and_retries_exactly_once():
    """The scene phase changes only after its public started shell is durable."""

    started_attempts: list[tuple[EventRecord, ...]] = []
    emitted: list[EventRecord] = []

    async def event_sink(events):
        batch = tuple(events)
        if any(event.type == "chat.private_started" for event in batch):
            started_attempts.append(batch)
            if len(started_attempts) == 1:
                raise HistoryWriteError("private started not durable")
        emitted.extend(batch)
        return batch

    scheduler, state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} after start")
        ]

    with pytest.raises(HistoryWriteError, match="private started not durable"):
        await scheduler.run(chat_id)

    assert state.phase == "day.discussion"
    assert state.active_scene == chat_id
    assert request.scene.pending_started_event is not None
    assert request.scene.started_event is None
    assert all(not agent.scenes for agent in agents.values())

    await scheduler.run(chat_id)

    assert len(started_attempts) == 2
    assert started_attempts[1] == started_attempts[0]
    assert sum(event.type == "chat.private_started" for event in emitted) == 1
    assert sum(len(agent.scenes) for agent in agents.values()) == 1


async def test_cancellation_waits_for_started_sink_checkpoint_before_propagating():
    """If the started sink succeeds under cancellation, reentry must not publish it twice."""

    started_entered = asyncio.Event()
    started_release = asyncio.Event()
    emitted: list[EventRecord] = []

    async def event_sink(events):
        batch = tuple(events)
        if any(event.type == "chat.private_started" for event in batch):
            started_entered.set()
            await started_release.wait()
        emitted.extend(batch)
        return batch

    scheduler, state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} after cancel")
        ]

    task = asyncio.create_task(scheduler.run(chat_id))
    await asyncio.wait_for(started_entered.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    started_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.phase == "day.private"
    assert request.scene.started_event is not None
    assert request.scene.pending_started_event is None
    assert sum(event.type == "chat.private_started" for event in emitted) == 1
    assert all(not agent.scenes for agent in agents.values())

    await scheduler.run(chat_id)

    assert sum(event.type == "chat.private_started" for event in emitted) == 1
    assert sum(len(agent.scenes) for agent in agents.values()) == 1


async def test_stopped_reentry_leaves_failed_private_action_pending_without_new_facts():
    """Calling the scheduler while stopped cannot discard or publish its pending action checkpoint."""

    action_attempts = 0

    async def event_sink(events):
        nonlocal action_attempts
        if any(event.type == "chat.private_message" for event in events):
            action_attempts += 1
            if action_attempts == 1:
                raise HistoryWriteError("private checkpoint unavailable")
        return tuple(events)

    scheduler, state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} pending")
        ]

    with pytest.raises(HistoryWriteError, match="private checkpoint unavailable"):
        await scheduler.run(chat_id)

    state.stopped = True
    assert await scheduler.run(chat_id) == []
    assert action_attempts == 1
    assert request.scene.action_count == 0
    assert scheduler.end_reason is None
    assert state.active_scene == chat_id

    state.stopped = False
    await scheduler.run(chat_id)
    assert action_attempts == 2
    assert sum(len(agent.scenes) for agent in agents.values()) == 1


async def test_invitation_history_failure_resumes_same_request_without_replaying_invitation():
    """An invitee notebook failure retries the response stage under the original request id."""

    emitted: list[EventRecord] = []

    async def event_sink(events):
        committed = tuple(
            event.model_copy(update={"seq": len(emitted) + index + 1})
            for index, event in enumerate(events)
        )
        emitted.extend(committed)
        return committed

    scheduler, state, agents = make_scheduler(event_sink=event_sink)
    agents["bob"].invitation = [HistoryWriteError("invite response not durable"), "accept"]

    with pytest.raises(HistoryWriteError, match="invite response not durable"):
        await scheduler.request("alice", "bob")

    first_invitation = next(event for event in emitted if event.type == "chat.private_invitation")
    assert state.active_scene is None

    recovered = await scheduler.request("alice", "bob")

    assert recovered.request_id == first_invitation.payload["request_id"]
    assert recovered.decision == "accept"
    assert recovered.scene is not None
    assert len(agents["bob"].invitations) == 2
    assert sum(event.type == "chat.private_invitation" for event in emitted) == 1
    assert sum(event.type == "chat.private_response" for event in emitted) == 1


async def test_selection_sink_failure_resumes_same_selection_without_replaying_prefix():
    """Selection persistence retries its exact fact before the selected normal model starts."""

    emitted: list[EventRecord] = []
    selection_attempts: list[tuple[EventRecord, ...]] = []
    failed_once = False

    async def event_sink(events):
        nonlocal failed_once
        batch = tuple(events)
        if any(event.type == "scheduler.selection" for event in batch):
            selection_attempts.append(batch)
            if not failed_once:
                failed_once = True
                raise HistoryWriteError("selection not durable")
        emitted.extend(batch)
        return batch

    scheduler, _state, agents = make_scheduler(action_budget=1, event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} selected")
        ]

    with pytest.raises(HistoryWriteError, match="selection not durable"):
        await scheduler.run(chat_id)

    assert sum(len(agent.probes) for agent in agents.values()) == 2
    assert all(not agent.scenes for agent in agents.values())

    await scheduler.run(chat_id)

    assert len(selection_attempts) == 2
    assert selection_attempts[1] == selection_attempts[0]
    assert sum(event.type == "scheduler.ranking" for event in emitted) == 1
    assert sum(event.type == "scheduler.probe_adjustment" for event in emitted) == 2
    assert sum(len(agent.probes) for agent in agents.values()) == 2
    assert sum(len(agent.scenes) for agent in agents.values()) == 1


async def test_cancel_after_selection_commit_resumes_selected_normal_without_replaying_prefix():
    """The in-memory selection cursor advances before awaiting its cancellable safe point."""

    emitted: list[EventRecord] = []
    selection_safe_started = asyncio.Event()
    block_selection_safe = False
    blocked_once = False

    async def event_sink(events):
        nonlocal block_selection_safe
        batch = tuple(events)
        emitted.extend(batch)
        if any(event.type == "scheduler.selection" for event in batch):
            block_selection_safe = True
        return batch

    async def safe_point():
        nonlocal blocked_once
        if block_selection_safe and not blocked_once:
            blocked_once = True
            selection_safe_started.set()
            await asyncio.Event().wait()

    scheduler, state, agents = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
        safe_point=safe_point,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} after cancel")
        ]

    task = asyncio.create_task(scheduler.run(chat_id))
    await asyncio.wait_for(selection_safe_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.active_scene == chat_id
    assert all(not agent.scenes for agent in agents.values())

    await scheduler.run(chat_id)

    assert sum(event.type == "scheduler.ranking" for event in emitted) == 1
    assert sum(event.type == "scheduler.probe_adjustment" for event in emitted) == 2
    assert sum(event.type == "scheduler.selection" for event in emitted) == 1
    assert sum(len(agent.probes) for agent in agents.values()) == 2
    assert sum(len(agent.scenes) for agent in agents.values()) == 1


async def test_stop_arriving_during_invitation_sink_prevents_invitee_model_call():
    """The private invitation must be durable before consent inference is allowed to start."""

    invitation_started = asyncio.Event()
    invitation_release = asyncio.Event()
    stop_requested = False

    async def event_sink(events):
        if any(event.type == "chat.private_invitation" for event in events):
            invitation_started.set()
            await invitation_release.wait()
        return tuple(events)

    async def safe_point():
        if stop_requested:
            state.stopped = True

    scheduler, state, agents = make_scheduler(
        event_sink=event_sink,
        safe_point=safe_point,
    )
    task = asyncio.create_task(scheduler.request("alice", "bob"))
    try:
        await asyncio.wait_for(invitation_started.wait(), timeout=1)
        stop_requested = True
        invitation_release.set()
        result = await asyncio.wait_for(task, timeout=1)

        assert agents["bob"].invitations == []
        assert result.scene is None
        assert state.active_scene is None
    finally:
        invitation_release.set()
        if not task.done():
            task.cancel()


async def test_stop_arriving_during_private_ranking_sink_prevents_any_probe():
    """Private ranking persistence is the safe boundary before participant probes."""

    ranking_started = asyncio.Event()
    ranking_release = asyncio.Event()
    stop_requested = False

    async def event_sink(events):
        if any(event.type == "scheduler.ranking" for event in events):
            ranking_started.set()
            await ranking_release.wait()
        return tuple(
            event.model_copy(update={"seq": index + 30})
            for index, event in enumerate(events)
        )

    async def safe_point():
        if stop_requested:
            state.stopped = True

    scheduler, state, agents = make_scheduler(
        event_sink=event_sink,
        safe_point=safe_point,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    task = asyncio.create_task(scheduler.run(request.scene.chat_id))
    try:
        await asyncio.wait_for(ranking_started.wait(), timeout=1)
        stop_requested = True
        ranking_release.set()
        events = await asyncio.wait_for(task, timeout=1)

        assert all(not agent.probes and not agent.scenes for agent in agents.values())
        assert any(event.type == "scheduler.stopped" for event in events)
    finally:
        ranking_release.set()
        if not task.done():
            task.cancel()


async def test_stop_arriving_during_private_selection_sink_prevents_normal_model_call():
    """A private selection cannot start its normal model before owner stop is observed."""

    selection_started = asyncio.Event()
    selection_release = asyncio.Event()
    stop_requested = False

    async def event_sink(events):
        if any(event.type == "scheduler.selection" for event in events):
            selection_started.set()
            await selection_release.wait()
        return tuple(
            event.model_copy(update={"seq": index + 40})
            for index, event in enumerate(events)
        )

    async def safe_point():
        if stop_requested:
            state.stopped = True

    scheduler, state, agents = make_scheduler(
        event_sink=event_sink,
        safe_point=safe_point,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    task = asyncio.create_task(scheduler.run(request.scene.chat_id))
    try:
        await asyncio.wait_for(selection_started.wait(), timeout=1)
        stop_requested = True
        selection_release.set()
        events = await asyncio.wait_for(task, timeout=1)

        assert sum(len(agent.probes) for agent in agents.values()) == 2
        assert all(not agent.scenes for agent in agents.values())
        assert any(event.type == "scheduler.stopped" for event in events)
    finally:
        selection_release.set()
        if not task.done():
            task.cancel()


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
    assert not agents["alice"].scenes and not agents["bob"].scenes


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
    events = await scheduler.run(result.scene.chat_id)
    assert scheduler.end_reason == "quiet"
    assert any(event.type == "scheduler.normal_action_failed" for event in events)
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


async def test_private_action_history_failure_preserves_scene_and_retries_same_selection():
    """A notebook/history failure may pause one action opportunity but cannot close or replay its audits."""

    emitted: list[EventRecord] = []

    async def event_sink(events):
        emitted.extend(events)
        return tuple(events)

    scheduler, state, agents = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [
            HistoryWriteError("notebook batch failed"),
            SpeakPrivate(actor=player_id, chat_id=chat_id, text=f"{player_id} recovered"),
        ]

    with pytest.raises(HistoryWriteError, match="notebook batch failed"):
        await scheduler.run(chat_id)

    selected = next(
        event.payload["player_id"]
        for event in emitted
        if event.type == "scheduler.selection"
    )
    assert state.phase == "day.private"
    assert state.active_scene == chat_id
    assert scheduler._scene is request.scene
    assert scheduler.end_reason is None
    assert not any(event.type == "chat.private_ended" for event in emitted)

    recovered = await scheduler.run(chat_id)

    assert len(agents[selected].scenes) == 2
    assert sum(event.type == "chat.private_started" for event in emitted) == 1
    assert sum(event.type == "scheduler.ranking" for event in emitted) == 1
    assert sum(event.type == "scheduler.selection" for event in emitted) == 1
    assert sum(event.type == "chat.private_message" for event in emitted) == 1
    assert sum(event.type == "chat.private_message" for event in recovered) == 1


async def test_private_probe_history_failure_resumes_without_replaying_ranking_or_prior_probe():
    """Probe persistence failure retries that participant, not the already committed scene prefix."""

    emitted: list[EventRecord] = []

    async def event_sink(events):
        emitted.extend(events)
        return tuple(events)

    scheduler, state, agents = make_scheduler(
        action_budget=1,
        event_sink=event_sink,
    )
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    chat_id = request.scene.chat_id
    agents["alice"].probe_result = ReactionProbe(
        decision="respond", urgency=15, action_type="speak"
    )
    agents["bob"].probe_result = [
        HistoryWriteError("probe segment failed"),
        ReactionProbe(decision="respond", urgency=-15, action_type="speak"),
    ]
    agents["alice"].actions = [
        SpeakPrivate(actor="alice", chat_id=chat_id, text="after probe recovery")
    ]

    with pytest.raises(HistoryWriteError, match="probe segment failed"):
        await scheduler.run(chat_id)

    assert state.phase == "day.private"
    assert sum(event.type == "scheduler.ranking" for event in emitted) == 1
    assert len(agents["alice"].probes) == 1
    assert len(agents["bob"].probes) == 1
    assert sum(event.type == "scheduler.probe_adjustment" for event in emitted) == 1

    await scheduler.run(chat_id)

    assert sum(event.type == "scheduler.ranking" for event in emitted) == 1
    assert len(agents["alice"].probes) == 1
    assert len(agents["bob"].probes) == 2
    assert sum(event.type == "scheduler.probe_adjustment" for event in emitted) == 2
    assert sum(event.type == "chat.private_message" for event in emitted) == 1


async def test_unknown_private_action_exception_preserves_scene_without_false_end():
    """Programming failures must propagate without manufacturing model_failed or ended facts."""

    emitted: list[EventRecord] = []

    async def event_sink(events):
        emitted.extend(events)
        return tuple(events)

    scheduler, state, agents = make_scheduler(event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    for player_id in ("alice", "bob"):
        agents[player_id].actions = [RuntimeError("programming defect")]

    with pytest.raises(RuntimeError, match="programming defect"):
        await scheduler.run(request.scene.chat_id)

    assert state.phase == "day.private"
    assert state.active_scene == request.scene.chat_id
    assert scheduler._scene is request.scene
    assert scheduler.end_reason is None
    assert not any(event.type == "chat.private_ended" for event in emitted)


async def test_cancelled_private_action_preserves_scene_without_ended_shell():
    """Task cancellation is not a gameplay end reason and must leave resumable ownership intact."""

    action_started = asyncio.Event()
    action_release = asyncio.Event()
    emitted: list[EventRecord] = []

    async def event_sink(events):
        emitted.extend(events)
        return tuple(events)

    class BlockingPrivateAgent(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            action_started.set()
            await action_release.wait()
            return AgentOutcome(action=YieldAction(actor="alice", reason="cancelled"), round_trips=1)

    scheduler, state, agents = make_scheduler(event_sink=event_sink)
    request = await scheduler.request("alice", "bob")
    assert request.scene is not None
    agents["alice"] = BlockingPrivateAgent()
    agents["bob"] = BlockingPrivateAgent()
    scheduler.agents = agents
    task = asyncio.create_task(scheduler.run(request.scene.chat_id))
    try:
        await asyncio.wait_for(action_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert state.phase == "day.private"
        assert state.active_scene == request.scene.chat_id
        assert scheduler._scene is request.scene
        assert scheduler.end_reason is None
        assert not any(event.type == "chat.private_ended" for event in emitted)
    finally:
        action_release.set()


async def test_private_transcript_reaches_the_other_participant_without_pretending_it_was_persisted():
    """The second participant needs the first message now, before Task 11 owns EventStream publication."""

    scheduler, _state, agents = make_scheduler(action_budget=2)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    chat_id = result.scene.chat_id
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["bob"].probe_result = ReactionProbe(decision="respond", urgency=-15, action_type="speak")
    agents["alice"].actions = [SpeakPrivate(actor="alice", chat_id=chat_id, text="first secret")]
    agents["bob"].actions = [SpeakPrivate(actor="bob", chat_id=chat_id, text="second secret")]

    events = await scheduler.run(chat_id)

    assert [event.payload["text"] for event in events if event.type == "chat.private_message"] == [
        "first secret",
        "second secret",
    ]
    bob_scene = agents["bob"].scenes[0]
    assert [event.payload["text"] for event in bob_scene.context_events] == ["first secret"]
    assert agents["bob"].probes[-1].payload["text"] == "first secret"
    assert all(event.seq == 0 for event in bob_scene.context_events)


async def test_request_revalidates_ownership_after_invitee_await_before_accepting():
    """A response cannot overwrite a scene or phase that changed while its model call awaited."""

    scheduler, state, agents = make_scheduler()

    class ChangesStateAfterAccept(ScriptedPrivateAgent):
        async def respond_private_invitation(self, invitation):
            state.active_scene = "external-scene"
            state.phase = "night"
            return "accept"

    agents["bob"] = ChangesStateAfterAccept()
    scheduler.agents = agents
    result = await scheduler.request("alice", "bob")

    assert result.decision == "defer"
    assert result.scene is None
    assert state.active_scene == "external-scene"
    assert state.phase == "night"


async def test_private_cleanup_does_not_restore_a_phase_or_scene_taken_over_externally():
    """A stale finally block must not undo Task 11's phase or scene transition."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class TakesOverState(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            state.phase = "night"
            state.active_scene = "external-scene"
            state.stopped = True
            return AgentOutcome(action=YieldAction(actor="alice", reason="external"), round_trips=1)

    agents["alice"] = TakesOverState()
    scheduler.agents = agents
    await scheduler.run(result.scene.chat_id)

    assert state.phase == "night"
    assert state.active_scene == "external-scene"


async def test_run_releases_its_stale_reservation_without_reclaiming_an_external_phase():
    """A pre-run night transition frees this slot but is never rewritten back to discussion."""

    scheduler, state, _agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    state.phase = "night"

    events = await scheduler.run(result.scene.chat_id)

    assert state.phase == "night"
    assert state.active_scene is None
    assert scheduler._scene is None
    assert [event.type for event in events] == ["chat.private_ended"]


async def test_run_releases_only_its_local_reference_when_external_scene_takes_over_before_start():
    """A stale chat id must not clear the new active scene or alter its phase."""

    scheduler, state, _agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    state.active_scene = "external-scene"
    state.phase = "night"

    events = await scheduler.run(result.scene.chat_id)

    assert state.active_scene == "external-scene"
    assert state.phase == "night"
    assert scheduler._scene is None
    assert [event.type for event in events] == ["chat.private_ended"]


async def test_one_participant_yield_cannot_quiet_the_other_before_their_turn():
    """Quiet is per participant, so Alice yielding leaves Bob a real normal-action opportunity."""

    scheduler, _state, agents = make_scheduler(action_budget=1)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    chat_id = result.scene.chat_id
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["bob"].probe_result = ReactionProbe(decision="respond", urgency=-15, action_type="speak")
    agents["alice"].actions = [YieldAction(actor="alice", reason="no news")]
    agents["bob"].actions = [SpeakPrivate(actor="bob", chat_id=chat_id, text="Bob still speaks")]

    await scheduler.run(chat_id)

    assert len(agents["bob"].scenes) == 1
    assert scheduler.end_reason == "action_budget"


@pytest.mark.parametrize(
    "forged_action",
    [
        YieldAction(actor="bob", reason="forged yield"),
        # update_notebook normally remains internal to PlayerAgent; an outward instance is rejected.
        UpdateNotebook(actor="bob", notebook=Notebook(notes="forged")),
    ],
)
async def test_every_outward_private_action_is_bound_to_the_selected_actor(forged_action):
    """Yield and notebook action types cannot bypass the selected-player identity check."""

    scheduler, _state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["bob"].probe_result = ReactionProbe(decision="defer", urgency=-15, action_type="yield")
    agents["alice"].actions = [forged_action]
    agents["bob"].actions = [LeavePrivateChat(actor="bob", chat_id=result.scene.chat_id)]

    events = await scheduler.run(result.scene.chat_id)

    rejected = [event for event in events if event.type == "scheduler.action_rejected"]
    assert rejected and rejected[0].payload["reason"] == "wrong_actor"


async def test_private_scores_keep_two_actions_per_player_reachable_before_global_cap():
    """Private cooldown is bounded: four actions can schedule two turns for each participant."""

    scheduler, _state, agents = make_scheduler(action_budget=4, per_player_action_limit=2)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class AlwaysSpeak(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            return AgentOutcome(
                action=SpeakPrivate(
                    actor=self.player_id,
                    chat_id=scene.details["chat_id"],
                    text=f"{self.player_id} turn {len(self.scenes)}",
                ),
                round_trips=1,
            )

    for player_id in ("alice", "bob"):
        agent = AlwaysSpeak()
        agent.player_id = player_id
        agents[player_id] = agent
    scheduler.agents = agents

    events = await scheduler.run(result.scene.chat_id)

    messages = [event for event in events if event.type == "chat.private_message"]
    assert len(messages) == 4
    assert [event.actor for event in messages].count("alice") == 2
    assert [event.actor for event in messages].count("bob") == 2
    assert scheduler.end_reason == "action_budget"


async def test_stop_during_probe_fallback_prevents_the_retry_and_later_player_probe():
    """A stop observed after a fallback is a safe boundary, not permission for one more short call."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class StopOnFallback(ScriptedPrivateAgent):
        async def probe(self, event):
            self.probes.append(event)
            state.stopped = True
            return ReactionProbe.fallback_silent()

    agents["alice"] = StopOnFallback()
    scheduler.agents = agents

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].probes) == 1
    assert not agents["bob"].probes
    assert scheduler.end_reason == "stopped"


async def test_invitation_parser_fallback_retries_once_before_accepting():
    """A malformed invitation response receives one short retry, without using a normal action call."""

    scheduler, state, agents = make_scheduler()
    agents["bob"].invitation = [PrivateInvitationResponse.fallback_defer(), "accept"]

    result = await scheduler.request("alice", "bob")

    assert result.decision == "accept"
    assert result.scene is not None
    assert len(agents["bob"].invitations) == 2
    assert state.active_scene == result.scene.chat_id


async def test_private_probe_parser_fallback_retries_once_before_its_adjustment_is_used():
    """Private probes retain Task 9's bounded parser retry without giving a fallback silent weight."""

    scheduler, _state, agents = make_scheduler(action_budget=1)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None
    agents["alice"].probe_result = [
        ReactionProbe.fallback_silent(),
        ReactionProbe(decision="respond", urgency=4, action_type="speak"),
    ]
    agents["alice"].actions = [SpeakPrivate(actor="alice", chat_id=result.scene.chat_id, text="retry")]

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].probes) == 2
    assert scheduler.probe_adjustments["alice"] == 4


async def test_stop_after_first_normal_model_failure_prevents_private_retry():
    """A normal retry may not begin after the failed first call has observed stop."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class StopOnNormalFailure(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            state.stopped = True
            raise ModelCallError("provider unavailable")

    agents["alice"] = StopOnNormalFailure()
    scheduler.agents = agents

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].scenes) == 1
    assert scheduler.end_reason == "stopped"


async def test_owner_safe_point_runs_before_private_probe_model_retry():
    """A pending Stop must be observed between failed short calls in a private scene."""

    stop_requested = False

    async def safe_point():
        if stop_requested:
            state.stopped = True

    scheduler, state, agents = make_scheduler(safe_point=safe_point)
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class RequestStopOnProbeFailure(ScriptedPrivateAgent):
        async def probe(self, event):
            nonlocal stop_requested
            self.probes.append(event)
            stop_requested = True
            raise ModelCallError("short unavailable")

    agents["alice"] = RequestStopOnProbeFailure()
    scheduler.agents = agents

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].probes) == 1
    assert not agents["bob"].probes
    assert scheduler.end_reason == "stopped"


@pytest.mark.parametrize("takeover", ["night", "external"])
async def test_probe_await_ownership_loss_never_selects_or_calls_a_normal_action(takeover: str):
    """A stale probe result cannot select a participant after Task 11 advances the scene."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class TakeOverDuringProbe(ScriptedPrivateAgent):
        async def probe(self, event):
            self.probes.append(event)
            if takeover == "night":
                state.phase = "night"
            else:
                state.active_scene = "external-scene"
                state.phase = "night"
            return ReactionProbe(decision="respond", urgency=15, action_type="speak")

    agents["alice"] = TakeOverDuringProbe()
    scheduler.agents = agents

    events = await scheduler.run(result.scene.chat_id)

    assert not any(event.type == "chat.private_message" for event in events)
    assert all(not agent.scenes for agent in agents.values())
    assert scheduler.end_reason == "ownership_lost"
    assert scheduler._scene is None
    assert state.phase == "night"
    assert state.active_scene == (None if takeover == "night" else "external-scene")


@pytest.mark.parametrize("takeover", ["night", "external"])
async def test_normal_await_ownership_loss_discards_the_stale_private_action(takeover: str):
    """A model response produced after scene loss cannot enter the transcript or event output."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class TakeOverDuringAction(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            if takeover == "night":
                state.phase = "night"
            else:
                state.active_scene = "external-scene"
                state.phase = "night"
            return AgentOutcome(
                action=SpeakPrivate(actor="alice", chat_id=scene.details["chat_id"], text="stale secret"),
                round_trips=1,
            )

    agents["alice"] = TakeOverDuringAction()
    scheduler.agents = agents
    agents["alice"].probe_result = ReactionProbe(decision="respond", urgency=15, action_type="speak")
    agents["bob"].probe_result = ReactionProbe(decision="defer", urgency=-15, action_type="yield")

    events = await scheduler.run(result.scene.chat_id)

    assert not any(event.type == "chat.private_message" for event in events)
    assert all("stale secret" not in repr(event.payload) for event in events)
    assert scheduler.end_reason == "ownership_lost"
    assert scheduler._scene is None
    assert state.phase == "night"
    assert state.active_scene == (None if takeover == "night" else "external-scene")


async def test_probe_fallback_does_not_retry_after_scene_ownership_is_lost():
    """Fallback retry uses the same ownership gate as model-error retry."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class LoseOwnershipOnFallback(ScriptedPrivateAgent):
        async def probe(self, event):
            self.probes.append(event)
            state.phase = "night"
            return ReactionProbe.fallback_silent()

    agents["alice"] = LoseOwnershipOnFallback()
    scheduler.agents = agents

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].probes) == 1
    assert scheduler.end_reason == "ownership_lost"
    assert state.active_scene is None
    assert state.phase == "night"


async def test_normal_model_error_does_not_retry_after_scene_ownership_is_lost():
    """Normal retry requires the private scene, not merely a non-stopped game."""

    scheduler, state, agents = make_scheduler()
    result = await scheduler.request("alice", "bob")
    assert result.scene is not None

    class LoseOwnershipOnFailure(ScriptedPrivateAgent):
        async def run_action(self, scene):
            self.scenes.append(scene)
            state.active_scene = "external-scene"
            state.phase = "night"
            raise ModelCallError("temporary")

    agents["alice"] = LoseOwnershipOnFailure()
    scheduler.agents = agents

    await scheduler.run(result.scene.chat_id)

    assert len(agents["alice"].scenes) == 1
    assert scheduler.end_reason == "ownership_lost"
    assert state.active_scene == "external-scene"
    assert state.phase == "night"
