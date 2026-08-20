from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Callable

from clocktower.agents.context import project_context
from clocktower.agents.player import AgentOutcome, ReactionProbe, segment_event
from clocktower.config import GameConfig
from clocktower.domain.actions import (
    CastVote,
    LeavePrivateChat,
    Nominate,
    RequestPrivateChat,
    SpeakPrivate,
    SpeakPublic,
    UseAbility,
    YieldAction,
)
from clocktower.event_stream import EventStream
from clocktower.history import HistoryWriter
from clocktower.models.protocol import ModelSegment
from clocktower.orchestrator import GameOrchestrator
from clocktower.rules.engine import RuleEngine


@dataclass
class GameScript:
    private_requested: bool = False
    private_message_sent: bool = False
    nomination_submitted: bool = False
    final_probe_player_ids: list[str] = field(default_factory=list)
    vote_observations: list[tuple[str, int]] = field(default_factory=list)
    notebook_updated: bool = False
    prompt_event_seqs: dict[str, set[int]] = field(default_factory=dict)


@dataclass
class CompleteGameFakeProvider:
    """Deterministic no-network provider trace used by the complete-game acceptance path."""

    call_count: int = 0

    async def record_call(
        self,
        history: HistoryWriter,
        *,
        player_id: str,
        purpose: str,
        phase: str,
    ) -> None:
        self.call_count += 1
        call_id = f"fake-{self.call_count}-{player_id}-{purpose}"
        specifications = (
            ("reasoning", "reasoning_content", "Consider the visible facts."),
            ("reasoning", "thinking", "Choose one bounded action."),
            ("tool_call", "tool_calls", '{"scripted":true}'),
            ("tool_result", "tool_result", '{"ok":true}'),
            ("final_message", "content", "Scripted provider turn complete."),
        )
        events = tuple(
            segment_event(
                player_id,
                purpose,
                ModelSegment(
                    call_id=call_id,
                    index=index,
                    kind=kind,
                    source_field=source_field,
                    text=text,
                    tool_index=0 if kind in {"tool_call", "tool_result"} else None,
                    tool_call_id="fake-tool" if kind in {"tool_call", "tool_result"} else None,
                    tool_name="scripted_decision" if kind in {"tool_call", "tool_result"} else None,
                    tool_type="function" if kind in {"tool_call", "tool_result"} else None,
                ),
                phase,
            )
            for index, (kind, source_field, text) in enumerate(specifications)
        )
        await history.append_many(events)


@dataclass
class CompleteGameAgent:
    player_id: str
    script: GameScript
    history: HistoryWriter | None = None
    state_provider: Callable | None = None
    provider: CompleteGameFakeProvider | None = None

    async def _record_provider_turn(self, purpose: str, *, trigger=None) -> None:
        assert self.history is not None
        assert self.state_provider is not None
        assert self.provider is not None
        state = self.state_provider()
        source = (trigger,) if trigger is not None else tuple(self.history.stream.after(0))
        context = project_context(self.player_id, state, source)
        self.script.prompt_event_seqs.setdefault(self.player_id, set()).update(
            event.seq for event in context.events if event.seq > 0
        )
        await self.provider.record_call(
            self.history,
            player_id=self.player_id,
            purpose=purpose,
            phase=trigger.phase if trigger is not None else state.phase,
        )

    async def respond_private_invitation(self, invitation):
        await self._record_provider_turn("private_invitation_response", trigger=invitation)
        return "accept"

    async def probe(self, event):
        await self._record_provider_turn("reaction_probe", trigger=event)
        if event.type == "day.final_nomination_probe":
            self.script.final_probe_player_ids.append(self.player_id)
            return ReactionProbe(decision="silent", urgency=0, action_type="yield")
        if event.phase == "day.private":
            return ReactionProbe(decision="respond", urgency=5, action_type="speak")
        if not self.script.private_requested:
            return ReactionProbe(decision="respond", urgency=10, action_type="private_chat")
        if not self.script.nomination_submitted:
            return ReactionProbe(decision="respond", urgency=10, action_type="nominate")
        return ReactionProbe(decision="silent", urgency=0, action_type="yield")

    async def run_action(self, scene):
        await self._record_provider_turn(scene.purpose)
        if self.player_id == "alice" and not self.script.notebook_updated:
            assert self.history is not None
            assert self.state_provider is not None
            state = self.state_provider()
            notebook = state.players[self.player_id].notebook.model_copy(deep=True)
            notebook.notes = "Track claims and revisit the private conversation."
            await self.history.update_notebook(state, self.player_id, notebook)
            self.script.notebook_updated = True
        if scene.purpose == "night_ability":
            return AgentOutcome(
                action=UseAbility(
                    actor=self.player_id,
                    action=scene.details["ability"],
                    targets=tuple(scene.details["legal_targets"][0]),
                ),
                round_trips=1,
            )
        if scene.purpose == "public_discussion":
            if not self.script.private_requested:
                self.script.private_requested = True
                target = next(
                    player_id
                    for player_id in ("alice", "bob", "carol", "david", "eve")
                    if player_id != self.player_id
                )
                return AgentOutcome(
                    action=RequestPrivateChat(actor=self.player_id, target_player=target),
                    round_trips=1,
                )
            if not self.script.nomination_submitted:
                self.script.nomination_submitted = True
                return AgentOutcome(
                    action=Nominate(actor=self.player_id, target="eve", accusation="The Imp."),
                    round_trips=1,
                )
            return AgentOutcome(
                action=YieldAction(actor=self.player_id, reason="quiet"),
                status="yielded",
                round_trips=1,
            )
        if scene.purpose == "private_chat":
            if not self.script.private_message_sent:
                self.script.private_message_sent = True
                return AgentOutcome(
                    action=SpeakPrivate(
                        actor=self.player_id,
                        chat_id=scene.details["chat_id"],
                        text="I trust this private channel.",
                    ),
                    round_trips=1,
                )
            return AgentOutcome(
                action=LeavePrivateChat(
                    actor=self.player_id,
                    chat_id=scene.details["chat_id"],
                ),
                round_trips=1,
            )
        if scene.purpose == "nomination_response":
            return AgentOutcome(
                action=SpeakPublic(actor=self.player_id, text="My defence."),
                round_trips=1,
            )
        if scene.purpose == "vote":
            assert self.history is not None
            earlier_votes = sum(
                event.type in {"vote.cast", "vote.resolved"}
                for event in self.history.stream.after(0)
            )
            self.script.vote_observations.append((self.player_id, earlier_votes))
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


async def run_complete_game(path):
    player_ids = ("alice", "bob", "carol", "david", "eve")
    rules = RuleEngine.start_game(player_ids, seed=17)
    history = HistoryWriter(path, EventStream())
    script = GameScript()
    provider = CompleteGameFakeProvider()
    agents = {
        player_id: CompleteGameAgent(player_id, script)
        for player_id in player_ids
    }
    for agent in agents.values():
        agent.history = history
        agent.state_provider = lambda: rules.state
        agent.provider = provider
    orchestrator = GameOrchestrator(
        rules=rules,
        agents=agents,
        history=history,
        game_config=GameConfig(
            seed=17,
            player_ids=player_ids,
            history_directory=path.parent,
            discussion_action_budget=4,
            discussion_quiet_windows=1,
            private_chat_action_budget=3,
            private_chat_quiet_windows=1,
        ),
        reload_model_config=lambda: None,
    )
    await orchestrator.run()
    return orchestrator, script


async def test_fixed_seed_five_player_game_is_reproducible_complete_and_network_free(tmp_path):
    """A lifecycle that skips private chat, final probes, or termination is not a complete game."""

    first, first_script = await run_complete_game(tmp_path / "first.jsonl")
    second, second_script = await run_complete_game(tmp_path / "second.jsonl")

    first_records = [
        json.loads(line)
        for line in (tmp_path / "first.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    second_records = [
        json.loads(line)
        for line in (tmp_path / "second.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    comparable = lambda records: [
        {key: value for key, value in record.items() if key != "time"}
        for record in records
    ]

    assert first.status().state == "ended"
    assert first.status().winner == "good"
    assert first.status().phase == "night"
    assert first.status().reason == "demon_dead"
    assert second.status().winner == "good"
    assert comparable(first_records) == comparable(second_records)
    assert [record["seq"] for record in first_records] == list(range(1, len(first_records) + 1))
    assert first_records[0]["type"] == "game.header"
    assert first_records[-1]["type"] == "game.ended"
    assert sum(record["type"] == "game.ended" for record in first_records) == 1

    types = [record["type"] for record in first_records]
    ordered = [
        "night.deaths_announced",
        "day.started",
        "chat.private_started",
        "chat.private_message",
        "chat.private_ended",
        "nomination.opened",
        "player.public_message",
        "vote.cast",
        "nomination.closed",
        "day.final_nomination_probe",
        "execution.resolved",
        "game.ended",
    ]
    positions = [types.index(event_type) for event_type in ordered]
    assert positions == sorted(positions)
    private_messages = [record for record in first_records if record["type"] == "chat.private_message"]
    assert len(private_messages) == 1
    assert private_messages[0]["audience"]["kind"] == "players"
    assert len(private_messages[0]["audience"]["player_ids"]) == 2

    opened = next(record for record in first_records if record["type"] == "nomination.opened")
    assert [player_id for player_id, _seen in first_script.vote_observations] == opened["payload"]["vote_order"]
    assert [seen for _player_id, seen in first_script.vote_observations] == [0, 1, 2, 3, 4]
    assert set(first_script.final_probe_player_ids) == {
        "alice", "bob", "carol", "david", "eve"
    } - {opened["payload"]["nominator"]}
    assert len(first_script.final_probe_player_ids) == 4
    assert len(second_script.final_probe_player_ids) == 4
