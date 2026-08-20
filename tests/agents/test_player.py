from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy

import pytest
from pydantic import ValidationError

from clocktower.agents.player import (
    AgentScene,
    PlayerAgent,
    PrivateInvitationResponse,
    ReactionProbe,
    segment_event,
)
from clocktower.config import ResolvedModel
from clocktower.domain.actions import Nominate, SpeakPublic, YieldAction
from clocktower.domain.events import Audience, EventRecord
from clocktower.event_stream import EventStream
from clocktower.history import HistoryWriter
from clocktower.models.protocol import ModelRequest, ModelSegment
from tests.builders import public_claim, sample_game_state


class ScriptedAdapter:
    """Socket-free adapter boundary; the PlayerAgent itself remains real."""

    def __init__(self, *scripts: tuple[ModelSegment, ...]) -> None:
        self.scripts = list(scripts)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelSegment]:
        self.requests.append(request)
        if not self.scripts:
            raise AssertionError("unexpected model call")
        for segment in self.scripts.pop(0):
            yield segment


class RecordingResolver:
    def __init__(self) -> None:
        self.short_flags: list[bool] = []

    def __call__(self, player_id: str, short: bool) -> ResolvedModel:
        assert player_id == "alice"
        self.short_flags.append(short)
        return ResolvedModel(
            provider="scripted",
            name="short" if short else "normal",
            base_url="https://unused.example/v1",
            api_key_env="UNUSED_KEY",
            api_key=None,
            reasoning_fields=("reasoning_content",),
            source="tests.short" if short else "tests.normal",
        )


def segment(
    index: int,
    kind: str,
    text: str,
    *,
    call_id: str = "call-1",
    source_field: str | None = None,
    tool_index: int | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_type: str | None = None,
    incomplete: bool = False,
) -> ModelSegment:
    return ModelSegment(
        call_id=call_id,
        index=index,
        kind=kind,  # type: ignore[arg-type]
        source_field=source_field or ("tool_calls" if kind == "tool_call" else "content"),
        text=text,
        incomplete=incomplete,
        tool_index=tool_index,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_type=tool_type,
    )


def tool_segment(
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str,
    index: int = 0,
    tool_index: int = 0,
    tool_call_id: str | None = None,
) -> ModelSegment:
    return segment(
        index,
        "tool_call",
        json.dumps(arguments, separators=(",", ":")),
        call_id=call_id,
        tool_index=tool_index,
        tool_call_id=tool_call_id or f"tool-{call_id}-{tool_index}",
        tool_name=name,
        tool_type="function",
    )


def notebook_payload(notes: str) -> dict[str, object]:
    return {
        "notebook": {
            "notes": notes,
            "attention": {
                "players": [],
                "pending_actions": [],
                "watch_triggers": [],
            },
        }
    }


def build_agent(tmp_path, adapter: ScriptedAdapter):
    game_state = sample_game_state()
    game_state.phase = "day.discussion"
    history = HistoryWriter(tmp_path / "game.jsonl", EventStream())
    resolver = RecordingResolver()
    agent = PlayerAgent(
        player_id="alice",
        game_state=game_state,
        resolve_model=resolver,
        adapter=adapter,
        history=history,
    )
    return agent, game_state, history, resolver


def history_records(history: HistoryWriter) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in history.path.read_text(encoding="utf-8").splitlines()
    ]


def test_reaction_probe_rejects_non_strict_decisions_and_urgency():
    """Coercion or an open decision string would let the short model control scheduling."""

    ReactionProbe(decision="respond", urgency=1, action_type="speak")

    try:
        ReactionProbe(decision="immediately", urgency=1, action_type="speak")
    except ValidationError:
        pass
    else:
        raise AssertionError("unknown decision was accepted")

    try:
        ReactionProbe(decision="respond", urgency="15", action_type="speak")
    except ValidationError:
        pass
    else:
        raise AssertionError("string urgency was coerced")


def test_segment_event_preserves_every_provider_field_as_observer_only():
    """Dropping tool identity fields prevents exact trace reconstruction and result binding."""

    provider_segment = segment(
        7,
        "tool_call",
        '{"text":"hi"}',
        call_id="provider-call",
        tool_index=3,
        tool_call_id="tool-42",
        tool_name="speak_public",
        tool_type="function",
        incomplete=True,
    )

    event = segment_event("alice", "formal_action", provider_segment, "day.discussion")

    assert event.audience == Audience.observer()
    assert event.type == "model.output_segment"
    assert event.payload == {
        "call_id": "provider-call",
        "player_id": "alice",
        "call_purpose": "formal_action",
        "segment_index": 7,
        "kind": "tool_call",
        "source_field": "tool_calls",
        "text": '{"text":"hi"}',
        "incomplete": True,
        "tool_index": 3,
        "tool_call_id": "tool-42",
        "tool_name": "speak_public",
        "tool_type": "function",
    }
    assert all(not isinstance(value, ModelSegment) for value in event.payload.values())


async def test_run_action_uses_normal_model_and_returns_one_canonical_action_without_publishing_body(tmp_path):
    """The agent boundary must propose an action; the owning scene decides how to publish it."""

    adapter = ScriptedAdapter(
        (
            segment(0, "reasoning", "private thought", source_field="reasoning_content"),
            tool_segment(
                "speak_public",
                {"text": "I claim Chef."},
                call_id="call-1",
                index=1,
                tool_call_id="tool-public",
            ),
        )
    )
    agent, _state, history, resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert resolver.short_flags == [False]
    assert outcome.action == SpeakPublic(actor="alice", text="I claim Chef.")
    assert outcome.round_trips == 1
    records = history_records(history)
    assert [record["type"] for record in records] == [
        "model.output_segment",
        "model.output_segment",
        "model.output_segment",
    ]
    assert all(record["audience"]["kind"] == "observer" for record in records)
    assert all(record["type"] != "player.public_message" for record in records)
    result = records[-1]["payload"]
    assert result["kind"] == "tool_result"
    assert result["tool_call_id"] == "tool-public"
    prompt = json.loads(adapter.requests[0].messages[1]["content"])
    assert prompt["scene"]["phase"] == "day.discussion"


async def test_run_action_allows_repeated_notebook_updates_with_adjacent_checkpoints(tmp_path):
    """Batching notebook changes would lose the required checkpoint after each accepted patch."""

    adapter = ScriptedAdapter(
        (tool_segment("update_notebook", notebook_payload("first"), call_id="call-1"),),
        (tool_segment("update_notebook", notebook_payload("second"), call_id="call-2"),),
        (
            tool_segment(
                "nominate",
                {"target_player": "bob", "accusation": "evasive"},
                call_id="call-3",
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert outcome.action == Nominate(actor="alice", target="bob", accusation="evasive")
    assert game_state.players["alice"].notebook.notes == "second"
    records = history_records(history)
    update_indexes = [index for index, record in enumerate(records) if record["type"] == "notebook.updated"]
    assert len(update_indexes) == 2
    assert all(records[index + 1]["type"] == "checkpoint" for index in update_indexes)
    tool_messages = [
        message
        for request in adapter.requests[1:]
        for message in request.messages
        if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "tool-call-1-0",
        "tool-call-1-0",
        "tool-call-2-0",
    ]


async def test_mixed_assistant_final_text_and_tool_identity_are_preserved_without_reasoning(tmp_path):
    """Tool continuation needs assistant content and exact identity, but never raw reasoning."""

    arguments = json.dumps(notebook_payload("one"), separators=(",", ":"))
    adapter = ScriptedAdapter(
        (
            segment(
                0,
                "reasoning",
                "RAW_REASONING_MUST_NOT_CONTINUE",
                call_id="call-1",
                source_field="reasoning_content",
            ),
            segment(
                1,
                "final_message",
                "Visible assistant text",
                call_id="call-1",
            ),
            segment(
                2,
                "tool_call",
                arguments,
                call_id="call-1",
                tool_index=4,
                tool_call_id="tool-mixed",
                tool_name="update_notebook",
                tool_type="provider_function",
            ),
        ),
        (
            tool_segment(
                "speak_public", {"text": "done"}, call_id="call-2"
            ),
        ),
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    await agent.run_action(AgentScene())

    assistant = next(
        message
        for message in adapter.requests[1].messages
        if message.get("role") == "assistant"
    )
    assert assistant == {
        "role": "assistant",
        "content": "Visible assistant text",
        "tool_calls": [
            {
                "id": "tool-mixed",
                "type": "provider_function",
                "function": {
                    "name": "update_notebook",
                    "arguments": arguments,
                },
            }
        ],
    }
    assert "RAW_REASONING_MUST_NOT_CONTINUE" not in json.dumps(
        adapter.requests[1].messages
    )


async def test_run_action_accepts_at_most_one_outward_action_from_parallel_calls(tmp_path):
    """Two outward intents invalidate the batch; neither may escape before correction."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "speak_public",
                {"text": "one"},
                call_id="call-1",
                tool_index=0,
                tool_call_id="tool-one",
            ),
            tool_segment(
                "nominate",
                {"target_player": "bob", "accusation": "two"},
                call_id="call-1",
                index=1,
                tool_index=1,
                tool_call_id="tool-two",
            ),
        ),
        (
            tool_segment(
                "speak_public",
                {"text": "corrected"},
                call_id="call-2",
                tool_call_id="tool-corrected",
            ),
        ),
    )
    agent, _state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert outcome.action == SpeakPublic(actor="alice", text="corrected")
    assert outcome.illegal_corrections == 1
    results = [
        record["payload"]
        for record in history_records(history)
        if record["type"] == "model.output_segment"
        and record["payload"]["kind"] == "tool_result"
    ]
    first_batch = [result for result in results if result["call_id"] == "call-1"]
    assert [result["tool_call_id"] for result in first_batch] == ["tool-one", "tool-two"]
    assert all(result["text"].startswith('{"error":') for result in first_batch)


async def test_mixed_legal_outward_and_unknown_tool_rejects_whole_batch(tmp_path):
    """A valid outward call cannot bypass an illegal sibling in the same assistant response."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "speak_public",
                {"text": "must not escape"},
                call_id="call-1",
                tool_index=0,
                tool_call_id="tool-valid",
            ),
            tool_segment(
                "read_global_state",
                {},
                call_id="call-1",
                index=1,
                tool_index=1,
                tool_call_id="tool-illegal",
            ),
        ),
        (
            tool_segment(
                "nominate",
                {"target_player": "bob", "accusation": "corrected"},
                call_id="call-2",
            ),
        ),
    )
    agent, _state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert outcome.action == Nominate(
        actor="alice", target="bob", accusation="corrected"
    )
    first_results = [
        record["payload"]
        for record in history_records(history)
        if record["payload"].get("kind") == "tool_result"
        and record["payload"].get("call_id") == "call-1"
    ]
    assert len(first_results) == 2
    assert all(json.loads(result["text"]).get("error") for result in first_results)


async def test_illegal_parallel_sibling_prevents_notebook_side_effect(tmp_path):
    """Validation must finish before a notebook update can create a checkpoint."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "update_notebook",
                notebook_payload("must not commit"),
                call_id="call-1",
                tool_index=0,
                tool_call_id="tool-note",
            ),
            tool_segment(
                "unknown",
                {},
                call_id="call-1",
                index=1,
                tool_index=1,
                tool_call_id="tool-bad",
            ),
        ),
        (
            tool_segment(
                "speak_public",
                {"text": "corrected"},
                call_id="call-2",
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert outcome.action == SpeakPublic(actor="alice", text="corrected")
    assert game_state.players["alice"].notebook.notes == ""
    assert all(record["type"] != "checkpoint" for record in history_records(history))


async def test_multiple_illegal_calls_in_first_response_consume_one_correction_round(tmp_path):
    """The correction budget is per assistant response, not per parallel tool call."""

    adapter = ScriptedAdapter(
        (
            tool_segment("unknown_one", {}, call_id="call-1", tool_index=0),
            tool_segment(
                "unknown_two",
                {},
                call_id="call-1",
                index=1,
                tool_index=1,
            ),
        ),
        (
            tool_segment(
                "speak_public", {"text": "corrected"}, call_id="call-2"
            ),
        ),
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert len(adapter.requests) == 2
    assert outcome.action == SpeakPublic(actor="alice", text="corrected")
    assert outcome.illegal_corrections == 1


async def test_duplicate_tool_call_id_invalidates_batch_with_one_unambiguous_result(tmp_path):
    """Duplicate result IDs cannot be paired back to two distinct calls."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "update_notebook",
                notebook_payload("one"),
                call_id="call-1",
                tool_index=0,
                tool_call_id="duplicate-id",
            ),
            tool_segment(
                "speak_public",
                {"text": "two"},
                call_id="call-1",
                index=1,
                tool_index=1,
                tool_call_id="duplicate-id",
            ),
        ),
        (
            tool_segment(
                "speak_public", {"text": "corrected"}, call_id="call-2"
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert outcome.action == SpeakPublic(actor="alice", text="corrected")
    assert game_state.players["alice"].notebook.notes == ""
    duplicate_results = [
        record["payload"]
        for record in history_records(history)
        if record["payload"].get("kind") == "tool_result"
        and record["payload"].get("tool_call_id") == "duplicate-id"
    ]
    assert len(duplicate_results) == 1
    assert "duplicate_tool_call_id" in duplicate_results[0]["text"]


async def test_unknown_then_out_of_phase_tool_uses_only_one_correction(tmp_path):
    """Repeated illegal calls must terminate instead of creating an unbounded correction loop."""

    adapter = ScriptedAdapter(
        (tool_segment("read_global_state", {}, call_id="call-1"),),
        (
            tool_segment(
                "cast_vote",
                {"nomination_id": "nom-1", "vote": True},
                call_id="call-2",
            ),
        ),
    )
    agent, _state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=False))

    assert len(adapter.requests) == 2
    assert outcome.action == YieldAction(actor="alice", reason="illegal_tool_call")
    assert outcome.illegal_corrections == 1
    errors = [
        record["payload"]
        for record in history_records(history)
        if record["payload"].get("kind") == "tool_result"
    ]
    assert len(errors) == 2
    assert all(payload["tool_call_id"] for payload in errors)


async def test_required_action_reports_failure_after_second_illegal_tool(tmp_path):
    """A required choice must stop upstream orchestration rather than silently yielding."""

    adapter = ScriptedAdapter(
        (tool_segment("unknown", {}, call_id="call-1"),),
        (tool_segment("unknown_again", {}, call_id="call-2"),),
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=True))

    assert outcome.action is None
    assert outcome.status == "required_action_failed"


async def test_required_action_without_a_tool_reports_failure_instead_of_yielding(tmp_path):
    """A required choice cannot be converted into an optional yield when the model only chats."""

    adapter = ScriptedAdapter(
        (segment(0, "final_message", "I forgot to choose.", call_id="call-1"),)
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=True))

    assert outcome.action is None
    assert outcome.status == "required_action_failed"


async def test_required_scene_never_offers_or_accepts_yield_action(tmp_path):
    """A required choice may stop upstream, but it cannot silently become a voluntary yield."""

    adapter = ScriptedAdapter(
        (tool_segment("yield_action", {"reason": "no"}, call_id="call-1"),),
        (tool_segment("yield_action", {"reason": "still no"}, call_id="call-2"),),
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=True))

    assert all(
        "yield_action"
        not in {tool["function"]["name"] for tool in request.tools}
        for request in adapter.requests
    )
    assert outcome.action is None
    assert outcome.status == "required_action_failed"
    assert outcome.illegal_corrections == 1


async def test_scene_identifier_binding_rejects_cross_chat_action_then_accepts_correction(tmp_path):
    """A phase-valid private tool must not act on a different private scene."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "speak_private",
                {"chat_id": "chat-other", "text": "leak"},
                call_id="call-1",
            ),
        ),
        (
            tool_segment(
                "speak_private",
                {"chat_id": "chat-allowed", "text": "hello"},
                call_id="call-2",
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)
    game_state.phase = "day.private"

    outcome = await agent.run_action(
        AgentScene(details={"chat_id": "chat-allowed"})
    )

    assert outcome.action is not None
    assert outcome.action.kind == "speak_private"
    assert outcome.action.chat_id == "chat-allowed"
    assert outcome.illegal_corrections == 1
    first_result = next(
        record["payload"]
        for record in history_records(history)
        if record["payload"].get("kind") == "tool_result"
    )
    assert "active scene" in first_result["text"]


async def test_tool_arguments_follow_json_schema_types_without_pydantic_coercion(tmp_path):
    """The string 'yes' is not a legal boolean vote even if a model class could coerce it."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "cast_vote",
                {"nomination_id": "nom-1", "vote": "yes"},
                call_id="call-1",
            ),
        ),
        (
            tool_segment(
                "cast_vote",
                {"nomination_id": "nom-1", "vote": True},
                call_id="call-2",
            ),
        ),
    )
    agent, game_state, _history, _resolver = build_agent(tmp_path, adapter)
    game_state.phase = "day.voting"

    outcome = await agent.run_action(
        AgentScene(details={"nomination_id": "nom-1"})
    )

    assert outcome.action is not None
    assert outcome.action.kind == "cast_vote"
    assert outcome.action.vote is True
    assert outcome.illegal_corrections == 1


async def test_run_action_stops_after_four_tool_round_trips(tmp_path):
    """Notebook-only responses must still obey the hard model/tool lifecycle budget."""

    scripts = tuple(
        (tool_segment("update_notebook", notebook_payload(f"note-{index}"), call_id=f"call-{index}"),)
        for index in range(1, 6)
    )
    adapter = ScriptedAdapter(*scripts)
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene())

    assert len(adapter.requests) == 4
    assert outcome.round_trips == 4
    assert outcome.action == YieldAction(actor="alice", reason="tool_round_trip_limit")
    assert game_state.players["alice"].notebook.notes == "note-4"
    assert sum(record["type"] == "checkpoint" for record in history_records(history)) == 4


@pytest.mark.parametrize("required", [False, True])
async def test_fourth_regular_round_illegal_gets_a_real_fifth_correction_response(
    tmp_path,
    required: bool,
):
    """The correction allowance must still exist when the first error lands at the limit."""

    adapter = ScriptedAdapter(
        *(
            (
                tool_segment(
                    "update_notebook",
                    notebook_payload(f"note-{index}"),
                    call_id=f"call-{index}",
                ),
            )
            for index in range(1, 4)
        ),
        (tool_segment("unknown", {}, call_id="call-4"),),
        (
            tool_segment(
                "speak_public",
                {"text": "corrected on the fifth request"},
                call_id="call-5",
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=required))

    assert len(adapter.requests) == 5
    assert outcome.round_trips == 5
    assert outcome.status == "completed"
    assert outcome.action == SpeakPublic(
        actor="alice", text="corrected on the fifth request"
    )
    assert outcome.illegal_corrections == 1
    assert game_state.players["alice"].notebook.notes == "note-3"
    assert sum(record["type"] == "checkpoint" for record in history_records(history)) == 3


@pytest.mark.parametrize(
    ("required", "expected_status", "expected_action"),
    [
        (
            False,
            "yielded",
            YieldAction(actor="alice", reason="tool_round_trip_limit"),
        ),
        (True, "required_action_failed", None),
    ],
)
async def test_early_illegal_response_adds_only_one_turn_to_the_regular_budget(
    tmp_path,
    required: bool,
    expected_status: str,
    expected_action: YieldAction | None,
):
    """An early correction is extra, while later legal notebook calls still stop at five total."""

    adapter = ScriptedAdapter(
        (tool_segment("unknown", {}, call_id="call-1"),),
        *(
            (
                tool_segment(
                    "update_notebook",
                    notebook_payload(f"note-{index}"),
                    call_id=f"call-{index}",
                ),
            )
            for index in range(2, 6)
        ),
        (
            tool_segment(
                "speak_public",
                {"text": "must not receive an unbounded sixth request"},
                call_id="call-6",
            ),
        ),
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=required))

    assert len(adapter.requests) == 5
    assert outcome.round_trips == 5
    assert outcome.status == expected_status
    assert outcome.action == expected_action
    assert outcome.illegal_corrections == 1
    assert game_state.players["alice"].notebook.notes == "note-5"
    assert sum(record["type"] == "checkpoint" for record in history_records(history)) == 4


@pytest.mark.parametrize(
    ("required", "expected_status", "expected_action"),
    [
        (False, "yielded", YieldAction(actor="alice", reason="illegal_tool_call")),
        (True, "required_action_failed", None),
    ],
)
async def test_illegal_correction_response_terminates_without_a_sixth_request(
    tmp_path,
    required: bool,
    expected_status: str,
    expected_action: YieldAction | None,
):
    """A second illegal response consumes the one correction turn and stops immediately."""

    adapter = ScriptedAdapter(
        *(
            (
                tool_segment(
                    "update_notebook",
                    notebook_payload(f"note-{index}"),
                    call_id=f"call-{index}",
                ),
            )
            for index in range(1, 4)
        ),
        (tool_segment("unknown", {}, call_id="call-4"),),
        (tool_segment("still_unknown", {}, call_id="call-5"),),
        (
            tool_segment(
                "speak_public",
                {"text": "must not run"},
                call_id="call-6",
            ),
        ),
    )
    agent, _game_state, _history, _resolver = build_agent(tmp_path, adapter)

    outcome = await agent.run_action(AgentScene(required=required))

    assert len(adapter.requests) == 5
    assert outcome.round_trips == 5
    assert outcome.status == expected_status
    assert outcome.action == expected_action
    assert outcome.illegal_corrections == 1


async def test_state_provider_refreshes_phase_tools_and_notebook_target_each_round(tmp_path):
    """Task 6 replaces GameState on commit, so a captured object becomes stale mid-lifecycle."""

    initial = sample_game_state()
    initial.phase = "day.discussion"
    replacement = initial.model_copy(update={"phase": "night"}, deep=True)
    state_box = {"current": initial}

    class ReplacingAdapter(ScriptedAdapter):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelSegment]:
            self.requests.append(request)
            if not self.scripts:
                raise AssertionError("unexpected model call")
            script = self.scripts.pop(0)
            if len(self.requests) == 1:
                state_box["current"] = replacement
            for scripted_segment in script:
                yield scripted_segment

    adapter = ReplacingAdapter(
        (tool_segment("update_notebook", notebook_payload("fresh"), call_id="call-1"),),
        (segment(0, "final_message", "done", call_id="call-2"),),
    )
    history = HistoryWriter(tmp_path / "game.jsonl", EventStream())
    resolver = RecordingResolver()
    agent = PlayerAgent(
        player_id="alice",
        state_provider=lambda: state_box["current"],
        resolve_model=resolver,
        adapter=adapter,
        history=history,
    )

    await agent.run_action(AgentScene())

    first_tools = {
        tool["function"]["name"] for tool in adapter.requests[0].tools
    }
    second_tools = {
        tool["function"]["name"] for tool in adapter.requests[1].tools
    }
    assert "speak_public" in first_tools
    assert "use_ability" not in first_tools
    assert "use_ability" in second_tools
    assert "speak_public" not in second_tools
    assert replacement.players["alice"].notebook.notes == "fresh"
    assert initial.players["alice"].notebook.notes == ""


async def test_probe_is_short_stateless_tool_free_and_does_not_mutate_agent_state(tmp_path):
    """A short probe must not join the normal continuation chain or gain game tools."""

    adapter = ScriptedAdapter(
        (
            segment(0, "reasoning", "brief thought", call_id="probe-1", source_field="reasoning_content"),
            segment(
                1,
                "final_message",
                '{"decision":"respond","urgency":4,"action_type":"speak"}',
                call_id="probe-1",
            ),
        ),
        (
            segment(
                0,
                "final_message",
                '{"decision":"defer","urgency":0,"action_type":"yield"}',
                call_id="probe-2",
            ),
        ),
    )
    agent, game_state, history, resolver = build_agent(tmp_path, adapter)
    game_state.players["alice"].notebook.notes = "keep me"
    before = deepcopy(agent.state)
    trigger = public_claim(actor="bob", mentions={"alice"})

    first = await agent.probe(trigger)
    second = await agent.probe(trigger)

    assert (first.decision, first.urgency, first.action_type) == ("respond", 4, "speak")
    assert second.decision == "defer"
    assert resolver.short_flags == [True, True]
    assert all(request.model.name == "short" and request.tools == () for request in adapter.requests)
    assert [len(request.messages) for request in adapter.requests] == [2, 2]
    assert agent.state == before
    assert game_state.players["alice"].notebook.notes == "keep me"
    probe_segments = [
        record["payload"]
        for record in history_records(history)
        if record["type"] == "model.output_segment"
    ]
    assert all(payload["call_purpose"] == "reaction_probe" for payload in probe_segments)


async def test_probe_malformed_output_safely_degrades_to_silent(tmp_path):
    """Malformed provider text must not become an unvalidated scheduler instruction."""

    adapter = ScriptedAdapter(
        (segment(0, "final_message", '{"decision":"rush"}', call_id="probe-bad"),)
    )
    agent, _state, _history, _resolver = build_agent(tmp_path, adapter)

    result = await agent.probe(public_claim(actor="bob", mentions={"alice"}))

    assert result.fallback is True
    assert result.decision == "silent"


async def test_probe_refuses_observer_only_event_type_even_if_audience_is_mislabeled_public(tmp_path):
    """A bad audience label must not turn raw observer reasoning into a short-call prompt."""

    adapter = ScriptedAdapter()
    agent, _state, _history, resolver = build_agent(tmp_path, adapter)
    observer_trace = EventRecord(
        phase="day.discussion",
        type="model.output_segment",
        actor="bob",
        audience=Audience.public(),
        payload={"kind": "reasoning", "text": "DO_NOT_SEND"},
    )

    result = await agent.probe(observer_trace)

    assert result == ReactionProbe(decision="silent", urgency=0, action_type="yield")
    assert adapter.requests == []
    assert resolver.short_flags == []


async def test_private_invitation_response_is_short_stateless_and_private_to_the_invitee(tmp_path):
    """Invitation consent is a dedicated short call, never a normal continuation or tool turn."""

    adapter = ScriptedAdapter(
        (
            segment(
                0,
                "final_message",
                '{"decision":"accept"}',
                call_id="invite-1",
            ),
        )
    )
    agent, game_state, history, resolver = build_agent(tmp_path, adapter)
    game_state.phase = "day.discussion"
    before = deepcopy(agent.state)
    invitation = EventRecord(
        phase="day.private_invite",
        type="chat.private_invitation",
        actor="bob",
        audience=Audience.player("alice"),
        payload={"request_id": "invite-1", "inviter": "bob"},
    )

    result = await agent.respond_private_invitation(invitation)

    assert result == PrivateInvitationResponse(decision="accept")
    assert resolver.short_flags == [True]
    assert adapter.requests[0].tools == ()
    assert len(adapter.requests[0].messages) == 2
    assert agent.state == before
    assert all(record["payload"]["call_purpose"] == "private_invitation_response" for record in history_records(history))
    prompt = json.loads(adapter.requests[0].messages[1]["content"])
    assert prompt["invitation"]["payload"]["inviter"] == "bob"


async def test_private_invitation_response_rejects_an_unauthorized_or_malformed_event_without_calling_a_model(tmp_path):
    """A public or malformed invitation cannot be promoted into private consent."""

    adapter = ScriptedAdapter()
    agent, _state, _history, resolver = build_agent(tmp_path, adapter)
    forged = EventRecord(
        phase="day.private_invite",
        type="chat.private_invitation",
        actor="bob",
        audience=Audience.public(),
        payload={"request_id": "invite-1", "inviter": "bob"},
    )

    result = await agent.respond_private_invitation(forged)

    assert result.fallback is True
    assert result.decision == "defer"
    assert adapter.requests == []
    assert resolver.short_flags == []


async def test_private_scene_context_excludes_public_and_other_chat_events(tmp_path):
    """Private model prompts may contain only the exact two-person chat event stream."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "speak_private",
                {"chat_id": "chat-a", "text": "only ours"},
                call_id="private-1",
            ),
        )
    )
    agent, game_state, history, _resolver = build_agent(tmp_path, adapter)
    game_state.phase = "day.private"
    await history.append(public_claim(actor="bob", mentions={"alice"}))
    await history.append(
        EventRecord(
            phase="day.private",
            type="chat.private_message",
            audience=Audience.players({"alice", "carol"}),
            payload={"chat_id": "chat-other", "text": "do not disclose"},
        )
    )
    await history.append(
        EventRecord(
            phase="day.private",
            type="chat.private_message",
            audience=Audience.players({"alice", "bob"}),
            payload={"chat_id": "chat-a", "text": "permitted"},
        )
    )

    await agent.run_action(
        AgentScene(
            phase="day.private",
            allowed_tools=("speak_private", "leave_private_chat", "update_notebook", "yield_action"),
            private_context_only=True,
            details={"chat_id": "chat-a", "participants": ["alice", "bob"]},
        )
    )

    prompt = json.loads(adapter.requests[0].messages[1]["content"])
    assert [event["payload"].get("chat_id") for event in prompt["events"]] == ["chat-a"]


async def test_private_scene_context_events_require_the_current_chat_id(tmp_path):
    """The in-memory transcript is authorization filtered like the stream, including chat identity."""

    adapter = ScriptedAdapter(
        (
            tool_segment(
                "speak_private",
                {"chat_id": "chat-current", "text": "reply"},
                call_id="private-context-1",
            ),
        )
    )
    agent, game_state, _history, _resolver = build_agent(tmp_path, adapter)
    game_state.phase = "day.private"
    participants = {"alice", "bob"}
    old_chat = EventRecord(
        phase="day.private",
        type="chat.private_message",
        audience=Audience.players(participants),
        payload={"chat_id": "chat-old", "text": "old secret"},
    )
    wrong_audience = EventRecord(
        phase="day.private",
        type="chat.private_message",
        audience=Audience.players({"alice", "carol"}),
        payload={"chat_id": "chat-current", "text": "other secret"},
    )
    current_chat = EventRecord(
        phase="day.private",
        type="chat.private_message",
        audience=Audience.players(participants),
        payload={"chat_id": "chat-current", "text": "current secret"},
    )

    await agent.run_action(
        AgentScene(
            phase="day.private",
            allowed_tools=("speak_private", "leave_private_chat", "update_notebook", "yield_action"),
            private_context_only=True,
            context_events=(old_chat, wrong_audience, current_chat),
            details={"chat_id": "chat-current", "participants": ["alice", "bob"]},
        )
    )

    prompt = json.loads(adapter.requests[0].messages[1]["content"])
    assert [event["payload"]["text"] for event in prompt["events"]] == ["current secret"]
