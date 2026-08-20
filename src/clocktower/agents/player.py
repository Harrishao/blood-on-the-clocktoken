"""Bounded ordinary-action and stateless short-probe lifecycle for one player."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from clocktower.agents.context import PlayerContext, project_context
from clocktower.agents.tools import ToolIntentError, parse_tool_intent
from clocktower.config import ResolvedModel
from clocktower.domain.actions import (
    CastVote,
    LeavePrivateChat,
    Nominate,
    PlayerAction,
    RequestPrivateChat,
    RespondPrivateChat,
    SpeakPrivate,
    UpdateNotebook,
    UseAbility,
    YieldAction,
)
from clocktower.domain.events import Audience, EventRecord, ModelOutputSegment
from clocktower.domain.state import GameState, Notebook
from clocktower.history import HistoryWriter
from clocktower.models.protocol import ModelAdapter, ModelRequest, ModelSegment


MAX_TOOL_ROUND_TRIPS = 4


class ModelResolver(Protocol):
    def __call__(self, player_id: str, short: bool) -> ResolvedModel: ...


class AgentScene(BaseModel):
    """The upper-layer scene contract for one model action opportunity."""

    model_config = ConfigDict(frozen=True)

    phase: str | None = None
    purpose: str = "formal_action"
    required: bool = False
    allowed_tools: tuple[str, ...] | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReactionProbe(BaseModel):
    """Strict bounded response used only to adjust deterministic scheduling."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decision: Literal["respond", "defer", "silent"]
    urgency: StrictInt = Field(ge=-15, le=15)
    action_type: Literal["speak", "private_chat", "nominate", "yield"]


class AgentOutcome(BaseModel):
    """One proposed canonical action plus observer-visible lifecycle records."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    action: PlayerAction | None = None
    status: Literal["completed", "yielded", "required_action_failed"] = "completed"
    round_trips: int
    illegal_corrections: int = 0
    events: tuple[EventRecord, ...] = ()


@dataclass
class PlayerAgentState:
    """Provider-neutral memory; raw reasoning is deliberately never stored here."""

    notebook: Notebook
    continuation: tuple[Mapping[str, Any], ...] = ()
    event_cursor: int = 0


@dataclass(slots=True)
class _ToolCall:
    call_id: str
    tool_index: int
    tool_call_id: str
    name: str
    arguments_text: str
    first_segment_index: int


def segment_event(
    player_id: str,
    purpose: str,
    segment: ModelSegment,
    phase: str,
) -> EventRecord:
    """Copy one provider segment into the typed observer-only domain boundary."""

    payload = ModelOutputSegment(
        call_id=segment.call_id,
        player_id=player_id,
        call_purpose=purpose,
        segment_index=segment.index,
        kind=segment.kind,
        source_field=segment.source_field,
        text=segment.text,
        incomplete=segment.incomplete,
        tool_index=segment.tool_index,
        tool_call_id=segment.tool_call_id,
        tool_name=segment.tool_name,
        tool_type=segment.tool_type,
    )
    return EventRecord(
        phase=phase,
        type="model.output_segment",
        actor=player_id,
        audience=Audience.observer(),
        payload=payload.model_dump(),
    )


class PlayerAgent:
    """Own one player's isolated prompt, notebook, and bounded call lifecycle."""

    def __init__(
        self,
        *,
        player_id: str,
        game_state: GameState,
        resolve_model: ModelResolver | object,
        adapter: ModelAdapter,
        history: HistoryWriter,
        event_source: Callable[[], Sequence[EventRecord]] | None = None,
    ) -> None:
        if player_id not in game_state.players:
            raise ValueError(f"unknown player: {player_id}")
        self.player_id = player_id
        self.game_state = game_state
        self._model_resolver = resolve_model
        self.adapter = adapter
        self.history = history
        self._event_source = event_source or (lambda: self.history.stream.after(0))
        self.state = PlayerAgentState(
            notebook=game_state.players[player_id].notebook.model_copy(deep=True)
        )
        self._call_numbers = count(1)
        self._lock = asyncio.Lock()

    async def run_action(self, scene: AgentScene | Mapping[str, Any] | None = None) -> AgentOutcome:
        """Run a normal-model tool loop without publishing the proposed game action."""

        scene_model = self._coerce_scene(scene)
        async with self._lock:
            start_seq = self.history.stream.next_seq - 1
            phase = scene_model.phase or self.game_state.phase
            scene_model = scene_model.model_copy(update={"phase": phase})
            context, source_events = self._project_context(phase, scene_model.allowed_tools)
            base_messages = self._action_messages(context, scene_model)
            messages: list[Mapping[str, Any]] = list(base_messages)
            illegal_calls = 0

            for round_number in range(1, MAX_TOOL_ROUND_TRIPS + 1):
                resolved = self._resolve_model(short=False)
                request = ModelRequest(
                    call_id=f"{self.player_id}-action-{next(self._call_numbers)}",
                    model=resolved,
                    messages=tuple(messages),
                    tools=tuple(context.tool_schemas()),
                    tool_choice="auto" if context.tools else None,
                )
                segments = await self._record_stream(request, scene_model.purpose, phase)
                calls = self._collect_tool_calls(segments)
                if not calls:
                    self._advance_cursor(source_events)
                    if scene_model.required:
                        return self._outcome(
                            action=None,
                            status="required_action_failed",
                            round_trips=round_number,
                            illegal_calls=illegal_calls,
                            start_seq=start_seq,
                        )
                    return self._outcome(
                        action=YieldAction(actor=self.player_id, reason="no_tool_call"),
                        status="yielded",
                        round_trips=round_number,
                        illegal_calls=illegal_calls,
                        start_seq=start_seq,
                    )

                result_messages: list[dict[str, Any]] = []
                assistant_calls: list[dict[str, Any]] = []
                outward: PlayerAction | None = None
                terminate_illegal = False
                next_result_index = max(segment.index for segment in segments) + 1

                allowed_names = frozenset(context.tool_names)
                for call in calls:
                    assistant_calls.append(self._assistant_tool_call(call))
                    try:
                        arguments = self._decode_arguments(call)
                        intent = parse_tool_intent(
                            call.name,
                            arguments,
                            player_id=self.player_id,
                            allowed_tools=allowed_names,
                        )
                        self._validate_intent_authority(intent, scene_model)
                        if isinstance(intent, UpdateNotebook):
                            notebook = Notebook(
                                notes=intent.patch,
                                attention=self.game_state.players[self.player_id].notebook.attention,
                            )
                            await self.history.update_notebook(
                                self.game_state, self.player_id, notebook
                            )
                            self.state.notebook = notebook.model_copy(deep=True)
                            result_payload = {"ok": True, "notebook_updated": True}
                        elif outward is None:
                            outward = intent
                            result_payload = {"ok": True, "accepted": intent.kind}
                        else:
                            result_payload = {"error": "outward_action_already_selected"}
                    except ToolIntentError as error:
                        illegal_calls += 1
                        result_payload = {"error": str(error)}
                        if illegal_calls > 1:
                            terminate_illegal = True

                    result_text = json.dumps(
                        result_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    result_segment = ModelSegment(
                        call_id=call.call_id,
                        index=next_result_index,
                        kind="tool_result",
                        source_field="tool_result",
                        text=result_text,
                        tool_index=call.tool_index,
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                        tool_type="function",
                    )
                    next_result_index += 1
                    await self.history.append(
                        segment_event(
                            self.player_id,
                            scene_model.purpose,
                            result_segment,
                            phase,
                        )
                    )
                    result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.tool_call_id,
                            "content": result_text,
                        }
                    )

                if outward is not None:
                    self._advance_cursor(source_events)
                    return self._outcome(
                        action=outward,
                        status="completed",
                        round_trips=round_number,
                        illegal_calls=illegal_calls,
                        start_seq=start_seq,
                    )
                if terminate_illegal:
                    self._advance_cursor(source_events)
                    return self._illegal_outcome(scene_model, round_number, start_seq)

                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": assistant_calls}
                )
                messages.extend(result_messages)

            self._advance_cursor(source_events)
            if scene_model.required:
                return self._outcome(
                    action=None,
                    status="required_action_failed",
                    round_trips=MAX_TOOL_ROUND_TRIPS,
                    illegal_calls=illegal_calls,
                    start_seq=start_seq,
                )
            return self._outcome(
                action=YieldAction(actor=self.player_id, reason="tool_round_trip_limit"),
                status="yielded",
                round_trips=MAX_TOOL_ROUND_TRIPS,
                illegal_calls=illegal_calls,
                start_seq=start_seq,
            )

    async def probe(self, event: EventRecord) -> ReactionProbe:
        """Run one fresh short-model call with no tools or continuation mutation."""

        projected_state = self.game_state.model_copy(
            update={"phase": event.phase or self.game_state.phase}, deep=True
        )
        probe_context = project_context(
            self.player_id,
            projected_state,
            (event,),
        )
        if not probe_context.events:
            return self._silent_probe()
        trigger_event = probe_context.events[0]
        phase = event.phase or self.game_state.phase
        player = self.game_state.players[self.player_id]
        prompt = {
            "identity": player.perceived_identity,
            "alignment": player.known_alignment,
            "ability_text": player.perceived_ability_text,
            "notebook": player.notebook.model_dump(mode="json"),
            "trigger_event": trigger_event.model_dump(mode="json"),
        }
        messages: tuple[Mapping[str, Any], ...] = (
            {
                "role": "system",
                "content": (
                    "Return only JSON with decision respond/defer/silent, integer urgency "
                    "from -15 to 15, and action_type speak/private_chat/nominate/yield."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
            },
        )
        resolved = self._resolve_model(short=True)
        request = ModelRequest(
            call_id=f"{self.player_id}-probe-{next(self._call_numbers)}",
            model=resolved,
            messages=messages,
            tools=(),
            tool_choice=None,
        )
        segments = await self._record_stream(request, "reaction_probe", phase)
        content = "".join(
            segment.text for segment in segments if segment.kind == "final_message"
        )
        try:
            return ReactionProbe.model_validate_json(content)
        except (ValidationError, ValueError):
            return self._silent_probe()

    def _resolve_model(self, *, short: bool) -> ResolvedModel:
        resolver = self._model_resolver
        if callable(resolver):
            return resolver(self.player_id, short)
        method = getattr(resolver, "resolve_model", None)
        if method is None:
            raise TypeError("resolve_model must be callable or expose resolve_model")
        return method(self.player_id, short=short)

    def _project_context(
        self,
        phase: str,
        allowed_tools: tuple[str, ...] | None,
    ) -> tuple[PlayerContext, tuple[EventRecord, ...]]:
        source_events = tuple(self._event_source())
        new_events = tuple(
            event
            for event in source_events
            if event.seq == 0 or event.seq > self.state.event_cursor
        )
        projected_state = self.game_state.model_copy(update={"phase": phase}, deep=True)
        context = project_context(self.player_id, projected_state, new_events)
        if allowed_tools is not None:
            allowed = frozenset(allowed_tools)
            context = context.model_copy(
                update={"tools": tuple(tool for tool in context.tools if tool.name in allowed)}
            )
        return context, new_events

    @staticmethod
    def _action_messages(
        context: PlayerContext,
        scene: AgentScene,
    ) -> tuple[Mapping[str, Any], ...]:
        prompt = {
            "identity": context.identity,
            "alignment": context.alignment,
            "ability_text": context.ability_text,
            "notebook": context.notebook.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in context.events],
            "scene": {
                "phase": scene.phase,
                "purpose": scene.purpose,
                "required": scene.required,
                "details": scene.details,
            },
            "available_tools": list(context.tool_names),
        }
        return (
            {
                "role": "system",
                "content": (
                    "You are one isolated Blood on the Clocktower player. Use only the "
                    "provided tools. You know only the identity, ability, notebook, and "
                    "authorized events in this prompt."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
            },
        )

    async def _record_stream(
        self,
        request: ModelRequest,
        purpose: str,
        phase: str,
    ) -> tuple[ModelSegment, ...]:
        segments: list[ModelSegment] = []
        async for provider_segment in self.adapter.stream(request):
            segments.append(provider_segment)
            await self.history.append(
                segment_event(self.player_id, purpose, provider_segment, phase)
            )
        return tuple(segments)

    @staticmethod
    def _collect_tool_calls(segments: Sequence[ModelSegment]) -> tuple[_ToolCall, ...]:
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for segment in segments:
            if segment.kind != "tool_call":
                continue
            tool_index = segment.tool_index if segment.tool_index is not None else segment.index
            key = (segment.call_id, tool_index)
            current = grouped.setdefault(
                key,
                {
                    "call_id": segment.call_id,
                    "tool_index": tool_index,
                    "tool_call_id": None,
                    "name": None,
                    "arguments": "",
                    "first_segment_index": segment.index,
                },
            )
            current["arguments"] += segment.text
            current["tool_call_id"] = segment.tool_call_id or current["tool_call_id"]
            current["name"] = segment.tool_name or current["name"]

        calls = []
        for current in sorted(grouped.values(), key=lambda value: value["first_segment_index"]):
            call_id = current["call_id"]
            tool_index = current["tool_index"]
            calls.append(
                _ToolCall(
                    call_id=call_id,
                    tool_index=tool_index,
                    tool_call_id=current["tool_call_id"]
                    or f"{call_id}:tool:{tool_index}",
                    name=current["name"] or "",
                    arguments_text=current["arguments"],
                    first_segment_index=current["first_segment_index"],
                )
            )
        return tuple(calls)

    @staticmethod
    def _decode_arguments(call: _ToolCall) -> dict[str, Any]:
        try:
            decoded = json.loads(call.arguments_text)
        except json.JSONDecodeError as error:
            raise ToolIntentError("tool arguments are not valid JSON") from error
        if not isinstance(decoded, dict):
            raise ToolIntentError("tool arguments must be a JSON object")
        return decoded

    def _validate_intent_authority(
        self,
        intent: PlayerAction,
        scene: AgentScene,
    ) -> None:
        players = self.game_state.players
        if isinstance(intent, (Nominate, RequestPrivateChat)):
            target = intent.target if isinstance(intent, Nominate) else intent.target_player
            if target not in players:
                raise ToolIntentError(f"unknown target player: {target}")
            if isinstance(intent, RequestPrivateChat) and target == self.player_id:
                raise ToolIntentError("cannot request private chat with yourself")
        if isinstance(intent, UseAbility):
            unknown = [target for target in intent.targets if target not in players]
            if unknown:
                raise ToolIntentError(f"unknown ability target: {unknown[0]}")
        details = scene.details
        if isinstance(intent, (SpeakPrivate, LeavePrivateChat)):
            expected_chat_id = details.get("chat_id")
            if not isinstance(expected_chat_id, str) or intent.chat_id != expected_chat_id:
                raise ToolIntentError("chat_id does not match the active scene")
        if isinstance(intent, RespondPrivateChat):
            expected_request_id = details.get("request_id")
            if (
                not isinstance(expected_request_id, str)
                or intent.request_id != expected_request_id
            ):
                raise ToolIntentError("request_id does not match the active scene")
        if isinstance(intent, CastVote):
            expected_nomination_id = details.get("nomination_id")
            if (
                not isinstance(expected_nomination_id, str)
                or intent.nomination_id != expected_nomination_id
            ):
                raise ToolIntentError("nomination_id does not match the active scene")
        if isinstance(intent, UseAbility):
            expected_action = details.get("ability", details.get("action"))
            if not isinstance(expected_action, str) or intent.action != expected_action:
                raise ToolIntentError("ability does not match the active scene")
        legal_targets = details.get("legal_targets")
        if legal_targets is not None and isinstance(intent, (Nominate, RequestPrivateChat)):
            target = intent.target if isinstance(intent, Nominate) else intent.target_player
            if target not in legal_targets:
                raise ToolIntentError("target is not authorized for the active scene")
        if legal_targets is not None and isinstance(intent, UseAbility):
            normalized = {
                tuple(targets) if isinstance(targets, (list, tuple)) else (targets,)
                for targets in legal_targets
            }
            if intent.targets not in normalized:
                raise ToolIntentError("ability targets are not authorized for the active scene")

    @staticmethod
    def _assistant_tool_call(call: _ToolCall) -> dict[str, Any]:
        return {
            "id": call.tool_call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments_text},
        }

    def _advance_cursor(self, source_events: Sequence[EventRecord]) -> None:
        sequenced = [event.seq for event in source_events if event.seq > 0]
        if sequenced:
            self.state.event_cursor = max(self.state.event_cursor, max(sequenced))

    def _illegal_outcome(
        self,
        scene: AgentScene,
        round_number: int,
        start_seq: int,
    ) -> AgentOutcome:
        if scene.required:
            return self._outcome(
                action=None,
                status="required_action_failed",
                round_trips=round_number,
                illegal_calls=2,
                start_seq=start_seq,
            )
        return self._outcome(
            action=YieldAction(actor=self.player_id, reason="illegal_tool_call"),
            status="yielded",
            round_trips=round_number,
            illegal_calls=2,
            start_seq=start_seq,
        )

    def _outcome(
        self,
        *,
        action: PlayerAction | None,
        status: Literal["completed", "yielded", "required_action_failed"],
        round_trips: int,
        illegal_calls: int,
        start_seq: int,
    ) -> AgentOutcome:
        return AgentOutcome(
            action=action,
            status=status,
            round_trips=round_trips,
            illegal_corrections=min(illegal_calls, 1),
            events=self.history.stream.after(start_seq),
        )

    @staticmethod
    def _coerce_scene(scene: AgentScene | Mapping[str, Any] | None) -> AgentScene:
        if scene is None:
            return AgentScene()
        if isinstance(scene, AgentScene):
            return scene
        return AgentScene.model_validate(scene)

    @staticmethod
    def _silent_probe() -> ReactionProbe:
        return ReactionProbe(decision="silent", urgency=0, action_type="yield")


__all__ = [
    "AgentOutcome",
    "AgentScene",
    "PlayerAgent",
    "PlayerAgentState",
    "ReactionProbe",
    "segment_event",
]
