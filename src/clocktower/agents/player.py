"""Bounded ordinary-action and stateless short-probe lifecycle for one player."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StrictInt, ValidationError

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
    private_context_only: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ReactionProbe(BaseModel):
    """Strict bounded response used only to adjust deterministic scheduling."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decision: Literal["respond", "defer", "silent"]
    urgency: StrictInt = Field(ge=-15, le=15)
    action_type: Literal["speak", "private_chat", "nominate", "yield"]
    _fallback: bool = PrivateAttr(default=False)

    @property
    def fallback(self) -> bool:
        """True only for a local parser fallback, never a provider-controlled schema field."""

        return self._fallback

    @classmethod
    def fallback_silent(cls) -> ReactionProbe:
        probe = cls.model_construct(decision="silent", urgency=0, action_type="yield")
        probe._fallback = True
        return probe


class PrivateInvitationResponse(BaseModel):
    """A stateless short-model decision about one private-chat invitation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decision: Literal["accept", "reject", "defer"]
    _fallback: bool = PrivateAttr(default=False)

    @property
    def fallback(self) -> bool:
        """Whether local parsing supplied the safe no-chat response."""

        return self._fallback

    @classmethod
    def fallback_defer(cls) -> PrivateInvitationResponse:
        response = cls.model_construct(decision="defer")
        response._fallback = True
        return response


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
    tool_type: str
    arguments_text: str
    first_segment_index: int


@dataclass(slots=True)
class _ValidatedToolCall:
    call: _ToolCall
    intent: PlayerAction | None = None
    error: str | None = None


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
        state_provider: Callable[[], GameState] | None = None,
        game_state: GameState | None = None,
        resolve_model: ModelResolver | object,
        adapter: ModelAdapter,
        history: HistoryWriter,
        event_source: Callable[[], Sequence[EventRecord]] | None = None,
    ) -> None:
        if state_provider is not None and game_state is not None:
            raise ValueError("provide state_provider or game_state, not both")
        if state_provider is None:
            if game_state is None:
                raise ValueError("state_provider is required")
            state_provider = lambda: game_state
        self._state_provider = state_provider
        initial_state = self._current_state()
        if player_id not in initial_state.players:
            raise ValueError(f"unknown player: {player_id}")
        self.player_id = player_id
        self._model_resolver = resolve_model
        self.adapter = adapter
        self.history = history
        self._event_source = event_source or (lambda: self.history.stream.after(0))
        self.state = PlayerAgentState(
            notebook=initial_state.players[player_id].notebook.model_copy(deep=True)
        )
        self._call_numbers = count(1)
        self._lock = asyncio.Lock()

    async def run_action(self, scene: AgentScene | Mapping[str, Any] | None = None) -> AgentOutcome:
        """Run a normal-model tool loop without publishing the proposed game action."""

        scene_model = self._coerce_scene(scene)
        async with self._lock:
            start_seq = self.history.stream.next_seq - 1
            continuation_messages: list[Mapping[str, Any]] = []
            illegal_calls = 0
            source_events: tuple[EventRecord, ...] = ()
            regular_round_trips = 0
            request_count = 0
            correction_pending = False

            while correction_pending or regular_round_trips < MAX_TOOL_ROUND_TRIPS:
                is_correction_response = correction_pending
                correction_pending = False
                if not is_correction_response:
                    regular_round_trips += 1
                request_count += 1
                round_state = self._current_state()
                phase = scene_model.phase or round_state.phase
                round_scene = scene_model.model_copy(update={"phase": phase})
                context, source_events = self._project_context(
                    round_state,
                    phase,
                    round_scene.allowed_tools,
                    private_context_only=round_scene.private_context_only,
                    participants=round_scene.details.get("participants"),
                )
                context = self._apply_scene_tool_policy(context, round_scene)
                messages = [
                    *self._action_messages(context, round_scene),
                    *continuation_messages,
                ]
                resolved = self._resolve_model(short=False)
                request = ModelRequest(
                    call_id=f"{self.player_id}-action-{next(self._call_numbers)}",
                    model=resolved,
                    messages=tuple(messages),
                    tools=tuple(context.tool_schemas()),
                    tool_choice="auto" if context.tools else None,
                )
                segments = await self._record_stream(request, round_scene.purpose, phase)
                calls = self._collect_tool_calls(segments)
                if not calls:
                    self._advance_cursor(source_events)
                    if scene_model.required:
                        return self._outcome(
                            action=None,
                            status="required_action_failed",
                            round_trips=request_count,
                            illegal_calls=illegal_calls,
                            start_seq=start_seq,
                        )
                    return self._outcome(
                        action=YieldAction(actor=self.player_id, reason="no_tool_call"),
                        status="yielded",
                        round_trips=request_count,
                        illegal_calls=illegal_calls,
                        start_seq=start_seq,
                    )

                validated = self._validate_tool_batch(calls, scene_model)
                next_result_index = max(segment.index for segment in segments) + 1
                batch_invalid = any(item.error is not None for item in validated)
                if batch_invalid:
                    illegal_calls += 1
                outward: PlayerAction | None = None
                result_messages: list[dict[str, Any]] = []
                assistant_calls: list[dict[str, Any]] = []
                seen_call_ids: set[str] = set()
                for item in validated:
                    call = item.call
                    if call.tool_call_id in seen_call_ids:
                        continue
                    seen_call_ids.add(call.tool_call_id)
                    assistant_calls.append(self._assistant_tool_call(call))
                    if batch_invalid:
                        result_payload = {
                            "error": item.error or "batch_rejected_due_to_invalid_sibling"
                        }
                    else:
                        intent = item.intent
                        if isinstance(intent, UpdateNotebook):
                            live_state = self._current_state()
                            notebook = intent.notebook.model_copy(deep=True)
                            await self.history.update_notebook(
                                live_state, self.player_id, notebook
                            )
                            self.state.notebook = notebook.model_copy(deep=True)
                            result_payload = {"ok": True, "notebook_updated": True}
                        else:
                            if intent is None:
                                raise AssertionError("validated tool call has no intent")
                            outward = intent
                            result_payload = {"ok": True, "accepted": intent.kind}
                    result_message = await self._record_tool_result(
                        call,
                        result_payload,
                        segment_index=next_result_index,
                        purpose=round_scene.purpose,
                        phase=phase,
                    )
                    next_result_index += 1
                    result_messages.append(result_message)

                if batch_invalid and (is_correction_response or illegal_calls > 1):
                    self._advance_cursor(source_events)
                    return self._illegal_outcome(round_scene, request_count, start_seq)
                if batch_invalid:
                    correction_pending = True
                if outward is not None:
                    self._advance_cursor(source_events)
                    return self._outcome(
                        action=outward,
                        status="completed",
                        round_trips=request_count,
                        illegal_calls=illegal_calls,
                        start_seq=start_seq,
                    )

                assistant_content = "".join(
                    segment.text
                    for segment in segments
                    if segment.kind == "final_message"
                )
                continuation_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content or None,
                        "tool_calls": assistant_calls,
                    }
                )
                continuation_messages.extend(result_messages)

            self._advance_cursor(source_events)
            if scene_model.required:
                return self._outcome(
                    action=None,
                    status="required_action_failed",
                    round_trips=request_count,
                    illegal_calls=illegal_calls,
                    start_seq=start_seq,
                )
            return self._outcome(
                action=YieldAction(actor=self.player_id, reason="tool_round_trip_limit"),
                status="yielded",
                round_trips=request_count,
                illegal_calls=illegal_calls,
                start_seq=start_seq,
            )

    async def probe(self, event: EventRecord) -> ReactionProbe:
        """Run one fresh short-model call with no tools or continuation mutation."""

        live_state = self._current_state()
        projected_state = live_state.model_copy(
            update={"phase": event.phase or live_state.phase}, deep=True
        )
        probe_context = project_context(
            self.player_id,
            projected_state,
            (event,),
        )
        if not probe_context.events:
            return self._silent_probe()
        trigger_event = probe_context.events[0]
        phase = event.phase or live_state.phase
        player = live_state.players[self.player_id]
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
            return self._silent_probe(fallback=True)

    async def respond_private_invitation(
        self,
        invitation: EventRecord,
    ) -> PrivateInvitationResponse:
        """Make one isolated, tool-free short decision for an invitee-only event."""

        live_state = self._current_state()
        projected_state = live_state.model_copy(
            update={"phase": invitation.phase or live_state.phase}, deep=True
        )
        invitation_context = project_context(self.player_id, projected_state, (invitation,))
        if (
            len(invitation_context.events) != 1
            or invitation_context.events[0].type != "chat.private_invitation"
            or invitation.audience.kind != "player"
            or invitation.audience.player_ids != frozenset({self.player_id})
        ):
            return PrivateInvitationResponse.fallback_defer()

        private_event = invitation_context.events[0]
        player = live_state.players[self.player_id]
        prompt = {
            "identity": player.perceived_identity,
            "alignment": player.known_alignment,
            "ability_text": player.perceived_ability_text,
            "notebook": player.notebook.model_dump(mode="json"),
            "invitation": private_event.model_dump(mode="json"),
        }
        messages: tuple[Mapping[str, Any], ...] = (
            {
                "role": "system",
                "content": (
                    "Return only JSON with decision accept, reject, or defer. "
                    "This is a private invitation response, not a game tool action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
            },
        )
        resolved = self._resolve_model(short=True)
        request = ModelRequest(
            call_id=f"{self.player_id}-private-invitation-{next(self._call_numbers)}",
            model=resolved,
            messages=messages,
            tools=(),
            tool_choice=None,
        )
        segments = await self._record_stream(
            request,
            "private_invitation_response",
            invitation.phase or live_state.phase,
        )
        content = "".join(
            segment.text for segment in segments if segment.kind == "final_message"
        )
        try:
            return PrivateInvitationResponse.model_validate_json(content)
        except (ValidationError, ValueError):
            return PrivateInvitationResponse.fallback_defer()

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
        game_state: GameState,
        phase: str,
        allowed_tools: tuple[str, ...] | None,
        *,
        private_context_only: bool = False,
        participants: object = None,
    ) -> tuple[PlayerContext, tuple[EventRecord, ...]]:
        source_events = tuple(self._event_source())
        new_events = tuple(
            event
            for event in source_events
            if event.seq == 0 or event.seq > self.state.event_cursor
        )
        if private_context_only:
            participant_ids = (
                frozenset(participants)
                if isinstance(participants, (list, tuple))
                and len(participants) == 2
                and all(isinstance(player_id, str) for player_id in participants)
                else frozenset()
            )
            new_events = tuple(
                event
                for event in new_events
                if event.audience.kind == "players"
                and event.audience.player_ids == participant_ids
            )
        projected_state = game_state.model_copy(update={"phase": phase}, deep=True)
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

    @staticmethod
    def _apply_scene_tool_policy(
        context: PlayerContext,
        scene: AgentScene,
    ) -> PlayerContext:
        if not scene.required:
            return context
        return context.model_copy(
            update={
                "tools": tuple(
                    tool for tool in context.tools if tool.name != "yield_action"
                )
            }
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
                    "tool_type": None,
                    "arguments": "",
                    "first_segment_index": segment.index,
                },
            )
            current["arguments"] += segment.text
            current["tool_call_id"] = segment.tool_call_id or current["tool_call_id"]
            current["name"] = segment.tool_name or current["name"]
            current["tool_type"] = segment.tool_type or current["tool_type"]

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
                    tool_type=current["tool_type"] or "function",
                    arguments_text=current["arguments"],
                    first_segment_index=current["first_segment_index"],
                )
            )
        return tuple(calls)

    def _validate_tool_batch(
        self,
        calls: Sequence[_ToolCall],
        scene: AgentScene,
    ) -> tuple[_ValidatedToolCall, ...]:
        call_id_counts: dict[str, int] = {}
        for call in calls:
            call_id_counts[call.tool_call_id] = call_id_counts.get(call.tool_call_id, 0) + 1

        validated: list[_ValidatedToolCall] = []
        for call in calls:
            if call_id_counts[call.tool_call_id] > 1:
                validated.append(
                    _ValidatedToolCall(call=call, error="duplicate_tool_call_id")
                )
                continue
            try:
                live_state = self._current_state()
                live_phase = scene.phase or live_state.phase
                live_scene = scene.model_copy(update={"phase": live_phase})
                live_context, _ = self._project_context(
                    live_state,
                    live_phase,
                    live_scene.allowed_tools,
                    private_context_only=live_scene.private_context_only,
                    participants=live_scene.details.get("participants"),
                )
                live_context = self._apply_scene_tool_policy(
                    live_context,
                    live_scene,
                )
                arguments = self._decode_arguments(call)
                intent = parse_tool_intent(
                    call.name,
                    arguments,
                    player_id=self.player_id,
                    allowed_tools=frozenset(live_context.tool_names),
                )
                self._validate_intent_authority(intent, live_scene, live_state)
                validated.append(_ValidatedToolCall(call=call, intent=intent))
            except ToolIntentError as error:
                validated.append(_ValidatedToolCall(call=call, error=str(error)))

        outward_items = [
            item
            for item in validated
            if item.error is None and not isinstance(item.intent, UpdateNotebook)
        ]
        if len(outward_items) > 1:
            for item in outward_items:
                item.error = "multiple_outward_actions"
        return tuple(validated)

    async def _record_tool_result(
        self,
        call: _ToolCall,
        payload: dict[str, Any],
        *,
        segment_index: int,
        purpose: str,
        phase: str,
    ) -> dict[str, Any]:
        result_text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result_segment = ModelSegment(
            call_id=call.call_id,
            index=segment_index,
            kind="tool_result",
            source_field="tool_result",
            text=result_text,
            tool_index=call.tool_index,
            tool_call_id=call.tool_call_id,
            tool_name=call.name,
            tool_type=call.tool_type,
        )
        await self.history.append(
            segment_event(self.player_id, purpose, result_segment, phase)
        )
        return {
            "role": "tool",
            "tool_call_id": call.tool_call_id,
            "content": result_text,
        }

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
        game_state: GameState,
    ) -> None:
        players = game_state.players
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
            "type": call.tool_type,
            "function": {"name": call.name, "arguments": call.arguments_text},
        }

    def _advance_cursor(self, source_events: Sequence[EventRecord]) -> None:
        sequenced = [event.seq for event in source_events if event.seq > 0]
        if sequenced:
            self.state.event_cursor = max(self.state.event_cursor, max(sequenced))

    def _current_state(self) -> GameState:
        state = self._state_provider()
        if not isinstance(state, GameState):
            raise TypeError("state_provider must return GameState")
        return state

    @property
    def game_state(self) -> GameState:
        """Compatibility view that always resolves the current authoritative state."""

        return self._current_state()

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
    def _silent_probe(*, fallback: bool = False) -> ReactionProbe:
        if fallback:
            return ReactionProbe.fallback_silent()
        return ReactionProbe(decision="silent", urgency=0, action_type="yield")


__all__ = [
    "AgentOutcome",
    "AgentScene",
    "PlayerAgent",
    "PlayerAgentState",
    "PrivateInvitationResponse",
    "ReactionProbe",
    "segment_event",
]
