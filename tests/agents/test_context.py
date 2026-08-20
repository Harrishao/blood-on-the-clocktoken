from __future__ import annotations

from clocktower.agents.context import project_context
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import Notebook
from tests.builders import sample_game_state


def event(
    event_type: str,
    audience: Audience,
    *,
    payload: dict[str, object] | None = None,
) -> EventRecord:
    return EventRecord(
        seq=1,
        phase="day.discussion",
        type=event_type,
        actor="bob",
        audience=audience,
        payload=payload or {},
    )


def test_context_filters_authorization_before_serializing_events():
    """Serializing first can leave another private chat or observer trace in the prompt."""

    state = sample_game_state()
    events = (
        event("chat.public_message", Audience.public(), payload={"text": "hello"}),
        event(
            "chat.private_message",
            Audience.players({"bob", "carol"}),
            payload={"text": "OTHER_PRIVATE_SECRET"},
        ),
        event(
            "model.output_segment",
            Audience.observer(),
            payload={"kind": "reasoning", "text": "OBSERVER_REASONING_SECRET"},
        ),
        event(
            "storyteller.decision",
            Audience.observer(),
            payload={"options": ["STORYTELLER_AUDIT_SECRET"]},
        ),
    )

    context = project_context("alice", state, events)

    assert [visible.type for visible in context.events] == ["chat.public_message"]
    serialized = context.model_dump_json()
    assert "OTHER_PRIVATE_SECRET" not in serialized
    assert "OBSERVER_REASONING_SECRET" not in serialized
    assert "STORYTELLER_AUDIT_SECRET" not in serialized


def test_context_keeps_public_and_own_private_events_only():
    """Dropping an authorized private observation would erase information the player earned."""

    state = sample_game_state()
    events = (
        event("chat.public_message", Audience.public()),
        event("chat.private_message", Audience.players({"alice", "bob"})),
        event("information.received", Audience.player("alice")),
        event("information.received", Audience.player("carol")),
    )

    context = project_context("alice", state, events)

    assert [visible.type for visible in context.events] == [
        "chat.public_message",
        "chat.private_message",
        "information.received",
    ]


def test_drunk_context_uses_only_perceived_identity_alignment_and_ability():
    """Reading PlayerState.role would reveal that a Drunk's perceived role is false."""

    state = sample_game_state(roles={"alice": "drunk", "bob": "chef", "eve": "imp"})
    alice = state.players["alice"]
    alice.perceived_identity = "washerwoman"
    alice.known_alignment = "good"
    alice.perceived_ability_text = "You start knowing that one of two players is a Townsfolk."

    context = project_context("alice", state, ())

    assert context.identity == "washerwoman"
    assert context.alignment == "good"
    assert context.ability_text == alice.perceived_ability_text
    assert "drunk" not in context.model_dump_json().lower()


def test_context_contains_only_the_players_own_notebook_copy():
    """Sharing a notebook reference or all-player mapping leaks private long-term memory."""

    state = sample_game_state()
    state.players["alice"].notebook = Notebook(notes="ALICE_NOTE")
    state.players["bob"].notebook = Notebook(notes="BOB_NOTE_SECRET")

    context = project_context("alice", state, ())
    state.players["alice"].notebook.notes = "mutated later"

    assert context.notebook.notes == "ALICE_NOTE"
    assert "BOB_NOTE_SECRET" not in context.model_dump_json()


def test_tools_are_restricted_by_phase_and_player_authority():
    """A broad static tool list lets a model attempt actions unavailable in its scene."""

    state = sample_game_state()
    player = state.players["alice"]

    expected = {
        "day.discussion": {
            "speak_public",
            "request_private_chat",
            "nominate",
            "update_notebook",
            "yield_action",
        },
        "day.private_invite": {"respond_private_chat", "update_notebook", "yield_action"},
        "day.private": {"speak_private", "leave_private_chat", "update_notebook", "yield_action"},
        "day.voting": {"cast_vote", "update_notebook", "yield_action"},
        "night": {"use_ability", "update_notebook", "yield_action"},
    }
    for phase, tool_names in expected.items():
        state.phase = phase
        assert set(project_context("alice", state, ()).tool_names) == tool_names

    state.phase = "day.discussion"
    player.alive = False
    assert "nominate" not in project_context("alice", state, ()).tool_names

    state.phase = "day.voting"
    player.dead_vote_available = False
    assert "cast_vote" not in project_context("alice", state, ()).tool_names


def test_unknown_phase_exposes_no_tools():
    """Defaulting to permissive tools makes a new phase unsafe until explicitly modeled."""

    state = sample_game_state()
    state.phase = "storyteller.audit"

    assert project_context("alice", state, ()).tool_names == ()
