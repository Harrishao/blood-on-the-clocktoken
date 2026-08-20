from __future__ import annotations

import pytest

from clocktower.agents.context import is_safe_public_event, project_context
from clocktower.domain.events import Audience, EventRecord
from clocktower.domain.state import AttentionState, Notebook, NotebookAttention
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


def test_notebook_attention_serializes_as_structured_metadata_with_safe_legacy_conversion():
    """A scalar attention flag cannot express event-scoped scheduling relevance."""

    structured = Notebook(
        notes="private",
        attention=NotebookAttention(
            players=["bob"], pending_actions=["nominate"], watch_triggers=["claim.public"]
        ),
    )
    legacy = Notebook(notes="old", attention=AttentionState.ACTIVE)

    assert structured.model_dump(mode="json")["attention"] == {
        "players": ["bob"],
        "pending_actions": ["nominate"],
        "watch_triggers": ["claim.public"],
    }
    assert legacy.attention == NotebookAttention()


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


@pytest.mark.parametrize(
    "event_type",
    [
        "model.output_segment",
        "checkpoint",
        "setup.completed",
        "storyteller.decision",
        "role.transformed",
        "vote.rule_resolved",
        "protection.applied",
        "poison.applied",
        "death.prevented",
        "death.redirected",
        "effect.suppressed",
        "butler.master_set",
    ],
)
def test_observer_fact_type_is_rejected_even_when_forged_as_public(event_type: str):
    """A bad publisher audience cannot downgrade hidden rule truth to player-visible data."""

    state = sample_game_state()
    forged = event(
        event_type,
        Audience.public(),
        payload={"hidden_truth": "DO_NOT_SEND"},
    )

    context = project_context("alice", state, (forged,))

    assert context.events == ()
    assert "DO_NOT_SEND" not in context.model_dump_json()


@pytest.mark.parametrize(
    "event_type",
    [
        "role.assigned",
        "evil.info_received",
        "information.received",
        "notebook.updated",
        "role.changed_private",
        "role.change_notified",
        "ability.choice_requested",
        "chat.private_message",
    ],
)
def test_player_private_type_rejects_public_audience_forgery(event_type: str):
    """Private event types require a player-scoped audience even if public is visible_to all."""

    state = sample_game_state()
    forged = event(
        event_type,
        Audience.public(),
        payload={"private_fact": "DO_NOT_SEND"},
    )

    assert project_context("alice", state, (forged,)).events == ()


@pytest.mark.parametrize("audience", [Audience.player("alice"), Audience.players({"alice", "bob"})])
def test_own_legal_player_private_event_remains_visible(audience: Audience):
    """Fail-closed classification must preserve legitimately addressed private information."""

    state = sample_game_state()
    private = event(
        "information.received",
        audience,
        payload={"number": 1},
    )

    assert project_context("alice", state, (private,)).events == (private,)


def test_unknown_event_type_is_rejected_even_when_public():
    """New event types must opt into a visibility class before entering prompts."""

    state = sample_game_state()
    unknown = event("future.hidden_fact", Audience.public(), payload={"secret": "x"})

    assert project_context("alice", state, (unknown,)).events == ()


def test_explicit_public_event_requires_public_audience():
    """A public-type event with a malformed narrower audience is rejected rather than guessed."""

    state = sample_game_state()
    malformed = event("player.public_message", Audience.player("alice"), payload={"text": "x"})

    assert project_context("alice", state, (malformed,)).events == ()


@pytest.mark.parametrize(
    "event_type",
    ["day.started", "day.discussion_resumed", "day.final_nomination_probe"],
)
def test_orchestrator_public_lifecycle_events_are_safe_player_context(event_type: str):
    """Fail-closed classification must not make the orchestrator's day loop invisible."""

    state = sample_game_state()
    lifecycle = event(event_type, Audience.public(), payload={"day": state.day})

    assert is_safe_public_event(lifecycle)
    assert project_context("alice", state, (lifecycle,)).events == (lifecycle,)
