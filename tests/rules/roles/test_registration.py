from __future__ import annotations

from clocktower.rules.roles.base import AbilityContext
from clocktower.rules.roles.registration import RegistrationQuery, registrations_for
from tests.builders import game_with_roles


def test_recluse_exposes_good_evil_and_minion_or_demon_registrations():
    """Restricting Recluse to its true identity removes legal Storyteller choices."""

    game = game_with_roles(alice="recluse", bob="imp", carol="poisoner")
    context = AbilityContext.from_state(game, "alice")

    alignments = registrations_for("alice", RegistrationQuery.ALIGNMENT, context)
    characters = registrations_for("alice", RegistrationQuery.CHARACTER, context)

    assert {registration.alignment for registration in alignments} == {"good", "evil"}
    assert {registration.character for registration in characters} >= {"recluse", "imp", "poisoner"}


def test_spy_exposes_good_and_townsfolk_or_outsider_registrations():
    """Treating Spy only as evil Minion makes information roles too reliable."""

    game = game_with_roles(alice="spy", bob="chef", carol="recluse", david="imp")
    context = AbilityContext.from_state(game, "alice")

    alignments = registrations_for("alice", RegistrationQuery.ALIGNMENT, context)
    characters = registrations_for("alice", RegistrationQuery.CHARACTER, context)

    assert {registration.alignment for registration in alignments} == {"evil", "good"}
    assert {registration.character for registration in characters} >= {"spy", "chef", "recluse"}


def test_ordinary_role_has_only_its_true_registration():
    """Giving ordinary Townsfolk alternate registrations would invent false outcomes."""

    game = game_with_roles(alice="chef", bob="imp")
    registrations = registrations_for(
        "alice", RegistrationQuery.CHARACTER, AbilityContext.from_state(game, "bob")
    )

    assert [(registration.character, registration.alignment) for registration in registrations] == [("chef", "good")]
