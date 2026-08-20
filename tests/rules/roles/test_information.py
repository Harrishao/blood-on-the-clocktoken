from __future__ import annotations

from clocktower.rules.roles.base import AbilityContext
from clocktower.rules.roles.information import (
    Chef,
    Empath,
    FortuneTeller,
    Investigator,
    Librarian,
    Ravenkeeper,
    Spy,
    Undertaker,
    Washerwoman,
)
from tests.builders import game_with_roles, game_with_seats


def test_empath_counts_only_alive_evil_neighbours_across_seating_edge():
    """Counting a dead neighbour would give Empath illegal information."""

    game = game_with_seats(["good", "evil", "good", "evil"])
    game.players["bob"].alive = False

    assert Empath().observe(game, actor_seat=0).number == 1


def test_empath_skips_dead_seats_to_each_nearest_living_neighbour():
    """Stopping at a dead seat hides the living neighbour immediately beyond it."""

    game = game_with_seats(["good", "good", "evil", "good", "good"])
    game.players["bob"].alive = False
    game.players["eve"].alive = False

    assert Empath().observe(game, actor_seat=0).number == 1


def test_empath_counts_a_lone_living_neighbour_once_and_keeps_zero_when_alone():
    """Duplicating a sole neighbour or emptying the Cartesian product is illegal."""

    game = game_with_seats(["good", "evil", "good"])
    game.players["carol"].alive = False
    assert Empath().observe(game, actor_seat=0).number == 1

    game.players["bob"].alive = False
    observations = Empath().legal_observations(AbilityContext.from_state(game, "alice"))
    assert {observation.number for observation in observations} == {0}


def test_chef_counts_evil_pairs_including_the_circular_seating_edge():
    """Ignoring the last-to-first pair undercounts Chef information."""

    game = game_with_seats(["evil", "good", "evil", "evil"])

    assert Chef().observe(game, actor_seat=0).number == 2


def test_fortune_teller_yes_includes_the_configured_red_herring():
    """Dropping the red herring would reveal the Fortune Teller's true result."""

    game = game_with_roles(alice="fortune_teller", bob="townsfolk", carol="imp")
    game.role_state.fortune_teller_red_herring = "bob"

    assert FortuneTeller().choose(game, "alice", ["bob", "alice"]).yes is True


def test_fortune_teller_leaves_recluse_registration_as_policy_options():
    """Resolving Recluse as a Demon inside the handler skips Storyteller policy."""

    game = game_with_roles(alice="fortune_teller", bob="recluse", carol="imp")
    observations = FortuneTeller().legal_observations(
        AbilityContext.from_state(game, "alice"), ["bob", "alice"]
    )

    assert {observation.yes for observation in observations} == {False, True}


def test_fortune_teller_cannot_receive_no_for_an_unregistered_demon():
    """Offering no for the actual Imp would invent a registration it lacks."""

    game = game_with_roles(alice="fortune_teller", bob="imp", carol="washerwoman")
    observations = FortuneTeller().legal_observations(
        AbilityContext.from_state(game, "alice"), ["bob", "alice"]
    )

    assert {observation.yes for observation in observations} == {True}


def test_librarian_can_receive_the_zero_outsider_observation():
    """Forcing a pair when no Outsider exists creates false rules information."""

    game = game_with_roles(alice="librarian", bob="washerwoman", carol="imp")
    observations = Librarian().legal_observations(AbilityContext.from_state(game, "alice"))

    assert [(observation.number, observation.player_ids) for observation in observations] == [(0, ())]


def test_librarian_keeps_zero_and_pair_for_a_recluse_registration_choice():
    """A Recluse may register non-Outsider, so zero cannot be discarded."""

    game = game_with_roles(alice="librarian", bob="recluse", carol="imp")
    observations = Librarian().legal_observations(AbilityContext.from_state(game, "alice"))

    assert any(observation.number == 0 for observation in observations)
    assert any(observation.character == "recluse" for observation in observations)


def test_librarian_keeps_zero_and_pair_when_only_spy_can_register_outsider():
    """A Spy's optional Outsider registration must remain a policy alternative."""

    game = game_with_roles(alice="librarian", bob="spy", carol="imp")
    observations = Librarian().legal_observations(AbilityContext.from_state(game, "alice"))

    assert any(observation.number == 0 for observation in observations)
    assert any(observation.character == "recluse" for observation in observations)


def test_washerwoman_can_show_its_only_townsfolk_actor_in_a_pair():
    """Excluding the actor removes legal first-night information in a small setup."""

    game = game_with_roles(alice="washerwoman", bob="imp", carol="poisoner")
    observations = Washerwoman().legal_observations(AbilityContext.from_state(game, "alice"))

    assert {observation.player_ids for observation in observations} == {
        ("alice", "bob"),
        ("alice", "carol"),
    }
    assert {observation.character for observation in observations} == {"washerwoman"}


def test_investigator_can_be_one_of_the_two_players_shown():
    """The same actor exclusion wrongly blocks Investigator's legal pair."""

    game = game_with_roles(alice="investigator", bob="poisoner", carol="imp")
    observations = Investigator().legal_observations(AbilityContext.from_state(game, "alice"))

    assert any(
        observation.character == "poisoner" and observation.player_ids == ("alice", "bob")
        for observation in observations
    )


def test_drunk_washerwoman_observations_use_the_perceived_handler_role():
    """Labeling an information result Drunk leaks the actor's true identity."""

    game = game_with_roles(alice="drunk", bob="imp", carol="poisoner")
    game.players["alice"].perceived_identity = "washerwoman"
    observations = Washerwoman().legal_observations(AbilityContext.from_state(game, "alice"))

    assert observations
    assert {observation.kind for observation in observations} == {"washerwoman"}


def test_poisoned_washerwoman_exposes_true_and_false_legal_observations():
    """Choosing a specific lie in the handler would bypass Storyteller policy."""

    game = game_with_roles(alice="washerwoman", bob="chef", carol="imp", david="empath")
    game.players["alice"].reminders.add("poisoned")

    observations = Washerwoman().legal_observations(AbilityContext.from_state(game, "alice"))

    assert any(observation.truthful for observation in observations)
    assert any(not observation.truthful for observation in observations)


def test_undertaker_exposes_recluse_character_registration_choices():
    """Collapsing Recluse to its true character would remove a legal registration."""

    game = game_with_roles(alice="undertaker", bob="recluse", carol="imp")
    observations = Undertaker().legal_observations(AbilityContext.from_state(game, "alice"), "bob")

    assert {observation.character for observation in observations} >= {"recluse", "imp"}


def test_ravenkeeper_requires_a_night_kill_and_offers_registered_characters():
    """Allowing a living Ravenkeeper to learn a role would grant an illegal ability."""

    game = game_with_roles(alice="ravenkeeper", bob="spy", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Ravenkeeper().legal_choices(context) == []
    game.players["alice"].alive = False
    game.players["alice"].reminders.add("killed_at_night")
    observation = Ravenkeeper().choose(AbilityContext.from_state(game, "alice"), "bob")

    assert {option.character for option in observation.options} >= {"spy", "ravenkeeper"}


def test_poisoned_ravenkeeper_exposes_false_character_options_without_selecting_one():
    """A poisoned Ravenkeeper restricted to real registrations gets protected information."""

    game = game_with_roles(alice="ravenkeeper", bob="chef", carol="imp")
    game.players["alice"].alive = False
    game.players["alice"].reminders.update({"killed_at_night", "poisoned"})
    observation = Ravenkeeper().choose(AbilityContext.from_state(game, "alice"), "bob")

    assert any(option.character == "imp" and not option.truthful for option in observation.options)


def test_spy_grimoire_is_a_private_snapshot_not_shared_game_state():
    """Returning the live grimoire would leak mutable truth into another context."""

    game = game_with_roles(alice="spy", bob="washerwoman", carol="imp")
    observation = Spy().observe(AbilityContext.from_state(game, "alice"))

    assert observation.private_to == "alice"
    assert observation.grimoire is not None
    assert observation.grimoire["bob"]["role"] == "washerwoman"
    observation.grimoire["bob"]["role"] = "imp"
    assert game.players["bob"].role == "washerwoman"
