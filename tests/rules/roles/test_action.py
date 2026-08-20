from __future__ import annotations

from clocktower.rules.roles.action import Mayor, Monk, Slayer, Soldier, Virgin
from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from tests.builders import game_with_roles


def test_monk_protects_one_other_living_player_until_dawn_when_healthy():
    game = game_with_roles(alice="monk", bob="chef", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Monk().legal_choices(context) == [AbilityChoice("alice", ("bob",)), AbilityChoice("alice", ("carol",))]
    assert Monk().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect("protect", {"target_id": "bob", "source": "monk", "expires": "dawn"})
    ]


def test_poisoned_monk_has_no_protection_choice():
    game = game_with_roles(alice="monk", bob="chef", carol="imp")
    game.players["alice"].reminders.add("poisoned")

    assert Monk().legal_choices(AbilityContext.from_state(game, "alice")) == []


def test_virgin_executes_a_healthy_first_townsfolk_nominator_and_ends_the_day():
    game = game_with_roles(alice="washerwoman", bob="virgin", carol="imp")
    context = AbilityContext.for_nomination(game, nominator="alice", nominee="bob")

    assert Virgin().on_nominated(context) == [
        RuleEffect("mark_used", {"player_id": "bob", "ability": "virgin"}),
        RuleEffect("execute", {"target_id": "alice", "reason": "virgin"}),
        RuleEffect("end_day", {"reason": "virgin"}),
    ]


def test_virgin_does_not_trigger_when_used_poisoned_or_nominated_by_non_townsfolk():
    for role, reminder in (("washerwoman", "virgin"), ("washerwoman", "poisoned"), ("imp", None)):
        game = game_with_roles(alice=role, bob="virgin", carol="chef")
        if reminder is not None:
            game.players["bob"].reminders.add("virgin_used" if reminder == "virgin" else reminder)

        assert Virgin().on_nominated(
            AbilityContext.for_nomination(game, nominator="alice", nominee="bob")
        ) == []


def test_slayer_marks_its_public_once_per_game_use_and_kills_a_legal_demon():
    game = game_with_roles(alice="slayer", bob="imp", carol="chef")
    context = AbilityContext.from_state(game, "alice")

    assert Slayer().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect("mark_used", {"player_id": "alice", "ability": "slayer"}),
        RuleEffect(
            "kill",
            {
                "target_id": "bob",
                "source": "slayer",
                "requires_registration_category": "demon",
            },
        ),
    ]

    game.players["alice"].reminders.add("slayer_used")
    assert Slayer().legal_choices(AbilityContext.from_state(game, "alice")) == []


def test_slayer_healthy_use_on_non_demon_only_marks_the_ability_used():
    game = game_with_roles(alice="slayer", bob="chef", carol="imp")

    assert Slayer().apply(
        AbilityContext.from_state(game, "alice"), AbilityChoice("alice", ("bob",))
    ) == [RuleEffect("mark_used", {"player_id": "alice", "ability": "slayer"})]


def test_soldier_prevents_only_healthy_demon_attacks():
    game = game_with_roles(alice="soldier", bob="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Soldier().on_demon_attack(context) == [
        RuleEffect("prevent_death", {"target_id": "alice", "source": "soldier"})
    ]
    game.players["alice"].reminders.add("poisoned")
    assert Soldier().on_demon_attack(AbilityContext.from_state(game, "alice")) == []


def test_mayor_offers_storyteller_legal_night_redirects_and_wins_at_three_after_no_execution():
    game = game_with_roles(alice="mayor", bob="chef", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Mayor().on_night_attack(context) == [
        RuleEffect(
            "redirect_death",
            {"from_player_id": "alice", "candidate_ids": ("bob", "carol"), "source": "mayor"},
        )
    ]
    assert Mayor().end_of_day_effects(context, no_execution=True) == [
        RuleEffect("declare_winner", {"winner": "good", "reason": "mayor"})
    ]


def test_mayor_does_not_win_with_an_execution_or_a_different_alive_count():
    game = game_with_roles(alice="mayor", bob="chef", carol="imp", david="poisoner")
    context = AbilityContext.from_state(game, "alice")

    assert Mayor().end_of_day_effects(context, no_execution=True) == []
    game.alive_count = 3
    assert Mayor().end_of_day_effects(AbilityContext.from_state(game, "alice"), no_execution=False) == []
