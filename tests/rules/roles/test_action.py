from __future__ import annotations

from clocktower.rules.roles.action import Mayor, Monk, Slayer, Soldier, Virgin
from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from tests.builders import game_with_roles


def test_monk_protects_one_other_living_player_until_dawn_when_healthy():
    game = game_with_roles(alice="monk", bob="chef", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Monk().legal_choices(context) == [AbilityChoice("alice", ("bob",)), AbilityChoice("alice", ("carol",))]
    assert Monk().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect(
            "protect",
            {"target_id": "bob", "source": "monk", "expires": "dawn", "requires_healthy": True},
        )
    ]


def test_poisoned_monk_keeps_the_same_choices_but_returns_a_health_gated_protection_intent():
    game = game_with_roles(alice="monk", bob="chef", carol="imp")
    game.players["alice"].reminders.add("poisoned")
    context = AbilityContext.from_state(game, "alice")

    assert Monk().legal_choices(context) == [AbilityChoice("alice", ("bob",)), AbilityChoice("alice", ("carol",))]
    assert Monk().apply(context, AbilityChoice("alice", ("bob",)))[0].payload["requires_healthy"] is True


def test_drunk_perceived_monk_keeps_dead_target_choices_without_revealing_its_real_role():
    game = game_with_roles(alice="drunk", bob="chef", carol="imp")
    game.players["alice"].perceived_identity = "monk"
    game.players["bob"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert Monk().legal_choices(context) == [AbilityChoice("alice", ("bob",)), AbilityChoice("alice", ("carol",))]
    assert Monk().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect(
            "protect",
            {"target_id": "bob", "source": "monk", "expires": "dawn", "requires_healthy": True},
        )
    ]


def test_virgin_marks_first_nomination_and_returns_registration_resolution_for_a_healthy_townsfolk():
    game = game_with_roles(alice="washerwoman", bob="virgin", carol="imp")
    context = AbilityContext.for_nomination(game, nominator="alice", nominee="bob")

    effects = Virgin().on_nominated(context)
    assert effects[0] == RuleEffect(
        "mark_used", {"player_id": "bob", "ability": "virgin", "public": False}
    )
    assert effects[1].kind == "resolve_virgin_trigger"
    assert effects[1].payload["nominator_id"] == "alice"
    assert effects[1].payload["required_category"] == "townsfolk"
    assert effects[1].payload["allows_no_trigger"] is False


def test_virgin_first_nomination_marks_used_when_poisoned_or_nominated_by_a_non_townsfolk():
    for role, reminder, effect_count in (("washerwoman", "poisoned", 1), ("imp", None, 1)):
        game = game_with_roles(alice=role, bob="virgin", carol="chef")
        if reminder:
            game.players["bob"].reminders.add(reminder)

        effects = Virgin().on_nominated(AbilityContext.for_nomination(game, nominator="alice", nominee="bob"))
        assert effects == [
            RuleEffect(
                "mark_used",
                {"player_id": "bob", "ability": "virgin", "public": False},
            )
        ] * effect_count


def test_virgin_leaves_spy_townsfolk_registration_and_nontrigger_as_policy_options():
    game = game_with_roles(alice="spy", bob="virgin", carol="imp")
    effects = Virgin().on_nominated(AbilityContext.for_nomination(game, nominator="alice", nominee="bob"))

    assert effects[0] == RuleEffect(
        "mark_used", {"player_id": "bob", "ability": "virgin", "public": False}
    )
    assert effects[1].kind == "resolve_virgin_trigger"
    assert effects[1].payload["allows_no_trigger"] is True
    assert {registration.category for registration in effects[1].payload["registration_options"]} >= {"minion", "townsfolk"}


def test_virgin_does_not_mark_or_trigger_after_its_first_nomination():
    game = game_with_roles(alice="washerwoman", bob="virgin", carol="chef")
    game.players["bob"].reminders.add("virgin_used")

    assert Virgin().on_nominated(AbilityContext.for_nomination(game, nominator="alice", nominee="bob")) == []


def test_slayer_marks_its_public_once_per_game_use_and_returns_a_health_gated_demon_kill():
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
                "requires_healthy": True,
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


def test_poisoned_slayer_can_choose_a_dead_demon_and_still_spends_its_once_per_game_use():
    game = game_with_roles(alice="slayer", bob="imp", carol="chef")
    game.players["alice"].reminders.add("poisoned")
    game.players["bob"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert AbilityChoice("alice", ("bob",)) in Slayer().legal_choices(context)
    assert Slayer().apply(context, AbilityChoice("alice", ("bob",)))[0] == RuleEffect(
        "mark_used", {"player_id": "alice", "ability": "slayer"}
    )


def test_soldier_prevents_only_healthy_demon_attacks():
    game = game_with_roles(alice="soldier", bob="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Soldier().on_demon_attack(context) == [
        RuleEffect("prevent_death", {"target_id": "alice", "source": "soldier", "requires_healthy": True})
    ]
    game.players["alice"].reminders.add("poisoned")
    assert Soldier().on_demon_attack(AbilityContext.from_state(game, "alice")) == [
        RuleEffect("prevent_death", {"target_id": "alice", "source": "soldier", "requires_healthy": True})
    ]


def test_mayor_offers_storyteller_legal_night_redirects_and_wins_at_three_after_no_execution():
    game = game_with_roles(alice="mayor", bob="soldier", carol="monk", david="imp")
    game.players["bob"].alive = False
    game.players["carol"].reminders.add("protected")
    game.alive_count = 3
    context = AbilityContext.from_state(game, "alice")

    assert Mayor().on_night_attack(context) == [
        RuleEffect(
            "redirect_death",
            {
                "from_player_id": "alice",
                "candidate_ids": ("bob", "carol", "david"),
                "normal_target_id": "alice",
                "allow_no_redirect": True,
                "source": "mayor",
                "requires_healthy": True,
            },
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
