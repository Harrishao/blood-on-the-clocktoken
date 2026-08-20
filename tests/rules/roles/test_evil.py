from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.evil import Baron, Imp, Poisoner, ScarletWoman
from clocktower.rules.setup import build_setup
from tests.builders import game_with_roles


def test_poisoner_keeps_dead_target_choices_when_poisoned_and_returns_a_health_gated_poison_intent():
    game = game_with_roles(alice="poisoner", bob="chef", carol="imp")
    game.players["bob"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert Poisoner().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect(
            "poison",
            {"target_id": "bob", "source": "poisoner", "expires": "next_day_end", "requires_healthy": True},
        )
    ]
    game.players["alice"].reminders.add("poisoned")
    assert AbilityChoice("alice", ("bob",)) in Poisoner().legal_choices(AbilityContext.from_state(game, "alice"))


def test_scarlet_woman_becomes_imp_only_when_a_demon_dies_with_at_least_five_alive():
    game = game_with_roles(
        alice="scarlet_woman", bob="imp", carol="chef", david="empath", eve="monk"
    )
    context = AbilityContext.from_state(game, "alice")

    assert ScarletWoman().on_demon_death(context, demon_id="bob") == [
        RuleEffect("transform_role", {"player_id": "alice", "role": "imp", "source": "scarlet_woman"})
    ]
    game.alive_count = 4
    assert ScarletWoman().on_demon_death(AbilityContext.from_state(game, "alice"), demon_id="bob") == []


def test_baron_exposes_the_same_setup_delta_used_by_core_setup_validation():
    roles = ["washerwoman", "chef", "empath", "recluse", "drunk", "baron", "imp"]

    assert Baron().setup_delta() == {"townsfolk": -2, "outsider": 2}
    assert build_setup(7, roles, seed=17).category_counts == (3, 2, 1, 1)


def test_imp_kills_dead_targets_and_prioritizes_a_healthy_living_scarlet_woman_as_self_kill_successor():
    game = game_with_roles(
        alice="imp", bob="poisoner", carol="scarlet_woman", david="chef", eve="monk"
    )
    game.players["david"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert Imp().apply(context, AbilityChoice("alice", ("david",))) == [
        RuleEffect("kill", {"target_id": "david", "source": "imp", "requires_healthy": True})
    ]
    self_kill = Imp().apply(context, AbilityChoice("alice", ("alice",)))
    assert self_kill[1].payload["candidate_ids"] == ("carol",)


def test_imp_self_kill_falls_back_to_other_living_minions_when_scarlet_is_poisoned_or_threshold_fails():
    game = game_with_roles(
        alice="imp", bob="poisoner", carol="scarlet_woman", david="chef", eve="monk"
    )
    game.players["carol"].reminders.add("poisoned")
    self_kill = Imp().apply(AbilityContext.from_state(game, "alice"), AbilityChoice("alice", ("alice",)))

    assert self_kill[1].payload["candidate_ids"] == ("bob", "carol")
    game.alive_count = 4
    game.players["carol"].reminders.clear()
    threshold_failed = Imp().apply(AbilityContext.from_state(game, "alice"), AbilityChoice("alice", ("alice",)))
    assert threshold_failed[1].payload["candidate_ids"] == ("bob", "carol")


def test_disabled_imp_self_kill_without_a_minion_never_returns_an_independently_applicable_good_win():
    game = game_with_roles(alice="imp", bob="chef", carol="saint")
    game.players["alice"].reminders.add("poisoned")
    context = AbilityContext.from_state(game, "alice")

    assert Imp().apply(context, AbilityChoice("alice", ("alice",))) == [
        RuleEffect("kill", {"target_id": "alice", "source": "imp", "requires_healthy": True}),
    ]
    assert Imp().on_demon_death(context, continuation_available=False) == []


def test_healthy_imp_death_without_a_continuation_declares_good_only_after_death_is_confirmed():
    game = game_with_roles(alice="imp", bob="chef", carol="saint")
    game.players["alice"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert Imp().on_demon_death(context, continuation_available=False) == [
        RuleEffect("declare_winner", {"winner": "good", "reason": "demon_dead"})
    ]
    assert Imp().on_demon_death(context, continuation_available=True) == []
