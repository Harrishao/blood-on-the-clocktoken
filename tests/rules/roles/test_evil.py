from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.evil import Baron, Imp, Poisoner, ScarletWoman
from clocktower.rules.setup import build_setup
from tests.builders import game_with_roles


def test_poisoner_poison_lasts_through_the_next_day_when_healthy():
    game = game_with_roles(alice="poisoner", bob="chef", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Poisoner().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect("poison", {"target_id": "bob", "source": "poisoner", "expires": "next_day_end"})
    ]
    game.players["alice"].reminders.add("poisoned")
    assert Poisoner().legal_choices(AbilityContext.from_state(game, "alice")) == []


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


def test_imp_kills_nightly_and_self_kill_offers_only_living_minions_as_successors():
    game = game_with_roles(alice="imp", bob="poisoner", carol="scarlet_woman", david="chef")
    game.players["carol"].alive = False
    context = AbilityContext.from_state(game, "alice")

    assert Imp().apply(context, AbilityChoice("alice", ("david",))) == [
        RuleEffect("kill", {"target_id": "david", "source": "imp"})
    ]
    assert Imp().apply(context, AbilityChoice("alice", ("alice",))) == [
        RuleEffect("kill", {"target_id": "alice", "source": "imp"}),
        RuleEffect(
            "transform_role",
            {"candidate_ids": ("bob",), "role": "imp", "source": "imp_self_kill"},
        ),
    ]


def test_imp_self_kill_or_demon_death_without_a_valid_continuation_declares_good_winner():
    game = game_with_roles(alice="imp", bob="chef", carol="saint")
    context = AbilityContext.from_state(game, "alice")

    assert Imp().apply(context, AbilityChoice("alice", ("alice",))) == [
        RuleEffect("kill", {"target_id": "alice", "source": "imp"}),
        RuleEffect("declare_winner", {"winner": "good", "reason": "demon_dead"}),
    ]
    assert Imp().on_demon_death(context, continuation_available=False) == [
        RuleEffect("declare_winner", {"winner": "good", "reason": "demon_dead"})
    ]
    assert Imp().on_demon_death(context, continuation_available=True) == []
