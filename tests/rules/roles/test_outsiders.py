from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.outsiders import Butler, Drunk, Saint
from tests.builders import game_with_roles


def test_butler_chooses_another_living_master_and_can_vote_only_with_that_master():
    game = game_with_roles(alice="butler", bob="chef", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    assert Butler().legal_choices(context) == [AbilityChoice("alice", ("bob",)), AbilityChoice("alice", ("carol",))]
    assert Butler().apply(context, AbilityChoice("alice", ("bob",))) == [
        RuleEffect("set_master", {"butler_id": "alice", "master_id": "bob"})
    ]
    assert Butler().may_vote(context, master_is_voting=True)
    assert not Butler().may_vote(context, master_is_voting=False)


def test_drunk_has_no_ability_and_can_only_receive_a_townsfolk_identity_not_in_play():
    game = game_with_roles(alice="drunk", bob="monk", carol="imp")
    context = AbilityContext.from_state(game, "alice")

    identities = Drunk().legal_perceived_identities(context)
    assert "monk" not in identities
    assert "washerwoman" in identities
    assert Drunk().legal_choices(context) == []


def test_saint_execution_declares_evil_winner_unless_the_saint_is_misinformed():
    game = game_with_roles(alice="saint", bob="imp")

    assert Saint().on_executed(AbilityContext.from_state(game, "alice")) == [
        RuleEffect("declare_winner", {"winner": "evil", "reason": "saint"})
    ]
    game.players["alice"].reminders.add("poisoned")
    assert Saint().on_executed(AbilityContext.from_state(game, "alice")) == []
