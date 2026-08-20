"""Pure Trouble Brewing Outsider handlers."""

from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.setup import ROLE_CATEGORIES


class Butler:
    role = "butler"
    first_night_order = 8
    other_night_order = 7

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if not ctx.actor.alive:
            return []
        return [
            AbilityChoice(ctx.actor_id, (player_id,))
            for player_id in ctx.state.players
            if player_id != ctx.actor_id
        ]

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        if choice.actor_id != ctx.actor_id or choice not in self.legal_choices(ctx):
            raise ValueError("illegal Butler master")
        return [
            RuleEffect(
                "set_master",
                {
                    "butler_id": ctx.actor_id,
                    "master_id": choice.targets[0],
                    "requires_healthy": True,
                },
            )
        ]

    def may_vote(self, ctx: AbilityContext, *, master_is_voting: bool) -> bool | None:
        """Return ``None`` when ordinary vote validation remains authoritative."""

        if not ctx.is_healthy:
            return None
        return master_is_voting


class Drunk:
    role = "drunk"
    first_night_order = None
    other_night_order = None

    def legal_perceived_identities(self, ctx: AbilityContext) -> tuple[str, ...]:
        """Return only Townsfolk identities absent from this setup."""

        in_play = {player.role for player in ctx.state.players.values()}
        return tuple(
            role
            for role, category in ROLE_CATEGORIES.items()
            if category == "townsfolk" and role not in in_play
        )

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Drunk has no ability")


class Saint:
    role = "saint"
    first_night_order = None
    other_night_order = None

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Saint has an execution trigger, not an activated choice")

    def on_executed(self, ctx: AbilityContext) -> list[RuleEffect]:
        if not ctx.is_healthy:
            return []
        return [RuleEffect("declare_winner", {"winner": "evil", "reason": self.role})]
