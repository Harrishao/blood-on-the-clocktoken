"""Pure Trouble Brewing Minion and Demon handlers."""

from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect


class Poisoner:
    role = "poisoner"
    first_night_order = 1
    other_night_order = 1

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if not ctx.actor.alive:
            return []
        return [
            AbilityChoice(ctx.actor_id, (player_id,))
            for player_id in ctx.state.players
        ]

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        if choice.actor_id != ctx.actor_id or choice not in self.legal_choices(ctx):
            raise ValueError("illegal Poisoner target")
        return [
            RuleEffect(
                "poison",
                {
                    "target_id": choice.targets[0],
                    "source": self.role,
                    "expires": "next_day_end",
                    "requires_healthy": True,
                },
            )
        ]


class ScarletWoman:
    role = "scarlet_woman"
    first_night_order = None
    other_night_order = None

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Scarlet Woman has a Demon-death trigger")

    def on_demon_death(self, ctx: AbilityContext, *, demon_id: str) -> list[RuleEffect]:
        demon = ctx.state.players.get(demon_id)
        if (
            not ctx.is_healthy
            or demon is None
            or demon.role != "imp"
            or ctx.state.alive_count < 5
        ):
            return []
        return [
            RuleEffect(
                "transform_role",
                {"player_id": ctx.actor_id, "role": "imp", "source": self.role},
            )
        ]


class Baron:
    role = "baron"
    first_night_order = None
    other_night_order = None

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Baron only changes setup")

    @staticmethod
    def setup_delta() -> dict[str, int]:
        """Document the modifier already enforced by ``build_setup``."""

        return {"townsfolk": -2, "outsider": 2}


class Imp:
    role = "imp"
    first_night_order = None
    other_night_order = 6

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if not ctx.actor.alive:
            return []
        return [
            AbilityChoice(ctx.actor_id, (player_id,))
            for player_id in ctx.state.players
        ]

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        if choice.actor_id != ctx.actor_id or choice not in self.legal_choices(ctx):
            raise ValueError("illegal Imp target")
        target_id = choice.targets[0]
        effects = [
            RuleEffect(
                "kill",
                {"target_id": target_id, "source": self.role, "requires_healthy": True},
            )
        ]
        if target_id != ctx.actor_id:
            return effects

        minion_successors = tuple(
            player_id
            for player_id, player in ctx.state.players.items()
            if player.alive and player_id != ctx.actor_id and player.role in {"poisoner", "spy", "scarlet_woman", "baron"}
        )
        scarlet_successors = tuple(
            player_id
            for player_id in minion_successors
            if ctx.state.players[player_id].role == "scarlet_woman"
            and not AbilityContext.from_state(ctx.state, player_id).is_misinformed
        )
        successors = scarlet_successors if ctx.state.alive_count >= 5 and scarlet_successors else minion_successors
        if successors:
            effects.append(
                RuleEffect(
                    "transform_role",
                    {
                        "candidate_ids": successors,
                        "role": "imp",
                        "source": "imp_self_kill",
                        "requires_healthy": True,
                    },
                )
            )
        return effects

    @staticmethod
    def on_demon_death(ctx: AbilityContext, *, continuation_available: bool) -> list[RuleEffect]:
        """Declare defeat only after the engine has atomically confirmed death."""

        if ctx.actor.alive or continuation_available:
            return []
        return [RuleEffect("declare_winner", {"winner": "good", "reason": "demon_dead"})]
