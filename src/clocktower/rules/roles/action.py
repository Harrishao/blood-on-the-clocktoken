"""Pure Trouble Brewing protection, execution, and public-action handlers."""

from __future__ import annotations

from clocktower.rules.roles.base import AbilityChoice, AbilityContext, RuleEffect
from clocktower.rules.roles.registration import RegistrationQuery, registrations_for
from clocktower.rules.setup import ROLE_CATEGORIES


class _ActionRole:
    first_night_order: int | None = None
    other_night_order: int | None = None

    @staticmethod
    def _living_targets(ctx: AbilityContext, *, include_actor: bool) -> list[AbilityChoice]:
        if not ctx.is_healthy:
            return []
        return [
            AbilityChoice(ctx.actor_id, (player_id,))
            for player_id, player in ctx.state.players.items()
            if player.alive and (include_actor or player_id != ctx.actor_id)
        ]

    @staticmethod
    def _validate_choice(ctx: AbilityContext, choice: AbilityChoice, choices: list[AbilityChoice]) -> None:
        if choice.actor_id != ctx.actor_id or choice not in choices:
            raise ValueError("illegal ability choice")


class Monk(_ActionRole):
    role = "monk"
    other_night_order = 2

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return self._living_targets(ctx, include_actor=False)

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        self._validate_choice(ctx, choice, self.legal_choices(ctx))
        return [
            RuleEffect(
                "protect",
                {"target_id": choice.targets[0], "source": self.role, "expires": "dawn"},
            )
        ]


class Virgin(_ActionRole):
    role = "virgin"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Virgin has a nomination trigger, not an activated choice")

    def on_nominated(self, ctx: AbilityContext) -> list[RuleEffect]:
        """Return the immediate effect only for the Virgin's first legal trigger."""

        if ctx.nominator_id is None or ctx.nominee_id != ctx.actor_id:
            raise ValueError("Virgin requires a nomination context for itself")
        nominator = ctx.state.players[ctx.nominator_id]
        if (
            not ctx.is_healthy
            or "virgin_used" in ctx.actor.reminders
            or ROLE_CATEGORIES.get(nominator.role) != "townsfolk"
        ):
            return []
        return [
            RuleEffect("mark_used", {"player_id": ctx.actor_id, "ability": self.role}),
            RuleEffect("execute", {"target_id": ctx.nominator_id, "reason": self.role}),
            RuleEffect("end_day", {"reason": self.role}),
        ]


class Slayer(_ActionRole):
    role = "slayer"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        if "slayer_used" in ctx.actor.reminders:
            return []
        return self._living_targets(ctx, include_actor=True)

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        self._validate_choice(ctx, choice, self.legal_choices(ctx))
        target_id = choice.targets[0]
        effects = [RuleEffect("mark_used", {"player_id": ctx.actor_id, "ability": self.role})]
        registrations = registrations_for(target_id, RegistrationQuery.CHARACTER, ctx)
        if any(registration.category == "demon" for registration in registrations):
            effects.append(
                RuleEffect(
                    "kill",
                    {
                        "target_id": target_id,
                        "source": self.role,
                        "requires_registration_category": "demon",
                    },
                )
            )
        return effects


class Soldier(_ActionRole):
    role = "soldier"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Soldier has a passive Demon-attack trigger")

    def on_demon_attack(self, ctx: AbilityContext) -> list[RuleEffect]:
        if not ctx.is_healthy:
            return []
        return [RuleEffect("prevent_death", {"target_id": ctx.actor_id, "source": self.role})]


class Mayor(_ActionRole):
    role = "mayor"

    def legal_choices(self, ctx: AbilityContext) -> list[AbilityChoice]:
        return []

    def apply(self, ctx: AbilityContext, choice: AbilityChoice) -> list[RuleEffect]:
        raise ValueError("Mayor has passive night and day-end triggers")

    def on_night_attack(self, ctx: AbilityContext) -> list[RuleEffect]:
        if not ctx.is_healthy:
            return []
        candidates = tuple(
            player_id
            for player_id, player in ctx.state.players.items()
            if player.alive and player_id != ctx.actor_id
        )
        if not candidates:
            return []
        return [
            RuleEffect(
                "redirect_death",
                {"from_player_id": ctx.actor_id, "candidate_ids": candidates, "source": self.role},
            )
        ]

    def end_of_day_effects(self, ctx: AbilityContext, *, no_execution: bool) -> list[RuleEffect]:
        if not ctx.is_healthy or not no_execution or ctx.state.alive_count != 3:
            return []
        return [RuleEffect("declare_winner", {"winner": "good", "reason": self.role})]
