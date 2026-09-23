from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ContextEffect:
    context: str
    action: str
    observations: int = 0
    successful_movements: int = 0
    failed_movements: int = 0
    energy_delta_total: float = 0.0
    orientation_changes: dict[str, int] | None = None

    def __post_init__(self):
        if self.orientation_changes is None:
            self.orientation_changes = {}

    @property
    def movement_success_rate(self):
        total = self.successful_movements + self.failed_movements

        if total == 0:
            return None

        return self.successful_movements / total

    @property
    def average_energy_delta(self):
        if self.observations == 0:
            return None

        return self.energy_delta_total / self.observations

    @property
    def confidence(self):
        if self.observations == 0:
            return 0.0

        return min(1.0, self.observations / 10.0)

    def to_dict(self):
        data = asdict(self)
        data["movement_success_rate"] = self.movement_success_rate
        data["average_energy_delta"] = self.average_energy_delta
        data["confidence"] = self.confidence
        return data


class ContextEffectModel:
    """
    Experimental context-aware consequence model for DIA.

    The model learns relationships between:

        context + action -> observed consequences

    It does not assume that one action always has one outcome.
    It tracks repeated observations and exposes uncertainty
    when evidence is limited.
    """

    def __init__(self):
        self.effects: dict[tuple[str, str], ContextEffect] = {}

    def _key(self, context, action):
        return (str(context), str(action))

    def learn(
        self,
        context,
        action,
        energy_before,
        energy_after,
        orientation_before,
        orientation_after,
        movement_succeeded,
    ):
        key = self._key(context, action)

        if key not in self.effects:
            self.effects[key] = ContextEffect(
                context=str(context),
                action=str(action),
            )

        effect = self.effects[key]

        energy_delta = energy_after - energy_before

        effect.observations += 1
        effect.energy_delta_total += energy_delta

        if movement_succeeded:
            effect.successful_movements += 1
        else:
            effect.failed_movements += 1

        orientation_key = (
            f"{orientation_before}->{orientation_after}"
        )

        effect.orientation_changes[orientation_key] = (
            effect.orientation_changes.get(orientation_key, 0) + 1
        )

        return effect.to_dict()

    def predict(self, context, action, energy_before):
        key = self._key(context, action)

        if key not in self.effects:
            return {
                "known": False,
                "context": context,
                "action": action,
                "prediction": None,
                "reason": "no_experience_for_context_action_pair",
            }

        effect = self.effects[key]

        average_delta = effect.average_energy_delta
        success_rate = effect.movement_success_rate

        predicted_energy = None

        if average_delta is not None:
            predicted_energy = energy_before + average_delta

        return {
            "known": True,
            "context": context,
            "action": action,
            "prediction": {
                "energy": predicted_energy,
                "movement_success_rate": success_rate,
                "observations": effect.observations,
                "confidence": effect.confidence,
                "orientation_changes": dict(
                    effect.orientation_changes
                ),
            },
            "reason": "prediction_based_on_observed_experiences",
        }

    def get_effect(self, context, action):
        key = self._key(context, action)

        effect = self.effects.get(key)

        if effect is None:
            return None

        return effect.to_dict()

    def known_contexts(self):
        return sorted(
            {
                context
                for context, action in self.effects.keys()
            }
        )

    def known_actions(self):
        return sorted(
            {
                action
                for context, action in self.effects.keys()
            }
        )

    def size(self):
        return len(self.effects)

    def total_observations(self):
        return sum(
            effect.observations
            for effect in self.effects.values()
        )

    def summary(self):
        return {
            "context_action_pairs": self.size(),
            "total_observations": self.total_observations(),
            "known_contexts": self.known_contexts(),
            "known_actions": self.known_actions(),
            "effects": {
                f"{context}|{action}": effect.to_dict()
                for (context, action), effect in self.effects.items()
            },
        }
