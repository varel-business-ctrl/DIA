class MultiEffectModel:
    def __init__(self):
        self.effects = {}

    def learn(
        self,
        action,
        energy_before,
        energy_after,
        orientation_before,
        orientation_after,
        movement_succeeded,
    ):
        self.effects[action] = {
            "energy_delta": energy_after - energy_before,
            "orientation_change": (
                orientation_before,
                orientation_after,
            ),
            "movement_succeeded": movement_succeeded,
        }

    def predict(
        self,
        action,
        energy_before,
    ):
        if action not in self.effects:
            return None

        effect = self.effects[action]

        return {
            "energy": energy_before + effect["energy_delta"],
            "orientation_change": effect["orientation_change"],
            "movement_succeeded": effect["movement_succeeded"],
        }

    def size(self):
        return len(self.effects)
