class EffectModel:
    def __init__(self):
        self.energy_deltas = {}

    def learn(self, action, energy_before, energy_after):
        delta = energy_after - energy_before
        self.energy_deltas[action] = delta

    def predict_energy(self, action, energy_before):
        if action not in self.energy_deltas:
            return None

        return energy_before + self.energy_deltas[action]

    def size(self):
        return len(self.energy_deltas)
