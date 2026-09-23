class FactorizedEnergyLearner:
    def __init__(self):
        self.energy_deltas = {}

    def learn(self, action, energy_before, energy_after):
        delta = energy_after - energy_before
        self.energy_deltas[action] = delta

    def predict(self, action, energy_before):
        if action not in self.energy_deltas:
            return None

        return energy_before + self.energy_deltas[action]


learner = FactorizedEnergyLearner()

# Training experience
learner.learn(
    action="move_forward",
    energy_before=100,
    energy_after=99,
)

# Previously unseen state
prediction = learner.predict(
    action="move_forward",
    energy_before=80,
)

print("LEARNED EFFECT:", learner.energy_deltas)
print("UNSEEN ENERGY:", 80)
print("PREDICTION:", prediction)
print("EXPECTED:", 79)
print("GENERALIZATION:", prediction == 79)
