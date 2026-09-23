class EnvironmentalLearner:
    def __init__(self):
        self.rules = {}

    def learn(self, action, obstacle_ahead, movement_succeeded):
        key = (action, obstacle_ahead)
        self.rules[key] = movement_succeeded

    def predict(self, action, obstacle_ahead):
        return self.rules.get((action, obstacle_ahead))


learner = EnvironmentalLearner()

# Training experience:
# move_forward with an obstacle directly ahead failed.
learner.learn(
    action="move_forward",
    obstacle_ahead=True,
    movement_succeeded=False,
)

tests = [
    ("training-like obstacle", "move_forward", True),
    ("clear path", "move_forward", False),
]

for name, action, obstacle_ahead in tests:
    prediction = learner.predict(
        action,
        obstacle_ahead,
    )

    print(
        name,
        "| obstacle_ahead =", obstacle_ahead,
        "| prediction =", prediction,
    )
