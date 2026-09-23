from organism.research.world_experiment import WorldExperimentRunner


def predict(initial, actions):
    energy = initial["organism"]["energy"]
    orientation = initial["organism"]["orientation"]

    directions = ["north", "east", "south", "west"]

    for action in actions:
        energy -= 1

        if action == "turn_right":
            index = directions.index(orientation)
            orientation = directions[(index + 1) % 4]

        elif action == "turn_left":
            index = directions.index(orientation)
            orientation = directions[(index - 1) % 4]

    return {
        "energy": energy,
        "orientation": orientation,
    }


experiments = [
    ["wait"],
    ["turn_right"],
    ["turn_left"],
    ["wait", "turn_right"],
    ["turn_right", "turn_right"],
    ["turn_left", "turn_left"],
    ["wait", "wait", "turn_right", "move_forward"],
]


total_error = 0.0

for number, actions in enumerate(experiments, start=1):

    runner = WorldExperimentRunner()

    initial = runner.observe()
    prediction = predict(initial, actions)

    result = runner.run(actions)
    final = result["final"]

    actual = {
        "energy": final["organism"]["energy"],
        "orientation": final["organism"]["orientation"],
    }

    differences = sum(
        prediction[key] != actual[key]
        for key in prediction
    )

    error = differences / len(prediction)
    total_error += error

    print()
    print(f"EXPERIMENT {number}")
    print("ACTIONS:", actions)
    print("PREDICTION:", prediction)
    print("ACTUAL:", actual)
    print("ERROR:", round(error, 3))


average_error = total_error / len(experiments)

print()
print("================================")
print("REPEATED EXPERIMENT SUMMARY")
print("EXPERIMENT COUNT:", len(experiments))
print("AVERAGE PREDICTION ERROR:", round(average_error, 3))
print("================================")
