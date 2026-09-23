from organism.learning.context_effect_model import ContextEffectModel


def show_prediction(model, context, action, energy):
    prediction = model.predict(
        context=context,
        action=action,
        energy_before=energy,
    )

    print(
        f"{context} | {action} | "
        f"prediction={prediction}"
    )


model = ContextEffectModel()


print("\n=== REPEATED SUCCESS EXPERIENCE ===")

for i in range(5):
    result = model.learn(
        context="clear_path",
        action="move_forward",
        energy_before=100 - i,
        energy_after=99 - i,
        orientation_before="east",
        orientation_after="east",
        movement_succeeded=True,
    )

    print(
        f"Experience {i + 1}: "
        f"observations={result['observations']} "
        f"success_rate={result['movement_success_rate']} "
        f"confidence={result['confidence']}"
    )


print("\n=== REPEATED FAILURE EXPERIENCE ===")

for i in range(5):
    result = model.learn(
        context="obstacle_ahead",
        action="move_forward",
        energy_before=100 - i,
        energy_after=99 - i,
        orientation_before="east",
        orientation_after="east",
        movement_succeeded=False,
    )

    print(
        f"Experience {i + 1}: "
        f"observations={result['observations']} "
        f"success_rate={result['movement_success_rate']} "
        f"confidence={result['confidence']}"
    )


print("\n=== MIXED EXPERIENCE ===")

mixed_results = [
    True,
    False,
    True,
    False,
    True,
]

for i, success in enumerate(mixed_results):
    result = model.learn(
        context="uncertain_path",
        action="move_forward",
        energy_before=100 - i,
        energy_after=99 - i,
        orientation_before="east",
        orientation_after="east",
        movement_succeeded=success,
    )

    print(
        f"Experience {i + 1}: "
        f"success={success} "
        f"observations={result['observations']} "
        f"success_rate={result['movement_success_rate']} "
        f"confidence={result['confidence']}"
    )


print("\n=== FINAL PREDICTIONS ===")

show_prediction(
    model,
    "clear_path",
    "move_forward",
    80,
)

show_prediction(
    model,
    "obstacle_ahead",
    "move_forward",
    80,
)

show_prediction(
    model,
    "uncertain_path",
    "move_forward",
    80,
)


print("\n=== FINAL MODEL SUMMARY ===")

print(model.summary())
