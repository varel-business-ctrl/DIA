from environment.world1.world import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


def configure_position(world, x, y, orientation):
    organism = world.organism
    organism.position.x = x
    organism.position.y = y
    organism.orientation = orientation
    organism.energy = 100


def run_trial(
    label,
    world,
    model,
    perception,
    extractor,
    x,
    y,
    orientation,
    train,
):
    configure_position(world, x, y, orientation)

    organism = world.organism

    observation_before = perception.perceive(world)
    context = extractor.extract(observation_before)

    energy_before = organism.energy
    orientation_before = organism.orientation

    position_before = (
        organism.position.x,
        organism.position.y,
    )

    prediction = model.predict(
        context=context,
        action="move_forward",
        energy_before=energy_before,
    )

    predicted_success = None

    if prediction["known"]:
        predicted_success = (
            prediction["prediction"]["movement_success_rate"] > 0.5
        )

    world.step("move_forward")

    energy_after = organism.energy
    orientation_after = organism.orientation

    position_after = (
        organism.position.x,
        organism.position.y,
    )

    actual_success = position_before != position_after

    if train:
        model.learn(
            context=context,
            action="move_forward",
            energy_before=energy_before,
            energy_after=energy_after,
            orientation_before=orientation_before,
            orientation_after=orientation_after,
            movement_succeeded=actual_success,
        )

    correct = (
        predicted_success is not None
        and predicted_success == actual_success
    )

    print(f"\n--- {label} ---")
    print("Position:", position_before)
    print("Orientation:", orientation)
    print("Context:", context)
    print("Prediction known:", prediction["known"])
    print("Predicted movement:", predicted_success)
    print("Actual movement:", actual_success)
    print("Prediction correct:", correct)
    print("Training mode:", train)

    return {
        "known": prediction["known"],
        "predicted": predicted_success,
        "actual": actual_success,
        "correct": correct,
    }


def main():
    world = World1()

    perception = EnhancedLocalPerception(radius=2)
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    print("DIA DIRECTIONAL GENERALIZATION TEST")
    print("=" * 55)

    # ---------------------------------------------------------
    # PHASE 1: Train only on north-facing experiences.
    # ---------------------------------------------------------
    print("\nPHASE 1: NORTH-ONLY TRAINING")
    print("=" * 55)

    training_cases = [
        ("North clear", 1, 1, "north"),
        ("North clear", 1, 1, "north"),
        ("North obstacle", 4, 3, "north"),
        ("North obstacle", 4, 3, "north"),
        ("North boundary", 1, 9, "north"),
        ("North boundary", 1, 9, "north"),
    ]

    for case in training_cases:
        run_trial(
            label=case[0],
            world=world,
            model=model,
            perception=perception,
            extractor=extractor,
            x=case[1],
            y=case[2],
            orientation=case[3],
            train=True,
        )

    # ---------------------------------------------------------
    # PHASE 2: Test unseen orientations without training.
    # ---------------------------------------------------------
    print("\n\nPHASE 2: UNSEEN ORIENTATION TESTING")
    print("=" * 55)

    test_cases = [
        # Clear path, east.
        ("East clear", 1, 1, "east"),

        # Clear path, south.
        ("South clear", 5, 5, "south"),

        # Obstacle ahead, east.
        ("East obstacle", 3, 4, "east"),

        # Obstacle ahead, south.
        ("South obstacle", 4, 5, "south"),

        # Boundary ahead, east.
        ("East boundary", 9, 1, "east"),

        # Boundary ahead, west.
        ("West boundary", 0, 1, "west"),
    ]

    known_predictions = 0
    correct_predictions = 0
    unknown_predictions = 0

    for case in test_cases:
        result = run_trial(
            label=case[0],
            world=world,
            model=model,
            perception=perception,
            extractor=extractor,
            x=case[1],
            y=case[2],
            orientation=case[3],
            train=False,
        )

        if result["known"]:
            known_predictions += 1

            if result["correct"]:
                correct_predictions += 1
        else:
            unknown_predictions += 1

    print("\n\nPHASE 3: EVALUATION")
    print("=" * 55)

    accuracy = (
        correct_predictions / known_predictions
        if known_predictions > 0
        else 0.0
    )

    print("Total test cases:", len(test_cases))
    print("Known predictions:", known_predictions)
    print("Unknown predictions:", unknown_predictions)
    print("Correct predictions:", correct_predictions)
    print("Accuracy among known predictions:", accuracy)

    print("\nMODEL SUMMARY:")
    print(model.summary())

    print("\nSCIENTIFIC INTERPRETATION:")
    print(
        "The model was trained using north-facing experiences "
        "and tested on different orientations."
    )
    print(
        "This tests whether the context representation transfers "
        "across directions."
    )
    print(
        "The test does not establish independent discovery "
        "or broad general intelligence."
    )


if __name__ == "__main__":
    main()
