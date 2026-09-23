from environment.world1.world import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


def configure_position(world, x, y, orientation):
    organism = world.organism
    organism.position.x = x
    organism.position.y = y
    organism.orientation = orientation


def collect_context(world, perception, extractor):
    observation = perception.perceive(world)
    return extractor.extract(observation)


def run_experience(
    label,
    world,
    model,
    perception,
    extractor,
    x,
    y,
    orientation,
):
    configure_position(world, x, y, orientation)

    organism = world.organism
    organism.energy = 100

    observation_before = perception.perceive(world)
    context = extractor.extract(observation_before)

    energy_before = organism.energy
    orientation_before = organism.orientation
    position_before = (
        organism.position.x,
        organism.position.y,
    )

    model_prediction = model.predict(
        context=context,
        action="move_forward",
        energy_before=energy_before,
    )

    predicted = None

    if model_prediction["known"]:
        predicted = (
            model_prediction["prediction"]["movement_success_rate"]
            > 0.5
        )

    result = world.step("move_forward")

    energy_after = organism.energy
    orientation_after = organism.orientation
    position_after = (
        organism.position.x,
        organism.position.y,
    )

    movement_succeeded = position_before != position_after

    model.learn(
        context=context,
        action="move_forward",
        energy_before=energy_before,
        energy_after=energy_after,
        orientation_before=orientation_before,
        orientation_after=orientation_after,
        movement_succeeded=movement_succeeded,
    )

    prediction_correct = (
        predicted is not None
        and predicted == movement_succeeded
    )

    print(f"\n--- {label} ---")
    print("Context:", context)
    print("Prediction before action:", predicted)
    print("Model knew context:", model_prediction["known"])
    print("Actual movement:", movement_succeeded)
    print("Position before:", position_before)
    print("Position after:", position_after)
    print("Prediction correct:", prediction_correct)

    return {
        "known": model_prediction["known"],
        "predicted": predicted,
        "actual": movement_succeeded,
        "correct": prediction_correct,
    }


def main():
    world = World1()

    perception = EnhancedLocalPerception(radius=2)
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    print("DIA PRE-ACTION PREDICTION TEST")
    print("=" * 50)

    # ---------------------------------------------------------
    # PHASE 1: Training
    # ---------------------------------------------------------
    print("\nPHASE 1: TRAINING")
    print("=" * 50)

    training_cases = [
        ("Clear training", 1, 1, "north"),
        ("Clear training", 1, 1, "north"),
        ("Clear training", 1, 1, "north"),
        ("Obstacle training", 4, 3, "north"),
        ("Obstacle training", 4, 3, "north"),
        ("Obstacle training", 4, 3, "north"),
        ("Boundary training", 1, 9, "north"),
        ("Boundary training", 1, 9, "north"),
        ("Boundary training", 1, 9, "north"),
    ]

    for case in training_cases:
        run_experience(
            label=case[0],
            world=world,
            model=model,
            perception=perception,
            extractor=extractor,
            x=case[1],
            y=case[2],
            orientation=case[3],
        )

    # ---------------------------------------------------------
    # PHASE 2: Prediction before action
    # ---------------------------------------------------------
    print("\n\nPHASE 2: PRE-ACTION TESTING")
    print("=" * 50)

    test_cases = [
        ("Clear test", 2, 1, "north"),
        ("Clear test", 2, 1, "north"),
        ("Obstacle test", 4, 3, "north"),
        ("Obstacle test", 4, 3, "north"),
        ("Boundary test", 1, 9, "north"),
        ("Boundary test", 1, 9, "north"),
    ]

    known_count = 0
    correct_count = 0
    total_count = 0

    for case in test_cases:
        result = run_experience(
            label=case[0],
            world=world,
            model=model,
            perception=perception,
            extractor=extractor,
            x=case[1],
            y=case[2],
            orientation=case[3],
        )

        total_count += 1

        if result["known"]:
            known_count += 1

        if result["correct"]:
            correct_count += 1

    # ---------------------------------------------------------
    # PHASE 3: Evaluation
    # ---------------------------------------------------------
    print("\n\nPHASE 3: EVALUATION")
    print("=" * 50)

    accuracy = (
        correct_count / known_count
        if known_count > 0
        else 0.0
    )

    print("Total test cases:", total_count)
    print("Known predictions:", known_count)
    print("Correct predictions:", correct_count)
    print("Prediction accuracy:", accuracy)

    print("\nMODEL SUMMARY:")
    print(model.summary())

    print("\nSCIENTIFIC INTERPRETATION:")
    if known_count == 0:
        print("No pre-action predictions were available.")
    else:
        print(
            "Accuracy measures whether the model's prediction "
            "matched the observed movement outcome."
        )

    print(
        "This test measures consequence prediction, "
        "not independent discovery of context concepts."
    )


if __name__ == "__main__":
    main()
