from environment.world1.world import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


def run_trial(
    label,
    world,
    model,
    perception,
    extractor,
):
    organism = world.organism

    observation_before = perception.perceive(world)
    context = extractor.extract(observation_before)

    energy_before = organism.energy
    orientation_before = organism.orientation
    position_before = (
        organism.position.x,
        organism.position.y,
    )

    result = world.step("move_forward")

    observation_after = perception.perceive(world)

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

    prediction = model.predict(
        context=context,
        action="move_forward",
        energy_before=energy_before,
    )

    print(f"\n--- {label} ---")
    print("Context:", context)
    print("Position before:", position_before)
    print("Position after:", position_after)
    print("Energy before:", energy_before)
    print("Energy after:", energy_after)
    print("Movement succeeded:", movement_succeeded)
    print("World result:", result)
    print("Prediction:", prediction)


def main():
    world = World1()

    perception = EnhancedLocalPerception(radius=2)
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    print("DIA ENHANCED CONTEXT LEARNING")
    print("=" * 45)

    organism = world.organism

    # Trial 1: Clear path.
    organism.position.x = 1
    organism.position.y = 1
    organism.orientation = "north"

    run_trial(
        "Clear path",
        world,
        model,
        perception,
        extractor,
    )

    # Trial 2: Obstacle ahead.
    organism.position.x = 4
    organism.position.y = 3
    organism.orientation = "north"

    run_trial(
        "Obstacle ahead",
        world,
        model,
        perception,
        extractor,
    )

    # Trial 3: Boundary ahead.
    organism.position.x = 1
    organism.position.y = 9
    organism.orientation = "north"

    run_trial(
        "Boundary ahead",
        world,
        model,
        perception,
        extractor,
    )

    print("\n" + "=" * 45)
    print("MODEL SUMMARY")
    print("=" * 45)
    print(model.summary())

    print("\nKNOWN CONTEXTS:")
    print(model.known_contexts())

    print("\nKNOWN ACTIONS:")
    print(model.known_actions())

    print("\nTOTAL OBSERVATIONS:")
    print(model.total_observations())

    print("\nMODEL SIZE:")
    print(model.size())


if __name__ == "__main__":
    main()
