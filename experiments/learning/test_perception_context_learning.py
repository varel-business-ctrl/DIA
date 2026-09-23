from environment.world1 import World1
from environment.world1.entities import Position
from environment.world1.perception import LocalPerception

from organism.learning.context_effect_model import ContextEffectModel
from organism.learning.context_extractor import ContextExtractor


def movement_changed(before, after):
    before_position = (
        before["organism"]["x"],
        before["organism"]["y"],
    )

    after_position = (
        after["organism"]["x"],
        after["organism"]["y"],
    )

    return before_position != after_position


def learn_from_experience(model, extractor, perception, world, action):
    before_world = world.observe()
    before = perception.perceive(world)

    context = extractor.extract(before)

    world.step(action)

    after_world = world.observe()
    after = perception.perceive(world)

    movement_succeeded = movement_changed(
        before_world,
        after_world,
    )

    result = model.learn(
        context=context,
        action=action,
        energy_before=before["energy"],
        energy_after=after["energy"],
        orientation_before=before["orientation"],
        orientation_after=after["orientation"],
        movement_succeeded=movement_succeeded,
    )

    print(
        "LEARNED:",
        "context=", context,
        "| action=", action,
        "| movement=", movement_succeeded,
        "| energy=", before["energy"], "->", after["energy"],
    )

    return result


def main():
    world = World1()
    perception = LocalPerception(radius=2)
    extractor = ContextExtractor()
    model = ContextEffectModel()

    print("\n=== CLEAR PATH EXPERIENCES ===")

    for i in range(3):
        world.organism.position = Position(1, 1)
        world.organism.orientation = "east"

        learn_from_experience(
            model,
            extractor,
            perception,
            world,
            "move_forward",
        )


    print("\n=== OBSTACLE EXPERIENCES ===")

    for i in range(3):
        world.organism.position = Position(3, 4)
        world.organism.orientation = "east"

        learn_from_experience(
            model,
            extractor,
            perception,
            world,
            "move_forward",
        )


    print("\n=== PREDICTION: CLEAR PATH ===")

    print(
        model.predict(
            context="clear_or_unknown_path",
            action="move_forward",
            energy_before=80,
        )
    )


    print("\n=== PREDICTION: OBSTACLE ===")

    print(
        model.predict(
            context="obstacle_ahead",
            action="move_forward",
            energy_before=80,
        )
    )


    print("\n=== PREDICTION: UNKNOWN CONTEXT ===")

    print(
        model.predict(
            context="water_environment",
            action="move_forward",
            energy_before=80,
        )
    )


    print("\n=== MODEL SUMMARY ===")

    print(model.summary())


if __name__ == "__main__":
    main()
