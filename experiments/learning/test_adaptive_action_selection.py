from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


ACTION = "move_forward"


def create_world():
    return World1()


def configure_agent(world, position, orientation):
    position_type = type(world.organism.position)

    world.organism.position = position_type(
        x=position[0],
        y=position[1],
    )

    world.organism.orientation = orientation
    world.organism.energy = 100


def get_context(perception, extractor, world):
    observation = perception.perceive(world)
    context = extractor.extract(observation)
    return context


def movement_succeeded(previous_position, current_position):
    return previous_position != current_position


def train_model(model, training_cases):
    perception = None
    extractor = ContextExtractorV2()

    for case in training_cases:
        world = create_world()
        configure_agent(
            world,
            case["position"],
            case["orientation"],
        )

        perception = EnhancedLocalPerception()
        context = get_context(perception, extractor, world)

        energy_before = world.organism.energy
        orientation_before = world.organism.orientation
        position_before = world.organism.position

        world.step(ACTION)

        energy_after = world.organism.energy
        orientation_after = world.organism.orientation
        position_after = world.organism.position

        succeeded = movement_succeeded(
            position_before,
            position_after,
        )

        model.learn(
            context=context,
            action=ACTION,
            energy_before=energy_before,
            energy_after=energy_after,
            orientation_before=orientation_before,
            orientation_after=orientation_after,
            movement_succeeded=succeeded,
        )

    print("Training completed.")
    print("Model summary:")
    print(model.summary())


def choose_action(model, context, energy_before):
    prediction = model.predict(
        context=context,
        action=ACTION,
        energy_before=energy_before,
    )

    if prediction["known"]:
        predicted_movement = prediction["prediction"]["movement_success_rate"] > 0.5

        if predicted_movement:
            return "move_forward", prediction

        return "turn_right", prediction

    return "move_forward", prediction


def run_blind_agent(test_cases):
    print("\n")
    print("=" * 55)
    print("BLIND AGENT: ALWAYS MOVE FORWARD")
    print("=" * 55)

    total_steps = 0
    successful_moves = 0
    final_energy = 0

    for index, case in enumerate(test_cases, start=1):
        world = create_world()
        configure_agent(
            world,
            case["position"],
            case["orientation"],
        )

        initial_position = world.organism.position
        initial_energy = world.organism.energy

        world.step("move_forward")

        final_position = world.organism.position
        energy = world.organism.energy

        moved = movement_succeeded(
            initial_position,
            final_position,
        )

        total_steps += 1
        successful_moves += int(moved)
        final_energy += energy

        print(f"\nCase {index}: {case['name']}")
        print(f"Initial position: {initial_position}")
        print(f"Orientation: {case['orientation']}")
        print(f"Final position: {final_position}")
        print(f"Movement succeeded: {moved}")
        print(f"Energy used: {initial_energy - energy}")

    print("\nBLIND AGENT RESULTS")
    print(f"Total action steps: {total_steps}")
    print(f"Successful movements: {successful_moves}")
    print(f"Failed movements: {total_steps - successful_moves}")
    print(f"Total final energy: {final_energy}")


def run_adaptive_agent(model, test_cases):
    print("\n")
    print("=" * 55)
    print("ADAPTIVE AGENT: PREDICT AND SELECT")
    print("=" * 55)

    extractor = ContextExtractorV2()

    total_steps = 0
    successful_moves = 0
    turns = 0
    final_energy = 0

    for index, case in enumerate(test_cases, start=1):
        world = create_world()
        configure_agent(
            world,
            case["position"],
            case["orientation"],
        )

        perception = EnhancedLocalPerception()

        initial_position = world.organism.position
        initial_energy = world.organism.energy

        context = get_context(perception, extractor, world)

        selected_action, prediction = choose_action(
            model,
            context,
            initial_energy,
        )

        print(f"\nCase {index}: {case['name']}")
        print(f"Initial position: {initial_position}")
        print(f"Initial orientation: {case['orientation']}")
        print(f"Context: {context}")
        print(f"Prediction known: {prediction['known']}")
        print(f"Selected action: {selected_action}")

        world.step(selected_action)
        total_steps += 1

        if selected_action == "turn_right":
            turns += 1

            position_after_turn = world.organism.position
            orientation_after_turn = world.organism.orientation

            print(f"Orientation after turn: {orientation_after_turn}")

            position_before_move = world.organism.position
            energy_before_move = world.organism.energy

            world.step("move_forward")
            total_steps += 1

            position_after_move = world.organism.position
            energy_after_move = world.organism.energy

            moved = movement_succeeded(
                position_before_move,
                position_after_move,
            )

            model.learn(
                context=context,
                action=selected_action,
                energy_before=initial_energy,
                energy_after=energy_before_move,
                orientation_before=case["orientation"],
                orientation_after=orientation_after_turn,
                movement_succeeded=True,
            )

            print(f"Position after turn: {position_after_turn}")
            print(f"Position after move: {position_after_move}")
            print(f"Movement after turn succeeded: {moved}")

            if moved:
                successful_moves += 1

        else:
            final_position = world.organism.position
            moved = movement_succeeded(
                initial_position,
                final_position,
            )

            if moved:
                successful_moves += 1

            print(f"Final position: {final_position}")
            print(f"Movement succeeded: {moved}")

        final_energy += world.organism.energy
        print(f"Energy remaining: {world.organism.energy}")

    print("\nADAPTIVE AGENT RESULTS")
    print(f"Total action steps: {total_steps}")
    print(f"Successful movements: {successful_moves}")
    print(f"Turns performed: {turns}")
    print(f"Failed movement attempts: {total_steps - turns - successful_moves}")
    print(f"Total final energy: {final_energy}")


def main():
    model = ContextEffectModel()

    training_cases = [
        {
            "name": "North clear training",
            "position": (1, 1),
            "orientation": "north",
        },
        {
            "name": "North clear training",
            "position": (1, 1),
            "orientation": "north",
        },
        {
            "name": "North obstacle training",
            "position": (4, 3),
            "orientation": "north",
        },
        {
            "name": "North obstacle training",
            "position": (4, 3),
            "orientation": "north",
        },
        {
            "name": "North boundary training",
            "position": (1, 9),
            "orientation": "north",
        },
        {
            "name": "North boundary training",
            "position": (1, 9),
            "orientation": "north",
        },
    ]

    test_cases = [
        {
            "name": "Clear path",
            "position": (1, 1),
            "orientation": "east",
        },
        {
            "name": "Obstacle ahead",
            "position": (3, 4),
            "orientation": "east",
        },
        {
            "name": "Boundary ahead",
            "position": (9, 1),
            "orientation": "east",
        },
        {
            "name": "Obstacle ahead",
            "position": (4, 5),
            "orientation": "south",
        },
        {
            "name": "Boundary ahead",
            "position": (0, 1),
            "orientation": "west",
        },
    ]

    print("=" * 55)
    print("DIA ADAPTIVE ACTION-SELECTION EXPERIMENT")
    print("=" * 55)

    train_model(model, training_cases)

    run_blind_agent(test_cases)

    run_adaptive_agent(model, test_cases)

    print("\n")
    print("=" * 55)
    print("SCIENTIFIC INTERPRETATION")
    print("=" * 55)
    print(
        "This experiment compares fixed behavior with "
        "context-informed action selection."
    )
    print(
        "The adaptive agent uses learned movement predictions "
        "to decide whether to move forward or turn right."
    )
    print(
        "The experiment does not yet establish autonomous "
        "goal formation, independent discovery, or general intelligence."
    )


if __name__ == "__main__":
    main()
