from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


TARGET = (5, 5)
MAX_ACTIONS = 100


def get_position(world):
    position = world.organism.position
    return position.x, position.y


def distance(position, target):
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def desired_direction(position, target):
    x, y = position
    target_x, target_y = target

    if x < target_x:
        return "east"

    if x > target_x:
        return "west"

    if y < target_y:
        return "north"

    if y > target_y:
        return "south"

    return None


def choose_turn(current, desired):
    directions = ["north", "east", "south", "west"]

    current_index = directions.index(current)
    desired_index = directions.index(desired)

    right_distance = (desired_index - current_index) % 4
    left_distance = (current_index - desired_index) % 4

    if right_distance <= left_distance:
        return "turn_right"

    return "turn_left"


def select_action(
    world,
    target,
    observation,
    context,
    model,
    failed_actions,
):
    position = get_position(world)
    orientation = world.organism.orientation

    desired = desired_direction(position, target)

    if desired is None:
        return "wait", "target_reached"

    if orientation != desired:
        return choose_turn(orientation, desired), "aligning"

    prediction = model.predict(
        context=context,
        action="move_forward",
        energy_before=world.organism.energy,
    )

    if prediction["known"] and prediction["prediction"] == "failure":
        return "turn_right", "learned_failure_avoidance"

    if observation["obstacle_ahead"]:
        return "turn_right", "obstacle_detected"

    if observation["boundary_ahead"]:
        return "turn_right", "boundary_detected"

    action_key = (position, orientation, "move_forward")

    if action_key in failed_actions:
        return "turn_right", "avoiding_repeated_failure"

    return "move_forward", "moving_toward_target"


def run_experiment():
    world = World1()

    perception = EnhancedLocalPerception()
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    failed_actions = set()

    action_count = 0
    movement_attempts = 0
    successful_movements = 0
    failed_movements = 0
    turns = 0
    repeated_failures_avoided = 0

    start = get_position(world)
    initial_distance = distance(start, TARGET)

    print("=== DIA OBSTACLE NAVIGATION ===")
    print("Starting position:", start)
    print("Target:", TARGET)
    print("Initial distance:", initial_distance)
    print("Initial energy:", world.organism.energy)
    print()

    while action_count < MAX_ACTIONS:
        position_before = get_position(world)
        orientation_before = world.organism.orientation
        energy_before = world.organism.energy

        if position_before == TARGET:
            print("TARGET REACHED")
            break

        observation = perception.perceive(world)
        context = extractor.extract(observation)

        action, reason = select_action(
            world=world,
            target=TARGET,
            observation=observation,
            context=context,
            model=model,
            failed_actions=failed_actions,
        )

        if action == "move_forward":
            action_key = (
                position_before,
                orientation_before,
                action,
            )

            if action_key in failed_actions:
                repeated_failures_avoided += 1

        result = world.step(action)

        position_after = get_position(world)
        orientation_after = world.organism.orientation
        energy_after = world.organism.energy

        moved = position_after != position_before

        if action in ("move_forward", "move_backward"):
            movement_attempts += 1

            if moved:
                successful_movements += 1
            else:
                failed_movements += 1

                failed_actions.add(
                    (
                        position_before,
                        orientation_before,
                        action,
                    )
                )

            model.learn(
                context=context,
                action=action,
                energy_before=energy_before,
                energy_after=energy_after,
                orientation_before=orientation_before,
                orientation_after=orientation_after,
                movement_succeeded=moved,
            )

        if action in ("turn_left", "turn_right"):
            turns += 1

        print(
            f"Step {action_count + 1}: "
            f"pos={position_before} -> {position_after}, "
            f"orientation={orientation_after}, "
            f"action={action}, "
            f"reason={reason}, "
            f"moved={moved}, "
            f"distance={distance(position_after, TARGET)}, "
            f"energy={energy_after}"
        )

        action_count += 1

        if result["done"]:
            print("WORLD STOPPED")
            break

    final_position = get_position(world)
    final_distance = distance(final_position, TARGET)

    print()
    print("=== OBSTACLE NAVIGATION RESULTS ===")
    print("Final position:", final_position)
    print("Target:", TARGET)
    print("Target reached:", final_position == TARGET)
    print("Actions taken:", action_count)
    print("Initial distance:", initial_distance)
    print("Final distance:", final_distance)
    print("Distance reduced:", initial_distance - final_distance)
    print("Movement attempts:", movement_attempts)
    print("Successful movements:", successful_movements)
    print("Failed movements:", failed_movements)
    print("Turns:", turns)
    print("Repeated failures avoided:", repeated_failures_avoided)
    print("Final energy:", world.organism.energy)
    print("Learned context-action pairs:", model.size())
    print("Total learned observations:", model.total_observations())

    print()
    print("=== LEARNING SUMMARY ===")
    print(model.summary())


if __name__ == "__main__":
    run_experiment()
