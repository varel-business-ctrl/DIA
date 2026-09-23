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


def turn_action(current, desired):
    directions = ["north", "east", "south", "west"]

    current_index = directions.index(current)
    desired_index = directions.index(desired)

    right_turns = (desired_index - current_index) % 4
    left_turns = (current_index - desired_index) % 4

    if right_turns <= left_turns:
        return "turn_right"

    return "turn_left"


def run_experiment():
    world = World1()

    perception = EnhancedLocalPerception()
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    # When greater than zero, DIA temporarily explores
    # the current direction rather than immediately aligning
    # with the target.
    avoidance_steps = 0

    action_count = 0
    movement_attempts = 0
    successful_movements = 0
    failed_movements = 0
    turns = 0
    obstacle_encounters = 0

    start = get_position(world)
    initial_distance = distance(start, TARGET)

    print("=== DIA OBSTACLE NAVIGATION V2 ===")
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

        desired = desired_direction(position_before, TARGET)

        # Stop if the target has been reached.
        if desired is None:
            print("TARGET REACHED")
            break

        # First priority: respond to an obstacle or boundary.
        if observation["obstacle_ahead"] or observation["boundary_ahead"]:
            action = "turn_right"
            reason = "obstacle_or_boundary_detected"

            # After turning, permit exploration in the new
            # direction instead of immediately turning back.
            avoidance_steps = 2
            obstacle_encounters += 1

        # Second priority: continue temporary avoidance.
        elif avoidance_steps > 0:
            action = "move_forward"
            reason = "temporary_obstacle_avoidance"
            avoidance_steps -= 1

        # Third priority: align with the target.
        elif orientation_before != desired:
            action = turn_action(
                orientation_before,
                desired,
            )
            reason = "aligning_to_target"

        # Otherwise, move toward the target.
        else:
            action = "move_forward"
            reason = "moving_toward_target"

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
    print("=== OBSTACLE NAVIGATION V2 RESULTS ===")
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
    print("Obstacle encounters:", obstacle_encounters)
    print("Final energy:", world.organism.energy)
    print("Learned context-action pairs:", model.size())
    print("Total learned observations:", model.total_observations())

    print()
    print("=== LEARNING SUMMARY ===")
    print(model.summary())


if __name__ == "__main__":
    run_experiment()
