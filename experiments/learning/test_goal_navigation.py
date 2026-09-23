from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel


TARGET = (7, 7)
MAX_ACTIONS = 100


def position_tuple(world):
    position = world.organism.position
    return position.x, position.y


def manhattan_distance(position, target):
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def direction_to_target(position, target):
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


def turn_action(current_orientation, desired_orientation):
    directions = ["north", "east", "south", "west"]

    current_index = directions.index(current_orientation)
    desired_index = directions.index(desired_orientation)

    right_turns = (desired_index - current_index) % 4
    left_turns = (current_index - desired_index) % 4

    if right_turns <= left_turns:
        return "turn_right", right_turns

    return "turn_left", left_turns


def choose_action(world, target, perception, extractor, model):
    position = position_tuple(world)
    orientation = world.organism.orientation

    desired_direction = direction_to_target(position, target)

    if desired_direction is None:
        return "wait", "target_reached"

    if orientation != desired_direction:
        action, turns = turn_action(
            orientation,
            desired_direction,
        )

        return action, "aligning_to_target"

    observation = perception.perceive(world)
    context = extractor.extract(observation)

    prediction = model.predict(
        context=context,
        action="move_forward",
        energy_before=world.organism.energy,
    )

    if prediction["known"] and prediction["prediction"] == "failure":
        return "turn_right", "avoiding_predicted_failure"

    if observation["obstacle_ahead"]:
        return "turn_right", "avoiding_obstacle"

    if observation["boundary_ahead"]:
        return "turn_right", "avoiding_boundary"

    return "move_forward", "moving_toward_target"


def run_navigation():
    world = World1()

    perception = EnhancedLocalPerception()
    extractor = ContextExtractorV2()
    model = ContextEffectModel()

    initial_position = position_tuple(world)
    initial_distance = manhattan_distance(initial_position, TARGET)

    print("=== DIA GOAL-DIRECTED NAVIGATION ===")
    print("Initial position:", initial_position)
    print("Target:", TARGET)
    print("Initial distance:", initial_distance)
    print("Initial energy:", world.organism.energy)
    print()

    action_count = 0
    movement_attempts = 0
    successful_movements = 0
    failed_movements = 0
    turns = 0
    history = []

    while action_count < MAX_ACTIONS:
        position_before = position_tuple(world)
        energy_before = world.organism.energy
        orientation_before = world.organism.orientation

        if position_before == TARGET:
            print("TARGET REACHED")
            break

        observation = perception.perceive(world)
        context = extractor.extract(observation)

        action, reason = choose_action(
            world,
            TARGET,
            perception,
            extractor,
            model,
        )

        result = world.step(action)

        position_after = position_tuple(world)
        energy_after = world.organism.energy
        orientation_after = world.organism.orientation

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

        distance_before = manhattan_distance(position_before, TARGET)
        distance_after = manhattan_distance(position_after, TARGET)

        record = {
            "step": action_count + 1,
            "position_before": position_before,
            "position_after": position_after,
            "orientation_before": orientation_before,
            "orientation_after": orientation_after,
            "action": action,
            "reason": reason,
            "context": context,
            "moved": moved,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "energy": energy_after,
        }

        history.append(record)

        print(
            f"Step {action_count + 1}: "
            f"pos={position_before} -> {position_after}, "
            f"orientation={orientation_after}, "
            f"action={action}, "
            f"reason={reason}, "
            f"distance={distance_after}, "
            f"energy={energy_after}"
        )

        action_count += 1

        if result["done"]:
            print("WORLD STOPPED")
            break

    final_position = position_tuple(world)
    final_distance = manhattan_distance(final_position, TARGET)

    print()
    print("=== NAVIGATION RESULTS ===")
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
    print("Final energy:", world.organism.energy)
print("Learned context-action pairs:", model.size())
print("Total learned observations:", model.total_observations())
    print()
    print("=== LEARNING SUMMARY ===")
    print(model.summary())


if __name__ == "__main__":
    run_navigation()
