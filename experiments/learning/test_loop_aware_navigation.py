from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2
from organism.learning.context_effect_model import ContextEffectModel

class LoopMemory:
    def __init__(self):
        self.visited_states = {}
        self.repeated_states = []

    def observe(self, position, orientation):
        state = (position, orientation)

        count = self.visited_states.get(state, 0)
        self.visited_states[state] = count + 1

        repeated = count > 0

        if repeated:
            self.repeated_states.append(state)

        return {
            "state": state,
            "visit_count": self.visited_states[state],
            "repeated": repeated,
        }

    def repetitions(self):
        return len(self.repeated_states)


def get_position(world):
    return (
        world.organism.position.x,
        world.organism.position.y,
    )


def get_orientation(world):
    return world.organism.orientation


def position_type(world):
    return type(world.organism.position)


def set_position(world, position):
    world.organism.position = position_type(world)(
        x=position[0],
        y=position[1],
    )


def distance_to_target(position, target):
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def choose_direction(position, target):
    """
    Basic target-directed direction selection.
    This is a hand-designed baseline.
    """

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


def direction_to_turns(current, desired):
    directions = ["north", "east", "south", "west"]

    if current == desired:
        return []

    current_index = directions.index(current)
    desired_index = directions.index(desired)

    right_turns = (desired_index - current_index) % 4
    left_turns = (current_index - desired_index) % 4

    if right_turns <= left_turns:
        return ["turn_right"] * right_turns

    return ["turn_left"] * left_turns


def main():
    print("=== DIA LOOP-AWARE NAVIGATION ===")

    world = World1()
    perception = EnhancedLocalPerception()
    extractor = ContextExtractorV2()
    model = ContextEffectModel()
    memory = LoopMemory()

    target = (5, 5)

    actions_taken = 0
    successful_movements = 0
    failed_movements = 0
    turns_taken = 0
    loop_responses = 0

    max_actions = 100

    while actions_taken < max_actions:
        position = get_position(world)
        orientation = get_orientation(world)

        if position == target:
            print("\nTARGET REACHED")
            break

        memory_result = memory.observe(position, orientation)

        print(
            f"Step {actions_taken + 1}: "
            f"Position={position}, "
            f"Orientation={orientation}, "
            f"Memory={memory_result}"
        )

        if memory_result["repeated"]:
            loop_responses += 1

            print("  LOOP DETECTED")
            print("  Response: turning right and moving forward")

            world.step("turn_right")
            actions_taken += 1
            turns_taken += 1

            if actions_taken >= max_actions:
                break

            before = get_position(world)
            energy_before = world.organism.energy
            orientation_before = get_orientation(world)

            observation = perception.perceive(world)
            context = extractor.extract(observation)

            world.step("move_forward")

            after = get_position(world)
            energy_after = world.organism.energy
            orientation_after = get_orientation(world)

            movement_succeeded = before != after

            model.learn(
                context=context,
                action="move_forward",
                energy_before=energy_before,
                energy_after=energy_after,
                orientation_before=orientation_before,
                orientation_after=orientation_after,
                movement_succeeded=movement_succeeded,
            )

            if movement_succeeded:
                successful_movements += 1
            else:
                failed_movements += 1

            actions_taken += 1
            continue

        desired_direction = choose_direction(position, target)

        if desired_direction is None:
            break

        turns = direction_to_turns(
            get_orientation(world),
            desired_direction,
        )

        if turns:
            action = turns[0]
            world.step(action)
            actions_taken += 1
            turns_taken += 1
            continue

        before = get_position(world)
        energy_before = world.organism.energy
        orientation_before = get_orientation(world)

        observation = perception.perceive(world)
        context = extractor.extract(observation)

        prediction = model.predict(
            context=context,
            action="move_forward",
            energy_before=energy_before,
        )

        world.step("move_forward")

        after = get_position(world)
        energy_after = world.organism.energy
        orientation_after = get_orientation(world)

        movement_succeeded = before != after

        model.learn(
            context=context,
            action="move_forward",
            energy_before=energy_before,
            energy_after=energy_after,
            orientation_before=orientation_before,
            orientation_after=orientation_after,
            movement_succeeded=movement_succeeded,
        )

        if movement_succeeded:
            successful_movements += 1
        else:
            failed_movements += 1

        print(
            f"  Context: {context}"
        )
        print(
            f"  Prediction: {prediction}"
        )
        print(
            f"  Movement successful: {movement_succeeded}"
        )

        actions_taken += 1

    final_position = get_position(world)

    print("\n=== FINAL RESULTS ===")
    print(f"Target: {target}")
    print(f"Final position: {final_position}")
    print(
        f"Final distance: "
        f"{distance_to_target(final_position, target)}"
    )
    print(f"Actions taken: {actions_taken}")
    print(f"Successful movements: {successful_movements}")
    print(f"Failed movements: {failed_movements}")
    print(f"Turns taken: {turns_taken}")
    print(f"Loop responses: {loop_responses}")
    print(f"Loop repetitions: {memory.repetitions()}")
    print(f"Final energy: {world.organism.energy}")

    print("\n=== MEMORY SUMMARY ===")
    print(memory.visited_states)


if __name__ == "__main__":
    main()
