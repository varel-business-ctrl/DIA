from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2


class ActionOutcomeMemory:
    def __init__(self):
        self.outcomes = {}

    def record(
        self,
        state,
        action,
        next_state,
        position_changed,
        orientation_changed,
        energy_before,
        energy_after,
        distance_before,
        distance_after,
    ):
        key = (state, action)

        outcome = {
            "next_state": next_state,
            "position_changed": position_changed,
            "orientation_changed": orientation_changed,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "energy_delta": energy_after - energy_before,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "improved": distance_after < distance_before,
        }

        self.outcomes.setdefault(key, []).append(outcome)

        return outcome

    def get(self, state, action):
        return self.outcomes.get((state, action), [])

    def tried_actions(self, state):
        return {
            action
            for (stored_state, action) in self.outcomes
            if stored_state == state
        }


class ActionSelectorV1:
    def __init__(self, actions, memory):
        self.actions = actions
        self.memory = memory

    def select_action(self, state):
        tried = self.memory.tried_actions(state)

        # First explore actions that have not been tried.
        for action in self.actions:
            if action not in tried:
                return {
                    "action": action,
                    "reason": "explore_untried_action",
                }

        # Evaluate known actions.
        candidates = []

        for action in self.actions:
            outcomes = self.memory.get(state, action)

            if not outcomes:
                continue

            improvements = sum(
                outcome["improved"]
                for outcome in outcomes
            )

            failures = sum(
                not outcome["position_changed"]
                for outcome in outcomes
            )

            candidates.append({
                "action": action,
                "improvements": improvements,
                "failures": failures,
                "attempts": len(outcomes),
            })

        if not candidates:
            return {
                "action": self.actions[0],
                "reason": "no_previous_experience",
            }

        # Prefer improvement, then fewer failures.
        candidates.sort(
            key=lambda item: (
                item["improvements"],
                -item["failures"],
            ),
            reverse=True,
        )

        return {
            "action": candidates[0]["action"],
            "reason": "choose_recorded_outcome",
            "evaluation": candidates[0],
        }


def get_position(world):
    return (
        world.organism.position.x,
        world.organism.position.y,
    )


def get_orientation(world):
    return world.organism.orientation


def distance_to_target(position, target):
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def get_state(world):
    return (
        get_position(world),
        get_orientation(world),
    )


def main():
    print("=== DIA REAL ACTION SELECTION TEST ===")

    world = World1()
    perception = EnhancedLocalPerception()
    extractor = ContextExtractorV2()

    memory = ActionOutcomeMemory()

    actions = [
        "move_forward",
        "turn_right",
        "turn_left",
        "wait",
    ]

    selector = ActionSelectorV1(actions, memory)

    target = (5, 5)
    max_actions = 40

    successful_position_changes = 0
    failed_position_changes = 0
    orientation_changes = 0

    for step in range(max_actions):
        position_before = get_position(world)

        if position_before == target:
            print("\nTARGET REACHED")
            break

        state_before = get_state(world)

        selection = selector.select_action(state_before)
        action = selection["action"]

        energy_before = world.organism.energy
        orientation_before = get_orientation(world)

        observation = perception.perceive(world)
        context = extractor.extract(observation)

        distance_before = distance_to_target(
            position_before,
            target,
        )

        print(
            f"\nStep {step + 1}"
        )
        print(f"State: {state_before}")
        print(f"Context: {context}")
        print(f"Selected action: {selection}")

        world.step(action)

        position_after = get_position(world)
        orientation_after = get_orientation(world)
        state_after = get_state(world)

        energy_after = world.organism.energy

        distance_after = distance_to_target(
            position_after,
            target,
        )

        position_changed = position_before != position_after
        orientation_changed = (
            orientation_before != orientation_after
        )

        if position_changed:
            successful_position_changes += 1
        else:
            failed_position_changes += 1

        if orientation_changed:
            orientation_changes += 1

        outcome = memory.record(
            state=state_before,
            action=action,
            next_state=state_after,
            position_changed=position_changed,
            orientation_changed=orientation_changed,
            energy_before=energy_before,
            energy_after=energy_after,
            distance_before=distance_before,
            distance_after=distance_after,
        )

        print(f"Outcome: {outcome}")

    final_position = get_position(world)
    final_distance = distance_to_target(
        final_position,
        target,
    )

    print("\n=== FINAL RESULTS ===")
    print(f"Target: {target}")
    print(f"Final position: {final_position}")
    print(f"Final distance: {final_distance}")
    print(f"Final energy: {world.organism.energy}")
    print(
        f"Successful position changes: "
        f"{successful_position_changes}"
    )
    print(
        f"Failed position changes: "
        f"{failed_position_changes}"
    )
    print(f"Orientation changes: {orientation_changes}")

    print("\n=== MEMORY SUMMARY ===")
    print(
        f"State-action pairs: "
        f"{len(memory.outcomes)}"
    )
    print(
        f"Total outcomes: "
        f"{sum(len(v) for v in memory.outcomes.values())}"
    )


if __name__ == "__main__":
    main()
