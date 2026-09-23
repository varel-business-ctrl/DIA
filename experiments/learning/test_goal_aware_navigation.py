from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = (5, 5)
MAX_STEPS = 150

ACTIONS = [
    "move_forward",
    "turn_right",
    "turn_left",
    "wait",
]

MOVEMENT_ACTIONS = {
    "move_forward",
    "move_backward",
}

TURN_ACTIONS = {
    "turn_right",
    "turn_left",
}


# ============================================================
# BASIC GEOMETRY
# ============================================================

DIRECTION_VECTORS = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
}


RIGHT_DIRECTION = {
    "north": "east",
    "east": "south",
    "south": "west",
    "west": "north",
}


LEFT_DIRECTION = {
    "north": "west",
    "west": "south",
    "south": "east",
    "east": "north",
}


def get_position(world) -> Tuple[int, int]:
    position = world.organism.position
    return position.x, position.y


def get_orientation(world) -> str:
    orientation = world.organism.orientation

    if hasattr(orientation, "value"):
        return str(orientation.value).lower()

    return str(orientation).lower()


def manhattan_distance(
    position: Tuple[int, int],
    target: Tuple[int, int],
) -> int:
    return abs(position[0] - target[0]) + abs(position[1] - target[1])


def state_key(world) -> Tuple[int, int, str]:
    position = get_position(world)
    orientation = get_orientation(world)
    return position[0], position[1], orientation


def position_type_from(world, position: Tuple[int, int]):
    current_position = world.organism.position
    position_type = type(current_position)

    return position_type(
        x=position[0],
        y=position[1],
    )


# ============================================================
# ACTUAL ACTION OUTCOME
# ============================================================

@dataclass
class ActionOutcome:
    state_before: Tuple[int, int, str]
    action: str
    state_after: Tuple[int, int, str]
    distance_before: int
    distance_after: int
    energy_before: float
    energy_after: float
    position_changed: bool
    improved: bool
    failed_movement: bool


# ============================================================
# MEMORY OF CONSEQUENCES
# ============================================================

class ConsequenceMemory:
    def __init__(self):
        self.records: Dict[
            Tuple[Tuple[int, int, str], str],
            List[ActionOutcome]
        ] = defaultdict(list)

    def record(self, outcome: ActionOutcome):
        key = (outcome.state_before, outcome.action)
        self.records[key].append(outcome)

    def get_records(
        self,
        state: Tuple[int, int, str],
        action: str,
    ) -> List[ActionOutcome]:
        return self.records.get((state, action), [])

    def has_seen(
        self,
        state: Tuple[int, int, str],
        action: str,
    ) -> bool:
        return len(self.get_records(state, action)) > 0

    def failure_count(
        self,
        state: Tuple[int, int, str],
        action: str,
    ) -> int:
        return sum(
            1
            for record in self.get_records(state, action)
            if record.failed_movement
        )

    def average_distance_change(
        self,
        state: Tuple[int, int, str],
        action: str,
    ) -> Optional[float]:
        records = self.get_records(state, action)

        if not records:
            return None

        changes = [
            record.distance_after - record.distance_before
            for record in records
        ]

        return sum(changes) / len(changes)

    def total_records(self) -> int:
        return sum(len(records) for records in self.records.values())

    def failed_actions(self) -> int:
        total = 0

        for records in self.records.values():
            for record in records:
                if record.failed_movement:
                    total += 1

        return total


# ============================================================
# GOAL-AWARE ACTION SELECTOR
# ============================================================

class GoalAwareActionSelector:
    """
    A first goal-aware selector.

    The selector uses:

    1. Current position and target distance.
    2. Current orientation.
    3. A simple directional heuristic.
    4. Actual recorded consequences.
    5. Failure penalties.
    6. Loop penalties.
    7. Limited exploration.

    This is still an engineered baseline.
    It is not autonomous general intelligence.
    """

    def __init__(self, target: Tuple[int, int]):
        self.target = target

    def desired_directions(
        self,
        position: Tuple[int, int],
    ) -> List[str]:
        x, y = position
        target_x, target_y = self.target

        directions = []

        if target_x > x:
            directions.append("east")
        elif target_x < x:
            directions.append("west")

        if target_y > y:
            directions.append("north")
        elif target_y < y:
            directions.append("south")

        return directions

    def direction_after_turn(
        self,
        orientation: str,
        action: str,
    ) -> str:
        if action == "turn_right":
            return RIGHT_DIRECTION[orientation]

        if action == "turn_left":
            return LEFT_DIRECTION[orientation]

        return orientation

    def movement_distance_estimate(
        self,
        position: Tuple[int, int],
        orientation: str,
        action: str,
    ) -> Tuple[int, int]:
        """
        Estimate the next position without changing the world.

        This is only a directional estimate.
        Obstacles are not assumed to be known in advance.
        """

        x, y = position

        if action == "move_forward":
            dx, dy = DIRECTION_VECTORS[orientation]
            return x + dx, y + dy

        if action == "move_backward":
            dx, dy = DIRECTION_VECTORS[orientation]
            return x - dx, y - dy

        return x, y

    def heuristic_score(
        self,
        world,
        action: str,
        memory: ConsequenceMemory,
        recent_states: List[Tuple[int, int, str]],
    ) -> float:
        position = get_position(world)
        orientation = get_orientation(world)

        current_distance = manhattan_distance(
            position,
            self.target,
        )

        score = 0.0

        # ----------------------------------------------------
        # Goal completion
        # ----------------------------------------------------

        if position == self.target:
            return 100000.0

        # ----------------------------------------------------
        # Turning logic
        # ----------------------------------------------------

        if action in TURN_ACTIONS:
            new_orientation = self.direction_after_turn(
                orientation,
                action,
            )

            desired = self.desired_directions(position)

            if new_orientation in desired:
                score += 30.0
            else:
                score += 2.0

            # Turning is useful, but has a small energy cost.
            score -= 1.0

        # ----------------------------------------------------
        # Movement logic
        # ----------------------------------------------------

        elif action == "move_forward":
            estimated_position = self.movement_distance_estimate(
                position,
                orientation,
                action,
            )

            estimated_distance = manhattan_distance(
                estimated_position,
                self.target,
            )

            distance_change = (
                current_distance - estimated_distance
            )

            # Reward estimated movement toward the target.
            score += distance_change * 20.0

            # Penalize movement in an irrelevant direction.
            if distance_change < 0:
                score -= 20.0

            # Small movement preference.
            score += 5.0

        elif action == "wait":
            score -= 15.0

        # ----------------------------------------------------
        # Actual learned consequences
        # ----------------------------------------------------

        average_change = memory.average_distance_change(
            state_key(world),
            action,
        )

        if average_change is not None:
            # Negative distance change means improvement.
            score -= average_change * 30.0

        failure_count = memory.failure_count(
            state_key(world),
            action,
        )

        score -= failure_count * 35.0

        # ----------------------------------------------------
        # Repeated-state penalty
        # ----------------------------------------------------

        predicted_orientation = self.direction_after_turn(
            orientation,
            action,
        )

        predicted_position = position

        if action == "move_forward":
            predicted_position = self.movement_distance_estimate(
                position,
                orientation,
                action,
            )

        predicted_state = (
            predicted_position[0],
            predicted_position[1],
            predicted_orientation,
        )

        if predicted_state in recent_states:
            score -= 40.0

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if not memory.has_seen(state_key(world), action):
            score += 8.0

        return score

    def choose_action(
        self,
        world,
        memory: ConsequenceMemory,
        recent_states: List[Tuple[int, int, str]],
    ) -> Tuple[str, str, float]:
        position = get_position(world)

        if position == self.target:
            return "wait", "target_reached", 100000.0

        scored_actions = []

        for action in ACTIONS:
            score = self.heuristic_score(
                world,
                action,
                memory,
                recent_states,
            )

            scored_actions.append((score, action))

        scored_actions.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_action = scored_actions[0]

        reason = "goal_aware_score"

        if not memory.has_seen(state_key(world), best_action):
            reason = "goal_aware_exploration"

        if memory.failure_count(
            state_key(world),
            best_action,
        ) > 0:
            reason = "avoiding_or_comparing_known_failures"

        return best_action, reason, best_score


# ============================================================
# WORLD EXECUTION
# ============================================================

def execute_action(world, action: str):
    """
    Execute exactly one action using World-1.

    The return value is preserved in case World-1 provides
    additional information.
    """

    return world.step(action)


def read_energy(world) -> float:
    organism = world.organism

    if hasattr(organism, "energy"):
        return float(organism.energy)

    return 0.0


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    world = World1()

    perception = EnhancedLocalPerception()
    context_extractor = ContextExtractorV2()

    memory = ConsequenceMemory()
    selector = GoalAwareActionSelector(TARGET)

    recent_states = []
    successful_movements = 0
    failed_movements = 0
    turns = 0
    waits = 0

    print("=" * 70)
    print("DIA GOAL-AWARE NAVIGATION EXPERIMENT")
    print("=" * 70)

    print(f"Target: {TARGET}")
    print(f"Maximum steps: {MAX_STEPS}")
    print(f"Starting position: {get_position(world)}")
    print(f"Starting orientation: {get_orientation(world)}")
    print(f"Starting energy: {read_energy(world)}")
    print()

    for step_number in range(1, MAX_STEPS + 1):
        position_before = get_position(world)
        orientation_before = get_orientation(world)
        state_before = state_key(world)

        distance_before = manhattan_distance(
            position_before,
            TARGET,
        )

        energy_before = read_energy(world)

        if position_before == TARGET:
            print()
            print("TARGET REACHED BEFORE ACTION")
            break

        # --------------------------------------------------------
        # ENERGY SAFETY CHECK
        # --------------------------------------------------------

        if energy_before <= 0:
            print()
            print("ENERGY DEPLETED BEFORE ACTION")
            print(f"Energy: {energy_before}")
            break

        observation = perception.perceive(world)
        context = context_extractor.extract(observation)

        action, reason, score = selector.choose_action(
            world,
            memory,
            recent_states,
        )

        print(
            f"Step {step_number:03d} | "
            f"Position {position_before} | "
            f"Facing {orientation_before:>5} | "
            f"Distance {distance_before:02d} | "
            f"Context {context:>24} | "
            f"Action {action:>13} | "
            f"Score {score:7.2f} | "
            f"Reason {reason}"
        )

        execute_action(world, action)

        position_after = get_position(world)
        orientation_after = get_orientation(world)

        state_after = state_key(world)

        distance_after = manhattan_distance(
            position_after,
            TARGET,
        )

        energy_after = read_energy(world)

        position_changed = (
            position_before != position_after
        )

        improved = (
            distance_after < distance_before
        )

        failed_movement = (
            action in MOVEMENT_ACTIONS
            and not position_changed
        )

        outcome = ActionOutcome(
            state_before=state_before,
            action=action,
            state_after=state_after,
            distance_before=distance_before,
            distance_after=distance_after,
            energy_before=energy_before,
            energy_after=energy_after,
            position_changed=position_changed,
            improved=improved,
            failed_movement=failed_movement,
        )

        memory.record(outcome)

        recent_states.append(state_after)

        if len(recent_states) > 12:
            recent_states.pop(0)

        if action in MOVEMENT_ACTIONS:
            if position_changed:
                successful_movements += 1
            else:
                failed_movements += 1

        elif action in TURN_ACTIONS:
            turns += 1

        elif action == "wait":
            waits += 1

        if position_after == TARGET:
            print()
            print("TARGET REACHED")
            break

    final_position = get_position(world)
    final_orientation = get_orientation(world)
    final_distance = manhattan_distance(
        final_position,
        TARGET,
    )

    print()
    print("=" * 70)
    print("FINAL EXPERIMENT REPORT")
    print("=" * 70)

    print(f"Target: {TARGET}")
    print(f"Final position: {final_position}")
    print(f"Final orientation: {final_orientation}")
    print(f"Final distance: {final_distance}")
    print(f"Reached target: {final_position == TARGET}")
    print(f"Successful movements: {successful_movements}")
    print(f"Failed movements: {failed_movements}")
    print(f"Turns: {turns}")
    print(f"Waits: {waits}")
    print(f"Total recorded consequences: {memory.total_records()}")
    print(f"Recorded failed actions: {memory.failed_actions()}")
    print(f"Final energy: {read_energy(world)}")

    if final_position == TARGET:
        print()
        print("RESULT: GOAL ACHIEVED")
    else:
        print()
        print("RESULT: GOAL NOT ACHIEVED")
        print("The failure is useful data for the next upgrade.")
    print(f"Energy exhausted: {read_energy(world) <= 0}")


if __name__ == "__main__":
    main()
