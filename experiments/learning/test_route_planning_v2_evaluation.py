
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional


from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception


# ============================================================
# DIA ROUTE PLANNING V2 — GENERALIZATION EVALUATION
# ============================================================

MAX_STEPS = 150

MIN_X = 0
MAX_X = 9
MIN_Y = 0
MAX_Y = 9

ACTIONS = (
    "move_forward",
    "turn_left",
    "turn_right",
)


# ============================================================
# BASIC GEOMETRY
# ============================================================

DIRECTIONS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}

DIRECTION_ORDER = (
    "north",
    "east",
    "south",
    "west",
)


def position_tuple(world: World1) -> tuple[int, int]:
    position = world.organism.position
    return position.x, position.y


def orientation(world: World1) -> str:
    return world.organism.orientation


def energy(world: World1) -> float:
    return float(world.organism.energy)


def manhattan_distance(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def in_bounds(position: tuple[int, int]) -> bool:
    x, y = position

    return (
        MIN_X <= x <= MAX_X
        and MIN_Y <= y <= MAX_Y
    )


def adjacent_position(
    position: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    dx, dy = DIRECTIONS[direction]

    return (
        position[0] + dx,
        position[1] + dy,
    )


def direction_between(
    current: tuple[int, int],
    next_position: tuple[int, int],
) -> Optional[str]:
    dx = next_position[0] - current[0]
    dy = next_position[1] - current[1]

    for direction, vector in DIRECTIONS.items():
        if vector == (dx, dy):
            return direction

    return None


def direction_index(direction: str) -> int:
    return DIRECTION_ORDER.index(direction)


def turn_action_toward(
    current_direction: str,
    desired_direction: str,
) -> str:
    current_index = direction_index(current_direction)
    desired_index = direction_index(desired_direction)

    clockwise_distance = (
        desired_index - current_index
    ) % 4

    counterclockwise_distance = (
        current_index - desired_index
    ) % 4

    if clockwise_distance == 0:
        raise ValueError(
            "turn_action_toward called when already aligned"
        )

    if clockwise_distance <= counterclockwise_distance:
        return "turn_right"

    return "turn_left"


# ============================================================
# CONSEQUENCE RECORD
# ============================================================

@dataclass
class Consequence:
    step: int
    before_position: tuple[int, int]
    after_position: tuple[int, int]
    before_orientation: str
    after_orientation: str
    action: str
    distance_before: int
    distance_after: int
    energy_before: float
    energy_after: float
    movement_succeeded: bool
    improved_distance: bool


# ============================================================
# SPATIAL WORLD MODEL
# ============================================================

class SpatialWorldModel:
    """
    A limited internal map.

    The model does not receive the complete obstacle map.

    It discovers blocked cells when an attempted movement fails.
    """

    def __init__(self) -> None:
        self.blocked_cells: set[tuple[int, int]] = set()
        self.visited_cells: set[tuple[int, int]] = set()
        self.observation_count = 0

    def observe_position(
        self,
        position: tuple[int, int],
    ) -> None:
        self.visited_cells.add(position)
        self.observation_count += 1

    def mark_blocked(
        self,
        position: tuple[int, int],
    ) -> None:
        if in_bounds(position):
            self.blocked_cells.add(position)

    def is_blocked(
        self,
        position: tuple[int, int],
    ) -> bool:
        return position in self.blocked_cells

    def summary(self) -> dict:
        return {
            "visited_cells": len(self.visited_cells),
            "blocked_cells": len(self.blocked_cells),
            "observations": self.observation_count,
            "known_blocked": sorted(self.blocked_cells),
        }


# ============================================================
# CONSEQUENCE MEMORY
# ============================================================

class ConsequenceMemory:
    def __init__(self) -> None:
        self.records: list[Consequence] = []

    def record(self, consequence: Consequence) -> None:
        self.records.append(consequence)

    def failed_actions(self) -> int:
        return sum(
            1
            for record in self.records
            if not record.movement_succeeded
            and record.action == "move_forward"
        )

    def successful_movements(self) -> int:
        return sum(
            1
            for record in self.records
            if record.movement_succeeded
            and record.action == "move_forward"
        )

    def turns(self) -> int:
        return sum(
            1
            for record in self.records
            if record.action in {
                "turn_left",
                "turn_right",
            }
        )

    def summary(self) -> dict:
        return {
            "records": len(self.records),
            "successful_movements": self.successful_movements(),
            "failed_movements": self.failed_actions(),
            "turns": self.turns(),
        }


# ============================================================
# BFS ROUTE PLANNER
# ============================================================

def neighbors(
    position: tuple[int, int],
    model: SpatialWorldModel,
):
    for direction in DIRECTION_ORDER:
        next_position = adjacent_position(
            position,
            direction,
        )

        if not in_bounds(next_position):
            continue

        if model.is_blocked(next_position):
            continue

        yield next_position


def bfs_route(
    start: tuple[int, int],
    target: tuple[int, int],
    model: SpatialWorldModel,
) -> Optional[list[tuple[int, int]]]:
    """
    Returns a route including the start and target.

    Unknown cells are treated as potentially traversable.
    Known blocked cells are excluded.
    """

    if start == target:
        return [start]

    queue = deque([start])
    previous: dict[
        tuple[int, int],
        Optional[tuple[int, int]],
    ] = {
        start: None,
    }

    while queue:
        current = queue.popleft()

        for next_position in neighbors(current, model):
            if next_position in previous:
                continue

            previous[next_position] = current

            if next_position == target:
                route = [target]
                cursor = target

                while previous[cursor] is not None:
                    cursor = previous[cursor]
                    route.append(cursor)

                route.reverse()
                return route

            queue.append(next_position)

    return None


# ============================================================
# WORLD STATE SETUP
# ============================================================

def set_position(
    world: World1,
    position: tuple[int, int],
) -> None:
    position_type = type(world.organism.position)

    world.organism.position = position_type(
        x=position[0],
        y=position[1],
    )


def configure_world(
    start: tuple[int, int],
    start_orientation: str,
) -> World1:
    world = World1()

    set_position(world, start)
    world.organism.orientation = start_orientation

    return world


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute_action(
    world: World1,
    action: str,
    target: tuple[int, int],
    step: int,
) -> Consequence:
    before_position = position_tuple(world)
    before_orientation = orientation(world)
    before_energy = energy(world)

    distance_before = manhattan_distance(
        before_position,
        target,
    )

    world.step(action)

    after_position = position_tuple(world)
    after_orientation = orientation(world)
    after_energy = energy(world)

    distance_after = manhattan_distance(
        after_position,
        target,
    )

    movement_succeeded = (
        action == "move_forward"
        and after_position != before_position
    )

    improved_distance = distance_after < distance_before

    return Consequence(
        step=step,
        before_position=before_position,
        after_position=after_position,
        before_orientation=before_orientation,
        after_orientation=after_orientation,
        action=action,
        distance_before=distance_before,
        distance_after=distance_after,
        energy_before=before_energy,
        energy_after=after_energy,
        movement_succeeded=movement_succeeded,
        improved_distance=improved_distance,
    )


# ============================================================
# SINGLE TRIAL
# ============================================================

@dataclass
class TrialResult:
    trial_id: int
    start: tuple[int, int]
    target: tuple[int, int]
    starting_orientation: str
    final_position: tuple[int, int]
    final_orientation: str
    final_distance: int
    reached_target: bool
    steps: int
    successful_movements: int
    failed_movements: int
    turns: int
    replans: int
    discovered_obstacles: int
    visited_cells: int
    final_energy: float


def run_trial(
    trial_id: int,
    start: tuple[int, int],
    target: tuple[int, int],
    start_orientation: str,
) -> TrialResult:
    world = configure_world(
        start=start,
        start_orientation=start_orientation,
    )

    perception = EnhancedLocalPerception()
    model = SpatialWorldModel()
    memory = ConsequenceMemory()

    initial_blocked_count = len(model.blocked_cells)

    print()
    print("=" * 64)
    print(f"TRIAL {trial_id}")
    print(f"Start: {start}")
    print(f"Target: {target}")
    print(f"Starting orientation: {start_orientation}")
    print(f"Starting energy: {energy(world):.1f}")
    print("=" * 64)

    for step in range(1, MAX_STEPS + 1):
        current_position = position_tuple(world)
        current_orientation = orientation(world)
        current_energy = energy(world)

        model.observe_position(current_position)

        distance = manhattan_distance(
            current_position,
            target,
        )

        if current_position == target:
            print(f"Step {step:03d}: TARGET REACHED")
            break

        if current_energy <= 0:
            print(f"Step {step:03d}: ENERGY EXHAUSTED")
            break

        # Perception is deliberately executed from the real world.
        # The complete world map is not passed into the planner.
        observation = perception.perceive(world)

        route = bfs_route(
            start=current_position,
            target=target,
            model=model,
        )

        if route is None or len(route) < 2:
            print(
                f"Step {step:03d}: NO ROUTE AVAILABLE"
            )
            break

        next_position = route[1]

        desired_direction = direction_between(
            current_position,
            next_position,
        )

        if desired_direction is None:
            print(
                f"Step {step:03d}: INVALID ROUTE TRANSITION"
            )
            break

        if current_orientation != desired_direction:
            action = turn_action_toward(
                current_direction=current_orientation,
                desired_direction=desired_direction,
            )
        else:
            action = "move_forward"

        consequence = execute_action(
            world=world,
            action=action,
            target=target,
            step=step,
        )

        memory.record(consequence)

        if (
            action == "move_forward"
            and not consequence.movement_succeeded
        ):
            attempted_cell = adjacent_position(
                consequence.before_position,
                consequence.before_orientation,
            )

            # The failed destination is added to the model.
            model.mark_blocked(attempted_cell)

            print(
                f"Step {step:03d}: movement failure "
                f"toward {attempted_cell}; "
                f"known obstacle added"
            )
        else:
            print(
                f"Step {step:03d}: "
                f"{consequence.before_position}"
                f"->{consequence.after_position}, "
                f"{consequence.before_orientation}"
                f"->{consequence.after_orientation}, "
                f"{action}, "
                f"distance "
                f"{consequence.distance_before}"
                f"->{consequence.distance_after}, "
                f"energy "
                f"{consequence.energy_before:.0f}"
                f"->{consequence.energy_after:.0f}"
            )

    final_position = position_tuple(world)
    final_orientation = orientation(world)
    final_energy = energy(world)

    final_distance = manhattan_distance(
        final_position,
        target,
    )

    reached_target = final_position == target

    discovered_obstacles = (
        len(model.blocked_cells)
        - initial_blocked_count
    )

    summary = memory.summary()

    print()
    print(f"Trial {trial_id} result:")
    print(f"Final position: {final_position}")
    print(f"Final orientation: {final_orientation}")
    print(f"Final distance: {final_distance}")
    print(f"Reached target: {reached_target}")
    print(f"Steps: {len(memory.records)}")
    print(
        f"Successful movements: "
        f"{summary['successful_movements']}"
    )
    print(
        f"Failed movements: "
        f"{summary['failed_movements']}"
    )
    print(f"Turns: {summary['turns']}")
    print(f"Discovered obstacles: {discovered_obstacles}")
    print(
        f"Visited cells: "
        f"{len(model.visited_cells)}"
    )
    print(f"Final energy: {final_energy:.1f}")
    print(f"World model: {model.summary()}")

    return TrialResult(
        trial_id=trial_id,
        start=start,
        target=target,
        starting_orientation=start_orientation,
        final_position=final_position,
        final_orientation=final_orientation,
        final_distance=final_distance,
        reached_target=reached_target,
        steps=len(memory.records),
        successful_movements=summary["successful_movements"],
        failed_movements=summary["failed_movements"],
        turns=summary["turns"],
        replans=len(memory.records),
        discovered_obstacles=discovered_obstacles,
        visited_cells=len(model.visited_cells),
        final_energy=final_energy,
    )


# ============================================================
# EVALUATION SUITE
# ============================================================

SCENARIOS = [
    {
        "start": (1, 1),
        "target": (5, 5),
        "orientation": "north",
    },
    {
        "start": (1, 1),
        "target": (8, 8),
        "orientation": "east",
    },
    {
        "start": (8, 8),
        "target": (2, 2),
        "orientation": "south",
    },
    {
        "start": (2, 7),
        "target": (7, 2),
        "orientation": "west",
    },
    {
        "start": (7, 1),
        "target": (2, 8),
        "orientation": "north",
    },
    {
        "start": (8, 1),
        "target": (1, 8),
        "orientation": "east",
    },
]


def print_aggregate_report(
    results: list[TrialResult],
) -> None:
    total_trials = len(results)
    successful_trials = sum(
        1
        for result in results
        if result.reached_target
    )

    total_steps = sum(
        result.steps
        for result in results
    )

    total_successful_movements = sum(
        result.successful_movements
        for result in results
    )

    total_failed_movements = sum(
        result.failed_movements
        for result in results
    )

    total_turns = sum(
        result.turns
        for result in results
    )

    total_obstacles = sum(
        result.discovered_obstacles
        for result in results
    )

    average_final_distance = (
        sum(result.final_distance for result in results)
        / total_trials
        if total_trials
        else 0
    )

    average_steps = (
        total_steps / total_trials
        if total_trials
        else 0
    )

    average_energy = (
        sum(result.final_energy for result in results)
        / total_trials
        if total_trials
        else 0
    )

    success_rate = (
        successful_trials / total_trials * 100
        if total_trials
        else 0
    )

    print()
    print()
    print("#" * 64)
    print("DIA ROUTE PLANNING V2 — AGGREGATE REPORT")
    print("#" * 64)

    print(f"Total trials: {total_trials}")
    print(f"Successful trials: {successful_trials}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per trial: {average_steps:.2f}")
    print(
        "Total successful movements: "
        f"{total_successful_movements}"
    )
    print(
        "Total failed movements: "
        f"{total_failed_movements}"
    )
    print(f"Total turns: {total_turns}")
    print(f"Total discovered obstacles: {total_obstacles}")
    print(
        "Average final distance: "
        f"{average_final_distance:.2f}"
    )
    print(f"Average final energy: {average_energy:.2f}")

    print()
    print("PER-TRIAL SUMMARY")
    print("-" * 64)

    for result in results:
        status = (
            "SUCCESS"
            if result.reached_target
            else "FAILURE"
        )

        print(
            f"Trial {result.trial_id}: "
            f"{status} | "
            f"{result.start} -> {result.target} | "
            f"final={result.final_position} | "
            f"distance={result.final_distance} | "
            f"steps={result.steps} | "
            f"failures={result.failed_movements}"
        )

    print()
    print("INTERPRETATION")
    print("-" * 64)

    if successful_trials == total_trials:
        print(
            "All configured trials reached their targets."
        )
    else:
        print(
            "Some configured trials did not reach "
            "their targets."
        )

    print(
        "This evaluation measures navigation performance "
        "inside the controlled World-1 environment."
    )

    print(
        "It does not establish general intelligence, "
        "independent goals, consciousness, or "
        "unrestricted real-world reasoning."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("DIA ROUTE PLANNING V2")
    print("Generalization and Replanning Evaluation")
    print(f"Maximum steps per trial: {MAX_STEPS}")

    results: list[TrialResult] = []

    for trial_id, scenario in enumerate(
        SCENARIOS,
        start=1,
    ):
        result = run_trial(
            trial_id=trial_id,
            start=scenario["start"],
            target=scenario["target"],
            start_orientation=scenario["orientation"],
        )

        results.append(result)

    print_aggregate_report(results)


if __name__ == "__main__":
    main()

