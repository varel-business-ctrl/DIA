"""
DIA Route Planning V1
=====================

Purpose:
    Test model-based navigation using:
    - An observed spatial world model
    - Breadth-first search (BFS)
    - Real action verification
    - Automatic replanning
    - Energy safety
    - Consequence recording

This experiment does not modify the previous navigation experiment.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from environment.world1 import World1
from organism.perception.enhanced_local import EnhancedLocalPerception


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = (5, 5)
MAX_STEPS = 150

MOVE_FORWARD = "move_forward"
TURN_RIGHT = "turn_right"
TURN_LEFT = "turn_left"

MOVEMENT_ACTIONS = {MOVE_FORWARD}

DIRECTIONS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}

DIRECTION_ORDER = ["north", "east", "south", "west"]

# Conservative default bounds for World1.
# The model can still learn blocked cells from consequences.
WORLD_MIN_X = 0
WORLD_MAX_X = 9
WORLD_MIN_Y = 0
WORLD_MAX_Y = 9


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Consequence:
    step: int
    position_before: Tuple[int, int]
    position_after: Tuple[int, int]
    orientation_before: str
    orientation_after: str
    action: str
    energy_before: float
    energy_after: float
    movement_succeeded: bool
    distance_before: int
    distance_after: int


class SpatialWorldModel:
    """
    Stores what the system has learned about the environment.

    Unknown cells are not automatically considered blocked.
    A cell becomes blocked when real movement evidence confirms it.
    """

    def __init__(self):
        self.blocked_cells: Set[Tuple[int, int]] = set()
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.observation_count = 0

    def observe_position(self, position: Tuple[int, int]):
        self.visited_cells.add(position)
        self.observation_count += 1

    def mark_blocked(self, position: Tuple[int, int]):
        if self.in_bounds(position):
            self.blocked_cells.add(position)

    def is_blocked(self, position: Tuple[int, int]) -> bool:
        return position in self.blocked_cells

    def in_bounds(self, position: Tuple[int, int]) -> bool:
        x, y = position

        return (
            WORLD_MIN_X <= x <= WORLD_MAX_X
            and WORLD_MIN_Y <= y <= WORLD_MAX_Y
        )

    def summary(self):
        return {
            "visited_cells": len(self.visited_cells),
            "blocked_cells": len(self.blocked_cells),
            "observations": self.observation_count,
            "known_blocked": sorted(self.blocked_cells),
        }


class ConsequenceMemory:
    """Stores actual consequences of executed actions."""

    def __init__(self):
        self.records: List[Consequence] = []

    def record(self, consequence: Consequence):
        self.records.append(consequence)

    def total_records(self) -> int:
        return len(self.records)

    def failed_movements(self) -> int:
        return sum(
            1
            for record in self.records
            if record.action in MOVEMENT_ACTIONS
            and not record.movement_succeeded
        )

    def successful_movements(self) -> int:
        return sum(
            1
            for record in self.records
            if record.action in MOVEMENT_ACTIONS
            and record.movement_succeeded
        )

    def replanning_evidence(self) -> int:
        return self.failed_movements()


# ============================================================
# WORLD ACCESS HELPERS
# ============================================================

def get_position(world: World1) -> Tuple[int, int]:
    position = world.organism.position
    return position.x, position.y


def get_orientation(world: World1) -> str:
    return world.organism.orientation


def read_energy(world: World1) -> float:
    return float(world.organism.energy)


def manhattan_distance(
    first: Tuple[int, int],
    second: Tuple[int, int],
) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def adjacent_position(
    position: Tuple[int, int],
    orientation: str,
) -> Tuple[int, int]:
    dx, dy = DIRECTIONS[orientation]

    return (
        position[0] + dx,
        position[1] + dy,
    )


def direction_between(
    current: Tuple[int, int],
    target: Tuple[int, int],
) -> Optional[str]:
    dx = target[0] - current[0]
    dy = target[1] - current[1]

    for direction, (direction_x, direction_y) in DIRECTIONS.items():
        if (dx, dy) == (direction_x, direction_y):
            return direction

    return None


# ============================================================
# BFS ROUTE PLANNER
# ============================================================

def neighboring_cells(
    position: Tuple[int, int],
    model: SpatialWorldModel,
):
    x, y = position

    for dx, dy in DIRECTIONS.values():
        neighbor = (x + dx, y + dy)

        if not model.in_bounds(neighbor):
            continue

        if model.is_blocked(neighbor):
            continue

        yield neighbor


def bfs_route(
    start: Tuple[int, int],
    target: Tuple[int, int],
    model: SpatialWorldModel,
) -> Optional[List[Tuple[int, int]]]:
    """
    Return a route including start and target.

    Unknown cells are treated as potentially traversable.
    Confirmed blocked cells are excluded.
    """

    if start == target:
        return [start]

    queue = deque([start])
    previous: Dict[
        Tuple[int, int],
        Optional[Tuple[int, int]],
    ] = {
        start: None,
    }

    while queue:
        current = queue.popleft()

        for neighbor in neighboring_cells(current, model):
            if neighbor in previous:
                continue

            previous[neighbor] = current

            if neighbor == target:
                route = []
                cursor = target

                while cursor is not None:
                    route.append(cursor)
                    cursor = previous[cursor]

                route.reverse()
                return route

            queue.append(neighbor)

    return None


# ============================================================
# ORIENTATION CONTROL
# ============================================================

def turn_action_toward(
    current_orientation: str,
    desired_orientation: str,
) -> Optional[str]:
    """
    Choose the shortest single turning action.

    The caller repeats this until orientation is correct.
    """

    current_index = DIRECTION_ORDER.index(current_orientation)
    desired_index = DIRECTION_ORDER.index(desired_orientation)

    clockwise_distance = (
        desired_index - current_index
    ) % 4

    counterclockwise_distance = (
        current_index - desired_index
    ) % 4

    if clockwise_distance == 0:
        return None

    if clockwise_distance <= counterclockwise_distance:
        return TURN_RIGHT

    return TURN_LEFT


# ============================================================
# ROUTE EXECUTION
# ============================================================

def execute_action(world: World1, action: str):
    """
    Execute one real action and return before/after state.
    """

    position_before = get_position(world)
    orientation_before = get_orientation(world)
    energy_before = read_energy(world)

    if energy_before <= 0:
        return {
            "position_before": position_before,
            "position_after": position_before,
            "orientation_before": orientation_before,
            "orientation_after": orientation_before,
            "energy_before": energy_before,
            "energy_after": energy_before,
            "executed": False,
        }

    world.step(action)

    position_after = get_position(world)
    orientation_after = get_orientation(world)
    energy_after = read_energy(world)

    return {
        "position_before": position_before,
        "position_after": position_after,
        "orientation_before": orientation_before,
        "orientation_after": orientation_after,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "executed": True,
    }


def main():
    world = World1()
    perception = EnhancedLocalPerception()

    model = SpatialWorldModel()
    memory = ConsequenceMemory()

    successful_movements = 0
    failed_movements = 0
    turns = 0
    replans = 0
    discovered_obstacles = 0

    print("=" * 70)
    print("DIA ROUTE PLANNING V1")
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
        energy_before = read_energy(world)

        model.observe_position(position_before)

        distance_before = manhattan_distance(
            position_before,
            TARGET,
        )

        # ----------------------------------------------------
        # STOP CONDITIONS
        # ----------------------------------------------------

        if position_before == TARGET:
            print()
            print("TARGET REACHED BEFORE ACTION")
            break

        if energy_before <= 0:
            print()
            print("ENERGY DEPLETED BEFORE ACTION")
            print(f"Energy: {energy_before}")
            break

        # ----------------------------------------------------
        # PERCEPTION
        # ----------------------------------------------------

        observation = perception.perceive(world)

        # ----------------------------------------------------
        # PLAN A ROUTE USING CURRENT MODEL
        # ----------------------------------------------------

        route = bfs_route(
            position_before,
            TARGET,
            model,
        )

        replans += 1

        if route is None:
            print()
            print("NO ROUTE AVAILABLE IN CURRENT MODEL")
            break

        # The first route item is current position.
        next_position = route[1]

        desired_orientation = direction_between(
            position_before,
            next_position,
        )

        if desired_orientation is None:
            print()
            print("PLANNER ERROR: NEXT POSITION IS NOT ADJACENT")
            break

        # ----------------------------------------------------
        # TURN TOWARD NEXT ROUTE CELL
        # ----------------------------------------------------

        action = turn_action_toward(
            orientation_before,
            desired_orientation,
        )

        if action is None:
            action = MOVE_FORWARD

        # ----------------------------------------------------
        # EXECUTE AND VERIFY CONSEQUENCE
        # ----------------------------------------------------

        result = execute_action(world, action)

        if not result["executed"]:
            print()
            print("ACTION BLOCKED BY ENERGY SAFETY")
            break

        position_after = result["position_after"]
        orientation_after = result["orientation_after"]
        energy_after = result["energy_after"]

        distance_after = manhattan_distance(
            position_after,
            TARGET,
        )

        movement_succeeded = (
            action == MOVE_FORWARD
            and position_after != position_before
        )

        consequence = Consequence(
            step=step_number,
            position_before=position_before,
            position_after=position_after,
            orientation_before=orientation_before,
            orientation_after=orientation_after,
            action=action,
            energy_before=energy_before,
            energy_after=energy_after,
            movement_succeeded=movement_succeeded,
            distance_before=distance_before,
            distance_after=distance_after,
        )

        memory.record(consequence)

        # ----------------------------------------------------
        # LEARN FROM REAL MOVEMENT CONSEQUENCES
        # ----------------------------------------------------

        if action == MOVE_FORWARD:
            if movement_succeeded:
                successful_movements += 1
            else:
                failed_movements += 1

                attempted_cell = adjacent_position(
                    position_before,
                    orientation_before,
                )

                was_new_obstacle = not model.is_blocked(
                    attempted_cell
                )

                model.mark_blocked(attempted_cell)

                if was_new_obstacle:
                    discovered_obstacles += 1

                print()
                print("MOVEMENT FAILURE DETECTED")
                print(f"Blocked cell learned: {attempted_cell}")
                print("World model updated.")
                print("Route will be recalculated.")

        elif action in {TURN_LEFT, TURN_RIGHT}:
            turns += 1

        print(
            f"Step {step_number:03d} | "
            f"Position {position_before} -> {position_after} | "
            f"Facing {orientation_before:>5} -> {orientation_after:>5} | "
            f"Distance {distance_before:02d} -> {distance_after:02d} | "
            f"Action {action:>13} | "
            f"Energy {energy_before:5.1f} -> {energy_after:5.1f}"
        )

        if position_after == TARGET:
            print()
            print("TARGET REACHED")
            break

    # ========================================================
    # FINAL REPORT
    # ========================================================

    final_position = get_position(world)
    final_orientation = get_orientation(world)
    final_distance = manhattan_distance(
        final_position,
        TARGET,
    )

    print()
    print("=" * 70)
    print("FINAL ROUTE PLANNING REPORT")
    print("=" * 70)

    print(f"Target: {TARGET}")
    print(f"Final position: {final_position}")
    print(f"Final orientation: {final_orientation}")
    print(f"Final distance: {final_distance}")
    print(f"Reached target: {final_position == TARGET}")
    print(f"Successful movements: {successful_movements}")
    print(f"Failed movements: {failed_movements}")
    print(f"Turns: {turns}")
    print(f"Replans: {replans}")
    print(f"New obstacles discovered: {discovered_obstacles}")
    print(f"Recorded consequences: {memory.total_records()}")
    print(f"Known visited cells: {len(model.visited_cells)}")
    print(f"Known blocked cells: {len(model.blocked_cells)}")
    print(f"Final energy: {read_energy(world)}")
    print(f"Energy exhausted: {read_energy(world) <= 0}")

    print()
    print("WORLD MODEL SUMMARY")
    print(model.summary())

    if final_position == TARGET:
        print()
        print("RESULT: GOAL ACHIEVED")
        print("The route planner reached the target.")
    else:
        print()
        print("RESULT: GOAL NOT ACHIEVED")
        print("The failure is recorded for further investigation.")


if __name__ == "__main__":
    main()
