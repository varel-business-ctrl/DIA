
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from environment.world1 import World1
from environment.world1.entities import Position, Obstacle


# ============================================================
# DIA ROUTE PLANNING V3
# CONFIGURABLE AND CHANGING ENVIRONMENTS
# ============================================================

MAX_STEPS = 150

MIN_X = 0
MAX_X = 9
MIN_Y = 0
MAX_Y = 9

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


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Consequence:
    action: str
    before_position: tuple[int, int]
    after_position: tuple[int, int]
    before_orientation: str
    after_orientation: str
    energy_before: float
    energy_after: float
    movement_succeeded: bool


@dataclass
class TrialResult:
    trial_id: int
    name: str
    start: tuple[int, int]
    target: tuple[int, int]
    reached: bool
    steps: int
    successful_movements: int
    failed_movements: int
    turns: int
    discovered_obstacles: int
    final_position: tuple[int, int]
    final_distance: int
    final_energy: float


# ============================================================
# GEOMETRY
# ============================================================

def position_tuple(world: World1) -> tuple[int, int]:
    position = world.organism.position
    return position.x, position.y


def get_orientation(world: World1) -> str:
    return world.organism.orientation


def get_energy(world: World1) -> float:
    return float(world.organism.energy)


def distance(
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


def adjacent(
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


def turn_toward(
    current: str,
    desired: str,
) -> str:
    current_index = direction_index(current)
    desired_index = direction_index(desired)

    clockwise = (desired_index - current_index) % 4
    counterclockwise = (current_index - desired_index) % 4

    if clockwise <= counterclockwise:
        return "turn_right"

    return "turn_left"


# ============================================================
# SPATIAL MODEL
# ============================================================

class SpatialWorldModel:
    """
    The planner only knows obstacles discovered through
    failed movement attempts.

    The actual obstacle map is NOT copied into this model.
    """

    def __init__(self) -> None:
        self.blocked_cells: set[tuple[int, int]] = set()
        self.visited_cells: set[tuple[int, int]] = set()
        self.observations = 0

    def observe(self, position: tuple[int, int]) -> None:
        self.visited_cells.add(position)
        self.observations += 1

    def mark_blocked(self, position: tuple[int, int]) -> None:
        if in_bounds(position):
            self.blocked_cells.add(position)

    def is_blocked(self, position: tuple[int, int]) -> bool:
        return position in self.blocked_cells


# ============================================================
# BFS PLANNER
# ============================================================

def get_neighbors(
    position: tuple[int, int],
    model: SpatialWorldModel,
):
    for direction in DIRECTION_ORDER:
        next_position = adjacent(position, direction)

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

        for next_position in get_neighbors(current, model):
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
# WORLD CONFIGURATION
# ============================================================

def set_position(
    world: World1,
    position: tuple[int, int],
) -> None:
    world.organism.position = Position(
        x=position[0],
        y=position[1],
    )


def create_world(
    start: tuple[int, int],
    orientation: str,
    obstacles: list[tuple[int, int]],
) -> World1:
    world = World1()

    set_position(world, start)
    world.organism.orientation = orientation

    world.obstacles = [
        Obstacle(
            position=Position(x=x, y=y),
        )
        for x, y in obstacles
    ]

    return world


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute(
    world: World1,
    action: str,
) -> Consequence:
    before_position = position_tuple(world)
    before_orientation = get_orientation(world)
    before_energy = get_energy(world)

    world.step(action)

    after_position = position_tuple(world)
    after_orientation = get_orientation(world)
    after_energy = get_energy(world)

    movement_succeeded = (
        action == "move_forward"
        and after_position != before_position
    )

    return Consequence(
        action=action,
        before_position=before_position,
        after_position=after_position,
        before_orientation=before_orientation,
        after_orientation=after_orientation,
        energy_before=before_energy,
        energy_after=after_energy,
        movement_succeeded=movement_succeeded,
    )


# ============================================================
# MAP DEFINITIONS
# ============================================================

MAPS = [
    {
        "name": "open_map",
        "start": (1, 1),
        "target": (8, 8),
        "orientation": "north",
        "obstacles": [],
    },
    {
        "name": "central_wall",
        "start": (1, 5),
        "target": (8, 5),
        "orientation": "east",
        "obstacles": [
            (4, 5),
            (5, 5),
            (6, 5),
        ],
    },
    {
        "name": "central_cluster",
        "start": (1, 1),
        "target": (8, 8),
        "orientation": "north",
        "obstacles": [
            (4, 4),
            (4, 5),
            (5, 4),
            (5, 5),
            (6, 5),
        ],
    },
    {
        "name": "narrow_passage",
        "start": (1, 5),
        "target": (8, 5),
        "orientation": "east",
        "obstacles": [
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 4),
            (4, 6),
            (5, 4),
            (5, 6),
            (6, 4),
            (6, 5),
            (6, 6),
        ],
    },
    {
        "name": "detour_required",
        "start": (2, 2),
        "target": (7, 2),
        "orientation": "east",
        "obstacles": [
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (4, 3),
            (5, 3),
        ],
    },
    {
        "name": "target_surrounding_obstacles",
        "start": (1, 1),
        "target": (7, 7),
        "orientation": "north",
        "obstacles": [
            (6, 7),
            (7, 6),
            (8, 7),
            (7, 8),
        ],
    },
]


# ============================================================
# SINGLE TRIAL
# ============================================================

def run_trial(
    trial_id: int,
    configuration: dict,
) -> TrialResult:
    name = configuration["name"]
    start = configuration["start"]
    target = configuration["target"]
    start_orientation = configuration["orientation"]
    obstacles = configuration["obstacles"]

    world = create_world(
        start=start,
        orientation=start_orientation,
        obstacles=obstacles,
    )

    model = SpatialWorldModel()
    consequences: list[Consequence] = []

    print()
    print("=" * 68)
    print(f"TRIAL {trial_id}: {name}")
    print(f"Start: {start}")
    print(f"Target: {target}")
    print(f"Orientation: {start_orientation}")
    print(f"Actual obstacle count: {len(obstacles)}")
    print("=" * 68)

    for step in range(1, MAX_STEPS + 1):
        current_position = position_tuple(world)
        current_orientation = get_orientation(world)
        current_energy = get_energy(world)

        model.observe(current_position)

        if current_position == target:
            print(f"Step {step:03d}: TARGET REACHED")
            break

        if current_energy <= 0:
            print(f"Step {step:03d}: ENERGY EXHAUSTED")
            break

        route = bfs_route(
            start=current_position,
            target=target,
            model=model,
        )

        if route is None or len(route) < 2:
            print(f"Step {step:03d}: NO ROUTE AVAILABLE")
            break

        next_position = route[1]

        desired_direction = direction_between(
            current_position,
            next_position,
        )

        if desired_direction is None:
            print(f"Step {step:03d}: INVALID ROUTE")
            break

        if current_orientation != desired_direction:
            action = turn_toward(
                current=current_orientation,
                desired=desired_direction,
            )
        else:
            action = "move_forward"

        consequence = execute(world, action)
        consequences.append(consequence)

        if (
            action == "move_forward"
            and not consequence.movement_succeeded
        ):
            attempted_cell = adjacent(
                consequence.before_position,
                consequence.before_orientation,
            )

            model.mark_blocked(attempted_cell)

            print(
                f"Step {step:03d}: FAILED MOVE "
                f"toward {attempted_cell}; "
                f"model updated"
            )
        else:
            print(
                f"Step {step:03d}: "
                f"{consequence.before_position}"
                f"->{consequence.after_position}, "
                f"{consequence.before_orientation}"
                f"->{consequence.after_orientation}, "
                f"{action}, "
                f"energy "
                f"{consequence.energy_before:.0f}"
                f"->{consequence.energy_after:.0f}"
            )

    final_position = position_tuple(world)
    final_orientation = get_orientation(world)
    final_energy = get_energy(world)

    successful_movements = sum(
        1
        for item in consequences
        if item.movement_succeeded
    )

    failed_movements = sum(
        1
        for item in consequences
        if (
            item.action == "move_forward"
            and not item.movement_succeeded
        )
    )

    turns = sum(
        1
        for item in consequences
        if item.action in {"turn_left", "turn_right"}
    )

    final_distance = distance(final_position, target)
    reached = final_position == target

    print()
    print(f"FINAL RESULT — {name}")
    print(f"Reached target: {reached}")
    print(f"Final position: {final_position}")
    print(f"Final orientation: {final_orientation}")
    print(f"Final distance: {final_distance}")
    print(f"Steps: {len(consequences)}")
    print(f"Successful movements: {successful_movements}")
    print(f"Failed movements: {failed_movements}")
    print(f"Turns: {turns}")
    print(f"Discovered obstacles: {len(model.blocked_cells)}")
    print(f"Visited cells: {len(model.visited_cells)}")
    print(f"Final energy: {final_energy:.1f}")
    print(f"Known blocked cells: {sorted(model.blocked_cells)}")

    return TrialResult(
        trial_id=trial_id,
        name=name,
        start=start,
        target=target,
        reached=reached,
        steps=len(consequences),
        successful_movements=successful_movements,
        failed_movements=failed_movements,
        turns=turns,
        discovered_obstacles=len(model.blocked_cells),
        final_position=final_position,
        final_distance=final_distance,
        final_energy=final_energy,
    )


# ============================================================
# AGGREGATE REPORT
# ============================================================

def aggregate_report(results: list[TrialResult]) -> None:
    total = len(results)

    successful = sum(
        1
        for result in results
        if result.reached
    )

    total_steps = sum(result.steps for result in results)
    total_successful = sum(
        result.successful_movements
        for result in results
    )
    total_failed = sum(
        result.failed_movements
        for result in results
    )
    total_turns = sum(result.turns for result in results)

    average_steps = (
        total_steps / total
        if total
        else 0
    )

    average_distance = (
        sum(result.final_distance for result in results) / total
        if total
        else 0
    )

    average_energy = (
        sum(result.final_energy for result in results) / total
        if total
        else 0
    )

    print()
    print()
    print("#" * 68)
    print("DIA ROUTE PLANNING V3 — AGGREGATE REPORT")
    print("#" * 68)

    print(f"Total trials: {total}")
    print(f"Successful trials: {successful}")
    print(
        f"Success rate: "
        f"{(successful / total * 100) if total else 0:.1f}%"
    )
    print(f"Total steps: {total_steps}")
    print(f"Average steps: {average_steps:.2f}")
    print(f"Successful movements: {total_successful}")
    print(f"Failed movements: {total_failed}")
    print(f"Turns: {total_turns}")
    print(f"Average final distance: {average_distance:.2f}")
    print(f"Average final energy: {average_energy:.2f}")

    print()
    print("PER-TRIAL SUMMARY")
    print("-" * 68)

    for result in results:
        status = "SUCCESS" if result.reached else "FAILURE"

        print(
            f"Trial {result.trial_id}: "
            f"{status} | "
            f"{result.name} | "
            f"{result.start}->{result.target} | "
            f"final={result.final_position} | "
            f"distance={result.final_distance} | "
            f"steps={result.steps} | "
            f"failed={result.failed_movements}"
        )

    print()
    print("SCIENTIFIC INTERPRETATION")
    print("-" * 68)

    print(
        "The planner receives the target and starting state "
        "in advance."
    )

    print(
        "The complete obstacle configuration is used only "
        "by the environment, not by the planner."
    )

    print(
        "The spatial model discovers obstacles through "
        "failed forward movements."
    )

    print(
        "Success in these trials demonstrates navigation "
        "performance under the configured conditions."
    )

    print(
        "It does not independently establish broad intelligence, "
        "consciousness, autonomy, or unrestricted discovery."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("DIA ROUTE PLANNING V3")
    print("Configurable and Changing Environments")
    print(f"Maximum steps per trial: {MAX_STEPS}")

    results: list[TrialResult] = []

    for trial_id, configuration in enumerate(
        MAPS,
        start=1,
    ):
        result = run_trial(
            trial_id=trial_id,
            configuration=configuration,
        )

        results.append(result)

    aggregate_report(results)


if __name__ == "__main__":
    main()

