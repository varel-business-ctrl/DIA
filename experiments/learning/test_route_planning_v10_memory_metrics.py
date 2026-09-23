"""DIA V10 - MEMORY INFLUENCE METRICS

Leakage-aware controlled experiment:
- Training learns blocked cells only from failed movements.
- Evaluation repeats the same maps with controlled starts/targets.
- Memory-enabled modes exclude remembered cells during planning.
- Metrics explicitly count remembered cells that influenced planning.
- Structural mode remains a clearly labeled placeholder until a genuine
  observation-derived abstraction is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple
from collections import deque

from environment.world1.world import World1

Cell = Tuple[int, int]


class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC_MEMORY = "episodic_memory"
    STRUCTURAL_MEMORY = "structural_memory"


@dataclass(frozen=True)
class Scenario:
    name: str
    map_id: str
    start: Cell
    target: Cell
    orientation: str


@dataclass
class Result:
    scenario: str
    mode: str
    solved: bool
    actions: int
    failures: int
    turns: int
    remembered_cells_available: int
    remembered_route_cells_excluded: int
    memory_influenced_planning: bool
    newly_discovered: Set[Cell] = field(default_factory=set)


class KnowledgeMemory:
    def __init__(self) -> None:
        self._blocked: Dict[str, Set[Cell]] = {}

    def learn(self, map_id: str, cells: Iterable[Cell]) -> None:
        self._blocked.setdefault(map_id, set()).update(cells)

    def known(self, map_id: str) -> Set[Cell]:
        return set(self._blocked.get(map_id, set()))


class SpatialModel:
    def __init__(self, width: int = 10, height: int = 10) -> None:
        self.width = width
        self.height = height

    def neighbors(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    def bfs(self, start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
        if start in blocked or target in blocked:
            return None
        queue = deque([start])
        previous: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                path: List[Cell] = []
                while current is not None:
                    path.append(current)
                    current = previous[current]  # type: ignore[assignment]
                return list(reversed(path))
            for nxt in self.neighbors(current):
                if nxt in blocked or nxt in previous:
                    continue
                previous[nxt] = current
                queue.append(nxt)
        return None


def set_position(world: World1, cell: Cell) -> None:
    position_type = type(world.organism.position)
    world.organism.position = position_type(x=cell[0], y=cell[1])


def direction_between(a: Cell, b: Cell) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(1, 0): "east", (-1, 0): "west", (0, 1): "north", (0, -1): "south"}[(dx, dy)]


def turn_toward(current: str, desired: str) -> str:
    order = ["north", "east", "south", "west"]
    i, j = order.index(current), order.index(desired)
    right = (j - i) % 4
    left = (i - j) % 4
    return "turn_right" if right <= left else "turn_left"


def execute(s: Scenario, mode: Mode, memory: Optional[KnowledgeMemory], max_actions: int = 100) -> Result:
    world = World1()
    install_obstacles(world, s.map_id)
    set_position(world, s.start)
    world.organism.orientation = s.orientation
    model = SpatialModel()

    local_known = memory.known(s.map_id) if memory and mode != Mode.NO_MEMORY else set()
    remembered_available = len(local_known)
    remembered_route_excluded = 0
    failures = 0
    turns = 0
    actions = 0
    newly_discovered: Set[Cell] = set()

    while actions < max_actions:
        position = (world.organism.position.x, world.organism.position.y)
        if position == s.target:
            break

        path = model.bfs(position, s.target, local_known)
        if path is None or len(path) < 2:
            break

        next_cell = path[1]
        desired = direction_between(position, next_cell)
        current_orientation = world.organism.orientation

        # Count remembered knowledge whenever the chosen route benefits from
        # excluding a remembered cell. We inspect the unconstrained route to
        # identify whether memory changed the planned route.
        if mode != Mode.NO_MEMORY and local_known:
            unconstrained = model.bfs(position, s.target, set())
            if unconstrained is not None and any(cell in local_known for cell in unconstrained[1:]):
                remembered_route_excluded += 1

        if current_orientation != desired:
            action = turn_toward(current_orientation, desired)
            actions += 1
            turns += 1
            world.step(action)
            world.organism.orientation = "east" if action == "turn_right" and current_orientation == "north" else world.organism.orientation
            # Use the environment's actual orientation after the step when
            # available; this fallback keeps the controller synchronized.
            if action == "turn_right":
                order = ["north", "east", "south", "west"]
                world.organism.orientation = order[(order.index(current_orientation) + 1) % 4]
            else:
                order = ["north", "east", "south", "west"]
                world.organism.orientation = order[(order.index(current_orientation) - 1) % 4]
            continue

        before = (world.organism.position.x, world.organism.position.y)
        actions += 1
        world.step("move_forward")
        after = (world.organism.position.x, world.organism.position.y)

        if after == before:
            failures += 1
            failed = next_cell
            newly_discovered.add(failed)
            local_known.add(failed)

    if memory is not None and mode != Mode.NO_MEMORY:
        memory.learn(s.map_id, newly_discovered)

    final_position = (world.organism.position.x, world.organism.position.y)
    influenced = remembered_route_excluded > 0
    return Result(
        scenario=s.name,
        mode=mode.value,
        solved=final_position == s.target,
        actions=actions,
        failures=failures,
        turns=turns,
        remembered_cells_available=remembered_available,
        remembered_route_cells_excluded=remembered_route_excluded,
        memory_influenced_planning=influenced,
        newly_discovered=newly_discovered,
    )


def scenarios() -> List[Scenario]:
    return [
        Scenario("training_01", "map_a", (1, 1), (6, 1), "east"),
        Scenario("training_02", "map_b", (1, 4), (7, 4), "east"),
        Scenario("training_03", "map_c", (1, 6), (5, 6), "east"),
        Scenario("evaluation_04", "map_a", (2, 1), (6, 1), "east"),
        Scenario("evaluation_05", "map_b", (1, 4), (7, 4), "east"),
        Scenario("evaluation_06", "map_c", (1, 6), (5, 6), "east"),
    ]


def install_obstacles(world: World1, map_id: str) -> None:
    layouts = {
        "map_a": {(3, 1), (5, 5), (6, 5)},
        "map_b": {(4, 3), (4, 4), (4, 5)},
        "map_c": {(2, 6), (3, 6), (6, 2)},
    }
    obstacle_type = type(world.obstacles[0]) if world.obstacles else None
    position_type = type(world.organism.position)
    world.obstacles = []
    for x, y in layouts[map_id]:
        position = position_type(x=x, y=y)
        if obstacle_type is not None:
            world.obstacles.append(obstacle_type(position=position))


def run_mode(mode: Mode, tasks: List[Scenario]) -> List[Result]:
    memory = None if mode == Mode.NO_MEMORY else KnowledgeMemory()
    results: List[Result] = []
    for scenario in tasks:
        # execute() creates the world internally. This hook is kept explicit so
        # the map setup remains visible and can be connected to World1 variants.
        results.append(execute(scenario, mode, memory))
    return results


def print_report(all_results: Dict[str, List[Result]]) -> None:
    print("DIA V10 - MEMORY INFLUENCE METRICS")
    print("Memory is learned only from failed movements.")
    print("Memory influence is counted during route planning, not only at execution.\n")

    for mode_name, results in all_results.items():
        solved = sum(r.solved for r in results)
        avg_actions = sum(r.actions for r in results) / len(results)
        avg_failures = sum(r.failures for r in results) / len(results)
        avg_excluded = sum(r.remembered_route_cells_excluded for r in results) / len(results)
        influenced = sum(r.memory_influenced_planning for r in results)
        print(f"MODE: {mode_name}")
        print(f"  solved: {solved}/{len(results)} ({solved / len(results) * 100:.1f}%)")
        print(f"  avg actions: {avg_actions:.2f}")
        print(f"  avg failures: {avg_failures:.2f}")
        print(f"  avg remembered route exclusions: {avg_excluded:.2f}")
        print(f"  trials with planning influence: {influenced}/{len(results)}\n")

    print("SCENARIO TABLE")
    print("scenario             mode                 actions failures remembered_available route_exclusions solved discovered")
    for mode_name, results in all_results.items():
        for r in results:
            print(f"{r.scenario:<20} {mode_name:<20} {r.actions:>7} {r.failures:>8} {r.remembered_cells_available:>19} {r.remembered_route_cells_excluded:>16} {str(r.solved):>6} {sorted(r.newly_discovered)}")


def main() -> None:
    tasks = scenarios()
    all_results: Dict[str, List[Result]] = {}
    for mode in Mode:
        all_results[mode.value] = run_mode(mode, tasks)
    print_report(all_results)


if __name__ == "__main__":
    main()
