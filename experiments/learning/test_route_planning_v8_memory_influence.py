"""DIA V8 - operational memory influence, repeated maps, leakage-free evaluation.

Memory may only store cells discovered through the controller's own failed moves.
The map identifier is an allowed environmental context label, not an obstacle map.
Training and evaluation use the same hidden map layouts with different tasks.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
import statistics
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from environment.world1.world import World1
from environment.world1.entities import Obstacle, Position

Cell = Tuple[int, int]
MOVE = "move_forward"
RIGHT = "turn_right"
LEFT = "turn_left"
ORIENTATIONS = ["north", "east", "south", "west"]
DIRECTIONS = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}

class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC = "episodic_memory"
    STRUCTURAL = "structural_memory"

@dataclass(frozen=True)
class Scenario:
    sid: str
    phase: str
    map_id: str
    obstacles: FrozenSet[Cell]
    start: Cell
    target: Cell
    orientation: str

@dataclass
class Result:
    sid: str
    mode: str
    reached: bool
    actions: int
    successes: int
    failures: int
    turns: int
    final: Cell
    discovered: Set[Cell]
    reused_cells: int

class KnowledgeMemory:
    """Stores only obstacle cells discovered by prior interaction."""
    def __init__(self):
        self.blocked_by_map: Dict[str, Set[Cell]] = {}
        self.discovery_count: Dict[str, int] = {}

    def known_blocked(self, map_id: str) -> Set[Cell]:
        return set(self.blocked_by_map.get(map_id, set()))

    def retrieve(self, map_id: str, candidate: Cell) -> bool:
        return candidate in self.blocked_by_map.get(map_id, set())

    def learn(self, map_id: str, cells: Set[Cell]) -> None:
        if not cells:
            return
        bucket = self.blocked_by_map.setdefault(map_id, set())
        before = len(bucket)
        bucket.update(cells)
        self.discovery_count[map_id] = self.discovery_count.get(map_id, 0) + max(0, len(bucket) - before)


def inside(c: Cell) -> bool:
    return 0 <= c[0] < 10 and 0 <= c[1] < 10


def neighbors(c: Cell):
    x, y = c
    yield (x + 1, y); yield (x - 1, y); yield (x, y + 1); yield (x, y - 1)


def bfs(start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
    if start in blocked or target in blocked:
        return None
    queue = deque([start])
    previous = {start: None}
    while queue:
        current = queue.popleft()
        if current == target:
            route = []
            node = current
            while node is not None:
                route.append(node)
                node = previous[node]
            return route[::-1]
        for nxt in neighbors(current):
            if inside(nxt) and nxt not in blocked and nxt not in previous:
                previous[nxt] = current
                queue.append(nxt)
    return None


def configure(world: World1, scenario: Scenario) -> None:
    world.obstacles.clear()
    for x, y in sorted(scenario.obstacles):
        world.obstacles.append(Obstacle(Position(x=x, y=y)))
    position_type = type(world.organism.position)
    world.organism.position = position_type(x=scenario.start[0], y=scenario.start[1])
    world.organism.orientation = scenario.orientation
    world.organism.energy = 100


def cell_of(world: World1) -> Cell:
    p = world.organism.position
    return (p.x, p.y)


def direction(a: Cell, b: Cell) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(1, 0): "east", (-1, 0): "west", (0, 1): "north", (0, -1): "south"}[(dx, dy)]


def turn_actions(current: str, desired: str) -> List[str]:
    if current == desired:
        return []
    ci, di = ORIENTATIONS.index(current), ORIENTATIONS.index(desired)
    right_steps = (di - ci) % 4
    left_steps = (ci - di) % 4
    return [RIGHT] * right_steps if right_steps <= left_steps else [LEFT] * left_steps


def build_scenarios() -> List[Scenario]:
    maps = {
        "map_A": frozenset({(4, 2), (4, 3), (4, 4), (6, 6), (6, 7)}),
        "map_B": frozenset({(2, 4), (3, 4), (4, 4), (5, 4), (7, 6)}),
        "map_C": frozenset({(3, 2), (3, 3), (5, 5), (6, 5), (7, 5)}),
    }
    tasks = [
        ("training", "map_A", (1, 2), (8, 2), "east"),
        ("training", "map_B", (1, 6), (8, 6), "east"),
        ("training", "map_C", (1, 1), (8, 8), "north"),
        ("training", "map_A", (8, 8), (1, 8), "west"),
        ("training", "map_B", (8, 1), (1, 1), "south"),
        ("training", "map_C", (8, 2), (1, 2), "west"),
        ("evaluation", "map_A", (1, 8), (8, 8), "north"),
        ("evaluation", "map_A", (8, 1), (1, 1), "south"),
        ("evaluation", "map_B", (1, 1), (8, 1), "east"),
        ("evaluation", "map_B", (8, 8), (1, 8), "west"),
        ("evaluation", "map_C", (1, 8), (8, 1), "south"),
        ("evaluation", "map_C", (8, 7), (1, 7), "west"),
    ]
    return [Scenario(f"{phase}_{i:02d}", phase, map_id, maps[map_id], start, target, orientation)
            for i, (phase, map_id, start, target, orientation) in enumerate(tasks, 1)]


def execute(s: Scenario, mode: Mode, memory: Optional[KnowledgeMemory], max_actions: int = 180) -> Result:
    world = World1()
    configure(world, s)
    local_known: Set[Cell] = set()
    failed_cells: Set[Cell] = set()
    reused = 0
    if memory is not None:
        local_known = memory.known_blocked(s.map_id)

    actions = successes = failures = turns = 0
    while actions < max_actions:
        position = cell_of(world)
        orientation = world.organism.orientation
        if position == s.target:
            break
        route = bfs(position, s.target, local_known)
        if route is None or len(route) < 2:
            break
        desired = direction(route[0], route[1])
        needed = turn_actions(orientation, desired)
        if needed:
            world.step(needed[0])
            actions += 1
            turns += 1
            continue

        predicted = (position[0] + DIRECTIONS[orientation][0], position[1] + DIRECTIONS[orientation][1])
        if mode != Mode.NO_MEMORY and memory is not None and memory.retrieve(s.map_id, predicted):
            reused += 1
            local_known.add(predicted)
            continue

        before = position
        world.step(MOVE)
        actions += 1
        after = cell_of(world)
        if after != before:
            successes += 1
        else:
            failures += 1
            failed_cells.add(predicted)
            local_known.add(predicted)

    if memory is not None:
        memory.learn(s.map_id, failed_cells)
    return Result(s.sid, mode.value, cell_of(world) == s.target, actions, successes, failures, turns,
                  cell_of(world), failed_cells, reused)


def summarize(results: List[Result], label: str) -> None:
    solved = sum(r.reached for r in results)
    print(f"\nMODE: {label}")
    print(f"  solved: {solved}/{len(results)} ({100 * solved / len(results):.1f}%)")
    print(f"  avg actions: {statistics.mean(r.actions for r in results):.2f}")
    print(f"  avg failures: {statistics.mean(r.failures for r in results):.2f}")
    print(f"  avg turns: {statistics.mean(r.turns for r in results):.2f}")
    print(f"  avg reused cells: {statistics.mean(r.reused_cells for r in results):.2f}")


def main() -> None:
    scenarios = build_scenarios()
    memories = {Mode.EPISODIC: KnowledgeMemory(), Mode.STRUCTURAL: KnowledgeMemory()}
    results: Dict[str, Dict[str, Result]] = {}
    print("DIA V8 - OPERATIONAL MEMORY INFLUENCE / REPEATED MAPS")
    print(f"Scenarios: {len(scenarios)} (training={sum(s.phase == 'training' for s in scenarios)}, evaluation={sum(s.phase == 'evaluation' for s in scenarios)})")
    print("Memory stores only cells discovered through failed movement; obstacle layouts remain hidden.")

    for scenario in scenarios:
        results[scenario.sid] = {}
        for mode in Mode:
            memory = None if mode == Mode.NO_MEMORY else memories[mode]
            results[scenario.sid][mode.value] = execute(scenario, mode, memory)

    for mode in Mode:
        summarize([results[s.sid][mode.value] for s in scenarios], mode.value)

    print("\nEVALUATION-ONLY PAIRED DIFFERENCES")
    evaluation = [s for s in scenarios if s.phase == "evaluation"]
    base = Mode.NO_MEMORY.value
    for mode in (Mode.EPISODIC.value, Mode.STRUCTURAL.value):
        action_diffs = [results[s.sid][mode].actions - results[s.sid][base].actions for s in evaluation]
        failure_diffs = [results[s.sid][mode].failures - results[s.sid][base].failures for s in evaluation]
        print(f"\n{mode}")
        print(f"  mean action difference: {statistics.mean(action_diffs):+.3f}")
        print(f"  mean failure difference: {statistics.mean(failure_diffs):+.3f}")
        print(f"  fewer actions: {sum(x < 0 for x in action_diffs)}/{len(action_diffs)}")
        print(f"  fewer failures: {sum(x < 0 for x in failure_diffs)}/{len(failure_diffs)}")

    print("\nSCENARIO TABLE")
    print("scenario             mode              actions failures reused final")
    for scenario in scenarios:
        for mode in Mode:
            r = results[scenario.sid][mode.value]
            print(f"{scenario.sid:20} {mode.value:18} {r.actions:7} {r.failures:8} {r.reused_cells:6} {r.final}")

    print("\nSCIENTIFIC NOTES")
    print("1. Training occurs before evaluation in the scenario sequence.")
    print("2. Evaluation comparisons are paired on the same task and hidden map.")
    print("3. No controller receives the obstacle set.")
    print("4. Structural mode is currently a controlled placeholder and must not be called discovered intelligence.")
    print("5. Memory influence is valid only if reused cells are nonzero and behavior differs from no_memory.")

if __name__ == "__main__":
    main()
