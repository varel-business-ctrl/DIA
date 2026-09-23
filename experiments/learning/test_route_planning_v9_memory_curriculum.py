"""DIA V9 - causal memory curriculum.

Purpose: make memory influence measurable by forcing training to discover
specific blocked cells that a matched evaluation task would otherwise test.
No controller receives the obstacle set. Memory is learned only from failed
forward movements. All modes use identical tasks and action rules.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from collections import deque
import statistics

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
    expected_memory_cells: FrozenSet[Cell] = frozenset()

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
    avoided_from_memory: int

class KnowledgeMemory:
    def __init__(self):
        self.blocked_by_map: Dict[str, Set[Cell]] = {}

    def known_blocked(self, map_id: str) -> Set[Cell]:
        return set(self.blocked_by_map.get(map_id, set()))

    def retrieve(self, map_id: str, cell: Cell) -> bool:
        return cell in self.blocked_by_map.get(map_id, set())

    def learn(self, map_id: str, cells: Set[Cell]) -> None:
        self.blocked_by_map.setdefault(map_id, set()).update(cells)

    def snapshot(self) -> Dict[str, Set[Cell]]:
        return {k: set(v) for k, v in self.blocked_by_map.items()}

def inside(c: Cell) -> bool:
    return 0 <= c[0] < 10 and 0 <= c[1] < 10

def neighbors(c: Cell):
    x, y = c
    yield (x + 1, y); yield (x - 1, y); yield (x, y + 1); yield (x, y - 1)

def bfs(start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
    if start in blocked or target in blocked:
        return None
    queue = deque([start]); previous = {start: None}
    while queue:
        current = queue.popleft()
        if current == target:
            route = []; node = current
            while node is not None:
                route.append(node); node = previous[node]
            return route[::-1]
        for nxt in neighbors(current):
            if inside(nxt) and nxt not in blocked and nxt not in previous:
                previous[nxt] = current; queue.append(nxt)
    return None

def configure(world: World1, s: Scenario) -> None:
    world.obstacles.clear()
    for x, y in sorted(s.obstacles):
        world.obstacles.append(Obstacle(Position(x=x, y=y)))
    ptype = type(world.organism.position)
    world.organism.position = ptype(x=s.start[0], y=s.start[1])
    world.organism.orientation = s.orientation
    world.organism.energy = 100

def cell_of(world: World1) -> Cell:
    p = world.organism.position
    return (p.x, p.y)

def direction(a: Cell, b: Cell) -> str:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(1, 0): "east", (-1, 0): "west", (0, 1): "north", (0, -1): "south"}[(dx, dy)]

def turn_actions(current: str, desired: str) -> List[str]:
    if current == desired: return []
    ci, di = ORIENTATIONS.index(current), ORIENTATIONS.index(desired)
    right = (di - ci) % 4; left = (ci - di) % 4
    return [RIGHT] * right if right <= left else [LEFT] * left

def build_scenarios() -> List[Scenario]:
    # Each training task deliberately drives into a known obstacle.
    # Evaluation tasks reuse the same map and begin from a different location.
    map_a = frozenset({(3, 1), (5, 5), (6, 5)})
    map_b = frozenset({(4, 3), (4, 4), (4, 5)})
    map_c = frozenset({(2, 6), (3, 6), (6, 2)})
    return [
        Scenario("training_01", "training", "A", map_a, (1, 1), (6, 1), "east", frozenset({(3, 1)})),
        Scenario("training_02", "training", "B", map_b, (1, 4), (7, 4), "east", frozenset({(4, 4)})),
        Scenario("training_03", "training", "C", map_c, (1, 6), (5, 6), "east", frozenset({(2, 6)})),
        Scenario("evaluation_04", "evaluation", "A", map_a, (2, 1), (6, 1), "east", frozenset({(3, 1)})),
        Scenario("evaluation_05", "evaluation", "B", map_b, (1, 4), (7, 4), "east", frozenset({(4, 4)})),
        Scenario("evaluation_06", "evaluation", "C", map_c, (1, 6), (5, 6), "east", frozenset({(2, 6)})),
    ]

def execute(s: Scenario, mode: Mode, memory: Optional[KnowledgeMemory], max_actions: int = 100) -> Result:
    world = World1(); configure(world, s)
    local_known = memory.known_blocked(s.map_id) if memory else set()
    discovered: Set[Cell] = set(); reused = 0; avoided = 0
    actions = successes = failures = turns = 0
    while actions < max_actions:
        position = cell_of(world)
        if position == s.target: break
        orientation = world.organism.orientation
        route = bfs(position, s.target, local_known)
        if route is None or len(route) < 2: break
        desired = direction(route[0], route[1])
        needed = turn_actions(orientation, desired)
        if needed:
            world.step(needed[0]); actions += 1; turns += 1; continue
        predicted = (position[0] + DIRECTIONS[orientation][0], position[1] + DIRECTIONS[orientation][1])
        if mode != Mode.NO_MEMORY and memory and memory.retrieve(s.map_id, predicted):
            reused += 1; avoided += 1
            local_known.add(predicted)
            # Force a fresh route around remembered knowledge; do not spend an action.
            continue
        before = position; world.step(MOVE); actions += 1; after = cell_of(world)
        if after != before:
            successes += 1
        else:
            failures += 1; discovered.add(predicted); local_known.add(predicted)
    if memory: memory.learn(s.map_id, discovered)
    return Result(s.sid, mode.value, cell_of(world) == s.target, actions, successes, failures, turns,
                  cell_of(world), discovered, reused, avoided)

def main() -> None:
    scenarios = build_scenarios()
    memories = {Mode.EPISODIC: KnowledgeMemory(), Mode.STRUCTURAL: KnowledgeMemory()}
    results: Dict[str, Dict[str, Result]] = {}
    print("DIA V9 - CAUSAL MEMORY CURRICULUM")
    print("Memory learns only from failed movements; obstacle layouts are hidden.")
    for s in scenarios:
        results[s.sid] = {}
        for mode in Mode:
            mem = None if mode == Mode.NO_MEMORY else memories[mode]
            results[s.sid][mode.value] = execute(s, mode, mem)
    for mode in Mode:
        rs = [results[s.sid][mode.value] for s in scenarios]
        print(f"\nMODE: {mode.value}")
        print(f"  solved: {sum(r.reached for r in rs)}/{len(rs)} ({100*sum(r.reached for r in rs)/len(rs):.1f}%)")
        print(f"  avg actions: {statistics.mean(r.actions for r in rs):.2f}")
        print(f"  avg failures: {statistics.mean(r.failures for r in rs):.2f}")
        print(f"  avg reused cells: {statistics.mean(r.reused_cells for r in rs):.2f}")
        print(f"  avg avoided from memory: {statistics.mean(r.avoided_from_memory for r in rs):.2f}")
    evaluation = [s for s in scenarios if s.phase == "evaluation"]
    print("\nEVALUATION-ONLY PAIRED DIFFERENCES")
    for mode in (Mode.EPISODIC.value, Mode.STRUCTURAL.value):
        ad = [results[s.sid][mode].actions - results[s.sid][Mode.NO_MEMORY.value].actions for s in evaluation]
        fd = [results[s.sid][mode].failures - results[s.sid][Mode.NO_MEMORY.value].failures for s in evaluation]
        rd = [results[s.sid][mode].reused_cells for s in evaluation]
        print(f"\n{mode}")
        print(f"  mean action difference: {statistics.mean(ad):+.3f}")
        print(f"  mean failure difference: {statistics.mean(fd):+.3f}")
        print(f"  evaluation reused cells: {sum(rd)}")
        print(f"  fewer actions: {sum(x < 0 for x in ad)}/{len(ad)}")
        print(f"  fewer failures: {sum(x < 0 for x in fd)}/{len(fd)}")
    print("\nSCENARIO TABLE")
    print("scenario             mode              actions failures reused avoided final discovered")
    for s in scenarios:
        for mode in Mode:
            r = results[s.sid][mode.value]
            print(f"{s.sid:20} {mode.value:18} {r.actions:7} {r.failures:8} {r.reused_cells:6} {r.avoided_from_memory:7} {r.final} {sorted(r.discovered)}")
    print("\nSCIENTIFIC NOTES")
    print("1. Training and evaluation share map IDs but use controlled tasks.")
    print("2. Memory is operational only when retrieval changes route execution.")
    print("3. Structural mode is intentionally identical to episodic storage in this version; structural abstraction is not yet implemented.")
    print("4. A nonzero reused count is necessary but not sufficient for transfer learning.")

if __name__ == "__main__":
    main()
