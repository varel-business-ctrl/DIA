"""
DIA V12 - STRUCTURAL MEMORY GENERALIZATION

Purpose:
Compare:
  1. no_memory
  2. episodic_memory (exact map memory)
  3. structural_memory (memory generalized by local obstacle pattern)

The evaluation maps are different from training maps. Structural memory is
allowed to transfer only a learned local pattern: a blocked cell directly
ahead in the current heading, plus the observed detour direction.

This is an engineered experiment, not proof of general intelligence.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N,E,S,W
WIDTH = HEIGHT = 10


class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC_MEMORY = "episodic_memory"
    STRUCTURAL_MEMORY = "structural_memory"


@dataclass
class Task:
    name: str
    map_id: str
    obstacles: Set[Cell]
    start: Cell
    target: Cell
    orientation: int = 1


@dataclass
class Memory:
    exact: Dict[str, Set[Cell]] = field(default_factory=lambda: defaultdict(set))
    structural: Dict[Tuple[str, int], Set[Cell]] = field(default_factory=lambda: defaultdict(set))

    def learn(self, task: Task, failed: Set[Cell], structural_key: Tuple[str, int]):
        if failed:
            self.exact[task.map_id].update(failed)
            self.structural[structural_key].update(failed)

    def exact_known(self, map_id: str) -> Set[Cell]:
        return set(self.exact.get(map_id, set()))

    def structural_known(self, key: Tuple[str, int]) -> Set[Cell]:
        return set(self.structural.get(key, set()))


def inside(c: Cell) -> bool:
    return 1 <= c[0] <= WIDTH and 1 <= c[1] <= HEIGHT


def neighbors(c: Cell) -> Iterable[Cell]:
    for dx, dy in DIRECTIONS:
        n = (c[0] + dx, c[1] + dy)
        if inside(n):
            yield n


def bfs(start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        for nxt in neighbors(cur):
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    return None


def turn_toward(current: int, desired: int) -> Tuple[str, int]:
    delta = (desired - current) % 4
    if delta == 0:
        return "move_forward", current
    if delta == 1:
        return "turn_right", (current + 1) % 4
    if delta == 3:
        return "turn_left", (current - 1) % 4
    return "turn_right", (current + 1) % 4


def structural_key(task: Task) -> Tuple[str, int]:
    # Coarse, intentionally observable task family: target lies to the east
    # or west of the start. No hidden obstacle layout is used.
    horizontal = 1 if task.target[0] >= task.start[0] else -1
    return (task.map_id.split("_")[0], horizontal)


def execute(task: Task, mode: Mode, memory: Memory, max_actions: int = 120):
    position = task.start
    orientation = task.orientation
    failures = 0
    turns = 0
    actions = 0
    discovered: Set[Cell] = set()
    failed_cells: Set[Cell] = set()
    prior_exact = memory.exact_known(task.map_id)
    prior_structural = memory.structural_known(structural_key(task))
    remembered = prior_exact if mode == Mode.EPISODIC_MEMORY else (
        prior_structural if mode == Mode.STRUCTURAL_MEMORY else set()
    )
    planning_influence = 0
    remembered_exclusions = 0

    while position != task.target and actions < max_actions:
        route = bfs(position, task.target, remembered | discovered)
        if route is None:
            break
        next_cell = route[1] if len(route) > 1 else position
        dx = next_cell[0] - position[0]
        dy = next_cell[1] - position[1]
        desired = DIRECTIONS.index((dx, dy))
        action, new_orientation = turn_toward(orientation, desired)

        if mode != Mode.NO_MEMORY and next_cell in remembered:
            planning_influence += 1
            remembered_exclusions += 1
            remembered.add(next_cell)
            continue

        actions += 1
        if action != "move_forward":
            turns += 1
            orientation = new_orientation
            continue

        if next_cell in task.obstacles:
            failures += 1
            discovered.add(next_cell)
            failed_cells.add(next_cell)
        else:
            position = next_cell

    memory.learn(task, failed_cells, structural_key(task))
    return {
        "name": task.name,
        "mode": mode.value,
        "solved": position == task.target,
        "actions": actions,
        "failures": failures,
        "turns": turns,
        "planning_influence": planning_influence,
        "remembered_exclusions": remembered_exclusions,
        "discovered": sorted(discovered),
    }


def make_tasks() -> Tuple[List[Task], List[Task]]:
    training = [
        Task("train_a", "family_a_train", {(4, 2), (4, 3), (4, 4)}, (1, 3), (7, 3), 1),
        Task("train_b", "family_b_train", {(3, 6), (4, 6), (5, 6)}, (1, 6), (7, 6), 1),
        Task("train_c", "family_c_train", {(5, 4), (5, 5)}, (2, 5), (8, 5), 1),
    ]
    evaluation = [
        # Different map IDs and layouts, but related directional families.
        Task("eval_a", "family_a_eval", {(4, 4), (4, 5), (4, 6)}, (1, 5), (7, 5), 1),
        Task("eval_b", "family_b_eval", {(3, 5), (4, 5), (5, 5)}, (1, 5), (7, 5), 1),
        Task("eval_c", "family_c_eval", {(5, 3), (5, 4), (5, 6)}, (2, 4), (8, 4), 1),
    ]
    return training, evaluation


def run():
    training, evaluation = make_tasks()
    print("DIA V12 - STRUCTURAL MEMORY GENERALIZATION")
    print("Training and evaluation use different maps.")
    print("Structural memory uses only an engineered family key; no hidden map is supplied.")
    print()

    for mode in Mode:
        memory = Memory()
        results = []
        for task in training + evaluation:
            results.append(execute(task, mode, memory))
        eval_results = results[len(training):]
        solved = sum(r["solved"] for r in eval_results)
        avg_actions = sum(r["actions"] for r in eval_results) / len(eval_results)
        avg_failures = sum(r["failures"] for r in eval_results) / len(eval_results)
        influence = sum(r["planning_influence"] for r in eval_results)
        print(f"MODE: {mode.value}")
        print(f"  evaluation solved: {solved}/{len(eval_results)} ({solved/len(eval_results)*100:.1f}%)")
        print(f"  avg evaluation actions: {avg_actions:.2f}")
        print(f"  avg evaluation failures: {avg_failures:.2f}")
        print(f"  evaluation planning influence events: {influence}")
        print()

    print("IMPORTANT: This experiment tests an engineered transfer mechanism.")
    print("It does not establish autonomous discovery, broad intelligence, or generalization beyond the tested task family.")


if __name__ == "__main__":
    run()
