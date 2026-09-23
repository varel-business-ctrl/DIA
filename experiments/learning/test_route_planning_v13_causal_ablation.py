"""
DIA V13 - CAUSAL STRUCTURAL MEMORY + ABLATION

This experiment fixes V12's measurement gap:
- compares no memory, episodic memory, structural memory
- runs a structural ablation (same controller, no structural retrieval)
- records prior memory, exclusions, route changes, and action outcomes
- structural memory stores obstacle offsets relative to the task start, grouped
  by an observable directional family; it never receives the hidden map.
This remains an engineered experiment, not proof of general intelligence.
"""
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
WIDTH = HEIGHT = 10

class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC_MEMORY = "episodic_memory"
    STRUCTURAL_MEMORY = "structural_memory"
    STRUCTURAL_ABLATION = "structural_ablation"

@dataclass
class Task:
    name: str
    map_id: str
    family: str
    obstacles: Set[Cell]
    start: Cell
    target: Cell
    orientation: int = 1

@dataclass
class Memory:
    exact: Dict[str, Set[Cell]] = field(default_factory=lambda: defaultdict(set))
    structural: Dict[Tuple[str, int], Set[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(set))

    def family_key(self, task: Task) -> Tuple[str, int]:
        direction = 1 if task.target[0] >= task.start[0] else -1
        return (task.family, direction)

    def learn(self, task: Task, failed: Set[Cell]) -> None:
        if not failed:
            return
        self.exact[task.map_id].update(failed)
        key = self.family_key(task)
        for cell in failed:
            self.structural[key].add((cell[0] - task.start[0], cell[1] - task.start[1]))

    def exact_known(self, map_id: str) -> Set[Cell]:
        return set(self.exact.get(map_id, set()))

    def structural_known(self, task: Task) -> Set[Cell]:
        key = self.family_key(task)
        return {(task.start[0] + dx, task.start[1] + dy)
                for dx, dy in self.structural.get(key, set())}

def inside(c: Cell) -> bool:
    return 1 <= c[0] <= WIDTH and 1 <= c[1] <= HEIGHT

def neighbors(c: Cell):
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

def turn_toward(current: int, desired: int):
    delta = (desired - current) % 4
    if delta == 0:
        return "move_forward", current
    if delta == 1:
        return "turn_right", (current + 1) % 4
    if delta == 3:
        return "turn_left", (current - 1) % 4
    return "turn_right", (current + 1) % 4

def execute(task: Task, mode: Mode, memory: Memory, max_actions: int = 160):
    position = task.start
    orientation = task.orientation
    actions = failures = turns = 0
    discovered: Set[Cell] = set()
    failed_cells: Set[Cell] = set()
    prior_exact = memory.exact_known(task.map_id)
    prior_structural = memory.structural_known(task)
    if mode == Mode.EPISODIC_MEMORY:
        remembered = prior_exact
    elif mode == Mode.STRUCTURAL_MEMORY:
        remembered = prior_structural
    else:
        remembered = set()
    prior_count = len(remembered)
    planning_influence = 0
    route_changes = 0
    trace = []

    while position != task.target and actions < max_actions:
        unconstrained = bfs(position, task.target, discovered)
        constrained = bfs(position, task.target, discovered | remembered)
        if constrained is None:
            break
        if unconstrained is not None and unconstrained != constrained:
            route_changes += 1
        route = constrained
        next_cell = route[1] if len(route) > 1 else position
        if next_cell in remembered:
            planning_influence += 1
            remembered.discard(next_cell)
            trace.append(("memory_skip", position, next_cell))
            continue
        dx = next_cell[0] - position[0]
        dy = next_cell[1] - position[1]
        desired = DIRECTIONS.index((dx, dy))
        action, new_orientation = turn_toward(orientation, desired)
        actions += 1
        if action != "move_forward":
            turns += 1
            orientation = new_orientation
            continue
        if next_cell in task.obstacles:
            failures += 1
            discovered.add(next_cell)
            failed_cells.add(next_cell)
            trace.append(("failure", position, next_cell))
        else:
            position = next_cell
            trace.append(("move", position, next_cell))

    memory.learn(task, failed_cells)
    return {
        "name": task.name, "mode": mode.value, "solved": position == task.target,
        "actions": actions, "failures": failures, "turns": turns,
        "prior_memory": prior_count, "planning_influence": planning_influence,
        "route_changes": route_changes, "discovered": sorted(discovered),
        "trace": trace,
    }

def make_tasks():
    training = [
        Task("train_a", "a_train", "horizontal", {(4,2),(4,3),(4,4)}, (1,3), (7,3)),
        Task("train_b", "b_train", "horizontal", {(3,6),(4,6),(5,6)}, (1,6), (7,6)),
        Task("train_c", "c_train", "horizontal", {(5,4),(5,5)}, (2,5), (8,5)),
    ]
    evaluation = [
        Task("eval_a", "a_eval", "horizontal", {(4,4),(4,5),(4,6)}, (1,5), (7,5)),
        Task("eval_b", "b_eval", "horizontal", {(3,5),(4,5),(5,5)}, (1,5), (7,5)),
        Task("eval_c", "c_eval", "horizontal", {(5,3),(5,4),(5,6)}, (2,4), (8,4)),
    ]
    return training, evaluation

def run_mode(mode):
    training, evaluation = make_tasks()
    memory = Memory()
    for task in training:
        execute(task, mode, memory)
    results = [execute(task, mode, memory) for task in evaluation]
    solved = sum(r["solved"] for r in results)
    return results, solved

def run():
    training, evaluation = make_tasks()
    print("DIA V13 - CAUSAL STRUCTURAL MEMORY + ABLATION")
    print("Different maps; no hidden obstacle layout is supplied to memory.")
    print("Structural memory stores translated obstacle offsets by family.")
    print()
    all_results = {}
    for mode in Mode:
        results, solved = run_mode(mode)
        all_results[mode.value] = results
        avg_actions = sum(r["actions"] for r in results) / len(results)
        avg_failures = sum(r["failures"] for r in results) / len(results)
        influence = sum(r["planning_influence"] for r in results)
        changes = sum(r["route_changes"] for r in results)
        prior = sum(r["prior_memory"] for r in results)
        print(f"MODE: {mode.value}")
        print(f"  evaluation solved: {solved}/{len(results)} ({solved/len(results)*100:.1f}%)")
        print(f"  avg actions: {avg_actions:.2f}")
        print(f"  avg failures: {avg_failures:.2f}")
        print(f"  prior memory cells available: {prior}")
        print(f"  planning influence events: {influence}")
        print(f"  route changes caused by constraints: {changes}")
        print()
    print("EVALUATION TRACE SUMMARY")
    for mode, results in all_results.items():
        print(f"{mode}:")
        for r in results:
            print(f"  {r['name']}: actions={r['actions']} failures={r['failures']} "
                  f"prior={r['prior_memory']} influence={r['planning_influence']} "
                  f"route_changes={r['route_changes']} solved={r['solved']}")
    print()
    print("INTERPRETATION RULE:")
    print("Treat structural memory as causally supported only when prior memory,")
    print("route changes, and behavioral differences are all visible.")
    print("This remains an engineered capability test, not general intelligence.")

if __name__ == "__main__":
    run()
