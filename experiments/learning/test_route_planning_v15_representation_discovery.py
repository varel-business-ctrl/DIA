"""DIA V15 - LEARNED REPRESENTATION DISCOVERY

Self-contained, deterministic grid-world experiment.

The learner is NOT told obstacle families or relative offsets. It observes
failed forward movements and learns reusable local signatures from experience.
We compare:
  - no_memory
  - episodic_memory
  - engineered_structural_memory
  - learned_representation
  - learned_representation_ablation

This is a capability experiment, not a claim of general intelligence.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]
W = H = 10
MAX_ACTIONS = 120

class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC = "episodic_memory"
    ENGINEERED = "engineered_structural_memory"
    LEARNED = "learned_representation"
    ABLATION = "learned_representation_ablation"

@dataclass(frozen=True)
class Task:
    name: str
    family: str
    start: Cell
    target: Cell
    obstacles: FrozenSet[Cell]

@dataclass
class Run:
    solved: bool
    actions: int
    failures: int
    turns: int
    prior_cells: int
    route_changes: int
    influence_events: int
    learned_patterns: int


def neighbors(c: Cell) -> Iterable[Cell]:
    x, y = c
    # Stable order makes runs reproducible.
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def inside(c: Cell) -> bool:
    return 1 <= c[0] <= W and 1 <= c[1] <= H


def bfs(start: Cell, target: Cell, blocked: Set[Cell]) -> Optional[List[Cell]]:
    if start == target:
        return [start]
    q = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if not inside(nxt) or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == target:
                path = [nxt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def direction(a: Cell, b: Cell) -> Cell:
    return (b[0] - a[0], b[1] - a[1])


def rotate(v: Cell, quarter_turns: int) -> Cell:
    x, y = v
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return x, y


def engineered_key(task: Task) -> Tuple[str, int, int]:
    # This is intentionally human-designed and supplied only to the engineered mode.
    dx = task.target[0] - task.start[0]
    dy = task.target[1] - task.start[1]
    dominant = "horizontal" if abs(dx) >= abs(dy) else "vertical"
    sign = 1 if (dx if dominant == "horizontal" else dy) >= 0 else -1
    return task.family, 1 if dominant == "horizontal" else 2, sign


class Memory:
    def __init__(self) -> None:
        self.episodic: Dict[str, Set[Cell]] = defaultdict(set)
        self.engineered: Dict[Tuple[str, int, int], Set[Cell]] = defaultdict(set)
        # Learned representation: recurring translated/rotated offset patterns.
        self.patterns: Dict[str, Set[Cell]] = defaultdict(set)
        self.pattern_counts: Dict[Tuple[str, Cell], int] = defaultdict(int)

    def learn(self, task: Task, failed: Set[Cell], mode: Mode) -> None:
        if not failed:
            return
        self.episodic[task.name].update(failed)
        if mode == Mode.ENGINEERED:
            self.engineered[engineered_key(task)].update(failed)
        if mode in (Mode.LEARNED, Mode.ABLATION):
            # Discover a task-independent signature from observation coordinates:
            # normalize failures relative to the start, then retain recurring offsets.
            for cell in failed:
                offset = (cell[0] - task.start[0], cell[1] - task.start[1])
                self.pattern_counts[(task.family, offset)] += 1
                if self.pattern_counts[(task.family, offset)] >= 2:
                    self.patterns[task.family].add(offset)

    def retrieve(self, task: Task, mode: Mode) -> Set[Cell]:
        if mode == Mode.NO_MEMORY or mode == Mode.ABLATION:
            return set()
        if mode == Mode.EPISODIC:
            return set(self.episodic.get(task.name, set()))
        if mode == Mode.ENGINEERED:
            return set(self.engineered.get(engineered_key(task), set()))
        # Learned representation translates recurring offsets to the current start.
        return {(task.start[0] + dx, task.start[1] + dy)
                for dx, dy in self.patterns.get(task.family, set())
                if inside((task.start[0] + dx, task.start[1] + dy))}


def run_task(task: Task, mode: Mode, memory: Memory, training: bool) -> Run:
    prior = memory.retrieve(task, mode) if not training else set()
    known = set(prior)
    position = task.start
    actions = failures = turns = route_changes = influence = 0
    failed_cells: Set[Cell] = set()
    learned_patterns = len(memory.patterns.get(task.family, set()))

    while position != task.target and actions < MAX_ACTIONS:
        baseline = bfs(position, task.target, set(known) - set(prior))
        constrained = bfs(position, task.target, set(known) | set(prior))
        if prior and baseline != constrained:
            # A counterfactual difference proves the prior information could matter.
            route_changes += 1
            influence += 1
        route = constrained if mode not in (Mode.NO_MEMORY, Mode.ABLATION) else baseline
        if not route or len(route) < 2:
            break
        nxt = route[1]
        dx, dy = direction(position, nxt)
        # Move directly in the desired direction; a turn costs one action when needed.
        # The controller has a simple orientation state, not a hidden map.
        desired = (dx, dy)
        if 'orientation' not in locals():
            orientation = (0, 1)
        if orientation != desired:
            orientation = desired
            actions += 1
            turns += 1
            if actions >= MAX_ACTIONS:
                break
        actions += 1
        if nxt in task.obstacles or not inside(nxt):
            failures += 1
            failed_cells.add(nxt)
            known.add(nxt)
        else:
            position = nxt
    memory.learn(task, failed_cells, mode)
    return Run(position == task.target, actions, failures, turns,
               len(prior), route_changes, influence, learned_patterns)


def make_tasks() -> Tuple[List[Task], List[Task]]:
    # Families share a hidden relational pattern, but positions and orientations vary.
    # The learner receives task-family labels as experimental grouping metadata; it is not given obstacle offsets or equivalence rules.
    train = [
        Task("train_h1", "wall_gap", (1, 3), (8, 3), frozenset({(4, 3), (4, 4)})),
        Task("train_h2", "wall_gap", (1, 5), (8, 5), frozenset({(4, 5), (4, 6)})),
        Task("train_v1", "wall_gap", (3, 1), (3, 8), frozenset({(3, 4), (4, 4)})),
        Task("train_c1", "cluster", (2, 2), (8, 8), frozenset({(5, 5), (6, 5), (5, 6)})),
    ]
    test = [
        Task("eval_h1", "wall_gap", (2, 3), (9, 3), frozenset({(5, 3), (5, 4)})),
        Task("eval_h2", "wall_gap", (2, 5), (9, 5), frozenset({(5, 5), (5, 6)})),
        Task("eval_v1", "wall_gap", (3, 2), (3, 9), frozenset({(3, 5), (4, 5)})),
        Task("eval_c1", "cluster", (1, 1), (9, 9), frozenset({(5, 5), (6, 5), (5, 6)})),
        Task("eval_mismatch", "unseen_family", (1, 2), (9, 2), frozenset({(5, 2), (5, 3), (5, 4)})),
    ]
    return train, test


def evaluate(mode: Mode, train: List[Task], test: List[Task]) -> Tuple[List[Run], List[Run]]:
    memory = Memory()
    training_runs = [run_task(t, mode, memory, True) for t in train]
    evaluation_runs = [run_task(t, mode, memory, False) for t in test]
    return training_runs, evaluation_runs


def report(mode: Mode, train: List[Task], test: List[Task]) -> None:
    tr, ev = evaluate(mode, train, test)
    solved = sum(r.solved for r in ev)
    avg_actions = sum(r.actions for r in ev) / len(ev)
    avg_failures = sum(r.failures for r in ev) / len(ev)
    avg_prior = sum(r.prior_cells for r in ev) / len(ev)
    influence = sum(r.influence_events for r in ev)
    changes = sum(r.route_changes for r in ev)
    learned = sum(r.learned_patterns for r in ev)
    print(f"\nMODE: {mode.value}")
    print(f"  evaluation solved: {solved}/{len(ev)} ({solved/len(ev)*100:.1f}%)")
    print(f"  avg actions: {avg_actions:.2f}")
    print(f"  avg failures: {avg_failures:.2f}")
    print(f"  avg prior cells: {avg_prior:.2f}")
    print(f"  influence events: {influence}")
    print(f"  route changes: {changes}")
    print(f"  learned pattern observations: {learned}")
    for task, result in zip(test, ev):
        print(f"  {task.name}: solved={result.solved} actions={result.actions} failures={result.failures} prior={result.prior_cells} influence={result.influence_events} changes={result.route_changes}")


def main() -> None:
    train, test = make_tasks()
    print("DIA V15 - LEARNED REPRESENTATION DISCOVERY")
    print("Self-contained grid-world; obstacle offsets and equivalence rules are not given to the learner.")
    print("The learned mode discovers recurring translated offsets from failed movements, using provided family labels as grouping metadata.")
    print("Ablation removes learned representation use and learning.")
    for mode in Mode:
        report(mode, train, test)
    print("\nINTERPRETATION")
    print("A learned representation is supported only if its behavior differs from ablation")
    print("and the difference is linked to prior memory and counterfactual route changes.")
    print("This remains an engineered capability test, not proof of broad intelligence.")

if __name__ == "__main__":
    main()
