"""
DIA V14 - CAUSAL MEMORY TRACE, COUNTERFACTUALS, AND ABLATION

Self-contained experiment. It intentionally uses a small grid-world so the
experiment can run reliably from any DIA checkout without hidden dependencies.

The test compares:
  no_memory
  episodic_memory
  structural_memory
  structural_ablation

It records counterfactual routes before acting, prior-memory exclusions,
actual memory-driven route changes, and behavioral outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Iterable

Cell = tuple[int, int]
WIDTH = HEIGHT = 10
MAX_ACTIONS = 120


class Mode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC = "episodic_memory"
    STRUCTURAL = "structural_memory"
    ABLATION = "structural_ablation"


DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    start: Cell
    target: Cell
    obstacles: frozenset[Cell]


@dataclass
class Trace:
    task: str
    mode: str
    solved: bool = False
    actions: int = 0
    failures: int = 0
    turns: int = 0
    prior_cells: int = 0
    influence_events: int = 0
    route_changes: int = 0
    counterfactual_checks: int = 0
    changed_steps: list[dict] = field(default_factory=list)


def neighbors(cell: Cell) -> Iterable[Cell]:
    x, y = cell
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT:
            yield (nx, ny)


def bfs(start: Cell, target: Cell, blocked: set[Cell]) -> list[Cell] | None:
    if start in blocked or target in blocked:
        return None
    queue = deque([start])
    previous: dict[Cell, Cell | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == target:
            route = []
            while current is not None:
                route.append(current)
                current = previous[current]
            return list(reversed(route))
        for nxt in neighbors(current):
            if nxt in blocked or nxt in previous:
                continue
            previous[nxt] = current
            queue.append(nxt)
    return None


def family_key(task: Task) -> str:
    # Deliberately engineered, and independent of absolute obstacle coordinates.
    dx = task.target[0] - task.start[0]
    dy = task.target[1] - task.start[1]
    direction = "horizontal" if abs(dx) >= abs(dy) else "vertical"
    return f"{task.family}:{direction}:{'right' if dx >= 0 else 'left'}"


def relative_offsets(task: Task) -> set[Cell]:
    sx, sy = task.start
    return {(x - sx, y - sy) for x, y in task.obstacles}


class Memory:
    def __init__(self) -> None:
        self.episodic: dict[str, set[Cell]] = {}
        self.structural: dict[str, set[Cell]] = {}

    def learn(self, task: Task, discovered: set[Cell]) -> None:
        self.episodic.setdefault(task.name, set()).update(discovered)
        self.structural.setdefault(family_key(task), set()).update(
            (x - task.start[0], y - task.start[1]) for x, y in discovered
        )

    def prior(self, task: Task, mode: Mode) -> set[Cell]:
        if mode == Mode.EPISODIC:
            return set(self.episodic.get(task.name, set()))
        if mode == Mode.STRUCTURAL:
            offsets = self.structural.get(family_key(task), set())
            return {
                (task.start[0] + dx, task.start[1] + dy)
                for dx, dy in offsets
                if 1 <= task.start[0] + dx <= WIDTH
                and 1 <= task.start[1] + dy <= HEIGHT
            }
        return set()


def direction_between(a: Cell, b: Cell) -> int:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return DIRS.index((dx, dy))


def run_task(task: Task, mode: Mode, memory: Memory) -> Trace:
    trace = Trace(task.name, mode.value)
    prior = memory.prior(task, mode)
    trace.prior_cells = len(prior)
    known: set[Cell] = set()
    position = task.start
    orientation = 0
    discovered: set[Cell] = set()

    for _ in range(MAX_ACTIONS):
        if position == task.target:
            trace.solved = True
            break

        baseline = bfs(position, task.target, known)
        constrained = bfs(position, task.target, known | prior)
        trace.counterfactual_checks += 1

        if baseline != constrained:
            trace.route_changes += 1
            trace.influence_events += 1
            trace.changed_steps.append({
                "position": position,
                "baseline_length": None if baseline is None else len(baseline) - 1,
                "constrained_length": None if constrained is None else len(constrained) - 1,
                "prior_cells": sorted(prior),
            })

        route = constrained if mode in (Mode.STRUCTURAL, Mode.EPISODIC) else baseline
        if route is None or len(route) < 2:
            break

        desired = direction_between(route[0], route[1])
        turn_delta = (desired - orientation) % 4
        if turn_delta == 0:
            next_cell = route[1]
            trace.actions += 1
            if next_cell in task.obstacles:
                trace.failures += 1
                discovered.add(next_cell)
                known.add(next_cell)
            else:
                position = next_cell
        else:
            # Rotate using the shortest direction.
            if turn_delta == 1 or turn_delta == 2:
                orientation = (orientation + 1) % 4
            else:
                orientation = (orientation - 1) % 4
            trace.actions += 1
            trace.turns += 1

    if position == task.target:
        trace.solved = True

    # Only failed movements create new knowledge. Ablation intentionally
    # suppresses all learning and all use of structural memory.
    if mode != Mode.ABLATION:
        memory.learn(task, discovered)
    return trace


def make_tasks() -> tuple[list[Task], list[Task]]:
    # Training obstacles are placed at translated positions that teach a
    # family-level pattern. Evaluation maps differ in absolute coordinates.
    training = [
        Task("train_a", "horizontal_barrier", (1, 2), (8, 2),
             frozenset({(4, 2), (4, 3)})),
        Task("train_b", "vertical_barrier", (2, 1), (2, 8),
             frozenset({(2, 4), (3, 4)})),
        Task("train_c", "horizontal_barrier", (1, 5), (8, 5),
             frozenset({(4, 5), (4, 6)})),
    ]
    evaluation = [
        Task("eval_a", "horizontal_barrier", (2, 3), (9, 3),
             frozenset({(5, 3), (5, 4)})),
        Task("eval_b", "vertical_barrier", (3, 2), (3, 9),
             frozenset({(3, 5), (4, 5)})),
        Task("eval_c", "horizontal_barrier", (1, 6), (8, 6),
             frozenset({(4, 6), (4, 7)})),
        Task("eval_mismatch", "horizontal_barrier", (1, 1), (8, 1),
             frozenset({(6, 4), (6, 5)})),
    ]
    return training, evaluation


def summarize(results: list[Trace]) -> None:
    solved = sum(r.solved for r in results)
    n = len(results)
    avg = lambda f: sum(f(r) for r in results) / n
    print(f"  evaluation solved: {solved}/{n} ({100*solved/n:.1f}%)")
    print(f"  avg actions: {avg(lambda r: r.actions):.2f}")
    print(f"  avg failures: {avg(lambda r: r.failures):.2f}")
    print(f"  avg prior cells: {avg(lambda r: r.prior_cells):.2f}")
    print(f"  influence events: {sum(r.influence_events for r in results)}")
    print(f"  route changes: {sum(r.route_changes for r in results)}")
    print(f"  counterfactual checks: {sum(r.counterfactual_checks for r in results)}")


def main() -> None:
    training, evaluation = make_tasks()
    print("DIA V14 - CAUSAL TRACE + COUNTERFACTUAL ABLATION")
    print("Self-contained deterministic grid-world; no hidden layout is supplied to memory.")
    print("Memory influence is counted only when unconstrained and constrained routes differ.")
    print()

    for mode in Mode:
        memory = Memory()
        for task in training:
            run_task(task, mode, memory)
        results = [run_task(task, mode, memory) for task in evaluation]
        print(f"MODE: {mode.value}")
        summarize(results)
        for r in results:
            print(
                f"  {r.task}: solved={r.solved} actions={r.actions} "
                f"failures={r.failures} prior={r.prior_cells} "
                f"influence={r.influence_events} changes={r.route_changes}"
            )
        print()

    print("INTERPRETATION")
    print("Influence is supported only when prior memory, counterfactual route differences,")
    print("and behavioral differences are all observed. This is an engineered capability test,")
    print("not evidence of autonomous discovery or broad intelligence.")


if __name__ == "__main__":
    main()
