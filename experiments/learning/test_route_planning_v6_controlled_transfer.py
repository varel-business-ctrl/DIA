"""
DIA V6 — Controlled Learning, Paired Baseline, and Structural Transfer

Purpose
-------
This experiment tests whether memory and reusable structural knowledge improve
navigation compared with a matched no-memory baseline.

Important scientific design
----------------------------
* The same generated scenarios are used for every controller.
* Each scenario is run with:
    1. NO_MEMORY: starts with an empty model.
    2. EPISODIC_MEMORY: may reuse prior experience.
    3. STRUCTURAL_MEMORY: reuses abstract obstacle-pattern knowledge.
* The environment is hidden from the controller except through observations
  and action consequences.
* Ground truth is used only by the evaluator, never by the controller.
* Results are paired by scenario ID.

This is still an engineered navigation experiment, not proof of general
intelligence or autonomous discovery.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import random
import statistics
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from environment.world1.world import World1
from environment.world1.entities import Obstacle, Position

Cell = Tuple[int, int]
Action = str
Orientation = str

MOVE_ACTION = "move_forward"
TURN_RIGHT = "turn_right"
TURN_LEFT = "turn_left"
WAIT = "wait"

DIRECTIONS: Dict[Orientation, Cell] = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
ORDERED_ORIENTATIONS = ["north", "east", "south", "west"]
TURN_RIGHT_MAP = {
    "north": "east",
    "east": "south",
    "south": "west",
    "west": "north",
}
TURN_LEFT_MAP = {
    "north": "west",
    "west": "south",
    "south": "east",
    "east": "north",
}


class ControllerMode(str, Enum):
    NO_MEMORY = "no_memory"
    EPISODIC_MEMORY = "episodic_memory"
    STRUCTURAL_MEMORY = "structural_memory"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    phase: str
    obstacles: FrozenSet[Cell]
    start: Cell
    target: Cell
    orientation: Orientation


@dataclass
class TrialResult:
    scenario_id: str
    mode: str
    phase: str
    reached: bool
    reachable: bool
    shortest_length: Optional[int]
    actions: int
    successful_moves: int
    failed_moves: int
    turns: int
    replans: int
    final_position: Cell
    discovered_blocked: Set[Cell]
    actual_obstacles: Set[Cell]
    used_transfer: bool
    route_efficiency: float
    action_overhead: Optional[float]
    structural_signature: Tuple
    classification: str = ""


class SpatialModel:
    """Controller-side partial map."""

    def __init__(self) -> None:
        self.known_blocked: Set[Cell] = set()
        self.known_free: Set[Cell] = set()
        self.observations: Dict[Cell, int] = defaultdict(int)

    def observe_cell(self, cell: Cell, blocked: bool) -> None:
        self.observations[cell] += 1
        if blocked:
            self.known_blocked.add(cell)
            self.known_free.discard(cell)
        else:
            if cell not in self.known_blocked:
                self.known_free.add(cell)

    def copy(self) -> "SpatialModel":
        other = SpatialModel()
        other.known_blocked = set(self.known_blocked)
        other.known_free = set(self.known_free)
        other.observations = defaultdict(int, self.observations)
        return other


class ExperienceMemory:
    """
    Generalized experience store.

    It does not key only on an exact obstacle layout. Instead, it stores
    structural signatures and successful local avoidance patterns.
    """

    def __init__(self) -> None:
        self.pattern_counts: Dict[Tuple, int] = defaultdict(int)
        self.pattern_failures: Dict[Tuple, int] = defaultdict(int)
        self.pattern_successes: Dict[Tuple, int] = defaultdict(int)
        self.directional_failures: Dict[Tuple[Orientation, Cell], int] = defaultdict(int)

    def learn_from_trial(
        self,
        signature: Tuple,
        orientation: Orientation,
        failed_cells: Iterable[Cell],
        reached: bool,
    ) -> None:
        self.pattern_counts[signature] += 1
        if reached:
            self.pattern_successes[signature] += 1
        else:
            self.pattern_failures[signature] += 1
        for cell in failed_cells:
            self.directional_failures[(orientation, cell)] += 1

    def confidence(self, signature: Tuple) -> float:
        total = self.pattern_counts.get(signature, 0)
        if total == 0:
            return 0.0
        return self.pattern_successes.get(signature, 0) / total


def inside(cell: Cell, width: int = 10, height: int = 10) -> bool:
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def neighbors(cell: Cell) -> Iterable[Cell]:
    x, y = cell
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def ground_truth_bfs(
    start: Cell,
    target: Cell,
    obstacles: Set[Cell],
    width: int = 10,
    height: int = 10,
) -> Optional[List[Cell]]:
    if start in obstacles or target in obstacles:
        return None
    queue = deque([start])
    previous: Dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            route: List[Cell] = []
            node: Optional[Cell] = current
            while node is not None:
                route.append(node)
                node = previous[node]
            return list(reversed(route))

        for nxt in neighbors(current):
            if not inside(nxt, width, height):
                continue
            if nxt in obstacles or nxt in previous:
                continue
            previous[nxt] = current
            queue.append(nxt)

    return None


def structural_signature(
    obstacles: Set[Cell],
    start: Cell,
    target: Cell,
    width: int = 10,
    height: int = 10,
) -> Tuple:
    """
    A deliberately simple abstract signature.

    It captures:
    * obstacle density bucket
    * relative target direction
    * whether the target has blocked neighbors
    * whether start/target lie near boundaries
    * coarse obstacle distribution by quadrant

    This is not a learned representation yet. It is a controlled engineered
    feature representation used to test whether abstraction can support reuse.
    """
    density = len(obstacles) / float(width * height)
    density_bucket = min(4, int(density * 10))

    dx = target[0] - start[0]
    dy = target[1] - start[1]
    horizontal = "right" if dx > 0 else "left" if dx < 0 else "same"
    vertical = "up" if dy > 0 else "down" if dy < 0 else "same"

    target_blocked_neighbors = sum(
        1 for n in neighbors(target) if n in obstacles
    )
    boundary_flags = (
        start[0] in (0, width - 1),
        start[1] in (0, height - 1),
        target[0] in (0, width - 1),
        target[1] in (0, height - 1),
    )

    quadrants = [0, 0, 0, 0]
    for x, y in obstacles:
        if x < width / 2 and y < height / 2:
            quadrants[0] += 1
        elif x >= width / 2 and y < height / 2:
            quadrants[1] += 1
        elif x < width / 2 and y >= height / 2:
            quadrants[2] += 1
        else:
            quadrants[3] += 1

    quadrant_bucket = tuple(min(3, q // 3) for q in quadrants)
    return (
        density_bucket,
        horizontal,
        vertical,
        min(4, target_blocked_neighbors),
        boundary_flags,
        quadrant_bucket,
    )


def configure_world(world: World1, scenario: Scenario) -> None:
    world.obstacles.clear()
    for x, y in sorted(scenario.obstacles):
        world.obstacles.append(Obstacle(Position(x=x, y=y)))

    position_type = type(world.organism.position)
    world.organism.position = position_type(
        x=scenario.start[0], y=scenario.start[1]
    )
    world.organism.orientation = scenario.orientation
    world.organism.energy = 100


def orientation_to_cell(orientation: Orientation, cell: Cell) -> Cell:
    dx, dy = DIRECTIONS[orientation]
    return (cell[0] + dx, cell[1] + dy)


def required_turns(current: Orientation, desired: Orientation) -> List[Action]:
    if current == desired:
        return []
    ci = ORDERED_ORIENTATIONS.index(current)
    di = ORDERED_ORIENTATIONS.index(desired)
    right_steps = (di - ci) % 4
    left_steps = (ci - di) % 4
    if right_steps <= left_steps:
        return [TURN_RIGHT] * right_steps
    return [TURN_LEFT] * left_steps


def direction_between(a: Cell, b: Cell) -> Orientation:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 1:
        return "east"
    if dx == -1:
        return "west"
    if dy == 1:
        return "north"
    if dy == -1:
        return "south"
    raise ValueError(f"Cells are not adjacent: {a}, {b}")


def internal_bfs(
    start: Cell,
    target: Cell,
    known_blocked: Set[Cell],
    width: int = 10,
    height: int = 10,
) -> Optional[List[Cell]]:
    """
    Optimistic planner: unknown cells are treated as potentially traversable.
    Known blocked cells are excluded.
    """
    if start in known_blocked or target in known_blocked:
        return None

    queue = deque([start])
    previous: Dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            route: List[Cell] = []
            node: Optional[Cell] = current
            while node is not None:
                route.append(node)
                node = previous[node]
            return list(reversed(route))

        for nxt in neighbors(current):
            if not inside(nxt, width, height):
                continue
            if nxt in known_blocked or nxt in previous:
                continue
            previous[nxt] = current
            queue.append(nxt)
    return None


def generate_scenarios(
    seed: int = 20260921,
    training_count: int = 12,
    transfer_count: int = 12,
    width: int = 10,
    height: int = 10,
) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []

    for phase, count, density in (
        ("training", training_count, 0.13),
        ("transfer", transfer_count, 0.20),
    ):
        for index in range(1, count + 1):
            start = (rng.randrange(width), rng.randrange(height))
            target = (rng.randrange(width), rng.randrange(height))
            while target == start:
                target = (rng.randrange(width), rng.randrange(height))

            obstacles: Set[Cell] = set()
            for x in range(width):
                for y in range(height):
                    cell = (x, y)
                    if cell in (start, target):
                        continue
                    if rng.random() < density:
                        obstacles.add(cell)

            orientation = rng.choice(ORDERED_ORIENTATIONS)
            scenarios.append(
                Scenario(
                    scenario_id=f"{phase}_{index:02d}",
                    phase=phase,
                    obstacles=frozenset(obstacles),
                    start=start,
                    target=target,
                    orientation=orientation,
                )
            )
    return scenarios


def choose_route(
    position: Cell,
    target: Cell,
    model: SpatialModel,
) -> Optional[List[Cell]]:
    return internal_bfs(position, target, model.known_blocked)


def execute_trial(
    scenario: Scenario,
    mode: ControllerMode,
    memory: Optional[ExperienceMemory],
    max_actions: int = 180,
) -> TrialResult:
    world = World1()
    configure_world(world, scenario)

    model = SpatialModel()
    position = scenario.start
    orientation = scenario.orientation
    actions = 0
    successful_moves = 0
    failed_moves = 0
    turns = 0
    replans = 0
    failed_cells: Set[Cell] = set()
    used_transfer = mode != ControllerMode.NO_MEMORY and memory is not None

    signature = structural_signature(set(scenario.obstacles), scenario.start, scenario.target)

    # Structural memory can supply cautious prior information only when a
    # pattern has repeatedly failed. It never receives ground-truth obstacles.
    prior_confidence = memory.confidence(signature) if memory else 0.0

    while actions < max_actions and position != scenario.target:
        route = choose_route(position, scenario.target, model)
        replans += 1

        if route is None:
            break

        if len(route) <= 1:
            break

        next_cell = route[1]
        desired_orientation = direction_between(position, next_cell)

        turns_needed = required_turns(orientation, desired_orientation)
        if turns_needed:
            action = turns_needed[0]
            if action == TURN_RIGHT:
                orientation = TURN_RIGHT_MAP[orientation]
            else:
                orientation = TURN_LEFT_MAP[orientation]
            turns += 1
            actions += 1
            continue

        # In structural mode, a repeated difficult pattern encourages a
        # conservative alternative only after an observed local failure.
        # This is intentionally modest and measurable, not omniscient.
        if (
            mode == ControllerMode.STRUCTURAL_MEMORY
            and prior_confidence < 0.5
            and next_cell in failed_cells
        ):
            model.known_blocked.add(next_cell)
            continue

        before = position
        predicted = orientation_to_cell(orientation, position)
        result = world.step(MOVE_ACTION)
        actions += 1

        current_position_obj = world.organism.position
        after = (current_position_obj.x, current_position_obj.y)
        position = after

        if after != before:
            successful_moves += 1
            model.observe_cell(after, blocked=False)
        else:
            failed_moves += 1
            failed_cells.add(predicted)
            model.observe_cell(predicted, blocked=True)

    actual_obstacles = set(scenario.obstacles)
    ground_route = ground_truth_bfs(
        scenario.start, scenario.target, actual_obstacles
    )
    reachable = ground_route is not None
    shortest_length = len(ground_route) - 1 if ground_route else None
    reached = position == scenario.target

    route_efficiency = 0.0
    action_overhead: Optional[float] = None
    if reached and shortest_length is not None:
        route_efficiency = shortest_length / actions if actions else 0.0
        action_overhead = actions / max(1, shortest_length)

    if reached and reachable:
        classification = "SUCCESS_SOLVABLE"
    elif not reachable and not reached:
        classification = "UNREACHABLE_OR_UNRESOLVED"
    elif reachable and not reached:
        classification = "FAILURE_SOLVABLE"
    else:
        classification = "UNCLASSIFIED"

    if memory is not None and mode != ControllerMode.NO_MEMORY:
        memory.learn_from_trial(
            signature=signature,
            orientation=orientation,
            failed_cells=failed_cells,
            reached=reached,
        )

    return TrialResult(
        scenario_id=scenario.scenario_id,
        mode=mode.value,
        phase=scenario.phase,
        reached=reached,
        reachable=reachable,
        shortest_length=shortest_length,
        actions=actions,
        successful_moves=successful_moves,
        failed_moves=failed_moves,
        turns=turns,
        replans=replans,
        final_position=position,
        discovered_blocked=set(model.known_blocked),
        actual_obstacles=actual_obstacles,
        used_transfer=used_transfer,
        route_efficiency=route_efficiency,
        action_overhead=action_overhead,
        structural_signature=signature,
        classification=classification,
    )


def mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def paired_report(
    scenarios: Sequence[Scenario],
    results: Dict[str, Dict[str, TrialResult]],
) -> None:
    print("\n" + "=" * 72)
    print("DIA V6 — PAIRED EVALUATION REPORT")
    print("=" * 72)

    modes = [m.value for m in ControllerMode]
    for mode in modes:
        mode_results = [results[s.scenario_id][mode] for s in scenarios]
        solved = sum(r.reached for r in mode_results)
        print(f"\nMODE: {mode}")
        print(f"  trials:                 {len(mode_results)}")
        print(f"  solved:                 {solved}")
        print(f"  success rate:           {100.0 * solved / len(mode_results):.1f}%")
        print(f"  average actions:        {mean([r.actions for r in mode_results]):.2f}")
        print(f"  average failed moves:   {mean([r.failed_moves for r in mode_results]):.2f}")
        print(f"  average turns:          {mean([r.turns for r in mode_results]):.2f}")
        print(f"  average efficiency:     {mean([r.route_efficiency for r in mode_results]):.3f}")

    print("\n" + "-" * 72)
    print("PAIRED DIFFERENCES: MEMORY MODES MINUS NO_MEMORY")
    print("-" * 72)

    baseline = ControllerMode.NO_MEMORY.value
    for mode in [ControllerMode.EPISODIC_MEMORY.value, ControllerMode.STRUCTURAL_MEMORY.value]:
        action_diffs = []
        failure_diffs = []
        efficiency_diffs = []
        for scenario in scenarios:
            base = results[scenario.scenario_id][baseline]
            trial = results[scenario.scenario_id][mode]
            action_diffs.append(trial.actions - base.actions)
            failure_diffs.append(trial.failed_moves - base.failed_moves)
            efficiency_diffs.append(trial.route_efficiency - base.route_efficiency)

        print(f"\n{mode}")
        print(f"  mean action difference:     {mean(action_diffs):+.3f}")
        print(f"  mean failed-move difference:{mean(failure_diffs):+.3f}")
        print(f"  mean efficiency difference: {mean(efficiency_diffs):+.3f}")
        print(f"  improved action count:      {sum(x < 0 for x in action_diffs)}/{len(action_diffs)}")
        print(f"  reduced failures:           {sum(x < 0 for x in failure_diffs)}/{len(failure_diffs)}")
        print(f"  improved efficiency:        {sum(x > 0 for x in efficiency_diffs)}/{len(efficiency_diffs)}")

    print("\n" + "-" * 72)
    print("SCENARIO-BY-SCENARIO PAIRED TABLE")
    print("-" * 72)
    print(
        f"{'scenario':<16}{'baseline':>10}{'episodic':>10}"
        f"{'structural':>12}{'base_fail':>10}{'struct_fail':>12}"
    )

    for scenario in scenarios:
        row = results[scenario.scenario_id]
        print(
            f"{scenario.scenario_id:<16}"
            f"{row[baseline].actions:>10}"
            f"{row[ControllerMode.EPISODIC_MEMORY.value].actions:>10}"
            f"{row[ControllerMode.STRUCTURAL_MEMORY.value].actions:>12}"
            f"{row[baseline].failed_moves:>10}"
            f"{row[ControllerMode.STRUCTURAL_MEMORY.value].failed_moves:>12}"
        )


def main() -> None:
    seed = 20260921
    scenarios = generate_scenarios(seed=seed)

    results: Dict[str, Dict[str, TrialResult]] = {}

    # Each mode receives its own memory instance. The baseline has no memory.
    memories = {
        ControllerMode.NO_MEMORY: None,
        ControllerMode.EPISODIC_MEMORY: ExperienceMemory(),
        ControllerMode.STRUCTURAL_MEMORY: ExperienceMemory(),
    }

    print("DIA V6 — CONTROLLED LEARNING AND STRUCTURAL TRANSFER")
    print(f"Random seed: {seed}")
    print(f"Scenarios: {len(scenarios)}")
    print("Each scenario is evaluated by all three controllers.")

    for scenario in scenarios:
        results[scenario.scenario_id] = {}
        for mode in ControllerMode:
            result = execute_trial(
                scenario=scenario,
                mode=mode,
                memory=memories[mode],
            )
            results[scenario.scenario_id][mode.value] = result

    paired_report(scenarios, results)

    print("\n" + "=" * 72)
    print("SCIENTIFIC INTERPRETATION GUIDANCE")
    print("=" * 72)
    print("1. Use paired differences, not separate random runs.")
    print("2. Negative action difference means the memory mode used fewer actions.")
    print("3. Negative failed-move difference means fewer failed movements.")
    print("4. Positive efficiency difference means better route efficiency.")
    print("5. Structural memory here is engineered, not independently discovered.")
    print("6. Success alone does not prove transfer learning.")
    print("7. Repeat with additional seeds before making broad claims.")


if __name__ == "__main__":
    main()
