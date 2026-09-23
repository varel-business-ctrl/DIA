"""
DIA Route Planning V5
=====================

Randomized environments, repeated trials, transfer measurement,
reachability-aware evaluation, route efficiency, and experience memory.

Run from ~/DIA:

    python -m py_compile experiments/learning/test_route_planning_v5_randomized.py

    python -m experiments.learning.test_route_planning_v5_randomized

This is an experimental benchmark. The planner is still explicitly designed;
the benchmark measures adaptation and transfer rather than proving general
intelligence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Optional

from environment.world1.world import World1
from environment.world1.entities import Obstacle, Position


WIDTH = 10
HEIGHT = 10
MAX_ACTIONS = 180
SEED = 20260921

MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"

DIRECTIONS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
DIRECTION_ORDER = ["north", "east", "south", "west"]
PositionTuple = tuple[int, int]


def inside(p: PositionTuple) -> bool:
    return 0 <= p[0] < WIDTH and 0 <= p[1] < HEIGHT


def step_position(p: PositionTuple, direction: str) -> PositionTuple:
    dx, dy = DIRECTIONS[direction]
    return p[0] + dx, p[1] + dy


def direction_between(a: PositionTuple, b: PositionTuple) -> Optional[str]:
    delta = (b[0] - a[0], b[1] - a[1])
    for name, vector in DIRECTIONS.items():
        if delta == vector:
            return name
    return None


def distance(a: PositionTuple, b: PositionTuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def position_tuple(position) -> PositionTuple:
    return position.x, position.y


def set_position(world: World1, p: PositionTuple) -> None:
    position_type = type(world.organism.position)
    world.organism.position = position_type(x=p[0], y=p[1])


def shortest_route(
    start: PositionTuple,
    target: PositionTuple,
    obstacles: set[PositionTuple],
) -> list[PositionTuple]:
    """Evaluator-only BFS using the actual obstacle map."""
    if start in obstacles or target in obstacles:
        return []

    queue = deque([start])
    previous: dict[PositionTuple, Optional[PositionTuple]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            break

        for direction in DIRECTION_ORDER:
            neighbor = step_position(current, direction)
            if not inside(neighbor):
                continue
            if neighbor in obstacles or neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)

    if target not in previous:
        return []

    route = []
    current: Optional[PositionTuple] = target
    while current is not None:
        route.append(current)
        current = previous[current]
    route.reverse()
    return route


@dataclass
class ExperienceMemory:
    """
    Persistent experience across trials.

    Stores environmental transitions and counts of blocked cells.
    The current planner uses this cautiously: known blocked cells are
    reused only when the same layout signature is encountered.
    """

    blocked_by_layout: dict[frozenset[PositionTuple], set[PositionTuple]] = field(
        default_factory=dict
    )
    trial_records: list[dict] = field(default_factory=list)

    def get_blocked(self, layout: frozenset[PositionTuple]) -> set[PositionTuple]:
        return set(self.blocked_by_layout.get(layout, set()))

    def store_blocked(
        self,
        layout: frozenset[PositionTuple],
        blocked: set[PositionTuple],
    ) -> None:
        self.blocked_by_layout[layout] = set(blocked)

    def add_record(self, record: dict) -> None:
        self.trial_records.append(record)

    def count(self) -> int:
        return len(self.trial_records)


@dataclass
class SpatialModel:
    blocked: set[PositionTuple] = field(default_factory=set)
    visited: set[PositionTuple] = field(default_factory=set)
    observations: list[dict] = field(default_factory=list)

    def mark_visited(self, p: PositionTuple) -> None:
        self.visited.add(p)

    def mark_blocked(self, p: PositionTuple) -> None:
        if inside(p):
            self.blocked.add(p)

    def record(
        self,
        before: PositionTuple,
        action: str,
        after: PositionTuple,
        succeeded: bool,
    ) -> None:
        self.observations.append(
            {
                "before": before,
                "action": action,
                "after": after,
                "succeeded": succeeded,
            }
        )


def internal_route(
    start: PositionTuple,
    target: PositionTuple,
    known_blocked: set[PositionTuple],
) -> list[PositionTuple]:
    """Planner BFS with incomplete knowledge."""
    if start in known_blocked or target in known_blocked:
        return []

    queue = deque([start])
    previous: dict[PositionTuple, Optional[PositionTuple]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == target:
            break

        for direction in DIRECTION_ORDER:
            neighbor = step_position(current, direction)
            if not inside(neighbor):
                continue
            if neighbor in known_blocked or neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)

    if target not in previous:
        return []

    route = []
    current: Optional[PositionTuple] = target
    while current is not None:
        route.append(current)
        current = previous[current]
    route.reverse()
    return route


def choose_turn(current: str, desired: str) -> Optional[str]:
    current_i = DIRECTION_ORDER.index(current)
    desired_i = DIRECTION_ORDER.index(desired)

    clockwise = (desired_i - current_i) % 4
    counterclockwise = (current_i - desired_i) % 4

    if clockwise == 0:
        return None
    return TURN_RIGHT if clockwise <= counterclockwise else TURN_LEFT


@dataclass
class Scenario:
    name: str
    start: PositionTuple
    target: PositionTuple
    orientation: str
    obstacles: set[PositionTuple]
    phase: str


def random_scenario(rng: random.Random, index: int, phase: str) -> Scenario:
    start = (rng.randrange(WIDTH), rng.randrange(HEIGHT))
    target = (rng.randrange(WIDTH), rng.randrange(HEIGHT))

    while target == start:
        target = (rng.randrange(WIDTH), rng.randrange(HEIGHT))

    obstacles: set[PositionTuple] = set()
    density = 0.13 if phase == "training" else 0.20

    for x in range(WIDTH):
        for y in range(HEIGHT):
            p = (x, y)
            if p in (start, target):
                continue
            if rng.random() < density:
                obstacles.add(p)

    orientation = rng.choice(DIRECTION_ORDER)

    return Scenario(
        name=f"{phase}_{index:02d}",
        start=start,
        target=target,
        orientation=orientation,
        obstacles=obstacles,
        phase=phase,
    )


def configure_world(world: World1, scenario: Scenario) -> None:
    set_position(world, scenario.start)
    world.organism.orientation = scenario.orientation
    world.obstacles.clear()

    for x, y in sorted(scenario.obstacles):
        world.obstacles.append(Obstacle(position=Position(x=x, y=y)))


def run_trial(
    scenario: Scenario,
    memory: ExperienceMemory,
    use_transfer: bool,
) -> dict:
    world = World1()
    configure_world(world, scenario)

    actual = set(scenario.obstacles)
    layout = frozenset(actual)
    truth = shortest_route(scenario.start, scenario.target, actual)
    reachable = bool(truth)

    model = SpatialModel()

    if use_transfer:
        model.blocked.update(memory.get_blocked(layout))

    actions = 0
    successful_moves = 0
    failed_moves = 0
    turns = 0
    replans = 0
    internal_no_route = False

    while actions < MAX_ACTIONS:
        current = position_tuple(world.organism.position)

        if current == scenario.target:
            break

        model.mark_visited(current)
        route = internal_route(current, scenario.target, model.blocked)
        replans += 1

        if len(route) < 2:
            internal_no_route = True
            break

        next_cell = route[1]
        desired = direction_between(current, next_cell)

        if desired is None:
            internal_no_route = True
            break

        orientation = world.organism.orientation

        if orientation != desired:
            action = choose_turn(orientation, desired)
            if action is None:
                internal_no_route = True
                break

            before = position_tuple(world.organism.position)
            world.step(action)
            after = position_tuple(world.organism.position)
            model.record(before, action, after, True)

            actions += 1
            turns += 1
            continue

        before = position_tuple(world.organism.position)
        world.step(MOVE_FORWARD)
        after = position_tuple(world.organism.position)
        moved = after != before

        model.record(before, MOVE_FORWARD, after, moved)
        actions += 1

        if moved:
            successful_moves += 1
            model.mark_visited(after)
        else:
            failed_moves += 1
            attempted = step_position(before, desired)
            if inside(attempted):
                model.mark_blocked(attempted)

    final = position_tuple(world.organism.position)
    reached = final == scenario.target
    budget_exhausted = (
        actions >= MAX_ACTIONS and not reached and not internal_no_route
    )

    if reached and reachable:
        classification = "SUCCESS_SOLVABLE"
    elif internal_no_route and not reachable:
        classification = "CORRECTLY_DETECTED_UNREACHABLE"
    elif internal_no_route and reachable:
        classification = "FALSE_NO_ROUTE_MODEL_ERROR"
    elif budget_exhausted and reachable:
        classification = "FAILURE_SOLVABLE"
    elif budget_exhausted and not reachable:
        classification = "UNRESOLVED_UNREACHABLE"
    else:
        classification = "UNCLASSIFIED"

    discovered = set(model.blocked)
    true_discovered = discovered.intersection(actual)
    false_positives = discovered.difference(actual)
    false_negatives = actual.difference(discovered)

    precision = len(true_discovered) / len(discovered) if discovered else 1.0
    recall = len(true_discovered) / len(actual) if actual else 1.0

    shortest_length = len(truth) - 1 if truth else None
    efficiency = None
    if reached and shortest_length is not None and actions > 0:
        efficiency = shortest_length / actions

    record = {
        "scenario": scenario.name,
        "phase": scenario.phase,
        "start": scenario.start,
        "target": scenario.target,
        "reachable": reachable,
        "shortest_length": shortest_length,
        "reached": reached,
        "final": final,
        "classification": classification,
        "actions": actions,
        "successful_moves": successful_moves,
        "failed_moves": failed_moves,
        "turns": turns,
        "replans": replans,
        "known_blocked": sorted(discovered),
        "actual_obstacles": sorted(actual),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "precision": precision,
        "recall": recall,
        "efficiency": efficiency,
        "used_transfer": use_transfer,
    }

    memory.store_blocked(layout, discovered)
    memory.add_record(record)
    return record


def print_result(record: dict) -> None:
    print("\n" + "=" * 72)
    print(f"SCENARIO: {record['scenario']}")
    print("=" * 72)
    print(f"phase:                  {record['phase']}")
    print(f"start:                  {record['start']}")
    print(f"target:                 {record['target']}")
    print(f"reachable:              {record['reachable']}")
    print(f"shortest_length:        {record['shortest_length']}")
    print(f"final:                  {record['final']}")
    print(f"reached:                {record['reached']}")
    print(f"classification:         {record['classification']}")
    print(f"actions:                {record['actions']}")
    print(f"successful_moves:       {record['successful_moves']}")
    print(f"failed_moves:           {record['failed_moves']}")
    print(f"turns:                  {record['turns']}")
    print(f"replans:                {record['replans']}")
    print(f"known_blocked:          {record['known_blocked']}")
    print(f"false_positives:        {record['false_positives']}")
    print(f"false_negatives:        {record['false_negatives']}")
    print(f"precision:              {record['precision']:.2f}")
    print(f"recall:                 {record['recall']:.2f}")
    print(f"efficiency:             {record['efficiency']}")
    print(f"used_transfer:          {record['used_transfer']}")


def summarize(records: list[dict], title: str) -> None:
    print("\n" + "#" * 72)
    print(title)
    print("#" * 72)

    total = len(records)
    reachable = [r for r in records if r["reachable"]]
    unreachable = [r for r in records if not r["reachable"]]

    solved = [
        r for r in reachable if r["classification"] == "SUCCESS_SOLVABLE"
    ]
    detected = [
        r for r in unreachable
        if r["classification"] == "CORRECTLY_DETECTED_UNREACHABLE"
    ]

    print(f"trials:                         {total}")
    print(f"reachable:                      {len(reachable)}")
    print(f"unreachable:                    {len(unreachable)}")
    print(f"solved reachable:               {len(solved)}")
    print(f"detected unreachable:            {len(detected)}")

    if reachable:
        print(f"reachable success rate:          {100 * len(solved) / len(reachable):.1f}%")
    if unreachable:
        print(f"unreachable detection rate:      {100 * len(detected) / len(unreachable):.1f}%")

    print(f"total actions:                   {sum(r['actions'] for r in records)}")
    print(f"successful moves:                {sum(r['successful_moves'] for r in records)}")
    print(f"failed moves:                    {sum(r['failed_moves'] for r in records)}")
    print(f"turns:                           {sum(r['turns'] for r in records)}")

    efficiencies = [r["efficiency"] for r in records if r["efficiency"] is not None]
    if efficiencies:
        print(f"average route efficiency:        {sum(efficiencies) / len(efficiencies):.3f}")

    precisions = [r["precision"] for r in records]
    recalls = [r["recall"] for r in records]
    print(f"average model precision:         {sum(precisions) / len(precisions):.3f}")
    print(f"average model recall:            {sum(recalls) / len(recalls):.3f}")

    counts: dict[str, int] = {}
    for record in records:
        key = record["classification"]
        counts[key] = counts.get(key, 0) + 1

    print("\nclassifications:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def compare_phases(records: list[dict]) -> None:
    print("\n" + "#" * 72)
    print("TRANSFER COMPARISON")
    print("#" * 72)

    training = [r for r in records if r["phase"] == "training"]
    transfer = [r for r in records if r["phase"] == "transfer"]

    for name, group in (("training", training), ("transfer", transfer)):
        reachable = [r for r in group if r["reachable"]]
        solved = [
            r for r in reachable
            if r["classification"] == "SUCCESS_SOLVABLE"
        ]

        print(f"\n{name}:")
        print(f"  trials:                 {len(group)}")
        print(f"  reachable:              {len(reachable)}")
        print(f"  solved:                 {len(solved)}")

        if reachable:
            print(f"  success rate:           {100 * len(solved) / len(reachable):.1f}%")

        if group:
            print(f"  average actions:        {sum(r['actions'] for r in group) / len(group):.2f}")
            print(f"  average failed moves:   {sum(r['failed_moves'] for r in group) / len(group):.2f}")


def main() -> None:
    print("=" * 72)
    print("DIA ROUTE PLANNING V5")
    print("Randomized Environments + Transfer Evaluation")
    print(f"Random seed: {SEED}")
    print("=" * 72)

    rng = random.Random(SEED)
    memory = ExperienceMemory()
    records: list[dict] = []

    training_count = 12
    transfer_count = 12

    print("\nTRAINING PHASE")
    for index in range(1, training_count + 1):
        scenario = random_scenario(rng, index, "training")
        result = run_trial(scenario, memory, use_transfer=False)
        records.append(result)
        print_result(result)

    print("\nTRANSFER PHASE")
    for index in range(1, transfer_count + 1):
        scenario = random_scenario(rng, index, "transfer")
        result = run_trial(scenario, memory, use_transfer=True)
        records.append(result)
        print_result(result)

    summarize(records, "V5 AGGREGATE REPORT")
    compare_phases(records)

    print("\n" + "#" * 72)
    print("SCIENTIFIC CAUTION")
    print("#" * 72)
    print("The planner remains hand-designed.")
    print("Randomization tests robustness, not consciousness or general intelligence.")
    print("A fixed random seed makes this run reproducible.")
    print("To test genuine learning, compare against a no-memory baseline")
    print("using the same generated scenarios and report paired differences.")


if __name__ == "__main__":
    main()
