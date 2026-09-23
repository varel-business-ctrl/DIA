from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ResearchCycle:
    cycle_id: str
    question_id: str
    hypothesis_ids: list
    experiment_ids: list
    result_ids: list
    conclusion: str
    created_at: str


class ResearchCycleStore:
    def __init__(self):
        self.cycles = []

    def record(
        self,
        cycle_id,
        question_id,
        hypothesis_ids,
        experiment_ids,
        result_ids,
        conclusion,
    ):
        cycle = ResearchCycle(
            cycle_id=cycle_id,
            question_id=question_id,
            hypothesis_ids=hypothesis_ids,
            experiment_ids=experiment_ids,
            result_ids=result_ids,
            conclusion=conclusion,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.cycles.append(cycle)
        return cycle

    def latest(self):
        if not self.cycles:
            return None

        return self.cycles[-1]

    def size(self):
        return len(self.cycles)
