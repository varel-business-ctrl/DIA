from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ExperimentResult:
    experiment_id: str
    outcome: str
    measurements: dict
    success: bool
    created_at: str


class ExperimentResultStore:
    def __init__(self):
        self.results = []

    def record(
        self,
        experiment_id,
        outcome,
        measurements,
        success,
    ):
        result = ExperimentResult(
            experiment_id=experiment_id,
            outcome=outcome,
            measurements=measurements,
            success=bool(success),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.results.append(result)
        return result

    def for_experiment(self, experiment_id):
        return [
            result
            for result in self.results
            if result.experiment_id == experiment_id
        ]

    def latest(self):
        if not self.results:
            return None

        return self.results[-1]

    def size(self):
        return len(self.results)
