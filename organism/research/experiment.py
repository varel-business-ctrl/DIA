from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Experiment:
    experiment_id: str
    hypothesis_id: str
    prediction: str
    observation: str
    conclusion: str
    support: float
    created_at: str


class ExperimentStore:
    def __init__(self):
        self.experiments = []

    def record(
        self,
        experiment_id,
        hypothesis_id,
        prediction,
        observation,
        conclusion,
        support=0.5,
    ):
        support = max(0.0, min(1.0, support))

        item = Experiment(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            prediction=prediction,
            observation=observation,
            conclusion=conclusion,
            support=support,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.experiments.append(item)
        return item

    def for_hypothesis(self, hypothesis_id):
        return [
            item
            for item in self.experiments
            if item.hypothesis_id == hypothesis_id
        ]

    def latest(self):
        if not self.experiments:
            return None

        return self.experiments[-1]

    def size(self):
        return len(self.experiments)
