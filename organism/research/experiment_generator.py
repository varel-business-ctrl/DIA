from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class GeneratedExperiment:
    experiment_id: str
    hypothesis_id: str
    objective: str
    prediction: str
    variables: list
    actions: list
    success_criteria: str
    created_at: str


class ExperimentGenerator:
    def __init__(self):
        self.experiments = []

    def generate(
        self,
        hypothesis_id,
        objective,
        prediction,
        variables,
        actions,
        success_criteria,
    ):
        experiment_id = f"GEN-EXP-{len(self.experiments) + 1:04d}"

        experiment = GeneratedExperiment(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            objective=objective,
            prediction=prediction,
            variables=variables,
            actions=actions,
            success_criteria=success_criteria,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.experiments.append(experiment)
        return experiment

    def latest(self):
        if not self.experiments:
            return None

        return self.experiments[-1]

    def size(self):
        return len(self.experiments)
