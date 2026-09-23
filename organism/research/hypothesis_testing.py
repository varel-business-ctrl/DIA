from organism.research.hypothesis import HypothesisStore
from organism.research.experiment import ExperimentStore


class HypothesisTestingSystem:
    def __init__(self):
        self.hypotheses = HypothesisStore()
        self.experiments = ExperimentStore()

    def test(
        self,
        hypothesis_id,
        prediction,
        observation,
        conclusion,
        support=0.5,
    ):
        result = self.experiments.record(
            experiment_id=f"EXP-{self.experiments.size() + 1:04d}",
            hypothesis_id=hypothesis_id,
            prediction=prediction,
            observation=observation,
            conclusion=conclusion,
            support=support,
        )

        return result

    def evidence_for(self, hypothesis_id):
        return self.experiments.for_hypothesis(hypothesis_id)

    def size(self):
        return self.experiments.size()
