from organism.research.hypothesis import HypothesisStore
from organism.research.experiment_generator import ExperimentGenerator
from organism.research.experiment_result import ExperimentResultStore
from organism.research.hypothesis_evaluator import HypothesisEvaluator


class ResearchLoop:
    def __init__(self):
        self.hypotheses = HypothesisStore()
        self.generator = ExperimentGenerator()
        self.results = ExperimentResultStore()
        self.evaluator = HypothesisEvaluator()

    def create_hypothesis(
        self,
        hypothesis_id,
        question_id,
        statement,
        confidence=0.5,
    ):
        return self.hypotheses.add(
            hypothesis_id,
            question_id,
            statement,
            confidence,
        )

    def generate_experiment(
        self,
        hypothesis_id,
        objective,
        prediction,
        variables,
        actions,
        success_criteria,
    ):
        return self.generator.generate(
            hypothesis_id,
            objective,
            prediction,
            variables,
            actions,
            success_criteria,
        )

    def record_result(
        self,
        experiment_id,
        outcome,
        measurements,
        success,
    ):
        return self.results.record(
            experiment_id,
            outcome,
            measurements,
            success,
        )

    def evaluate_hypothesis(self, hypothesis_id):
        tests = self.results.for_experiment(
            self.generator.latest().experiment_id
        )

        if not tests:
            return {
                "status": "no_result",
                "confidence": 0.0,
            }

        class TestAdapter:
            def __init__(self, result):
                self.support = 1.0 if result.success else 0.0

        adapted = [TestAdapter(result) for result in tests]

        return self.evaluator.evaluate(adapted)
