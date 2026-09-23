from organism.learning.predictor import TransitionPredictor
from organism.learning.error import prediction_error


class Learner:
    def __init__(self):
        self.predictor = TransitionPredictor()
        self.total_errors = 0
        self.learning_events = 0

    def process(
        self,
        observation,
        action,
        actual_next_observation,
    ):
        prediction = self.predictor.predict(
            observation,
            action,
        )

        error = prediction_error(
            prediction,
            actual_next_observation,
        )

        if error is not None:
            self.total_errors += error["total_differences"]

        self.predictor.learn(
            observation,
            action,
            actual_next_observation,
        )

        self.learning_events += 1

        return {
            "prediction": prediction,
            "actual": actual_next_observation,
            "error": error,
            "model_size": self.predictor.size(),
            "total_errors": self.total_errors,
            "learning_events": self.learning_events,
        }
