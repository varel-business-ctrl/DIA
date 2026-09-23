from environment.world1 import World1
from environment.world1.entities import Position
from environment.world1.perception import LocalPerception

from organism.learning.context_effect_model import ContextEffectModel
from organism.learning.context_extractor import ContextExtractor


class ClosedLoopLearningExperiment:
    def __init__(self):
        self.world = World1()
        self.perception = LocalPerception(radius=2)
        self.extractor = ContextExtractor()
        self.model = ContextEffectModel()

        self.total_predictions = 0
        self.correct_predictions = 0
        self.unknown_predictions = 0

    def movement_changed(self, before, after):
        before_position = (
            before["organism"]["x"],
            before["organism"]["y"],
        )

        after_position = (
            after["organism"]["x"],
            after["organism"]["y"],
        )

        return before_position != after_position

    def prepare_situation(self, situation):
        self.world.organism.energy = 100

        if situation == "clear_path":
            self.world.organism.position = Position(1, 1)
            self.world.organism.orientation = "east"

        elif situation == "obstacle_ahead":
            self.world.organism.position = Position(3, 4)
            self.world.organism.orientation = "east"

    def run_trial(self, situation, action="move_forward"):
        self.prepare_situation(situation)

        before_world = self.world.observe()
        before_perception = self.perception.perceive(self.world)

        context = self.extractor.extract(before_perception)

        prediction = self.model.predict(
            context=context,
            action=action,
            energy_before=before_perception["energy"],
        )

        if prediction["known"]:
            predicted_success_rate = prediction["prediction"][
                "movement_success_rate"
            ]

            predicted_success = predicted_success_rate >= 0.5

        else:
            predicted_success = None
            self.unknown_predictions += 1

        self.world.step(action)

        after_world = self.world.observe()
        after_perception = self.perception.perceive(self.world)

        actual_success = self.movement_changed(
            before_world,
            after_world,
        )

        if predicted_success is not None:
            self.total_predictions += 1

            if predicted_success == actual_success:
                self.correct_predictions += 1

        self.model.learn(
            context=context,
            action=action,
            energy_before=before_perception["energy"],
            energy_after=after_perception["energy"],
            orientation_before=before_perception["orientation"],
            orientation_after=after_perception["orientation"],
            movement_succeeded=actual_success,
        )

        print("\nTRIAL")
        print("Situation:", situation)
        print("Context:", context)
        print("Prediction known:", prediction["known"])
        print("Predicted movement:", predicted_success)
        print("Actual movement:", actual_success)
        print(
            "Energy:",
            before_perception["energy"],
            "->",
            after_perception["energy"],
        )

    def accuracy(self):
        if self.total_predictions == 0:
            return None

        return self.correct_predictions / self.total_predictions

    def run(self):
        situations = [
            "clear_path",
            "obstacle_ahead",
            "clear_path",
            "obstacle_ahead",
            "clear_path",
            "obstacle_ahead",
            "clear_path",
            "obstacle_ahead",
        ]

        print("\n=== CLOSED-LOOP LEARNING ===")

        for situation in situations:
            self.run_trial(situation)

        print("\n=== LEARNING RESULTS ===")
        print("Total stored effects:", self.model.size())
        print("Total observations:", self.model.total_observations())
        print("Known contexts:", self.model.known_contexts())
        print("Known actions:", self.model.known_actions())

        print("\n=== PREDICTION METRICS ===")
        print("Known predictions:", self.total_predictions)
        print("Unknown predictions:", self.unknown_predictions)
        print("Correct predictions:", self.correct_predictions)
        print("Prediction accuracy:", self.accuracy())

        print("\n=== FINAL CLEAR PATH PREDICTION ===")
        print(
            self.model.predict(
                context="clear_or_unknown_path",
                action="move_forward",
                energy_before=80,
            )
        )

        print("\n=== FINAL OBSTACLE PREDICTION ===")
        print(
            self.model.predict(
                context="obstacle_ahead",
                action="move_forward",
                energy_before=80,
            )
        )

        print("\n=== FINAL MODEL SUMMARY ===")
        print(self.model.summary())


if __name__ == "__main__":
    experiment = ClosedLoopLearningExperiment()
    experiment.run()
