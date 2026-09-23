from environment.world1 import World1
from environment.world1.entities import Position
from environment.world1.perception import LocalPerception

from organism.learning.context_effect_model import ContextEffectModel
from organism.learning.context_extractor import ContextExtractor


class HeldOutGeneralizationExperiment:
    def __init__(self):
        self.world = World1()
        self.perception = LocalPerception(radius=2)
        self.extractor = ContextExtractor()
        self.model = ContextEffectModel()

        self.predictions = 0
        self.correct = 0
        self.unknown = 0

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

    def set_position(self, x, y, orientation):
        self.world.organism.position = Position(x, y)
        self.world.organism.orientation = orientation
        self.world.organism.energy = 100

    def observe_context(self):
        observation = self.perception.perceive(self.world)
        context = self.extractor.extract(observation)
        return observation, context

    def train(self, name, x, y, orientation, action):
        self.set_position(x, y, orientation)

        before_world = self.world.observe()
        before, context = self.observe_context()

        self.world.step(action)

        after_world = self.world.observe()
        after = self.perception.perceive(self.world)

        actual_success = self.movement_changed(
            before_world,
            after_world,
        )

        self.model.learn(
            context=context,
            action=action,
            energy_before=before["energy"],
            energy_after=after["energy"],
            orientation_before=before["orientation"],
            orientation_after=after["orientation"],
            movement_succeeded=actual_success,
        )

        print(
            "TRAIN:",
            name,
            "| context=", context,
            "| success=", actual_success,
        )

    def test(self, name, x, y, orientation, action):
        self.set_position(x, y, orientation)

        before_world = self.world.observe()
        before, context = self.observe_context()

        prediction = self.model.predict(
            context=context,
            action=action,
            energy_before=before["energy"],
        )

        if prediction["known"]:
            predicted_rate = prediction["prediction"][
                "movement_success_rate"
            ]

            predicted_success = predicted_rate >= 0.5
        else:
            predicted_success = None
            self.unknown += 1

        self.world.step(action)

        after_world = self.world.observe()

        actual_success = self.movement_changed(
            before_world,
            after_world,
        )

        if predicted_success is not None:
            self.predictions += 1

            if predicted_success == actual_success:
                self.correct += 1

        print(
            "TEST:",
            name,
            "| context=", context,
            "| predicted=", predicted_success,
            "| actual=", actual_success,
            "| known=", prediction["known"],
        )

    def accuracy(self):
        if self.predictions == 0:
            return None

        return self.correct / self.predictions

    def run(self):
        print("\n=== TRAINING PHASE ===")

        self.train(
            name="training clear location",
            x=1,
            y=1,
            orientation="east",
            action="move_forward",
        )

        self.train(
            name="training obstacle location",
            x=3,
            y=4,
            orientation="east",
            action="move_forward",
        )

        print("\n=== HELD-OUT TEST PHASE ===")

        self.test(
            name="new clear location",
            x=1,
            y=2,
            orientation="east",
            action="move_forward",
        )

        self.test(
            name="new obstacle location",
            x=4,
            y=3,
            orientation="north",
            action="move_forward",
        )

        print("\n=== GENERALIZATION RESULTS ===")

        print("Known predictions:", self.predictions)
        print("Unknown predictions:", self.unknown)
        print("Correct predictions:", self.correct)
        print("Accuracy:", self.accuracy())

        print("\n=== FINAL MODEL SUMMARY ===")
        print(self.model.summary())


if __name__ == "__main__":
    experiment = HeldOutGeneralizationExperiment()
    experiment.run()
