from organism.core.organism import Organism
from organism.learning.learner import Learner
from environment.world1.perception import LocalPerception


class OrganismRuntime:
    def __init__(self, world):
        self.world = world
        self.organism = Organism("O-001")
        self.perception = LocalPerception(radius=2)
        self.learner = Learner()

    def step(self, action):
        observation = self.perception.perceive(self.world)

        self.organism.observe(
            observation,
            advance_age=True,
        )

        result = self.world.step(action)

        next_observation = self.perception.perceive(self.world)

        consequence = {
            "energy_change": (
                next_observation["energy"]
                - observation["energy"]
            ),
            "position_changed": (
                observation.get("position")
                != next_observation.get("position")
            ),
        }

        self.organism.memory.store(
            observation,
            action,
            next_observation,
            consequence,
        )

        learning = self.learner.process(
            observation,
            action,
            next_observation,
        )

        self.organism.state.last_action = action
        self.organism.state.experiences += 1

        if learning["error"] is not None:
            self.organism.state.prediction_error = (
                learning["error"]["total_differences"]
            )

        self.organism.observe(next_observation)

        return {
            "organism": self.organism.status(),
            "observation": observation,
            "action": action,
            "result": result,
            "next_observation": next_observation,
            "learning": learning,
        }
