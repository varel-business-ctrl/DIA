from .state import OrganismInternalState
from organism.memory.episodic import EpisodicMemory


class Organism:
    def __init__(self, organism_id="O-001"):
        self.organism_id = organism_id
        self.state = OrganismInternalState()
        self.memory = EpisodicMemory()

    def observe(self, observation, advance_age=False):
        self.state.previous_observation = (
            self.state.current_observation.copy()
        )

        self.state.current_observation = observation.copy()

        self.state.energy = observation.get(
            "energy",
            self.state.energy,
        )

        self.state.orientation = observation.get(
            "orientation",
            self.state.orientation,
        )

        if advance_age:
            self.state.age += 1

    def remember(self, action, next_observation, consequence):
        previous_observation = self.state.previous_observation.copy()

        self.memory.store(
            previous_observation,
            action,
            next_observation.copy(),
            consequence,
        )

        self.state.last_action = action
        self.state.experiences += 1

    def status(self):
        return {
            "id": self.organism_id,
            "age": self.state.age,
            "energy": self.state.energy,
            "orientation": self.state.orientation,
            "experiences": self.state.experiences,
            "memory_size": self.memory.size(),
        }
