from dataclasses import dataclass
from typing import Any


@dataclass
class Experience:
    observation: dict[str, Any]
    action: str
    next_observation: dict[str, Any]
    consequence: dict[str, Any]


class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.experiences = []

    def store(
        self,
        observation,
        action,
        next_observation,
        consequence,
    ):
        experience = Experience(
            observation=observation,
            action=action,
            next_observation=next_observation,
            consequence=consequence,
        )

        self.experiences.append(experience)

        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def recall_recent(self, count=5):
        return self.experiences[-count:]

    def size(self):
        return len(self.experiences)
