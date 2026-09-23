from dataclasses import dataclass
from organism.decision.scorer import DecisionScorer


@dataclass
class Decision:
    action: str
    reason: str
    score: float
    expected_energy: float
    expected_orientation: str


class DecisionEngine:
    ACTIONS = [
        "wait",
        "turn_left",
        "turn_right",
        "move_forward",
    ]

    DIRECTIONS = ["north", "east", "south", "west"]

    def __init__(self):
        self.scorer = DecisionScorer()

    def movement_delta(self, orientation, action):
        if action == "move_forward":
            direction = orientation

        elif action == "move_backward":
            opposites = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            }
            direction = opposites[orientation]

        else:
            return 0, 0

        return {
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0),
        }[direction]

    def predict(self, observation, action):
        energy = observation["energy"]
        orientation = observation["orientation"]

        energy -= 1

        if action == "turn_right":
            index = self.DIRECTIONS.index(orientation)
            orientation = self.DIRECTIONS[
                (index + 1) % len(self.DIRECTIONS)
            ]

        elif action == "turn_left":
            index = self.DIRECTIONS.index(orientation)
            orientation = self.DIRECTIONS[
                (index - 1) % len(self.DIRECTIONS)
            ]

        movement_succeeded = False
        predicted_position = {"relative_x": 0, "relative_y": 0}
        resource_detected = False

        if action in ("move_forward", "move_backward"):
            dx, dy = self.movement_delta(orientation, action)

            obstacle_ahead = any(
                obstacle["relative_x"] == dx
                and obstacle["relative_y"] == dy
                for obstacle in observation.get("obstacles", [])
            )

            if not obstacle_ahead:
                movement_succeeded = True
                predicted_position = {
                    "relative_x": dx,
                    "relative_y": dy,
                }

                for resource in observation.get("resources", []):
                    if (
                        resource["relative_x"] == dx
                        and resource["relative_y"] == dy
                    ):
                        resource_detected = True
                        break

        return {
            "energy": energy,
            "orientation": orientation,
            "movement_succeeded": movement_succeeded,
            "predicted_position": predicted_position,
            "resource_detected": resource_detected,
        }

    def resource_value(self, observation):
        resources = observation.get("resources", [])

        if not resources:
            return 0.0

        nearest_distance = min(
            abs(resource["relative_x"]) + abs(resource["relative_y"])
            for resource in resources
        )

        if nearest_distance == 0:
            return 1.0

        return max(
            0.0,
            1.0 - (nearest_distance / 4.0),
        )

    def choose(self, observation):
        candidates = []

        resource_value = self.resource_value(observation)

        for action in self.ACTIONS:
            prediction = self.predict(observation, action)

            score = self.scorer.score(
                predicted_energy=prediction["energy"],
                exploration_value=0.2,
                resource_value=resource_value,
            )

            candidates.append(
                (
                    score,
                    action,
                    prediction,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item[0],
                self.ACTIONS.index(item[1]),
            )
        )

        score, action, prediction = candidates[0]

        return Decision(
            action=action,
            reason="selected_highest_scoring_predicted_outcome",
            score=score,
            expected_energy=prediction["energy"],
            expected_orientation=prediction["orientation"],
        )
