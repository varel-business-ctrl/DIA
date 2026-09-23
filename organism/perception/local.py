from dataclasses import dataclass


@dataclass
class LocalPerception:
    energy: float
    orientation: str
    resources: list
    obstacles: list


class PerceptionSystem:
    def __init__(self, radius=2):
        self.radius = radius

    def perceive(self, world_state):
        organism = world_state["organism"]

        ox = organism["x"]
        oy = organism["y"]

        nearby_resources = []
        nearby_obstacles = []

        for resource in world_state.get("resources", []):
            dx = resource["x"] - ox
            dy = resource["y"] - oy

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                nearby_resources.append(
                    {
                        "relative_x": dx,
                        "relative_y": dy,
                        "energy": resource["energy"],
                    }
                )

        for obstacle in world_state.get("obstacles", []):
            dx = obstacle["x"] - ox
            dy = obstacle["y"] - oy

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                nearby_obstacles.append(
                    {
                        "relative_x": dx,
                        "relative_y": dy,
                    }
                )

        return {
            "energy": organism["energy"],
            "orientation": organism["orientation"],
            "resources": nearby_resources,
            "obstacles": nearby_obstacles,
        }
