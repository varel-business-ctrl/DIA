from dataclasses import dataclass

from .config import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    INITIAL_ENERGY,
    RESOURCE_ENERGY,
    MAX_STEPS,
)

from .entities import Position, Resource, Obstacle


@dataclass
class OrganismState:
    position: Position
    energy: int
    orientation: str = "north"


class World1:
    def __init__(self):
        self.width = WORLD_WIDTH
        self.height = WORLD_HEIGHT
        self.max_steps = MAX_STEPS

        self.organism = OrganismState(
            position=Position(1, 1),
            energy=INITIAL_ENERGY,
            orientation="north",
        )

        self.resources = [
            Resource(
                position=Position(7, 7),
                energy=RESOURCE_ENERGY,
            )
        ]

        self.obstacles = [
            Obstacle(position=Position(4, 4)),
            Obstacle(position=Position(4, 5)),
            Obstacle(position=Position(5, 4)),
        ]

        self.step_count = 0

    def reset(self):
        self.__init__()

    def inside_world(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def blocked(self, x, y):
        return any(
            obstacle.position.x == x
            and obstacle.position.y == y
            for obstacle in self.obstacles
        )

    def observe(self):
        return {
            "organism": {
                "x": self.organism.position.x,
                "y": self.organism.position.y,
                "energy": self.organism.energy,
                "orientation": self.organism.orientation,
            },
            "resources": [
                {
                    "x": resource.position.x,
                    "y": resource.position.y,
                    "energy": resource.energy,
                }
                for resource in self.resources
            ],
            "obstacles": [
                {
                    "x": obstacle.position.x,
                    "y": obstacle.position.y,
                }
                for obstacle in self.obstacles
            ],
            "step": self.step_count,
        }

    def step(self, action):
        self.step_count += 1

        old_position = Position(
            self.organism.position.x,
            self.organism.position.y,
        )

        if action == "turn_left":
            directions = ["north", "west", "south", "east"]
            current = directions.index(self.organism.orientation)
            self.organism.orientation = directions[(current + 1) % 4]

        elif action == "turn_right":
            directions = ["north", "east", "south", "west"]
            current = directions.index(self.organism.orientation)
            self.organism.orientation = directions[(current + 1) % 4]

        elif action in ("move_forward", "move_backward"):
            direction = self.organism.orientation

            if action == "move_backward":
                opposites = {
                    "north": "south",
                    "south": "north",
                    "east": "west",
                    "west": "east",
                }
                direction = opposites[direction]

            dx, dy = {
                "north": (0, 1),
                "south": (0, -1),
                "east": (1, 0),
                "west": (-1, 0),
            }[direction]

            new_x = old_position.x + dx
            new_y = old_position.y + dy

            if self.inside_world(new_x, new_y):
                if not self.blocked(new_x, new_y):
                    self.organism.position = Position(
                        new_x,
                        new_y,
                    )

        elif action == "wait":
            pass

        elif action == "interact":
            self.collect_resource()

        self.organism.energy -= 1

        done = (
            self.organism.energy <= 0
            or self.step_count >= self.max_steps
        )

        return {
            "observation": self.observe(),
            "previous_position": {
                "x": old_position.x,
                "y": old_position.y,
            },
            "action": action,
            "done": done,
        }

    def collect_resource(self):
        current = self.organism.position

        for resource in self.resources[:]:
            if (
                resource.position.x == current.x
                and resource.position.y == current.y
            ):
                self.organism.energy += resource.energy
                self.resources.remove(resource)
