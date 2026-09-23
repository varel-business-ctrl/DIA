from dataclasses import dataclass


@dataclass
class EnhancedLocalPerception:
    """
    Enhanced local perception for DIA.

    Reports information available from the immediate environment,
    including whether the organism is facing a world boundary.

    This is an experimental baseline. It does not provide the
    organism with unrestricted access to the complete world state.
    """

    radius: int = 2

    DIRECTIONS = {
        "north": (0, 1),
        "east": (1, 0),
        "south": (0, -1),
        "west": (-1, 0),
    }

    def perceive(self, world):
        organism = world.organism

        x = organism.position.x
        y = organism.position.y
        orientation = organism.orientation

        visible_resources = []
        visible_obstacles = []

        for resource in world.resources:
            dx = resource.position.x - x
            dy = resource.position.y - y

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                visible_resources.append(
                    {
                        "relative_x": dx,
                        "relative_y": dy,
                        "energy": resource.energy,
                    }
                )

        for obstacle in world.obstacles:
            dx = obstacle.position.x - x
            dy = obstacle.position.y - y

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                visible_obstacles.append(
                    {
                        "relative_x": dx,
                        "relative_y": dy,
                    }
                )

        dx, dy = self.DIRECTIONS[orientation]

        target_x = x + dx
        target_y = y + dy

        boundary_ahead = not world.inside_world(
            target_x,
            target_y,
        )

        obstacle_ahead = any(
            obstacle["relative_x"] == dx
            and obstacle["relative_y"] == dy
            for obstacle in visible_obstacles
        )

        return {
            "energy": organism.energy,
            "orientation": orientation,
            "resources": visible_resources,
            "obstacles": visible_obstacles,
            "boundary_ahead": boundary_ahead,
            "obstacle_ahead": obstacle_ahead,
        }
