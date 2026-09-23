class LocalPerception:
    def __init__(self, radius=2):
        self.radius = radius

    def perceive(self, world):
        organism = world.organism
        ox = organism.position.x
        oy = organism.position.y

        visible_resources = []
        for resource in world.resources:
            dx = resource.position.x - ox
            dy = resource.position.y - oy

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                visible_resources.append({
                    "relative_x": dx,
                    "relative_y": dy,
                    "energy": resource.energy,
                })

        visible_obstacles = []
        for obstacle in world.obstacles:
            dx = obstacle.position.x - ox
            dy = obstacle.position.y - oy

            if abs(dx) <= self.radius and abs(dy) <= self.radius:
                visible_obstacles.append({
                    "relative_x": dx,
                    "relative_y": dy,
                })

        return {
            "orientation": organism.orientation,
            "energy": organism.energy,
            "resources": visible_resources,
            "obstacles": visible_obstacles,
        }
