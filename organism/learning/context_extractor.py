class ContextExtractor:
    """
    Extracts a basic action-relevant context from local perception.

    This is a hand-designed baseline, not autonomous discovery.
    The context is inferred from locally observed obstacles.
    """

    DIRECTIONS = {
        "north": (0, 1),
        "east": (1, 0),
        "south": (0, -1),
        "west": (-1, 0),
    }

    def extract(self, observation):
        orientation = observation.get("orientation", "north")

        dx, dy = self.DIRECTIONS[orientation]

        obstacles = observation.get("obstacles", [])

        obstacle_ahead = any(
            obstacle.get("relative_x") == dx
            and obstacle.get("relative_y") == dy
            for obstacle in obstacles
        )

        if obstacle_ahead:
            return "obstacle_ahead"

        return "clear_or_unknown_path"
