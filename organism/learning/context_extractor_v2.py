class ContextExtractorV2:
    """
    Extracts an action-relevant context from enhanced perception.

    Priority:
    1. Boundary ahead
    2. Obstacle ahead
    3. Clear or currently unknown path

    This is a designed baseline, not autonomous concept discovery.
    """

    def extract(self, observation):
        if observation.get("boundary_ahead", False):
            return "boundary_ahead"

        if observation.get("obstacle_ahead", False):
            return "obstacle_ahead"

        return "clear_or_unknown_path"
