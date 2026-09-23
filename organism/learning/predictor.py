class TransitionPredictor:
    def __init__(self):
        self.model = {}

    def _state_key(self, observation):
        return (
            observation.get("orientation"),
            observation.get("energy"),
            tuple(
                sorted(
                    (
                        item["relative_x"],
                        item["relative_y"],
                    )
                    for item in observation.get("resources", [])
                )
            ),
            tuple(
                sorted(
                    (
                        item["relative_x"],
                        item["relative_y"],
                    )
                    for item in observation.get("obstacles", [])
                )
            ),
        )

    def predict(self, observation, action):
        key = (
            self._state_key(observation),
            action,
        )

        return self.model.get(key)

    def learn(
        self,
        observation,
        action,
        next_observation,
    ):
        key = (
            self._state_key(observation),
            action,
        )

        self.model[key] = next_observation.copy()

    def size(self):
        return len(self.model)
