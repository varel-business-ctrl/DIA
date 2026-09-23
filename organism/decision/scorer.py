class DecisionScorer:

    def score(
        self,
        predicted_energy,
        exploration_value=0.0,
        resource_value=0.0,
    ):
        energy_value = max(0.0, min(1.0, predicted_energy / 100.0))
        exploration_value = max(0.0, min(1.0, exploration_value))
        resource_value = max(0.0, min(1.0, resource_value))

        return round(
            energy_value * 0.4
            + exploration_value * 0.2
            + resource_value * 0.4,
            3,
        )
