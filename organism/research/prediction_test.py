class PredictionTest:
    def compare(self, prediction, observation):
        differences = {}

        keys = set(prediction.keys()) | set(observation.keys())

        for key in keys:
            predicted = prediction.get(key)
            actual = observation.get(key)

            if predicted != actual:
                differences[key] = {
                    "predicted": predicted,
                    "actual": actual,
                }

        return {
            "matches": len(differences) == 0,
            "difference_count": len(differences),
            "differences": differences,
        }

    def error_score(self, comparison):
        total = comparison["difference_count"]

        if total == 0:
            return 0.0

        return round(min(1.0, total / 10), 3)
