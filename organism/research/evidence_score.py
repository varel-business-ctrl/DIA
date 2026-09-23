class EvidenceScorer:
    def score(
        self,
        prediction_error_before,
        prediction_error_after,
        trials=1,
        control_quality=0.5,
        measurement_quality=0.5,
    ):
        trials = max(1, trials)
        control_quality = max(0.0, min(1.0, control_quality))
        measurement_quality = max(0.0, min(1.0, measurement_quality))

        before = max(0.0, prediction_error_before)
        after = max(0.0, prediction_error_after)

        if before == 0:
            improvement = 0.0
        else:
            improvement = max(0.0, min(1.0, (before - after) / before))

        repetition = min(1.0, trials / 10)

        score = (
            improvement * 0.4
            + repetition * 0.2
            + control_quality * 0.2
            + measurement_quality * 0.2
        )

        return round(score, 3)
