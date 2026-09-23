class BeliefUpdater:
    def update(self, current_confidence, evidence_score, supports=True):
        current_confidence = max(0.0, min(1.0, current_confidence))
        evidence_score = max(0.0, min(1.0, evidence_score))

        if supports:
            new_confidence = current_confidence + (
                (1.0 - current_confidence) * evidence_score * 0.5
            )
        else:
            new_confidence = current_confidence - (
                current_confidence * evidence_score * 0.5
            )

        return round(max(0.0, min(1.0, new_confidence)), 3)

    def status(self, confidence):
        if confidence >= 0.8:
            return "strong"
        if confidence >= 0.6:
            return "supported"
        if confidence >= 0.4:
            return "uncertain"
        if confidence >= 0.2:
            return "weak"
        return "rejected"
