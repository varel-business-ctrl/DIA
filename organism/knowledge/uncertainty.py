class UncertaintyEvaluator:
    def evaluate(self, evidence_items):
        if not evidence_items:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "evidence_count": 0,
            }

        average_confidence = sum(
            item.confidence
            for item in evidence_items
        ) / len(evidence_items)

        sources = set(
            item.source
            for item in evidence_items
        )

        if average_confidence >= 0.8:
            status = "strongly_supported"
        elif average_confidence >= 0.6:
            status = "supported"
        elif average_confidence >= 0.4:
            status = "uncertain"
        else:
            status = "weak_evidence"

        return {
            "status": status,
            "confidence": round(
                average_confidence,
                3,
            ),
            "evidence_count": len(evidence_items),
            "source_count": len(sources),
        }
