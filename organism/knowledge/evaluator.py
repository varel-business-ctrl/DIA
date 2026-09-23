class KnowledgeEvaluator:
    def evaluate(self, items):
        if not items:
            return {
                "status": "no_evidence",
                "confidence": 0.0,
                "sources": [],
            }

        sources = list({
            item.source
            for item in items
        })

        confidence = sum(
            item.confidence
            for item in items
        ) / len(items)

        if len(sources) >= 2:
            confidence = min(
                1.0,
                confidence + 0.1
            )

        return {
            "status": "supported",
            "confidence": round(
                confidence,
                3
            ),
            "sources": sources,
            "evidence_count": len(items),
        }
