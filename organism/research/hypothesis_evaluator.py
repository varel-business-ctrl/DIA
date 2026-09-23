class HypothesisEvaluator:
    def evaluate(self, tests):
        if not tests:
            return {
                "status": "untested",
                "confidence": 0.0,
                "test_count": 0,
            }

        average_support = sum(
            test.support for test in tests
        ) / len(tests)

        if average_support >= 0.8:
            status = "strongly_supported"
        elif average_support >= 0.6:
            status = "supported"
        elif average_support >= 0.4:
            status = "uncertain"
        else:
            status = "weakly_supported"

        return {
            "status": status,
            "confidence": round(average_support, 3),
            "test_count": len(tests),
        }
