class ResearchPriority:
    def calculate(
        self,
        uncertainty=0.5,
        importance=0.5,
        information_gain=0.5,
        knowledge_gap=0.5,
    ):
        values = [
            uncertainty,
            importance,
            information_gain,
            knowledge_gap,
        ]

        values = [
            max(0.0, min(1.0, value))
            for value in values
        ]

        priority = sum(values) / len(values)

        return round(priority, 3)
