class ResearchQuestionGenerator:
    def generate(
        self,
        topic,
        uncertainty,
        importance,
        information_gain,
        knowledge_gap,
    ):
        uncertainty = max(0.0, min(1.0, uncertainty))
        importance = max(0.0, min(1.0, importance))
        information_gain = max(0.0, min(1.0, information_gain))
        knowledge_gap = max(0.0, min(1.0, knowledge_gap))

        priority = round(
            (
                uncertainty
                + importance
                + information_gain
                + knowledge_gap
            ) / 4,
            3,
        )

        question = (
            f"What should DIA investigate about {topic} "
            f"to reduce its uncertainty?"
        )

        return {
            "topic": topic,
            "question": question,
            "priority": priority,
            "uncertainty": uncertainty,
            "importance": importance,
            "information_gain": information_gain,
            "knowledge_gap": knowledge_gap,
        }
