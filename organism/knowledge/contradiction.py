class ContradictionDetector:
    def detect(self, items):
        contradictions = []

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                first = items[i]
                second = items[j]

                if first.topic != second.topic:
                    continue

                first_text = first.content.lower()
                second_text = second.content.lower()

                negations = [
                    ("is", "is not"),
                    ("are", "are not"),
                    ("can", "cannot"),
                    ("does", "does not"),
                    ("true", "false"),
                ]

                for positive, negative in negations:
                    if (
                        positive in first_text
                        and negative in second_text
                    ) or (
                        negative in first_text
                        and positive in second_text
                    ):
                        contradictions.append({
                            "topic": first.topic,
                            "source_a": first.source,
                            "source_b": second.source,
                        })
                        break

        return contradictions
