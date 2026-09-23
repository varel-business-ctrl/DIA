from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Hypothesis:
    hypothesis_id: str
    question_id: str
    statement: str
    confidence: float
    status: str
    created_at: str


class HypothesisStore:
    def __init__(self):
        self.hypotheses = []

    def add(
        self,
        hypothesis_id,
        question_id,
        statement,
        confidence=0.5,
        status="untested",
    ):
        confidence = max(0.0, min(1.0, confidence))

        item = Hypothesis(
            hypothesis_id=hypothesis_id,
            question_id=question_id,
            statement=statement,
            confidence=confidence,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.hypotheses.append(item)
        return item

    def for_question(self, question_id):
        return [
            item
            for item in self.hypotheses
            if item.question_id == question_id
        ]

    def size(self):
        return len(self.hypotheses)
