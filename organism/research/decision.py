from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ResearchDecision:
    question_id: str
    reason: str
    expected_information_gain: float
    uncertainty: float
    decision: str
    created_at: str


class ResearchDecisionStore:
    def __init__(self):
        self.decisions = []

    def create(
        self,
        question_id,
        reason,
        expected_information_gain,
        uncertainty,
        decision,
    ):
        item = ResearchDecision(
            question_id=question_id,
            reason=reason,
            expected_information_gain=max(
                0.0, min(1.0, expected_information_gain)
            ),
            uncertainty=max(0.0, min(1.0, uncertainty)),
            decision=decision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.decisions.append(item)
        return item

    def latest(self):
        if not self.decisions:
            return None

        return self.decisions[-1]

    def size(self):
        return len(self.decisions)
