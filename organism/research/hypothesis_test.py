from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class HypothesisTest:
    test_id: str
    hypothesis_id: str
    evidence: str
    result: str
    support: float
    created_at: str


class HypothesisTestStore:
    def __init__(self):
        self.tests = []

    def add(
        self,
        test_id,
        hypothesis_id,
        evidence,
        result,
        support=0.5,
    ):
        support = max(0.0, min(1.0, support))

        item = HypothesisTest(
            test_id=test_id,
            hypothesis_id=hypothesis_id,
            evidence=evidence,
            result=result,
            support=support,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.tests.append(item)
        return item

    def for_hypothesis(self, hypothesis_id):
        return [
            item
            for item in self.tests
            if item.hypothesis_id == hypothesis_id
        ]

    def size(self):
        return len(self.tests)
