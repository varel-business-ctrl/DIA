from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Contradiction:
    hypothesis_id: str
    evidence: str
    reason: str
    severity: float
    created_at: str


class ContradictionStore:
    def __init__(self):
        self.items = []

    def add(
        self,
        hypothesis_id,
        evidence,
        reason,
        severity=0.5,
    ):
        severity = max(0.0, min(1.0, severity))

        item = Contradiction(
            hypothesis_id=hypothesis_id,
            evidence=evidence,
            reason=reason,
            severity=severity,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.items.append(item)
        return item

    def for_hypothesis(self, hypothesis_id):
        return [
            item
            for item in self.items
            if item.hypothesis_id == hypothesis_id
        ]

    def size(self):
        return len(self.items)
