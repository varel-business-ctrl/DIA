from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evidence:
    claim: str
    source: str
    evidence_type: str
    confidence: float
    timestamp: str
    metadata: dict[str, Any]


class EvidenceStore:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.items = []

    def add(
        self,
        claim,
        source,
        evidence_type="observation",
        confidence=0.5,
        metadata=None,
    ):
        confidence = max(0.0, min(1.0, confidence))

        evidence = Evidence(
            claim=claim,
            source=source,
            evidence_type=evidence_type,
            confidence=confidence,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            metadata=metadata or {},
        )

        self.items.append(evidence)

        if len(self.items) > self.capacity:
            self.items.pop(0)

        return evidence

    def search(self, text):
        text = text.lower()

        return [
            item
            for item in self.items
            if text in item.claim.lower()
            or text in item.source.lower()
        ]

    def size(self):
        return len(self.items)
