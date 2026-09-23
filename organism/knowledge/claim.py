from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Claim:
    claim_id: str
    statement: str
    topic: str
    status: str
    confidence: float
    created_at: str


class ClaimStore:
    def __init__(self):
        self.claims = []

    def add(
        self,
        claim_id,
        statement,
        topic,
        confidence=0.5,
        status="uncertain",
    ):
        confidence = max(0.0, min(1.0, confidence))

        claim = Claim(
            claim_id=claim_id,
            statement=statement,
            topic=topic,
            status=status,
            confidence=confidence,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self.claims.append(claim)

        return claim

    def get(self, claim_id):
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim

        return None

    def search(self, topic):
        topic = topic.lower()

        return [
            claim
            for claim in self.claims
            if topic in claim.topic.lower()
            or topic in claim.statement.lower()
        ]

    def size(self):
        return len(self.claims)
