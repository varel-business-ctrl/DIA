from organism.knowledge.claim import ClaimStore
from organism.knowledge.evidence import EvidenceStore
from organism.knowledge.claim_evidence import ClaimEvidenceLinker
from organism.knowledge.belief import BeliefEvaluator
from organism.knowledge.persistent_claims import PersistentClaimStore
from organism.knowledge.persistent_evidence import PersistentEvidenceStore
from organism.knowledge.persistent_beliefs import PersistentBeliefStore


class KnowledgeSystem:
    def __init__(self):
        self.claims = ClaimStore()
        self.evidence = EvidenceStore()
        self.links = ClaimEvidenceLinker()
        self.beliefs = BeliefEvaluator()

        self.persistent_claims = PersistentClaimStore()
        self.persistent_evidence = PersistentEvidenceStore()
        self.persistent_beliefs = PersistentBeliefStore()

    def create_claim(
        self,
        claim_id,
        statement,
        topic,
        confidence=0.5,
        status="uncertain",
    ):
        claim = self.claims.add(
            claim_id,
            statement,
            topic,
            confidence,
            status,
        )

        self.persistent_claims.add({
            "claim_id": claim.claim_id,
            "statement": claim.statement,
            "topic": claim.topic,
            "status": claim.status,
            "confidence": claim.confidence,
            "created_at": claim.created_at,
        })

        return claim

    def add_evidence(
        self,
        claim_id,
        source,
        evidence,
        confidence=0.5,
        relationship="supports",
    ):
        item = self.evidence.add(
            claim=evidence,
            source=source,
            evidence_type="research",
            confidence=confidence,
            metadata={
                "claim_id": claim_id,
            },
        )

        self.links.link(
            claim_id,
            source,
            relationship,
        )

        self.persistent_evidence.add({
            "claim_id": claim_id,
            "source": source,
            "evidence": evidence,
            "confidence": confidence,
            "relationship": relationship,
        })

        return item

    def evaluate(self, claim_id):
        result = self.beliefs.evaluate(
            claim_id,
            self.evidence.items,
            self.links.links,
        )

        self.persistent_beliefs.add(result)

        return result
