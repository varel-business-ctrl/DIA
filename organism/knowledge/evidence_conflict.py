class EvidenceConflictDetector:
    def detect(self, evidence_items):
        conflicts = []

        for i in range(len(evidence_items)):
            for j in range(i + 1, len(evidence_items)):
                first = evidence_items[i]
                second = evidence_items[j]

                if first.source == second.source:
                    continue

                first_claim = first.claim.lower()
                second_claim = second.claim.lower()

                if self._appears_opposite(
                    first_claim,
                    second_claim,
                ):
                    conflicts.append({
                        "claim_a": first.claim,
                        "source_a": first.source,
                        "confidence_a": first.confidence,
                        "claim_b": second.claim,
                        "source_b": second.source,
                        "confidence_b": second.confidence,
                    })

        return conflicts

    def _appears_opposite(self, first, second):
        pairs = [
            (" is ", " is not "),
            (" are ", " are not "),
            (" can ", " cannot "),
            (" does ", " does not "),
            (" true", " false"),
        ]

        for positive, negative in pairs:
            if (
                positive in first
                and negative in second
            ) or (
                negative in first
                and positive in second
            ):
                return True

        return False
