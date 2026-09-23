class BeliefEvaluator:
    def evaluate(self, claim_id, evidence_items, links):
        supporting = []
        contradicting = []

        for link in links:
            if link["claim_id"] != claim_id:
                continue

            for evidence in evidence_items:
                if evidence.source != link["evidence_source"]:
                    continue

                if link["relationship"] == "supports":
                    supporting.append(evidence)

                elif link["relationship"] == "contradicts":
                    contradicting.append(evidence)

        support_score = sum(
            evidence.confidence
            for evidence in supporting
        )

        contradiction_score = sum(
            evidence.confidence
            for evidence in contradicting
        )

        total = (
            support_score
            + contradiction_score
        )

        if total == 0:
            net_confidence = 0.0
        else:
            net_confidence = (
                support_score
                / total
            )

        if supporting and contradicting:
            status = "uncertain"
        elif supporting:
            status = "supported"
        elif contradicting:
            status = "contradicted"
        else:
            status = "unknown"

        return {
            "claim_id": claim_id,
            "status": status,
            "supporting_evidence": len(supporting),
            "contradicting_evidence": len(contradicting),
            "net_confidence": round(
                net_confidence,
                3,
            ),
        }
