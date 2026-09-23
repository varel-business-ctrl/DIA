class ClaimEvidenceLinker:
    def __init__(self):
        self.links = []

    def link(
        self,
        claim_id,
        evidence_source,
        relationship,
    ):
        if relationship not in (
            "supports",
            "contradicts",
        ):
            raise ValueError(
                "Relationship must be "
                "'supports' or 'contradicts'."
            )

        link = {
            "claim_id": claim_id,
            "evidence_source": evidence_source,
            "relationship": relationship,
        }

        self.links.append(link)

        return link

    def get_for_claim(self, claim_id):
        return [
            link
            for link in self.links
            if link["claim_id"] == claim_id
        ]

    def size(self):
        return len(self.links)
