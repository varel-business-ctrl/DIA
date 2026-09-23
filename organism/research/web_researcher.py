from organism.web.explorer import WebExplorer
from organism.web.parser import WebParser
from organism.knowledge.store import KnowledgeStore
from organism.knowledge.evidence import EvidenceStore


class WebResearcher:
    def __init__(self):
        self.explorer = WebExplorer()
        self.parser = WebParser()

        self.knowledge = KnowledgeStore()
        self.evidence = EvidenceStore()

    def research(
        self,
        url,
        topic,
        confidence=0.5,
    ):
        page = self.explorer.fetch(url)

        text = self.parser.extract_text(
            page["content"]
        )

        self.knowledge.add(
            topic=topic,
            content=text,
            source=url,
            confidence=confidence,
        )

        self.evidence.add(
            claim=text,
            source=url,
            evidence_type="web_source",
            confidence=confidence,
            metadata={
                "topic": topic,
                "status": page["status"],
                "content_type": page["content_type"],
            },
        )

        return {
            "topic": topic,
            "source": url,
            "status": page["status"],
            "characters": len(text),
            "knowledge_size": self.knowledge.size(),
            "evidence_size": self.evidence.size(),
        }
