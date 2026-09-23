from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Knowledge:
    topic: str
    content: str
    source: str
    confidence: float
    timestamp: str


class KnowledgeStore:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.items = []

    def add(
        self,
        topic,
        content,
        source,
        confidence=0.5,
    ):
        confidence = max(0.0, min(1.0, confidence))

        item = Knowledge(
            topic=topic,
            content=content,
            source=source,
            confidence=confidence,
            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self.items.append(item)

        if len(self.items) > self.capacity:
            self.items.pop(0)

    def search(self, topic):
        topic = topic.lower()

        return [
            item
            for item in self.items
            if topic in item.topic.lower()
            or topic in item.content.lower()
        ]

    def size(self):
        return len(self.items)
