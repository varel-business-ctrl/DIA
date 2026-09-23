from dataclasses import dataclass
from datetime import datetime, timezone

from organism.research.priority import ResearchPriority
from organism.research.persistent_questions import PersistentResearchQuestionStore


@dataclass
class ResearchQuestion:
    question_id: str
    question: str
    priority: float
    reason: str
    created_at: str


class ResearchQuestionStore:
    def __init__(self):
        self.priority_engine = ResearchPriority()
        self.persistence = PersistentResearchQuestionStore()
        self.questions = self._load_questions()

    def _load_questions(self):
        questions = []

        for item in self.persistence.load():
            questions.append(
                ResearchQuestion(
                    question_id=item["question_id"],
                    question=item["question"],
                    priority=item["priority"],
                    reason=item.get("reason", "unknown"),
                    created_at=item["created_at"],
                )
            )

        return questions

    def add(
        self,
        question_id,
        question,
        priority=None,
        reason="unknown",
        uncertainty=0.5,
        importance=0.5,
        information_gain=0.5,
        knowledge_gap=0.5,
    ):
        if priority is None:
            priority = self.priority_engine.calculate(
                uncertainty=uncertainty,
                importance=importance,
                information_gain=information_gain,
                knowledge_gap=knowledge_gap,
            )

        priority = max(0.0, min(1.0, priority))

        item = ResearchQuestion(
            question_id=question_id,
            question=question,
            priority=priority,
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.questions.append(item)

        self.persistence.add({
            "question_id": item.question_id,
            "question": item.question,
            "priority": item.priority,
            "reason": item.reason,
            "created_at": item.created_at,
        })

        return item

    def highest_priority(self):
        if not self.questions:
            return None

        return max(self.questions, key=lambda item: item.priority)

    def size(self):
        return len(self.questions)

    def status(self):
        next_question = self.highest_priority()

        return {
            "question_count": len(self.questions),
            "next_question": None if next_question is None else {
                "question_id": next_question.question_id,
                "question": next_question.question,
                "priority": next_question.priority,
                "reason": next_question.reason,
            },
        }
