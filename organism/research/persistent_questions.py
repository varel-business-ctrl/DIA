import json
import os


class PersistentResearchQuestionStore:
    def __init__(self, path="data/research_questions.json"):
        self.path = path
        self._ensure_directory()

    def _ensure_directory(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def save(self, questions):
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(questions, file, indent=2, ensure_ascii=False)

    def load(self):
        if not os.path.exists(self.path):
            return []

        with open(self.path, "r", encoding="utf-8") as file:
            return json.load(file)

    def add(self, question):
        questions = self.load()
        questions.append(question)
        self.save(questions)
        return question

    def size(self):
        return len(self.load())

    def highest_priority(self):
        questions = self.load()

        if not questions:
            return None

        return max(
            questions,
            key=lambda question: question.get("priority", 0.0)
        )
