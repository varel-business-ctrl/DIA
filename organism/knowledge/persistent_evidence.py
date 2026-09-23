import json
import os


class PersistentEvidenceStore:
    def __init__(self, path="data/evidence.json"):
        self.path = path
        self._ensure_directory()

    def _ensure_directory(self):
        directory = os.path.dirname(self.path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    def save(self, evidence):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                evidence,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def load(self):
        if not os.path.exists(self.path):
            return []

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def add(self, evidence):
        items = self.load()

        items.append(evidence)

        self.save(items)

        return evidence

    def size(self):
        return len(self.load())
