import json
import os


class PersistentKnowledge:
    def __init__(self, path="data/knowledge.json"):
        self.path = path
        self._ensure_directory()

    def _ensure_directory(self):
        directory = os.path.dirname(self.path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    def save(self, knowledge):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                knowledge,
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
