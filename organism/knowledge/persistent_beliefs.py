import json
import os


class PersistentBeliefStore:
    def __init__(self, path="data/beliefs.json"):
        self.path = path
        self._ensure_directory()

    def _ensure_directory(self):
        directory = os.path.dirname(self.path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    def save(self, beliefs):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                beliefs,
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

    def add(self, belief):
        beliefs = self.load()

        beliefs.append(belief)

        self.save(beliefs)

        return belief

    def latest(self, claim_id):
        beliefs = [
            belief
            for belief in self.load()
            if belief.get("claim_id") == claim_id
        ]

        if not beliefs:
            return None

        return beliefs[-1]

    def history(self, claim_id):
        return [
            belief
            for belief in self.load()
            if belief.get("claim_id") == claim_id
        ]

    def size(self):
        return len(self.load())
