import json
import os


class PersistentClaimStore:
    def __init__(self, path="data/claims.json"):
        self.path = path
        self._ensure_directory()

    def _ensure_directory(self):
        directory = os.path.dirname(self.path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

    def save(self, claims):
        with open(
            self.path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                claims,
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

    def add(self, claim):
        claims = self.load()

        claims.append(claim)

        self.save(claims)

        return claim

    def size(self):
        return len(self.load())
