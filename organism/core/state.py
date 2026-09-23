from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrganismInternalState:
    age: int = 0
    energy: int = 100
    orientation: str = "north"

    current_observation: dict[str, Any] = field(default_factory=dict)

    previous_observation: dict[str, Any] = field(default_factory=dict)

    last_action: str | None = None

    prediction_error: float = 0.0

    experiences: int = 0
