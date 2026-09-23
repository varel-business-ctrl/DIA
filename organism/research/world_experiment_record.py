from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class WorldExperimentRecord:
    experiment_id: str
    hypothesis_id: str
    actions: list
    initial_state: dict
    final_state: dict
    prediction: dict
    prediction_error: float
    created_at: str


class WorldExperimentRecordStore:
    def __init__(self):
        self.records = []

    def record(
        self,
        experiment_id,
        hypothesis_id,
        actions,
        initial_state,
        final_state,
        prediction,
        prediction_error,
    ):
        record = WorldExperimentRecord(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            prediction=prediction,
            prediction_error=prediction_error,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.records.append(record)
        return record

    def latest(self):
        if not self.records:
            return None

        return self.records[-1]

    def size(self):
        return len(self.records)
