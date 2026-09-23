from organism.research.world_experiment import WorldExperimentRunner
from organism.research.world_experiment_record import WorldExperimentRecordStore


runner = WorldExperimentRunner()
records = WorldExperimentRecordStore()

initial = runner.observe()

actions = [
    "wait",
    "wait",
    "turn_right",
    "move_forward",
]

prediction = {
    "energy": 96,
    "orientation": "east",
}

result = runner.run(actions)
final = result["final"]

prediction_error = 0

for key, value in prediction.items():
    if final["organism"].get(key) != value:
        prediction_error += 1

prediction_error = round(
    min(1.0, prediction_error / len(prediction)),
    3,
)

record = records.record(
    experiment_id="EXP-W1-001",
    hypothesis_id="H-001",
    actions=actions,
    initial_state=initial,
    final_state=final,
    prediction=prediction,
    prediction_error=prediction_error,
)

print("INITIAL:", initial)
print("FINAL:", final)
print("PREDICTION:", prediction)
print("PREDICTION ERROR:", prediction_error)
print("RECORD:", record)
