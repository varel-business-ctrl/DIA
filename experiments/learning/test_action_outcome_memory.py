class ActionOutcomeMemory:
    def __init__(self):
        self.outcomes = {}

    def record(
        self,
        state,
        action,
        next_state,
        movement_succeeded,
        distance_before,
        distance_after,
    ):
        key = (state, action)

        outcome = {
            "next_state": next_state,
            "movement_succeeded": movement_succeeded,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "improved": distance_after < distance_before,
        }

        if key not in self.outcomes:
            self.outcomes[key] = []

        self.outcomes[key].append(outcome)

        return outcome

    def get_outcomes(self, state, action):
        return self.outcomes.get((state, action), [])

    def action_count(self, state, action):
        return len(self.get_outcomes(state, action))

    def has_tried(self, state, action):
        return self.action_count(state, action) > 0

    def successful_actions(self, state):
        results = []

        for (stored_state, action), outcomes in self.outcomes.items():
            if stored_state != state:
                continue

            if any(outcome["movement_succeeded"] for outcome in outcomes):
                results.append(action)

        return results

    def failed_actions(self, state):
        results = []

        for (stored_state, action), outcomes in self.outcomes.items():
            if stored_state != state:
                continue

            if all(
                not outcome["movement_succeeded"]
                for outcome in outcomes
            ):
                results.append(action)

        return results

    def summary(self):
        return {
            "state_action_pairs": len(self.outcomes),
            "total_outcomes": sum(
                len(outcomes)
                for outcomes in self.outcomes.values()
            ),
            "outcomes": self.outcomes,
        }


def main():
    print("=== DIA ACTION OUTCOME MEMORY TEST ===")

    memory = ActionOutcomeMemory()

    state = ((5, 3), "north")

    print("\nRecording first action...")

    result = memory.record(
        state=state,
        action="move_forward",
        next_state=((5, 3), "north"),
        movement_succeeded=False,
        distance_before=2,
        distance_after=2,
    )

    print(result)

    print("\nRecording second action...")

    result = memory.record(
        state=state,
        action="turn_right",
        next_state=((6, 3), "east"),
        movement_succeeded=True,
        distance_before=2,
        distance_after=3,
    )

    print(result)

    print("\nRecording third action...")

    result = memory.record(
        state=state,
        action="turn_left",
        next_state=((4, 3), "west"),
        movement_succeeded=True,
        distance_before=2,
        distance_after=1,
    )

    print(result)

    print("\n=== MEMORY QUERIES ===")

    print(
        "Tried move_forward:",
        memory.has_tried(state, "move_forward"),
    )

    print(
        "Tried wait:",
        memory.has_tried(state, "wait"),
    )

    print(
        "Successful actions:",
        memory.successful_actions(state),
    )

    print(
        "Failed actions:",
        memory.failed_actions(state),
    )

    print(
        "Move forward outcomes:",
        memory.get_outcomes(state, "move_forward"),
    )

    print("\n=== MEMORY SUMMARY ===")
    print(memory.summary())


if __name__ == "__main__":
    main()
