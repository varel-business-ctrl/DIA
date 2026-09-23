class ActionSelectorV1:
    def __init__(self, actions):
        self.actions = actions
        self.tried_actions = {}
        self.action_results = {}

    def record_outcome(
        self,
        state,
        action,
        movement_succeeded,
        improved,
    ):
        key = (state, action)

        if key not in self.action_results:
            self.action_results[key] = []

        self.action_results[key].append({
            "movement_succeeded": movement_succeeded,
            "improved": improved,
        })

        self.tried_actions.setdefault(state, set()).add(action)

    def select_action(self, state):
        tried = self.tried_actions.get(state, set())

        # Explore actions that have not been tried in this state.
        for action in self.actions:
            if action not in tried:
                return {
                    "action": action,
                    "reason": "explore_untried_action",
                }

        # Evaluate actions that have already been tried.
        candidates = []

        for action in self.actions:
            key = (state, action)
            outcomes = self.action_results.get(key, [])

            if not outcomes:
                continue

            successful = sum(
                1
                for outcome in outcomes
                if outcome["movement_succeeded"]
            )

            improved = sum(
                1
                for outcome in outcomes
                if outcome["improved"]
            )

            failed = len(outcomes) - successful

            candidates.append({
                "action": action,
                "successful": successful,
                "improved": improved,
                "failed": failed,
                "attempts": len(outcomes),
            })

        if not candidates:
            return {
                "action": self.actions[0],
                "reason": "no_previous_experience",
            }

        # Prefer actions that have improved the situation.
        candidates.sort(
            key=lambda item: (
                item["improved"],
                item["successful"],
                -item["failed"],
            ),
            reverse=True,
        )

        best = candidates[0]

        return {
            "action": best["action"],
            "reason": "choose_best_recorded_outcome",
            "evaluation": best,
        }

    def summary(self):
        return {
            "known_state_action_pairs": len(self.action_results),
            "recorded_outcomes": sum(
                len(results)
                for results in self.action_results.values()
            ),
        }


def main():
    print("=== DIA ACTION SELECTOR V1 TEST ===")

    actions = [
        "move_forward",
        "turn_right",
        "turn_left",
        "wait",
    ]

    selector = ActionSelectorV1(actions)

    state = ((5, 3), "north")

    print("\nRecording experiences...")

    selector.record_outcome(
        state=state,
        action="move_forward",
        movement_succeeded=False,
        improved=False,
    )

    selector.record_outcome(
        state=state,
        action="turn_right",
        movement_succeeded=True,
        improved=False,
    )

    selector.record_outcome(
        state=state,
        action="turn_left",
        movement_succeeded=True,
        improved=True,
    )

    print("\nSelecting action after experience:")

    # All actions have not been tried yet.
    # The selector should explore an untried action.
    print(selector.select_action(state))

    selector.record_outcome(
        state=state,
        action="wait",
        movement_succeeded=False,
        improved=False,
    )

    print("\nSelecting action after all actions were tried:")

    print(selector.select_action(state))

    print("\n=== SELECTOR SUMMARY ===")
    print(selector.summary())


if __name__ == "__main__":
    main()
