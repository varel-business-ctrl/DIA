class LoopMemory:
    def __init__(self):
        self.visited_states = {}
        self.repeated_states = []

    def observe(self, position, orientation):
        state = (position, orientation)

        count = self.visited_states.get(state, 0)
        self.visited_states[state] = count + 1

        repeated = count > 0

        if repeated:
            self.repeated_states.append(state)

        return {
            "state": state,
            "visit_count": self.visited_states[state],
            "repeated": repeated,
        }

    def has_repeated(self, position, orientation):
        state = (position, orientation)
        return self.visited_states.get(state, 0) > 1

    def size(self):
        return len(self.visited_states)

    def repetitions(self):
        return len(self.repeated_states)

    def summary(self):
        return {
            "unique_states": self.size(),
            "repeated_state_events": self.repetitions(),
            "visited_states": self.visited_states,
        }


def run_test():
    memory = LoopMemory()

    test_states = [
        ((5, 3), "north"),
        ((5, 3), "east"),
        ((6, 3), "east"),
        ((7, 3), "south"),
        ((7, 3), "west"),
        ((6, 3), "west"),
        ((5, 3), "north"),
        ((5, 3), "east"),
    ]

    print("=== DIA LOOP MEMORY TEST ===")

    for position, orientation in test_states:
        result = memory.observe(position, orientation)
        print(
            "Position:",
            position,
            "Orientation:",
            orientation,
            "Result:",
            result,
        )

    print()
    print("=== MEMORY SUMMARY ===")
    print(memory.summary())


if __name__ == "__main__":
    run_test()
