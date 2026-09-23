from environment.world1 import World1


class WorldExperimentRunner:
    def __init__(self):
        self.world = World1()

    def reset(self):
        return self.world.reset()

    def observe(self):
        return self.world.observe()

    def run_actions(self, actions):
        results = []

        for action in actions:
            result = self.world.step(action)

            results.append({
                "action": action,
                "result": result,
            })

        return results

    def run(self, actions):
        initial = self.observe()

        results = self.run_actions(actions)

        final = self.observe()

        return {
            "initial": initial,
            "actions": results,
            "final": final,
            "steps": len(actions),
        }
