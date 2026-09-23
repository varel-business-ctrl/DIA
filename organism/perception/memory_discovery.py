from environment.world1 import World1
from organism.perception.local import PerceptionSystem
from organism.core.organism import Organism


world = World1()
perception = PerceptionSystem(radius=2)
organism = Organism()

previous = perception.perceive(world.observe())
organism.observe(previous, advance_age=True)

actions = [
    "turn_right",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "turn_left",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
]

for action in actions:
    world.step(action)

    current = perception.perceive(world.observe())

    consequence = {
        "resource_detected": len(current["resources"]) > 0,
        "energy_change": current["energy"] - previous["energy"],
    }

    organism.remember(
        action,
        current,
        consequence,
    )

    organism.observe(current, advance_age=True)

    previous = current


print("FINAL PERCEPTION:")
print(organism.state.current_observation)

print()
print("ORGANISM STATUS:")
print(organism.status())

print()
print("MEMORY SIZE:")
print(organism.memory.size())

print()
print("LAST MEMORY:")
print(organism.memory.recall_recent(1)[0])
