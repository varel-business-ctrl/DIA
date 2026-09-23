from environment.world1 import World1
from organism.perception.local import PerceptionSystem
from organism.core.organism import Organism


world = World1()
perception = PerceptionSystem(radius=2)
organism = Organism()

# Initial perception
previous = perception.perceive(world.observe())
organism.observe(previous, advance_age=True)

# Move to the resource
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

# Record experiences
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


# Resource is currently remembered AND perceived
print("AT DISCOVERY:")
print("Perception:", organism.state.current_observation)
print("Memory size:", organism.memory.size())

# Move away until resource leaves perception
for action in [
    "move_forward",
    "move_forward",
    "move_forward",
]:
    world.step(action)

    current = perception.perceive(world.observe())

    organism.observe(current, advance_age=True)

print()
print("AFTER MOVING AWAY:")
print("Current perception:", organism.state.current_observation)

print()
print("RECALLED MEMORY:")
for experience in organism.memory.recall_recent(12):
    if experience.consequence["resource_detected"]:
        print(experience)
