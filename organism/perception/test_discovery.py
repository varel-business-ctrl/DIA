from environment.world1 import World1
from organism.perception.local import PerceptionSystem


world = World1()
perception = PerceptionSystem(radius=2)

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


print("STEP 0:")
print("LOCAL PERCEPTION:", perception.perceive(world.observe()))

for action in actions:
    world.step(action)

    world_state = world.observe()
    local = perception.perceive(world_state)

    print()
    print("ACTION:", action)
    print(
        "WORLD POSITION:",
        world_state["organism"]["x"],
        world_state["organism"]["y"],
    )
    print("STEP:", world_state["step"])
    print("LOCAL PERCEPTION:", local)
