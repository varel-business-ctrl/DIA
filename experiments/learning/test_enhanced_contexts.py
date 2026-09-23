from environment.world1.world import World1
from organism.perception.enhanced_local import EnhancedLocalPerception
from organism.learning.context_extractor_v2 import ContextExtractorV2


def show_context(label, world):
    perception = EnhancedLocalPerception(radius=2)
    extractor = ContextExtractorV2()

    observation = perception.perceive(world)
    context = extractor.extract(observation)

    print(f"\n--- {label} ---")
    print("Position:", world.organism.position)
    print("Orientation:", observation["orientation"])
    print("Boundary ahead:", observation["boundary_ahead"])
    print("Obstacle ahead:", observation["obstacle_ahead"])
    print("Extracted context:", context)

    return context


def main():
    world = World1()

    print("DIA ENHANCED CONTEXT TEST")
    print("=" * 40)

    print("\nInitial world state:")
    print(world.observe())

    print("\nTesting initial position...")
    show_context("Initial position", world)

    print("\nTesting boundary detection...")
    organism = world.organism

    # Move organism to the northern boundary.
    organism.position.x = 1
    organism.position.y = 9
    organism.orientation = "north"

    show_context("Northern boundary", world)

    print("\nTesting obstacle detection...")
    # Place organism directly below an existing obstacle.
    organism.position.x = 4
    organism.position.y = 3
    organism.orientation = "north"

    show_context("Obstacle ahead", world)

    print("\nTesting clear path detection...")
    organism.position.x = 1
    organism.position.y = 1
    organism.orientation = "north"

    show_context("Clear path", world)

    print("\nTEST COMPLETE")


if __name__ == "__main__":
    main()
