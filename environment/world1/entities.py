from dataclasses import dataclass


@dataclass
class Position:
    x: int
    y: int


@dataclass
class Resource:
    position: Position
    energy: int = 25


@dataclass
class Obstacle:
    position: Position
