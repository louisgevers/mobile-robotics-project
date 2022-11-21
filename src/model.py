from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    @property
    def v(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance(self, other) -> float:
        return np.linalg.norm(self.v - other.v)


@dataclass
class Robot:
    position: Point
    angle: float


@dataclass
class Obstacle:
    corners: Sequence[Point]


@dataclass
class World:
    robot: Robot
    goal: Point
    obstacles: Sequence[Obstacle]

    def robot_at_goal(self, tolerance: float = 0.01) -> bool:
        return self.robot.position.distance(self.goal) <= tolerance


@dataclass
class MotorSpeed:
    left: float
    right: float


@dataclass
class HorizontalSensor:
    v: np.ndarray

    @property
    def left(self) -> float:
        return self.v[0]

    @property
    def center_left(self) -> float:
        return self.v[1]

    @property
    def center(self) -> float:
        return self.v[2]

    @property
    def center_right(self) -> float:
        return self.v[3]

    @property
    def right(self) -> float:
        return self.v[4]


@dataclass
class GroundSensor:
    v: np.ndarray

    @property
    def left(self) -> float:
        return self.v[0]

    @property
    def right(self) -> float:
        return self.v[1]


@dataclass
class SensorReading:
    horizontal: HorizontalSensor
    ground: GroundSensor
    motor: MotorSpeed
