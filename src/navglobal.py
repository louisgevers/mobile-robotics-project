from typing import Sequence
from src import model, utils
import numpy as np

ANGLE_TOLERANCE = np.deg2rad(5)
SPEED = 100


class GlobalNavigation:
    def __init__(self, path: Sequence[model.Point]) -> None:
        self.next_index = 1
        self.path = path

    def next_command(self, robot: model.Robot) -> model.MotorSpeed:
        if robot.position.distance(self.get_next_point()) <= 5:
            self.next_index += 1
        angle_error = self.get_angle_difference(robot)
        if abs(angle_error) > ANGLE_TOLERANCE:
            return self.turn(angle_error)
        else:
            # Go straight
            return model.MotorSpeed(left=SPEED, right=SPEED)

    def get_angle_difference(self, robot: model.Robot) -> float:
        point = self.get_next_point()
        # Target angle is the angle between y-axis and vector between robot and point
        target_angle = utils.get_global_angle(
            robot.position.x, robot.position.y, point.x, point.y
        )
        error = target_angle - robot.angle
        # Correct error in [-pi pi] range
        if abs(error) > np.pi:
            error = error - np.sign(error) * 2 * np.pi
        return error

    def get_next_point(self) -> model.Point:
        return self.path[self.next_index]

    def turn(self, angle_error: float) -> model.MotorSpeed:
        if angle_error < 0:
            # Turn right
            return model.MotorSpeed(left=SPEED / 2, right=-SPEED / 2)
        else:
            # Turn left
            return model.MotorSpeed(left=-SPEED / 2, right=SPEED / 2)


if __name__ == "__main__":
    robot = model.Robot(position=model.Point(x=0.6, y=0.6), angle=np.deg2rad(42))
    path = [
        model.Point(x=0, y=0),
        model.Point(x=1, y=1),
        model.Point(x=0, y=1),
        model.Point(x=3, y=5),
    ]
    navigator = GlobalNavigation(path)
    command = navigator.next_command(robot)
    print(command)
