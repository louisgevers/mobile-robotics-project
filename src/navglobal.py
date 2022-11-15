from typing import Sequence
from src import model, thymio
import time


def follow_path(robot: model.Robot, path: Sequence[model.Point]) -> model.MotorSpeed:
    ### Access the robot's pose
    # robot.position.x
    # robot.position.y
    # robot.position.z

    ### Return the command to send to the motors
    return model.MotorSpeed(left=100, right=100)
    # pass


# You can run this file directly to test your code on the thymio
if __name__ == "__main__":
    thymio.initialize()
    # Create a custom robot pose
    robot = model.Robot(position=model.Point(0, 0), angle=0)
    # Create custom path
    path = [model.Point(0, 0), model.Point(0.2, 0.5), model.Point(1, 1)]

    for i in range(100):
        command = follow_path(robot, path)
        thymio.process_command(command)
        time.sleep(0.1)
