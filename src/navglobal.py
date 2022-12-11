from typing import Sequence
from src import model, thymio
import time
import numpy as np

# initialized values

# reference speed
mean_speed = 100

# Ziegler-Nichols method for PID

# position
PID_errors_position = np.zeros((3, 1))

# ====
# For Louis : here you can change Ku & Tu (you can see in PID_coefficients_position how it impacts the coefficients)
# high Ku, it will converge faster / too high, you will oscillate
# normally, you won't have to touch Tu because we only use a proportional coefficient (it's enough)
# ===
Ku_position, Tu_position = 0.8, 100
PID_coefficients_position = np.array([0.6 * Ku_position, 1.2 * Ku_position / Tu_position, 3 * Ku_position * Tu_position / 40])  # [Kp, Ki, Kd]

#angle
PID_errors_angle = np.zeros((3, 1))

# ====
# For Louis : same as for position
# # ===
Ku_angle, Tu_angle = 0.8, 100
PID_coefficients_angle = np.array([0.6 * Ku_angle, 1.2 * Ku_angle / Tu_angle, 3 * Ku_angle * Tu_angle / 40])  # [Kp, Ki, Kd]

# collective PID
sensibility_position = 20
sensibility_angle = np.pi/32
PID_weight_angle_position = np.array([0.25, 0.75])
correction = np.zeros((4, 1))


def follow_path(robot: model.Robot, path: Sequence[model.Point]) -> model.MotorSpeed:
    ### Access the robot's pose
    # robot.position.x
    # robot.position.y
    # robot.position.z

    # find the closest index in the path array with the current robot position
    index = min(range(len(path)), key=lambda i: abs(path[i].x - robot.position.x))

    # Calcul of the perpendicular distance between the robot position and the path
    PathPosH, PathPosL = index + 1, index
    if index + 1 >= len(path) - 1:
        PathPosH, PathPosL = len(path) - 1, len(path) - 2
    p1 = np.array(
        [path[PathPosL].x, path[PathPosL].y]
    )  # previous correct position on the path
    p2 = np.array(
        [path[PathPosH].x, path[PathPosH].y]
    )  # previous correct position on the path
    p3 = np.array([robot.position.x, robot.position.y])  # array of the robot position


    # PID on the position
    # -------------------
    error_position = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

    # error > 0 robot over the correct path
    # error < 0 robot below the correct path
    if p3[1] >= p1[1] or p3[1] >= p2[1]:
        correction[0] = -1
    else:
        correction[0] = +1


    # PID correction for position
    PID_errors_position[0], PID_errors_position[1], PID_errors_position[2] = (
        error_position,
        0,  # PID_errors[1] + error_position,
        0,  # error_position - PID_errors[0],
    )
    # basically it does what is below
    # proportional, p = error_position
    # integrator, i += p
    # derivative, d = p - lp
    # update previous error_position, lp = p


    correction[1] = np.dot(PID_coefficients_position, PID_errors_position)


    # PID on the angle
    # ----------------
    error_angle = robot.angle - np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
    if error_angle>=0:
        correction[2] = -1
    else:
        correction[2] = +1


        # PID correction for angle
    PID_errors_angle[0], PID_errors_angle[1], PID_errors_angle[2] = (
        error_angle,
        0,  # PID_errors[1] + error_angle,
        0,  # error_angle - PID_errors[0],
    )

    correction[3] = np.dot(PID_coefficients_angle, PID_errors_angle)


    #if error_angle is too high, priority to correct the angle first
    if abs(error_angle) > sensibility_angle and abs(error_position) < sensibility_position:
        if error_angle>0:
            rspeed = -255
            lspeed = 255
        else:
            rspeed = -255
            lspeed = 255
    else:
        # correction sent to the robot
        if robot.angle > 0 and robot.angle < np.pi:
            rspeed = mean_speed + correction[0]*correction[1] + correction[2]*correction[3]*PID_weight_angle_position[0]
            lspeed = mean_speed - correction[0]*correction[1] - correction[2]*correction[3]*PID_weight_angle_position[0]
        else:
            rspeed = mean_speed - correction[0]*correction[1] - correction[2]*correction[3]*PID_weight_angle_position[0]
            lspeed = mean_speed + correction[0]*correction[1] + correction[2]*correction[3]*PID_weight_angle_position[0]


    # restricting speed of motors between 255 and -255
    if rspeed > 255:
        rspeed = 255
    if lspeed > 255:
        lspeed = 255
    if rspeed < -255:
        rspeed = -255
    if lspeed < -255:
        lspeed = -255

    ### Return the command to send to the motors
    return model.MotorSpeed(lspeed, rspeed)
    # pass


# You can run this file directly to test your code on the thymio
if __name__ == "__main__":
    th = thymio.Thymio()
    # Create a custom robot pose
    robot = model.Robot(position=model.Point(0, 0), angle=0)
    # Create custom path
    path = [model.Point(0, 0), model.Point(0.2, 0.5), model.Point(1, 1)]

    for i in range(10):
        command = follow_path(robot, path)
        th.process_command(command)
        time.sleep(0.1)

    th.stop()
