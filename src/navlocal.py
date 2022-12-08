from src import model, thymio
import time
import math
import numpy as np

TRESHOLD_IR_SENSOR = 1100
TURN_SPEED = 150
DELTA_ANGLE_SENSOR = 80 / 5  # angle between distance sensors (in degree)

def set_motor_speed(th: thymio.Thymio, speed: model.MotorSpeed):
    command = speed
    th.process_command(command)
    return

def avoid_obstacle(th: thymio.Thymio, robot_position):
    sensor_data = th.read_sensor_data()
    pos_data = robot_position()
    #last_speed_l = th.read_sensor_data()
    #last_speed_r = th.read_sensor_data()
    last_speed = 100 #set last speed, shoud be last know speed unless it's the start
    time.sleep(0.1)
    while sees_obstacle(sensor_data):
        sensor_data = th.read_sensor_data()
        pos_data = robot_position()
        last_angle_pos = pos_data.angle
        sum_angle = 0.0
        angle = get_angle(sensor_data)
        current_angle_pos = last_angle_pos
        print("new loop")
        while True:
            current_angle_pos = robot_position().angle
            delta = abs(current_angle_pos - last_angle_pos)
            if delta > 2: #any large number in radian -> simply detect if there is a discontinuity
                delta = delta - 2 * 3.14
            if angle >= 0:
                set_motor_speed(
                    th, calculate_speed(last_speed, -TURN_SPEED)
                )
            else:
                set_motor_speed(
                    th, calculate_speed(last_speed, TURN_SPEED)
                )
            last_angle_pos = current_angle_pos
            sum_angle = sum_angle + delta
            if sum_angle >= abs(angle):
                set_motor_speed(th, calculate_speed(last_speed, 0)) #stop turning -> Go straight
                break
    return


def get_angle(sensor_data: model.SensorReading) -> float:
    alpha = DELTA_ANGLE_SENSOR * math.pi / 180
    alpha_array = [-2 * alpha, 1 * alpha, 0, 1 * alpha, 2 * alpha]
    tmp1 = 0
    tmp2 = 10e-7
    for i in range(5):
        tmp1 = tmp1 + alpha_array[i] * get_array(sensor_data)[i]
        tmp2 = tmp2 + get_array(sensor_data)[i]
    return tmp1 / tmp2


def calculate_speed(linear, angular) -> model.MotorSpeed:
    rspeed = linear + angular / 2
    lspeed = linear - angular / 2
    return model.MotorSpeed(left=rspeed, right=lspeed)


def get_array(sensor_data: model.SensorReading):
    tmp = sensor_data.horizontal
    return np.array(
        [tmp.left, tmp.center_left, tmp.center, tmp.center_right, tmp.right]
    )


def sees_obstacle(sensor_data: model.SensorReading) -> bool:
    if any(get_array(sensor_data) > TRESHOLD_IR_SENSOR):
        return True
    return False


#Run this file directly to test your code on the thymio
if __name__ == "__main__":
    # Create the thymio connection
    th = thymio.Thymio()
    while True:
        avoid_obstacle(th)
    # Stop the robot
    th.stop()
