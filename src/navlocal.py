from src import model, thymio
import time
import math
import numpy as np

TRESHOLD_IR_SENSOR = 1100 #Treshold used for IR Sensor
TURN_SPEED = 150 #This parameter determine the turn speed, determined by trial and error.
DELTA_ANGLE_SENSOR = 80 / 5  # angle between distance sensors (in degree)
LINEAR_SPEED = 100 #This parameter determine the linear speed, determined by trial and error.

#Set the speed
def set_motor_speed(th: thymio.Thymio, speed: model.MotorSpeed):
    command = speed
    th.process_command(command)
    return


def avoid_obstacle(th: thymio.Thymio, robot_position):
    sensor_data = th.read_sensor_data()
    pos_data = robot_position()
    while sees_obstacle(sensor_data):
        sensor_data = th.read_sensor_data()
        pos_data = robot_position()
        last_angle_pos = pos_data.angle
        sum_angle = 0.0 #Reset
        angle = get_angle(sensor_data) #Calculate angle where there is the least density of obstacles
        current_angle_pos = last_angle_pos
        #Turn until we reached the angle of least density
        while True:
            current_angle_pos = robot_position().angle
            #Caculate angle displacement between each loop
            delta = abs(current_angle_pos - last_angle_pos)
            if delta > 2: #Detect if there is a discontinuity (If displacement is too big)
                delta = delta - 2 * 3.14
            if angle >= 0: #Turn right
                set_motor_speed(
                    th, calculate_speed(LINEAR_SPEED, -TURN_SPEED)
                )
            else: #Turn left
                set_motor_speed(
                    th, calculate_speed(LINEAR_SPEED, TURN_SPEED)
                )
            last_angle_pos = current_angle_pos
            sum_angle = sum_angle + delta
            if sum_angle >= abs(angle): #Reached Goal
                set_motor_speed(th, calculate_speed(LINEAR_SPEED, 0)) #stop turning -> Go straight
                break
            time.sleep(0.1)
    return

#Calculate angle where there is the least density of obstacles
def get_angle(sensor_data: model.SensorReading) -> float:
    alpha = DELTA_ANGLE_SENSOR * math.pi / 180 #angle between IR sensors
    #Array of index for each sensor
    alpha_array = [-2 * alpha, 1 * alpha, 0, 1 * alpha, 2 * alpha]
    tmp1 = 0
    tmp2 = 10e-7
    #Average
    for i in range(5):
        tmp1 = tmp1 + alpha_array[i] * get_array(sensor_data)[i]
        tmp2 = tmp2 + get_array(sensor_data)[i]
    return tmp1 / tmp2

#Calculate the speed of each motor based on the linear speed and angular speed that we wish for
def calculate_speed(linear, angular) -> model.MotorSpeed:
    rspeed = linear + angular / 2
    lspeed = linear - angular / 2
    return model.MotorSpeed(left=rspeed, right=lspeed)

#Return array of all IR sensors
def get_array(sensor_data: model.SensorReading):
    tmp = sensor_data.horizontal
    return np.array(
        [tmp.left, tmp.center_left, tmp.center, tmp.center_right, tmp.right]
    )

#If IR sensor reads over a treshold, turn true
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
