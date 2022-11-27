from src import model, thymio
import time
import math

TRESHOLD_IR_SENSOR = 900

def set_motor_speed(th : thymio.Thymio, speed : model.MotorSpeed):
    command = speed
    th.process_command(command)

def avoid_obstacle(th : thymio.Thymio):
    sensor_data = th.read_sensor_data()
    pos_data = th.read_robot_position()
    time.sleep(0.1)
    while sees_obstacle(sensor_data):
        sensor_data = th.read_sensor_data()
        pos_data = th.read_robot_position()
        last_angle_pos = pos_data.angle
        sum_angle=0.
        angle = get_angle(sensor_data)
        current_angle_pos = last_angle_pos
        while True:
            current_angle_pos = th.read_robot_position().angle
            delta = abs(current_angle_pos-last_angle_pos)
            #if delta>3: uncomment if discontinuity problem
            #    delta = delta - 2*3.14
            last_angle_pos=current_angle_pos
            sum_angle=sum_angle+delta
            if angle>0:
                set_motor_speed(th,calculate_speed(1,2))#try different values
            else:
                set_motor_speed(th,calculate_speed(1,-2))#try different values
            if sum_angle>=angle:
                set_motor_speed(th,calculate_speed(0,0))
                break
            time.sleep(0.1)
    pass

def get_angle(sensor_data: model.SensorReading) -> float:
    alpha = 80/5*math.pi/180
    alpha_array =  [-2*alpha,1*alpha,0,1*alpha,2*alpha]
    tmp1 = 0
    tmp2 = 0.000000001
    for i in range(5):
        tmp1=tmp1+alpha_array[i]*get_array(sensor_data)[i]
        tmp2=tmp2+get_array(sensor_data)[i]
    return (tmp1/tmp2)

def calculate_speed(linear, angular) -> model.MotorSpeed:
    rspeed = linear + angular/2
    lspeed = linear - angular/2
    return model.MotorSpeed(left=rspeed, right=lspeed)

def get_array(sensor_data: model.SensorReading):
    tmp = sensor_data.horizontal
    return [tmp.left,tmp.center_left,tmp.center,tmp.center_right,tmp.right]

def sees_obstacle(sensor_data: model.SensorReading) -> bool:
    if sensor_data.horizontal.left>TRESHOLD_IR_SENSOR or \
        sensor_data.horizontal.center_left>TRESHOLD_IR_SENSOR or \
        sensor_data.horizontal.center>TRESHOLD_IR_SENSOR or \
        sensor_data.horizontal.center_right>TRESHOLD_IR_SENSOR or \
        sensor_data.horizontal.right>TRESHOLD_IR_SENSOR :
        return True
    return False


# You can run this file directly to test your code on the thymio
if __name__ == "__main__":
    # Create the thymio connection
    th = thymio.Thymio()
    while True:
        avoid_obstacle(th)  
    # Stop the robot
    th.stop()