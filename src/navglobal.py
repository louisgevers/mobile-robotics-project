from typing import Sequence
from src import model, thymio
import time
import numpy as np

#initialized values
PID_errors = np.zeros((1,3))
base_speed = 100

#Ziegler-Nichols method for PID
Ku, Tu = 1, 1

def follow_path(robot: model.Robot, path: Sequence[model.Point]) -> model.MotorSpeed:
    ### Access the robot's pose
    # robot.position.x
    # robot.position.y
    # robot.position.z

    index = min(range(len(path)), key=lambda i: abs(path[i].x-robot.position.x))
    devH, devL = index+1, index-1
    if index>=len(path)-1:
        devH = len(path)-1
    if index==0:
        devL = 0
    theta = np.arctan( (path[devH].y-path[devL].y) / (path[devH].x-path[devL].x) )

    error = (robot.position.y-path[index].y) + (robot.angle-theta) # error of position + error of angle

    # PID correction
    PID_errors[0], PID_errors[1]; PID_errors[2] = error, PID_errors[1]+error, error-PID_errors[0]
    # basically it does what is below
    #proportional, p = error
    #integrator, i += p
    #derivative, d = p - lp
    #update previous error, lp = p

    
    PID_coefficients = np.array([0.6*Ku,1.2*Ku/Tu,3*Ku*Tu/40]) #[Kp, Ki, Kd]
    #adding the correction to the base_speed for the left and right motor
    correction = int(PID_coefficients*PID_errors)
    rspeed = base_speed + correction
    lspeed = base_speed - correction

    #restricting speeds of motors between 255 and -255
    if (rspeed > 255):
        rspeed = 255    
    if (lspeed > 255):
        lspeed = 255    
    if (rspeed < -255):
        rspeed = -255    
    if (lspeed < -255):
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
