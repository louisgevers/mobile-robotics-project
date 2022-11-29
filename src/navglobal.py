from typing import Sequence
from src import model, thymio
import time
import numpy as np

#initialized values
PID_errors = np.zeros((1,3))
# reference speed
max_speed = 255
#Ziegler-Nichols method for PID
Ku, Tu = 1, 1
PID_coefficients = np.array([0.6*Ku,1.2*Ku/Tu,3*Ku*Tu/40]) #[Kp, Ki, Kd]

def follow_path(robot: model.Robot, path: Sequence[model.Point]) -> model.MotorSpeed:
    ### Access the robot's pose
    # robot.position.x
    # robot.position.y
    # robot.position.z

    #find the closest index in the path array with the current robot position
    index = min(range(len(path)), key=lambda i: abs(path[i].x-robot.position.x))

    # Calcul of the perpendicular distance between the robot position and the path
    PathPosH, PathPosL = index+1,index
    if index+1>=len(path)-1:
        PathPosH, PathPosL = len(path)-1,len(path)-2  
    p1 = np.array([path[index].x, path[index].y]) #previous correct position on the path
    p2 = np.array([path[index+1].x, path[index+1].y]) #previous correct position on the path
    p3 = np.array([robot.position.x, robot.position.y]) #array of the robot position

    d=abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

    # error > 0 robot over the correct path
    # error > à robot below the correct path
    if p3(1)>=p1(1) or p3(1)>=p2(1):
        error = d # error of position
    else:
        error = -d

    # PID correction
    PID_errors[0], PID_errors[1]; PID_errors[2] = error, PID_errors[1]+error, error-PID_errors[0]
    # basically it does what is below
    #proportional, p = error
    #integrator, i += p
    #derivative, d = p - lp
    #update previous error, lp = p


    #adding the correction to the base_speed for the left and right motor
    correction = int(PID_coefficients*PID_errors)
    
    # checking the direction of the robot to reduce the speed of the closest motor engine to the correct path
    if robot.angle>0 and robot.angle<np.pi:
        if error>0:
            rspeed = max_speed - correction
            lspeed = max_speed
        else:
            rspeed = max_speed
            lspeed = max_speed - correction
    else:
        if error>0:
            rspeed = max_speed
            lspeed = max_speed - correction
        else:
            rspeed = max_speed - correction 
            lspeed = max_speed       

    #restricting speed of motors between 255 and -255
    # not necessary for the first two conditions but still in the project just for safety
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
