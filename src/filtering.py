from src import model


def update_robot(
    robot: model.Robot, command: model.MotorSpeed, sensors: model.SensorReading
):
    ### Access the sent command (i.e. sent velocities to the motors)
    # command.left
    # command.right

    ### Access the read data
    # sensors.ground.left or sensors.ground.v[0]
    # sensors.ground.right or sensors.ground.v[1]

    # sensors.horizontal.left or sensors.horizontal.v[0]
    # sensors.horizontal.center_left or sensors.horizontal.v[1]
    # sensors.horizontal.center or sensors.horizontal.v[2]
    # sensors.horizontal.center_right or sensors.horizontal.v[3]
    # sensors.horizontal.right or sensors.horizontal.v[4]

    # sensors.motor.left
    # sensors.motor.right

    # ---

    ### Update robot pose (no need to return)
    # robot.angle =
    # robot.position.x =
    # robot.position.y =

    pass
