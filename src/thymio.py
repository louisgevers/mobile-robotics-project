import tdmclient.notebook
import numpy as np
from src import model


# TODO use thymio interface instead
# This does not work outside of a notebook...


async def initialize():
    await tdmclient.notebook.start()


async def stop():
    tdmclient.notebook.stop()


@tdmclient.notebook.sync_var
def process_command(command: model.MotorSpeed):
    global motor_left_target, motor_right_target
    motor_left_target = command.left
    motor_right_target = command.right


@tdmclient.notebook.sync_var
def read_sensor_data() -> model.SensorReading:
    global prox_horizontal, prox_ground_delta, motor_left_speed, motor_right_speed
    horizontal = np.array(prox_horizontal)
    ground = np.array(prox_ground_delta)
    return model.SensorReading(
        horizontal=model.HorizontalSensor(horizontal),
        vertical=model.GroundSensor(ground),
        motor=model.MotorSpeed(motor_left_speed, motor_right_speed),
    )
