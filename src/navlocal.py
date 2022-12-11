from src import model, navglobal
import numpy as np
import time

SPEED = navglobal.SPEED / 2


class LocalNavigation:
    def __init__(
        self, ir_sensor_threshold=1100, delta_angle=80 / 5, turn_threshold=25, timeout=4 #Key parameters
    ) -> None:
        self.ir_sensor_threshold = ir_sensor_threshold
        self.delta_angle = np.deg2rad(delta_angle)
        self.turn_threshold = np.deg2rad(turn_threshold)
        self.timeout = timeout
        self.last_seen = None
    
    #Main function running the local obstacle avoidance
    def avoidance_mode(self, sensor_data: model.SensorReading) -> bool:
        if self.sees_obstacle(sensor_data): #Run until no obstacle is found
            self.last_seen = time.time()
            return True
        elif self.last_seen is not None:
            return time.time() - self.last_seen < self.timeout
        else:
            return False

    def next_command(self, sensor_data: model.SensorReading) -> model.MotorSpeed:
        if not self.sees_obstacle(sensor_data):
            return model.MotorSpeed(left=SPEED, right=SPEED)
        angle = self.get_angle(sensor_data)
        if abs(angle) < self.turn_threshold or max(sensor_data.horizontal.v) > 4000:
            return self.turn(angle)
        else:
            return model.MotorSpeed(left=SPEED, right=SPEED)

    def sees_obstacle(self, sensor_data: model.SensorReading) -> bool: #Check for obstacles
        return any(sensor_data.horizontal.v > self.ir_sensor_threshold)

    def get_angle(self, sensor_data: model.SensorReading) -> float: #Angle calculation function
        sum_alpha = 0
        for alpha, sensor in zip([-2, -1, 0, 1, 2], sensor_data.horizontal.v):
            sum_alpha += alpha * self.delta_angle * sensor

        return sum_alpha / sum(sensor_data.horizontal.v)

    def turn(self, angle: float) -> model.MotorSpeed:
        if angle > 0:
            return model.MotorSpeed(left=-SPEED, right=SPEED)
        else:
            return model.MotorSpeed(left=SPEED, right=-SPEED)
