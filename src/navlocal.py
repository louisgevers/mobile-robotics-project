from src import model, thymio
import time


def avoid_obstacle(sensor_data: model.SensorReading) -> model.MotorSpeed:
    ### Access the read data
    # sensors.ground.left or sensors.ground.v[0]
    # sensors.ground.right or sensors.ground.v[1]

    # sensors.horizontal.left or sensors.horizontal.v[0]
    # sensors.horizontal.center_left or sensors.horizontal.v[1]
    # sensors.horizontal.center or sensors.horizontal.v[2]
    # sensors.horizontal.center_right or sensors.horizontal.v[3]
    # sensors.horizontal.right or sensors.horizontal.v[4]

    ### Return the command to send to the motors
    # return model.MotorSpeed(left=0, right=0)
    pass


def sees_obstacle(sensor_data: model.SensorReading) -> bool:
    ### Access the read data
    # sensors.ground.left or sensors.ground.v[0]
    # sensors.ground.right or sensors.ground.v[1]

    # sensors.horizontal.left or sensors.horizontal.v[0]
    # sensors.horizontal.center_left or sensors.horizontal.v[1]
    # sensors.horizontal.center or sensors.horizontal.v[2]
    # sensors.horizontal.center_right or sensors.horizontal.v[3]
    # sensors.horizontal.right or sensors.horizontal.v[4]

    ### Return whether you see an obstacle
    # return False
    pass


# You can run this file directly to test your code on the thymio
if __name__ == "__main__":
    thymio.initialize()
    # Read the sensor data
    sensor_data = thymio.read_sensor_data()
    # Avoid obstacle while you see it
    while sees_obstacle(sensor_data):
        command = avoid_obstacle(sensor_data)
        thymio.process_command(command)
        time.sleep(0.1)
    thymio.stop()
