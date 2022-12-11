from tdmclient import ClientAsync, aw
import numpy as np
from src import model
from threading import Thread
import asyncio
import time


class Thymio:

    _node = None
    _variables = None

    def __init__(self) -> None:
        # Create initial variables
        self._variables = {
            "motor.left.speed": [0],
            "motor.right.speed": [0],
            "prox.horizontal": [0, 0, 0, 0, 0],
            "prox.ground.delta": [0, 0],
        }
        # Create thread for reading variables in background
        t = Thread(target=self._start)
        t.start()
        # Wait a bit for initialization
        time.sleep(1)
        self.kill_lamps()

    def _start(self):
        """
        Starts an asynchronous connection with the Thymio.
        This has to be run inside a thread to not block the application.
        """
        try:
            with ClientAsync() as client:

                async def prog():
                    with await client.lock() as self._node:
                        await self._node.watch(variables=True)
                        self._node.add_variables_changed_listener(
                            self.on_variables_changed
                        )
                        await client.sleep()

                asyncio.run(prog())
        except:
            print("Connection to Thymio failed!")

    def stop(self):
        """
        Stops the thymio and unlocks the connection.
        """
        self.process_command(model.MotorSpeed(0, 0))
        aw(self._node.unlock())

    def on_variables_changed(self, node, variables):
        """
        Callback for variable changes from thymio.
        """
        if self._variables is None:
            self._variables = variables
        else:
            self._variables.update(variables)

    def process_command(self, command: model.MotorSpeed):
        """
        Send a motor speed command to the thymio.
        """
        if self._node is None:
            raise "Node has not been initialized before starting robot command"
        variables = {
            "motor.left.target": [int(command.left)],
            "motor.right.target": [int(command.right)],
        }
        aw(self._node.set_variables(variables))

    def kill_lamps(self):
        aw(
            self._node.set_variables(
                {
                    "leds.top": [0, 0, 0],
                    # "leds.prox.h": [0, 0, 0, 0, 0, 0, 0, 0],
                    # "leds.prox.v": [0, 0],
                    # "leds.buttons": [0, 0, 0, 0],
                    # "leds.rc": [0, 0],
                    # "leds.circle": [0, 0, 0, 0, 0, 0, 0, 0],
                    # "leds.bottom.left": [0, 0, 0],
                    # "leds.bottom.right": [0, 0, 0],
                    # "leds.temperature": [0, 0],
                    # "leds.sound": [0],
                }
            )
        )

    def read_sensor_data(self) -> model.SensorReading:
        """
        Obtain the latest sensor data from the thymio
        """
        if self._node is None:
            raise "Node has not been initialized before starting robot command"
        horizontal = np.array(self._variables["prox.horizontal"])
        ground = np.array(self._variables["prox.ground.delta"])
        return model.SensorReading(
            horizontal=model.HorizontalSensor(horizontal),
            ground=model.GroundSensor(ground),
            motor=model.MotorSpeed(
                self._variables["motor.left.speed"][0],
                self._variables["motor.right.speed"][0],
            ),
        )

    def read_robot_position(self) -> model.Robot:
        """
        Obtain the latest sensor data from the robot position
        """
        if self._node is None:
            raise "Node has not been initialized before starting robot command"
        # position = #I have no idea how to get this data
        # angle = #I have no idea how to get this data
        return model.Robot(
            position=model.Point(0.0, 0.0),
            angle=0.0,
        )
