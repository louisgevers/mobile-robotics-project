from typing import Sequence
import cv2
import time
from src import model, vision, navglobal, navlocal, thymio, pathfinder


class Recorder:
    def __init__(
        self,
        viz: vision.VisionPipeline,
        filename: str,
        start: float,
    ) -> None:
        self.viz = viz
        self.filename = filename
        self.start = start

    def run(self, world: model.World, path: Sequence[model.Point]):
        # Create an array to record all states
        record = []
        # Connect to the thymio
        th = thymio.Thymio()
        # Record the scene
        record.append(world)
        # Record the path
        record.append(path)
        # Create global navigation
        navigator = navglobal.GlobalNavigation(path)
        # Create local navigation
        avoider = navlocal.LocalNavigation()
        was_avoiding = False
        # Initial sensor_data
        sensor_data = th.read_sensor_data()
        # Global navigation until the goal
        while not world.robot_at_goal(60):
            world.robot = self.viz.get_robot_pose()
            if avoider.avoidance_mode(sensor_data):
                command = avoider.next_command(sensor_data)
                was_avoiding = True
            else:
                if was_avoiding:
                    path = pathfinder.find_path(world)
                    navigator = navglobal.GlobalNavigation(path)
                    was_avoiding = False
                command = navigator.next_command(world.robot)
            th.process_command(command)
            sensor_data = th.read_sensor_data()
            # Add timestep to record
            elapsed = time.time() - self.start
            record.append({elapsed: (world.robot, command, sensor_data.motor, path)})
        # Disconnect thymio
        th.stop()
        # Dump to file
        self.dump_to_file(record)

    def dump_to_file(self, record):
        with open(self.filename, "w") as f:
            lines = [str(item) + "\n" for item in record]
            for line in lines:
                f.write(line)
