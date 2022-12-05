from src import vision, pathfinder, model, navglobal, thymio
from enum import Enum
import numpy as np
import cv2


class DemoState(Enum):
    START = 0
    ARUCO = 1
    RED = 2
    BLUE = 3
    WORLD = 4
    PATH = 5
    GLOBAL = 6
    STOP = 7


class FrameHandler:
    def handle(self, frame: np.ndarray):
        cv2.imshow("Frame", frame)


class ArucoFrameHandler(FrameHandler):
    def __init__(self) -> None:
        super().__init__()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    def handle(self, frame: np.ndarray):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        super().handle(frame)


class ColorFrameHandler(FrameHandler):
    def __init__(self, color: str) -> None:
        super().__init__()
        self.color = color

    def handle(self, frame: np.ndarray):
        calibrated = vision.calibrate(frame)
        mask = vision.get_color_mask(calibrated, self.color)
        super().handle(mask)


class WorldFrameHandler(FrameHandler):
    def __init__(self) -> None:
        super().__init__()
        self.world = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        calibrated = vision.calibrate(frame)
        self.world = vision.analyze_scene(frame)
        vision.draw_world(calibrated, self.world)
        return calibrated

    def handle(self, frame: np.ndarray):
        calibrated = self.preprocess(frame)
        super().handle(calibrated)


class PathFrameHandler(WorldFrameHandler):
    def __init__(self) -> None:
        super().__init__()
        self.path = None

    def handle(self, frame: np.ndarray):
        calibrated = self.preprocess(frame)
        path = pathfinder.find_path(self.world)
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            cv2.line(
                calibrated,
                np.int32(start.v),
                np.int32(end.v),
                color=(255, 0, 255),
                thickness=2,
            )
        self.path = path
        cv2.imshow("Frame", calibrated)


class GlobalFrameHandler(FrameHandler):
    def __init__(self, th: thymio.Thymio, prev_handler: PathFrameHandler) -> None:
        self.prev_handler = prev_handler
        self.th = th

    def handle(self, frame: np.ndarray):
        path = self.prev_handler.path
        robot = vision.get_robot_pose(frame)
        if robot is not None:
            self.prev_handler.world.robot = robot
            command = navglobal.follow_path(robot, path)
            self.th.process_command(command)
        calibrated = vision.calibrate(frame)
        vision.draw_world(calibrated, self.prev_handler.world)
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            cv2.line(
                calibrated,
                np.int32(start.v),
                np.int32(end.v),
                color=(255, 0, 255),
                thickness=2,
            )
        super().handle(calibrated)


class StopHandler(FrameHandler):
    def __init__(self, th: thymio.Thymio) -> None:
        self.th = th

    def handle(self, frame: np.ndarray):
        self.th.stop()


def main(th: thymio.Thymio):
    path = PathFrameHandler()

    FRAME_HANDLERS = {
        DemoState.START: FrameHandler(),
        DemoState.ARUCO: ArucoFrameHandler(),
        DemoState.RED: ColorFrameHandler("red"),
        DemoState.BLUE: ColorFrameHandler("blue"),
        DemoState.WORLD: WorldFrameHandler(),
        DemoState.PATH: path,
        DemoState.GLOBAL: GlobalFrameHandler(th, path),
        DemoState.STOP: StopHandler(th),
    }

    state = DemoState.START

    source = vision.get_webcam_capture(builtin=True)

    while True:

        _, frame = source.read()

        handler = FRAME_HANDLERS[state]
        handler.handle(frame)

        # Press q to quit
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q"):
            break
        elif k & 0xFF == ord(" "):
            state = DemoState(state.value + 1)


if __name__ == "__main__":
    th = thymio.Thymio()
    main(th)
    th.stop()
