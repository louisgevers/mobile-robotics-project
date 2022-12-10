from typing import Mapping
import cv2
import numpy as np
from src import model, utils
from dataclasses import dataclass
import threading


class TakeLatestFrameThread(threading.Thread):
    def __init__(self, source: cv2.VideoCapture, recording: str = None):
        self.source = source
        self.frame = None
        self.recording = recording
        self.out = None
        super().__init__()
        self.start()

    def setup_recording(self):
        frame_width = int(self.source.get(3))
        frame_height = int(self.source.get(4))

        self.out = cv2.VideoWriter(
            self.recording,
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (frame_width, frame_height),
        )

    def run(self):
        while True:
            ret, frame = self.source.read()
            if ret:
                self.frame = frame
                if self.out is not None:
                    self.out.write(frame)


class FrameSource:
    """
    Interface for retrieving a frame for vision.
    """

    def get_frame(self) -> np.ndarray:
        """
        Returns the frame from retrieved from the source.

        Returns:
            np.ndarray: OpenCV image
        """
        pass


class ImageSource(FrameSource):
    def __init__(self, path: str) -> None:
        """
        Uses an image at the provided path as source for vision.

        Args:
            path (str): Path of the image
        """
        self.path = path

    def get_frame(self) -> np.ndarray:
        return cv2.imread(self.path)


class VideoSource(FrameSource):
    def __init__(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)
        self.ret = True

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        self.ret = ret
        return frame


class WebcamSource(FrameSource):
    def __init__(self, builtin: bool, recording: str = None) -> None:
        """
        Uses a webcam as source of frames.

        Args:
            builtin (bool): Whether to use the laptop webcam.
        """
        index = 0 if builtin else 2
        cap = cv2.VideoCapture(index)
        self.camera_thread = TakeLatestFrameThread(cap, recording)

    def get_frame(self) -> np.ndarray:
        return self.camera_thread.frame


@dataclass
class HSVBound:
    """
    Determines the upper and lower bounds for HSV filtering.
    Each bound should be an numpy array of three elements corresponding
    to the H, S, and V value respectively.
    """

    lb: np.ndarray
    ub: np.ndarray


class VisionTools:
    def __init__(
        self, aruco_dict=cv2.aruco.DICT_4X4_50, target_resolution=(840, 600)
    ) -> None:
        """
        Collection of utility functions for the vision pipeline. This class encapsulates
        some state variables such as the latest warp coordinates to handle cases where
        features are dropped from certain frames for robustness.

        Args:
            aruco_dict (int, optional): The type of aruco markers. Defaults to cv2.aruco.DICT_4X4_50.
            target_resolution (tuple, optional): Target resolution after callibration. Defaults to (840, 600).
        """
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.target_resolution = target_resolution
        self.target_transform = np.float32(
            [
                [0, 0],
                [target_resolution[0], 0],
                [0, target_resolution[1]],
                [target_resolution[0], target_resolution[1]],
            ]
        )
        self.latest_warp_coordinates = None

    def get_aruco_markers(self, img: np.ndarray):
        """
        Returns the detected aruco corners and ids from the image.

        Args:
            img (np.ndarray): Image to detect the aruco markers from.

        Returns:
            tuple: Tuple of aruco corners and ids
        """
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict)
        return corners, ids

    def get_aruco_dict(self, img: np.ndarray) -> Mapping[int, np.ndarray]:
        # Retrieve the markers and corresponding ids
        corners, ids = self.get_aruco_markers(img)

        # If none found return empty dict
        if corners is None or ids is None:
            return {}

        # Create a mapping from aruco marker id to the corresponding corner
        # They are wrapped in an array, therefore remove that layer by accessing first index
        return dict([(marker[0], corner[0]) for corner, marker in zip(corners, ids)])

    def get_aruco_calibrated(
        self, img: np.ndarray, corners_by_id: Mapping[int, np.ndarray]
    ) -> np.ndarray:
        # top right, top left, bottom right, bottom left
        id_order = [0, 2, 1, 3]
        if all([aruco_id in corners_by_id for aruco_id in id_order]):
            # Retrieve the corner coordinates
            corners = np.float32([corners_by_id[aruco_id][0] for aruco_id in id_order])
            self.last_warp_coordinates = corners
        else:
            # Corners were not found
            corners = None
        return self.get_warped_image(img, corners)

    def get_warped_image(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if corners is not None:
            self.last_warp_coordinates = corners
        elif self.last_warp_coordinates is None:
            # Cannot transform
            print("Warning: could not calibrate image!")
            return img
        # Transform the image
        transform = cv2.getPerspectiveTransform(
            self.last_warp_coordinates, self.target_transform
        )
        return cv2.warpPerspective(img, transform, self.target_resolution)

    def get_color_mask(self, img: np.ndarray, bounds: HSVBound) -> np.ndarray:
        # Blur the image to get rid of noise
        blur = cv2.GaussianBlur(img, (15, 15), 0)
        # Convert to HSV color space
        hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # Filter pixels in the chosen color range
        mask = cv2.inRange(hsv_img, bounds.lb, bounds.ub)
        return mask

    def get_centroids(self, img: np.ndarray) -> np.ndarray:
        # Returns all centroids of detected contours
        # Filter color before this to extract centroids of a given color

        # Use Canny edge detection
        edges = cv2.Canny(img, 100, 100)

        # Get all (external) contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by descending area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        centroids = []
        for contour in contours:
            # Make sure it's not noise
            if cv2.contourArea(contour) > 20:
                # Compute moments for centroids
                moment = cv2.moments(contour)
                # Add small value for division by 0 errors
                cx = moment["m10"] / (moment["m00"] + 1e-6)
                cy = moment["m01"] / (moment["m00"] + 1e-6)
                centroids.append([cx, cy])

        return np.array(centroids)

    def get_polygon_contours(self, img: np.ndarray, dilate=0) -> np.ndarray:
        # Returns the (approximated) contours
        # Filter color before this to extract contours of a given color

        # Dilate the shapes before finding contours
        if dilate > 0:
            kernel = np.ones((dilate, dilate))
            img = cv2.dilate(img, kernel, iterations=1)

        # Get all (external) contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours with polygon
        polygons = []
        for contour in contours:
            # Make sure it's not noise
            if cv2.contourArea(contour) > 20:
                approximation = cv2.approxPolyDP(contour, 10, True)
                polygons.append(approximation)
        return polygons


class VisionPipeline:
    def __init__(
        self,
        source: FrameSource,
        tools: VisionTools = VisionTools(),
        red_bounds=HSVBound(lb=np.array([130, 25, 90]), ub=np.array([180, 255, 255])),
        blue_bounds=HSVBound(lb=np.array([50, 70, 70]), ub=np.array([110, 255, 255])),
        dilate_factor=80,
    ) -> None:
        self.source = source
        self.tools = tools
        self.latest_frame = None
        self.red_bounds = red_bounds
        self.blue_bounds = blue_bounds
        self.dilate_factor = dilate_factor
        self.last_robot_pose = None

    def analyze_scene(self) -> model.World:
        frame = self.__read_frame(with_calibration=True)
        return self.__compute_world(frame)

    def get_robot_pose(self) -> model.Robot:
        frame = self.__read_frame(with_calibration=True)
        return self.__compute_robot_pose(frame)

    def __read_frame(self, with_calibration: bool = False) -> np.ndarray:
        frame = self.source.get_frame()
        if with_calibration:
            frame = self.__calibrate(frame)
        self.latest_frame = frame
        return self.latest_frame

    def __calibrate(self, img: np.ndarray) -> np.ndarray:
        markers = self.tools.get_aruco_dict(img)
        return self.tools.get_aruco_calibrated(img, markers)

    def __compute_world(self, img: np.ndarray) -> model.World:
        # Compute pose
        pose = self.__compute_robot_pose(img)

        # Compute goal
        mask_goal = self.tools.get_color_mask(img, self.blue_bounds)
        centroids = self.tools.get_centroids(mask_goal)
        if len(centroids) == 0:
            raise Exception(
                "No goal detected! Please calibrate colors and verify goal is in view"
            )
        else:
            x, y = centroids[0]
            goal = model.Point(x, y)

        # Compute obstacles
        obstacles = []
        mask_obstacles = self.tools.get_color_mask(img, self.red_bounds)
        contours = self.tools.get_polygon_contours(
            mask_obstacles, dilate=self.dilate_factor
        )
        for contour in contours:
            points = []
            for point in contour:
                x, y = point[0]
                points.append(model.Point(x, y))
            obstacles.append(points)

        return model.World(
            robot=pose,
            goal=goal,
            obstacles=obstacles,
        )

    def __compute_robot_pose(self, img: np.ndarray) -> model.Robot:
        # Get aruco markers
        markers = self.tools.get_aruco_dict(img)
        # Thymio aruco has ID 4
        if 4 not in markers:
            print("Warning: could not find robot pose!")
            return None
        # Get marker
        aruco_marker = markers[4]

        # Top left corner
        x1, y1 = aruco_marker[0]
        # Bottom right corner
        x2, y2 = aruco_marker[2]

        # Centroid is the position of the robot
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        # Compute angle between y-axis and vector
        alpha = utils.get_global_angle(x2, y2, x1, y1)

        # We computed the angle of the diagonal, so add 45 degrees for correction
        alpha -= np.pi / 4

        self.last_robot_pose = model.Robot(position=model.Point(x, y), angle=alpha)
        return self.last_robot_pose


def draw_world(img: np.ndarray, world: model.World):
    # Draw a green circle at the goal
    cv2.circle(img, np.int32(world.goal.v), 4, (0, 255, 0), 2)

    # Convert obstacle points to contours
    contours = []
    for obstacle in world.obstacles:
        contour = []
        for point in obstacle:
            contour.append([point.v])
        contours.append(np.array(contour))
    # Draw blue contours for obstacles
    cv2.drawContours(img, contours, -1, (255, 0, 0))

    # Draw robot position and orientation
    # Convert to array for drawing
    position = np.int32(world.robot.position.v)

    # Compute direction vector for drawing
    alpha = world.robot.angle
    direction_vector = np.array([-np.sin(-alpha), np.cos(-alpha)])

    # Draw a circle at the detected position
    cv2.circle(img, position, 4, color=(0, 255, 255), thickness=8)
    cv2.arrowedLine(
        img, position, position + np.int32(100 * direction_vector), (0, 0, 255), 2
    )
