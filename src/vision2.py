import operator
from typing import Literal, Mapping
import cv2
import numpy as np
from src import model

# Resolution after callibration
TARGET_RESOLUTION = (840, 600)  # A1 sheet


def get_aruco_dict(img: np.ndarray) -> Mapping[int, np.ndarray]:
    # The printed markers are 4x4 types
    markers = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # Retrieve the markers and corresponding ids
    corners, ids, _ = cv2.aruco.detectMarkers(img, markers)

    # Create a mapping from aruco marker id to the corresponding corner
    # They are wrapped in an array, therefore remove that layer by accessing first index
    return dict([(marker[0], corner[0]) for corner, marker in zip(corners, ids)])


def get_warped_image(
    img: np.ndarray, corners_by_id: Mapping[int, np.ndarray]
) -> np.ndarray:
    # top right, top left, bottom right, bottom left
    id_order = [0, 2, 1, 3]
    # Retrieve the corner coordinates
    corners = np.float32([corners_by_id[aruco_id][0] for aruco_id in id_order])
    # Define target
    target_transform = get_target_transform(TARGET_RESOLUTION)
    # Transform the image
    transform = cv2.getPerspectiveTransform(corners, target_transform)
    return cv2.warpPerspective(img, transform, TARGET_RESOLUTION)


def compute_robot_pose(aruco_marker: np.ndarray) -> model.Robot:
    # Top left corner
    x1, y1 = aruco_marker[0]
    # Bottom right corner
    x2, y2 = aruco_marker[2]

    # Centroid is the position of the robot
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    # Compute angle between y-axis and vector
    v = np.array([x1 - x2, y1 - y2])
    v = v / np.linalg.norm(v)
    alpha = -np.arctan2(-v[0], v[1])

    # We computed the angle of the diagonal, so add 45 degrees for correction
    alpha -= np.pi / 4

    return model.Robot(position=model.Point(x, y), angle=alpha)


def get_color_mask(img: np.ndarray, color: Literal["red", "blue"]) -> np.ndarray:
    # Use the bounds according to given color.
    # Note that these might have to be recalibrated in different lighting conditions.
    # TODO make these top-level file constants and add calibration script
    if color == "red":
        lower_bound = np.array([130, 25, 90])
        upper_bound = np.array([180, 255, 255])
    elif color == "blue":
        lower_bound = np.array([50, 70, 70])
        upper_bound = np.array([110, 255, 255])
    else:
        raise Exception(f'Color "{color}" is not a valid color to filter on.')

    # Blur the image to get rid of noise
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Filter pixels in the chosen color range
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return mask


def get_centroids(img: np.ndarray) -> np.ndarray:
    # Returns all centroids of detected contours
    # Filter color before this to extract centroids of a given color

    # Use Canny edge detection
    edges = cv2.Canny(img, 100, 100)

    # Get all (external) contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def get_target_transform(resolution) -> np.ndarray:
    # Has to be float32 for perspectiveTransform
    return np.float32(
        [
            [0, 0],
            [resolution[0], 0],
            [0, resolution[1]],
            [resolution[0], resolution[1]],
        ]
    )
