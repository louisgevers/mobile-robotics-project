from typing import Literal, Mapping
import cv2
import numpy as np
from src import model

# Resolution after callibration
TARGET_RESOLUTION = (840, 600)  # A1 sheet

# For obstacles
DILATE_FACTOR = 80

# Keep last warp coordinates
last_warp_coordinates = None

# Keep last robot position
last_robot_position = None


def get_webcam_capture(builtin: bool) -> cv2.VideoCapture:
    index = 0 if builtin else 2
    return cv2.VideoCapture(index)


def analyze_scene(img: np.ndarray) -> model.World:
    # All the algorithms assume calibrated image
    img = calibrate(img)

    # Compute pose
    pose = get_robot_pose(img, should_calibrate=False)  # We already calibrated

    # Compute goal
    mask_goal = get_color_mask(img, "blue")
    centroids = get_centroids(mask_goal)
    if len(centroids) > 1:
        raise Exception("More than 1 goal detected! Please calibrate colors")
    elif len(centroids) == 0:
        raise Exception(
            "No goal detected! Please calibrate colors and verify goal is in view"
        )
    else:
        x, y = centroids[0]
        goal = model.Point(x, y)

    # Compute obstacles
    obstacles = []
    mask_obstacles = get_color_mask(img, "red")
    contours = get_polygon_contours(mask_obstacles, dilate=DILATE_FACTOR)
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


def get_robot_pose(img: np.ndarray, should_calibrate=True) -> model.Robot:
    # TODO make this a class
    global last_robot_position
    if should_calibrate:
        img = calibrate(img)
    # Get aruco markers
    markers = get_aruco_dict(img)
    # Thymio aruco has ID 4
    if 4 in markers:
        thymio_marker = markers[4]
        last_robot_position = compute_robot_pose(thymio_marker)
    else:
        print("Warning: could not find robot pose!")
    return last_robot_position


def calibrate(img: np.ndarray) -> np.ndarray:
    markers = get_aruco_dict(img)
    return get_warped_image(img, markers)


def get_aruco_dict(img: np.ndarray) -> Mapping[int, np.ndarray]:
    # The printed markers are 4x4 types
    markers = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # Retrieve the markers and corresponding ids
    corners, ids, _ = cv2.aruco.detectMarkers(img, markers)

    if corners is None or ids is None:
        return {}

    # Create a mapping from aruco marker id to the corresponding corner
    # They are wrapped in an array, therefore remove that layer by accessing first index
    return dict([(marker[0], corner[0]) for corner, marker in zip(corners, ids)])


def get_warped_image(
    img: np.ndarray, corners_by_id: Mapping[int, np.ndarray]
) -> np.ndarray:
    # TODO make a class
    global last_warp_coordinates
    # top right, top left, bottom right, bottom left
    id_order = [0, 2, 1, 3]
    if all([aruco_id in corners_by_id for aruco_id in id_order]):
        # Retrieve the corner coordinates
        corners = np.float32([corners_by_id[aruco_id][0] for aruco_id in id_order])
        last_warp_coordinates = corners
    elif last_warp_coordinates is not None:
        # Use previous
        corners = last_warp_coordinates
    else:
        # Cannot transform
        print("Warning: could not calibrate image!")
        return img

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


def get_polygon_contours(img: np.ndarray, dilate=0) -> np.ndarray:
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
