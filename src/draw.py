from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src import vision, model


def plot_image(img: np.ndarray):
    plt.figure(figsize=(12, 7))
    # Matplotlib uses RGB format
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb, aspect="auto")
    plt.show()


def display_aruco_markers(source: vision.FrameSource, tools: vision.VisionTools):
    img = source.get_frame()
    corners, ids = tools.get_aruco_markers(img)

    result = img.copy()
    cv2.aruco.drawDetectedMarkers(result, corners, ids)
    plot_image(result)


def draw_path(img: np.ndarray, path: Sequence[model.Point]):
    # Draw the path
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        cv2.line(
            img, np.int32(start.v), np.int32(end.v), color=(255, 0, 255), thickness=2
        )


def draw_robot_pose(img: np.ndarray, robot: model.Robot) -> np.ndarray:
    result = img.copy()
    # Convert to array for drawing
    position = np.int32(robot.position.v)

    # Compute direction vector for drawing
    alpha = robot.angle
    direction_vector = np.array([-np.sin(-alpha), np.cos(-alpha)])

    # Draw a circle at the detected position
    cv2.circle(result, position, 4, color=(0, 255, 255), thickness=8)
    cv2.arrowedLine(
        result, position, position + np.int32(100 * direction_vector), (0, 0, 255), 2
    )
    return result


def draw_centroids(img: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    result = img.copy()
    for centroid in centroids:
        cv2.circle(result, np.int32(centroid), 4, color=(0, 255, 0), thickness=2)
    return result


def draw_contours(img: np.ndarray, contours: np.ndarray) -> np.ndarray:
    result = img.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0))
    return result


def draw_world(img: np.ndarray, world: model.World) -> np.ndarray:
    result = img.copy()
    # Draw a green circle at the goal
    cv2.circle(result, np.int32(world.goal.v), 4, (0, 255, 0), 2)

    # Convert obstacle points to contours
    contours = []
    for obstacle in world.obstacles:
        contour = []
        for point in obstacle:
            contour.append([point.v])
        contours.append(np.array(contour))
    # Draw blue contours for obstacles
    cv2.drawContours(result, contours, -1, (255, 0, 0))

    # Draw robot position and orientation
    # Convert to array for drawing
    position = np.int32(world.robot.position.v)

    # Compute direction vector for drawing
    alpha = world.robot.angle
    direction_vector = np.array([-np.sin(-alpha), np.cos(-alpha)])

    # Draw a circle at the detected position
    cv2.circle(result, position, 4, color=(0, 255, 255), thickness=8)
    cv2.arrowedLine(
        result, position, position + np.int32(100 * direction_vector), (0, 0, 255), 2
    )
    return result
