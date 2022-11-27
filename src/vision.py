from typing import Literal
from src import model
import cv2
import numpy as np
import operator

# Vision parameters
# TODO: adjust when we have definitive board dimensions
BOARD_DIMENSIONS = (500, 500)

WARPED_GOAL = np.float32(
    [
        [0, 0],
        [BOARD_DIMENSIONS[0], 0],
        [0, BOARD_DIMENSIONS[1]],
        [BOARD_DIMENSIONS[0], BOARD_DIMENSIONS[1]],
    ]
)

# TODO cleanup global variables
current_warp_coordinates = None


def analyze_scene() -> model.World:
    # TODO dummy data
    return model.World(
        robot=model.Robot(
            position=model.Point(0, 0),
            angle=0,
        ),
        goal=model.Point(0, 0),
        obstacles=[],
    )


def get_robot_pose(img: np.ndarray) -> model.Robot:
    # Filter the blue (front) and yellow (back) squares
    mask_blue = get_color_mask(img, "blue")
    mask_yellow = get_color_mask(img, "yellow")

    # Find their centroids
    centroid_blue = get_centroids(mask_blue)
    centroid_yellow = get_centroids(mask_yellow)

    if len(centroid_blue) != 1 or len(centroid_yellow) != 1:
        # Failed to find the blue and yellow centroids
        return None

    # Get the coordinates of the centroids
    x1, y1 = centroid_yellow[0]
    x2, y2 = centroid_blue[0]

    # The robot position corresponds to the center of the line between the two squares
    x = x1 + (x2 - x1) / 2
    y = y1 + (y2 - y1) / 2

    # Compute signed angle between the y-axis and the robot's orientation
    v = np.array([x2 - x1, y2 - y1])
    v = v / np.linalg.norm(v)
    alpha = -np.arctan2(-v[0], v[1])

    return model.Robot(position=model.Point(x, y), angle=alpha)


def get_webcam_capture(builtin: bool) -> cv2.VideoCapture:
    index = 0 if builtin else 2
    return cv2.VideoCapture(index)


def calibrate_frame(img: np.ndarray) -> np.ndarray:
    global current_warp_coordinates
    mask = get_color_mask(img, color="red")
    centroids = get_centroids(mask, frame=True)
    if len(centroids) == 4:
        current_warp_coordinates = centroids
    return (
        get_warped_image(img, current_warp_coordinates)
        if current_warp_coordinates is not None
        else img
    )


def get_color_mask(
    img: np.ndarray, color: Literal["red", "blue", "yellow"]
) -> np.ndarray:
    # Use the bounds according to given color.
    # Note that these might have to be recalibrated in different lighting conditions.
    # TODO make these top-level file constants
    if color == "red":
        lower_bound = np.array([130, 25, 90])
        upper_bound = np.array([180, 255, 255])
    elif color == "blue":
        lower_bound = np.array([90, 150, 240])
        upper_bound = np.array([180, 255, 255])
    elif color == "yellow":
        lower_bound = np.array([0, 60, 240])
        upper_bound = np.array([90, 255, 255])
    else:
        raise Exception(f'Color "{color}" is not a valid color to filter on.')

    # Blur the image to get rid of noise
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Filter pixels in the chosen color range
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return mask


def get_centroids(img: np.ndarray, frame: bool = False) -> np.ndarray:
    # Returns all centroids of detected contours
    # Filter color before this to extract centroids of a given color

    # Use Canny edge detection
    edges = cv2.Canny(img, 100, 100)

    # Get all (external) contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Squares on the robot are smaller than the frame squares
    min_area = 100 if frame else 20

    centroids = []
    for contour in contours:
        # Make sure it's not noise
        if cv2.contourArea(contour) > min_area:
            # Compute moments for centroids
            moment = cv2.moments(contour)
            # Add small value for division by 0 errors
            cx = moment["m10"] / (moment["m00"] + 1e-6)
            cy = moment["m01"] / (moment["m00"] + 1e-6)
            centroids.append([cx, cy])

    # If these are the centroids for a frame, sort them
    if frame:
        # Sort by top left, top right, bottom left, bottom right
        centroids.sort(key=operator.itemgetter(1))
        top = sorted(centroids[:2], key=operator.itemgetter(0))
        bottom = sorted(centroids[2:], key=operator.itemgetter(0))
        centroids = top + bottom

    return np.array(centroids)


def get_warped_image(img: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Need conversion to float32 for perspectiveTransform to work
    centroids = np.float32(centroids)
    # Define transformation from centroids to goal
    transform = cv2.getPerspectiveTransform(centroids, WARPED_GOAL)
    # Warp the perspective to match board dimensions
    return cv2.warpPerspective(img, transform, BOARD_DIMENSIONS)


def main():

    source = get_webcam_capture(builtin=False)

    cv2.namedWindow("Original")
    cv2.namedWindow("Result")

    while True:

        _, frame = source.read()
        warped = calibrate_frame(frame)
        pose = get_robot_pose(warped)

        # Draw the pose
        if pose is not None:
            position = np.int32([pose.position.x, pose.position.y])

            alpha = pose.angle
            direction_vector = np.array([-np.sin(-alpha), np.cos(-alpha)])

            cv2.circle(
                warped,
                position,
                2,
                (0, 0, 255),
                thickness=4,
            )
            cv2.arrowedLine(
                warped,
                position,
                position + np.int32(100 * direction_vector),
                (0, 0, 255),
                2,
            )

        cv2.imshow("Original", frame)
        cv2.imshow("Result", warped)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
