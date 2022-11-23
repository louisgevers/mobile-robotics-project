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


def get_color_mask(img: np.ndarray, color: Literal["red"]) -> np.ndarray:
    # Use the bounds according to given color.
    # Note that these might have to be recalibrated in different lighting conditions.
    if color == "red":
        lower_bound = np.array([130, 25, 90])
        upper_bound = np.array([180, 255, 255])
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

    centroids = []
    for contour in contours:
        # Make sure it's not noise
        if cv2.contourArea(contour) > 100:
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

    source = get_webcam_capture(builtin=True)

    cv2.namedWindow("Original")
    cv2.namedWindow("Result")

    while True:

        _, frame = source.read()
        warped = calibrate_frame(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Result", warped)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
