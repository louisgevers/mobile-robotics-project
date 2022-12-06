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
