import numpy as np


def get_global_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    # Angle between y-axis and given vector
    v = np.array([x2 - x1, y2 - y1])
    v = v / np.linalg.norm(v)
    angle = -np.arctan2(-v[0], v[1])
    return angle
