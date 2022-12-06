import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_image(img: np.ndarray):
    plt.figure(figsize=(12, 7))
    # Matplotlib uses RGB format
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb, aspect="auto")
    plt.show()
