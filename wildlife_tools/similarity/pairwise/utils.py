import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualise_matches(img0: Image, keypoints0: np.ndarray, img1: Image, keypoints1: list, ax=None):
    """
    Visualise matches between two images.

    Args:
        img0 (np.array or PIL Image): First image.
        keypoints0 (np.array): Keypoints in the first image.
        img1 (np.array): Second image.
        keypoints1 (np.array): Keypoints in the second image.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to draw on. If None, a new axis is created.
    """

    # Convert images to numpy arrays
    img0 = np.array(img0)
    img1 = np.array(img1)

    keypoints0 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints0]
    keypoints1 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints1]

    # Create dummy matches (DMatch objects)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]

    # Draw matches
    img_matches = cv2.drawMatches(
        img0,
        keypoints0,
        img1,
        keypoints1,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Plotting
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(img_matches)
    ax.axis("off")
