import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from PIL import Image


def set_seed(seed=0, device="cuda"):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def frame_image(img, frame_width, color=(255, 0, 0)):
    b = frame_width
    ny, nx = img.shape[0], img.shape[1]
    framed_img = np.array(Image.new("RGB", (b + ny + b, b + nx + b), color))
    framed_img[b:-b, b:-b] = img
    return framed_img


def plot_predictions(images, labels, figsize=(15, 3), frame_size=5):
    N = len(images)
    fig, ax = plt.subplots(1, N, figsize=figsize)
    for i in range(N):
        img = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

        if labels[i] == labels[0]:
            color = "green"
        else:
            color = "red"
        if i == 0:
            color = "yellow"

        img = frame_image(img, frame_size, color=color)
        ax[i].imshow(img)
        ax[i].axis("off")
        if len(labels[i]) > 10:
            label = labels[i][:10] + "..."
        else:
            label = labels[i]
        ax[i].text(0.5, 1.05, label, transform=ax[i].transAxes, rotation=30)
    fig.subplots_adjust(wspace=0.0)
    return fig
