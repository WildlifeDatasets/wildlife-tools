from typing import Callable, Tuple
import torch
import numpy as np
import pandas as pd
import cv2

from wildlife_tools.data import WildlifeDataset


class NumpyDataset(WildlifeDataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        img_size: Tuple,
        root: str | None = None,
        transform: Callable | None = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
    ):
        super().__init__(
            metadata=metadata,
            root=root,
            transform=transform,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label,
            img_load=img_load,
        )
        self.img_size = img_size

    def get_image(self, path: str):
        img = cv2.imread(path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class BalancedImageDataset(WildlifeDataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | None = None,
        transform: Callable | None = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
    ):
        super().__init__(
            metadata=metadata,
            root=root,
            transform=transform,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label,
            img_load=img_load,
        )
        torch_labels = torch.from_numpy(self.labels)
        class_counts = torch.bincount(torch_labels)
        class_weights = 1.0 / class_counts
        self.sample_weights = torch.zeros_like(torch_labels, dtype=torch.float32)
        for label in np.unique(self.labels):
            label_mask = self.labels == label
            self.sample_weights[label_mask] = class_weights[label]

    def get_image(self, path: str):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
