import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from PIL import Image
from wildlife_datasets import datasets

Transform = Callable[[Image.Image], Any]


class WildlifeDataset(datasets.WildlifeDataset):
    """
    PyTorch-style dataset for image datasets. Wrapper around `WildlifeDataset` from wildlife-datasets.

    Args:
        metadata (pd.DataFrame): A pandas dataframe containing image metadata.
        root (str | None, optional): Root directory if paths in metadata are relative. If None, absolute paths in metadata are used.
        transform (Transform | None, optional): A function that takes in a PIL image and returns its transformed version.
        img_load (str, optional): Method to load images. See wildlife-datasets for available options.
        col_path (str, optional): Column name in the metadata containing image file paths.
        col_label (str, optional): Column name in the metadata containing class labels.
        load_label (bool, optional): If False, `__getitem__` returns only image instead of (image, label) tuple.
        **kwargs: Additional arguments passed to `WildlifeDataset` from wildlife-datasets.

    Attributes:
        labels (np.ndarray): An integers array of ordinal encoding of labels.
        labels_string (np.ndarray): A strings array of original labels.
        labels_map (dict): A mapping between labels and their ordinal encoding.
        num_classes (int): Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | None = None,
        transform: Transform | None = None,
        img_load: str = "full",
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            df=metadata,
            transform=transform,
            img_load=img_load,
            col_path=col_path,
            col_label=col_label,
            load_label=load_label,
            factorize_label=True,
            **kwargs,
        )


ImageDataset = WildlifeDataset


class FeatureDataset:
    """
    PyTorch-style dataset for a extracted features. Couples features with metadata.

    Args:
        features (Sequence): Index of features should match with metadata.
        metadata (pd.DataFrame): A pandas dataframe containing image metadata.
        col_label (str, optional): Column name in the metadata containing class labels.
        load_label (bool, optional): If False, `__getitem__` returns only image instead of (image, label) tuple.

    Attributes:
        labels (np.ndarray): An integers array of ordinal encoding of labels.
        labels_string (np.ndarray): A strings array of original labels.
        labels_map (dict): A mapping between labels and their ordinal encoding.
        num_classes (int): Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        features: Sequence,
        metadata: pd.DataFrame,
        col_label: str = "identity",
        load_label: bool = True,
    ):

        if len(features) != len(metadata):
            raise ValueError("Features and metadata (lables) have different length ! ")

        self.load_label = load_label
        self.features = features
        self.metadata = metadata.reset_index(drop=True)
        self.col_label = col_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].values)

    @property
    def labels_string(self):
        return self.metadata[self.col_label].astype(str).values

    def __getitem__(self, idx: int):
        if self.load_label:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    def __len__(self):
        return len(self.metadata)

    @property
    def num_classes(self):
        return len(self.labels_map)

    def save(self, path: str):
        data = {
            "features": self.features,
            "metadata": self.metadata,
            "col_label": self.col_label,
            "load_label": self.load_label,
        }
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, path: str, **config):
        with open(path, "rb") as file:
            data = pickle.load(file)
        return cls(**data, **config)

    @classmethod
    def from_config(cls, config):
        path = config.pop("path")
        return cls.load(path, **config)


class FeatureDatabase(FeatureDataset):
    """Alias for FeatureDataset"""

    pass
