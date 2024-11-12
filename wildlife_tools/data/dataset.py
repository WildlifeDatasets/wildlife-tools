import json
import os
import pickle
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_coco
from PIL import Image


class ImageDataset:
    """
    PyTorch-style dataset for a image datasets

    Args:
        metadata: A pandas dataframe containing image metadata.
        root: Root directory if paths in metadata are relative. If None, absolute paths in metadata are used.
        transform: A function that takes in an image and returns its transformed version.
        col_path: Column name in the metadata containing image file paths.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, \_\_getitem\_\_ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | None = None,
        transform: Callable | None = None,
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.col_path = col_path
        self.col_label = col_label
        self.load_label = load_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].values)

    @property
    def labels_string(self):
        return self.metadata[self.col_label].astype(str).values

    @property
    def num_classes(self):
        return len(self.labels_map)

    def __len__(self):
        return len(self.metadata)

    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        if self.root:
            img_path = os.path.join(self.root, data[self.col_path])
        else:
            img_path = data[self.col_path]
        img = self.get_image(img_path)

        if self.transform:
            img = self.transform(img)

        if self.load_label:
            return img, self.labels[idx]
        else:
            return img


class WildlifeDataset(ImageDataset):
    """
    PyTorch-style dataset for a datasets from wildlife-datasets library.

    Args:
        metadata: A pandas dataframe containing image metadata.
        root: Root directory if paths in metadata are relative. If None, absolute paths in metadata are used.
        transform: A function that takes in an image and returns its transformed version.
        img_load: Method to load images.
            Options: 'full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide',
                      and 'crop_black'.
        col_path: Column name in the metadata containing image file paths.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, \_\_getitem\_\_ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """

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
        self.metadata = metadata.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.col_path = col_path
        self.col_label = col_label
        self.load_label = load_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].values)

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        if self.root:
            img_path = os.path.join(self.root, data[self.col_path])
        else:
            img_path = data[self.col_path]
        img = self.get_image(img_path)

        if self.img_load in ["full_mask", "full_hide", "bbox_mask", "bbox_hide", "mask_crop"]:
            if not ("segmentation" in data):
                raise ValueError(f"{self.img_load} selected but no segmentation found.")
            if type(data["segmentation"]) == str:
                segmentation = eval(data["segmentation"])
            else:
                segmentation = data["segmentation"]
            if isinstance(segmentation, list) or isinstance(segmentation, np.ndarray):
                # Convert polygon to compressed RLE
                w, h = img.size
                rles = mask_coco.frPyObjects([segmentation], h, w)
                segmentation = mask_coco.merge(rles)
            if isinstance(segmentation, dict) and (
                isinstance(segmentation["counts"], list) or isinstance(segmentation["counts"], np.ndarray)
            ):
                # Convert uncompressed RLE to compressed RLE
                h, w = segmentation["size"]
                segmentation = mask_coco.frPyObjects(segmentation, h, w)

        if self.img_load in ["bbox"]:
            if not ("bbox" in data):
                raise ValueError(f"{self.img_load} selected but no bbox found.")
            if type(data["bbox"]) == str:
                bbox = json.loads(data["bbox"])
            else:
                bbox = data["bbox"]

        # Load full image as it is.
        if self.img_load == "full":
            img = img

        # Mask background using segmentation mask.
        elif self.img_load == "full_mask":
            if not np.any(pd.isnull(segmentation)):
                mask = mask_coco.decode(segmentation).astype("bool")
                img = Image.fromarray(img * mask[..., np.newaxis])

        # Hide object using segmentation mask
        elif self.img_load == "full_hide":
            if not np.any(pd.isnull(segmentation)):
                mask = mask_coco.decode(segmentation).astype("bool")
                img = Image.fromarray(img * ~mask[..., np.newaxis])

        # Crop to bounding box
        elif self.img_load == "bbox":
            if not np.any(pd.isnull(bbox)):
                img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Mask background using segmentation mask and crop to bounding box.
        elif self.img_load == "bbox_mask":
            if not np.any(pd.isnull(segmentation)):
                mask = mask_coco.decode(segmentation).astype("bool")
                img = Image.fromarray(img * mask[..., np.newaxis])
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img.crop(
                    (
                        np.min(x_nonzero),
                        np.min(y_nonzero),
                        np.max(x_nonzero),
                        np.max(y_nonzero),
                    )
                )

        # Hide object using segmentation mask and crop to bounding box.
        elif self.img_load == "bbox_hide":
            if not np.any(pd.isnull(segmentation)):
                mask = mask_coco.decode(segmentation).astype("bool")
                img = Image.fromarray(img * ~mask[..., np.newaxis])
                y_nonzero, x_nonzero, _ = np.nonzero(img)
                img = img.crop(
                    (
                        np.min(x_nonzero),
                        np.min(y_nonzero),
                        np.max(x_nonzero),
                        np.max(y_nonzero),
                    )
                )

        # Crop black background around images
        elif self.img_load == "crop_black":
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            img = img.crop(
                (
                    np.min(x_nonzero),
                    np.min(y_nonzero),
                    np.max(x_nonzero),
                    np.max(y_nonzero),
                )
            )

        else:
            raise ValueError(f"Invalid img_load argument: {self.img_load}")

        if self.transform:
            img = self.transform(img)

        if self.load_label:
            return img, self.labels[idx]
        else:
            return img


class FeatureDataset:
    """
    PyTorch-style dataset for a extracted features. Couples features with metadata.

    Args:
        features: list, np.array or tensor of features. Index of features should match with metadata.
        metadata: A pandas dataframe containing features metadata.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, \_\_getitem\_\_ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        features: list,
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

    def __getitem__(self, idx):
        if self.load_label:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    def __len__(self):
        return len(self.metadata)

    @property
    def num_classes(self):
        return len(self.labels_map)

    def save(self, path):
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
    def from_file(cls, path, **config):
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
