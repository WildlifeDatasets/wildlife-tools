from typing import Protocol

from ..data import FeatureDataset, ImageDataset


class FeatureExtractor(Protocol):
    """Interface for extracting features from an image dataset."""

    def __call__(self, dataset: ImageDataset) -> FeatureDataset: ...
