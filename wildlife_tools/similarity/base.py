from typing import Protocol, TypeVar

import numpy as np

from ..data import FeatureDataset, ImageDataset

DatasetT = TypeVar("DatasetT", bound=FeatureDataset | ImageDataset, contravariant=True)


class Matcher(Protocol[DatasetT]):
    """Interface for computing similarity scores between two feature (or image) datasets."""

    def __call__(self, query: DatasetT, database: DatasetT, pairs: np.ndarray | None = None) -> np.ndarray: ...
