from typing import Protocol

import numpy as np

from ..data import FeatureDataset


class Matcher(Protocol):
    """Interface for computing similarity scores between two feature datasets."""

    def __call__(
        self, query: FeatureDataset, database: FeatureDataset, pairs: np.ndarray | None = None
    ) -> np.ndarray: ...
