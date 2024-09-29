import os
import pickle
from wildlife_tools.data.dataset import WildlifeDataset, FeatureDataset


class FeatureExtractor:
    def __call__(self, dataset: WildlifeDataset) -> FeatureDataset:
        raise NotImplementedError()

