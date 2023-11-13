import torch
import os
import numpy as np
from tqdm import tqdm 
from wildlife_tools.features.base import FeatureExtractor


class InMemoryFeatures(FeatureExtractor):
    ''' Loads dataset in memory for faster access.'''

    def __call__(self, dataset):
        features = []
        for x, y in tqdm(dataset, mininterval=1, ncols=100):
            features.append(x)
        return features
