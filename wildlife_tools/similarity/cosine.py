import numpy as np
import torch
import torch.nn.functional as F

from wildlife_tools.similarity.base import Similarity
from wildlife_tools.data import FeatureDataset


class CosineSimilarity(Similarity):
    """
    Calculates cosine similarity, equivalently to `sklearn.metrics.pairwise.cosine_similarity`

    Args:
        query (FeatureDataset): Query dataset of deep features.
        database (FeatureDataset): Database dataset of deep features.

    Returns:
        dict: dictionary with `cosine` key. Value is 2D array with cosine similarity.

    """
    def __call__(self, query: FeatureDataset, database: FeatureDataset, pairs: tuple | None = None) -> np.ndarray:
        return self.cosine_similarity(query.features, database.features)


    def cosine_similarity(self, a, b):
        a, b = torch.tensor(a), torch.tensor(b)
        similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
        return similarity.numpy()
