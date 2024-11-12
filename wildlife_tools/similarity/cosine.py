import numpy as np
import torch
import torch.nn.functional as F

from ..data import FeatureDataset


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two sets of vectors.
    Pytorch Equivalent to `sklearn.metrics.pairwise.cosine_similarity`.
    """

    a, b = torch.tensor(a), torch.tensor(b)
    similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
    return similarity.numpy()


class CosineSimilarity:
    """Wraps cosine similarity to be usable in SimilarityPipeline."""

    def __call__(self, query: FeatureDataset, database: FeatureDataset, **kwargs) -> np.ndarray:
        """
        Calculates cosine similarity given query and database feature datasets.

        Args:
            query (FeatureDataset): Query dataset of deep features.
            database (FeatureDataset): Database dataset of deep features.

        Returns:
            similarity (np.array): 2D numpy array with cosine similarity.

        """

        return cosine_similarity(query.features, database.features)
