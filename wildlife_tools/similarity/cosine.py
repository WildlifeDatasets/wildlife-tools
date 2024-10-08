import numpy as np
import torch
import torch.nn.functional as F

from wildlife_tools.similarity.base import Similarity
from wildlife_tools.data import FeatureDataset


class CosineSimilarity(Similarity):
    ''' Cosine similarity between query and database feature datasets. '''

    def __call__(self, query: FeatureDataset, database: FeatureDataset, **kwargs) -> np.ndarray:
        """
        Calculates cosine similarity, equivalently to `sklearn.metrics.pairwise.cosine_similarity`

        Args:
            query (FeatureDataset): Query dataset of deep features.
            database (FeatureDataset): Database dataset of deep features.

        Returns:
            similarity (np.array): 2D numpy array with cosine similarity.

        """

        return self.cosine_similarity(query.features, database.features)


    def cosine_similarity(self, a, b):
        a, b = torch.tensor(a), torch.tensor(b)
        similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
        return similarity.numpy()
