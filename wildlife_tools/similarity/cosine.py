import torch
import torch.nn.functional as F

from wildlife_tools.similarity.base import Similarity


class CosineSimilarity(Similarity):
    """
    Calculates cosine similarity, equivalently to `sklearn.metrics.pairwise.cosine_similarity`

    Returns:
        dict: dictionary with `cosine` key. Value is 2D array with cosine similarity.

    """

    def __call__(self, query, database):
        return {"cosine": self.cosine_similarity(query, database)}

    def cosine_similarity(self, a, b):
        a, b = torch.tensor(a), torch.tensor(b)
        similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
        return similarity.numpy()
