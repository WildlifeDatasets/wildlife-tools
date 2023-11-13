import os
import torch
import torch.nn.functional as F
from wildlife_tools.similarity.base import Similarity


def cosine_similarity(a, b):
    '''
    Equivalent to sklearn.metrics.pairwise.cosine_similarity
    '''
    a, b = torch.tensor(a), torch.tensor(b)
    similarity = torch.matmul(F.normalize(a), F.normalize(b).T)
    return similarity.numpy()


class CosineSimilarity(Similarity):
    def __call__(self, query, database):
        return {'default': cosine_similarity(query, database) }
