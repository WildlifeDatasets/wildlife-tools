import torch
import numpy as np
import pandas as pd
from collections import defaultdict



class NearestClassifier():
    def __call__(self, similarity, labels=None):

        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=1, dim=1)
        pred = pred.numpy().flatten()

        if labels is not None:
            pred = labels[pred]
        return pred




class KnnClassifier():
    '''
    Predict query label as k labels of nearest matches in database. If there is tie at given k, prediction from k-1 is used.
    Input is similarity matrix with `n_query` x `n_database` shape.
    

    Args:
        k: use k nearest in database for the majority voting.
        labels_map: If provided, decode predictions to database string labels.
    Returns:
        1D array with length `n_query` of database labels (Ordinally encoded by default - col index of the similarity matrix).
    '''

    def __init__(self, k: int = 1, labels_map: dict | None = None):
        self.k = k
        self.labels_map = labels_map


    def __call__(self, similarity, ):
        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=self.k, dim=1)
        pred = self.aggregate(idx)[:, self.k-1]

        if self.labels_map is not None:
            pred = self.labels_map[pred]
        return pred


    def aggregate(self, predictions):
        '''
        Aggregates array of nearest neigbours to single prediction for each k.
        If there is tie at given k, prediction from k-1 is used.

        Args:
            array of with shape [n_query, k] of nearest neighbours.
        Returns:
            array of shape [n_query, k] of predicitons. Column dimensions are predictions for [k=1, k=2 ... k=k]
        '''

        results = defaultdict(list)
        for k in range(1, predictions.shape[1] + 1):
            for row in predictions[:, :k]:
                vals, counts = np.unique(row, return_counts=True)
                best = vals[np.argmax(counts)]

                counts_sorted = sorted(counts)
                if (len(counts_sorted)) > 1 and (counts_sorted[0] == counts_sorted[1]):
                    best = None
                results[k].append(best)

        results = pd.DataFrame(results).T.fillna(method='ffill').T
        return results.values



