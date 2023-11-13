import torch
import numpy as np
import pandas as pd
from collections import defaultdict



class NearestClassifier():
    def __call__(self, similarity, labels=None):
        '''
        Input is query x database similarity matrix and optionally mapping to database labels.
        '''
        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=1, dim=1)
        pred = pred.numpy().flatten()

        if labels is not None:
            pred = labels[pred]
        return pred




class KnnClassifier():

    def __init__(self, k: int = 1):
        self.k = k


    def __call__(self, similarity, labels=None):
        '''
        Input is query x database similarity matrix and optionally mapping to database labels.
        '''
        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=self.k, dim=1)
        pred = self.aggregate(idx)[:, self.k-1]

        if labels is not None:
            pred = labels[pred]
        return pred


    def aggregate(self, preds):
        '''
        Aggregates array of nearest neigbours to single prediction for each k.
        If there is tie at given k, prediction from k-1 is used.

        Input:
            - array of with shape [n_query, k] of nearest neighbours.

        Output: 
            - array of shape [n_query, k] of predicitons.
            - Column dimensions corresponds to a prediction for [k=1, k=2 ... k=k]
        '''

        results = defaultdict(list)
        for k in range(1, preds.shape[1] + 1):
            for row in preds[:, :k]:
                vals, counts = np.unique(row, return_counts=True)
                best = vals[np.argmax(counts)]

                counts_sorted = sorted(counts)
                if (len(counts_sorted)) > 1 and (counts_sorted[0] == counts_sorted[1]):
                    best = None
                results[k].append(best)

        results = pd.DataFrame(results).T.fillna(method='ffill').T
        return results.values



