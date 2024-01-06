from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from wildlife_tools.similarity import CosineSimilarity


class KnnClassifier:
    """
    Predict query label as k labels of nearest matches in database. If there is tie at given k,
    prediction from k-1 is used. Input is similarity matrix with `n_query` x `n_database` shape.


    Args:
        k: use k nearest in database for the majority voting.
        database_labels: list of labels in database. If provided, decode predictions to database
        (e.g. string) labels.
    Returns:
        1D array with length `n_query` of database labels (col index of the similarity matrix).
    """

    def __init__(self, k: int = 1, database_labels: np.array | None = None):
        self.k = k
        self.database_labels = database_labels

    def __call__(self, similarity):
        similarity = torch.tensor(similarity, dtype=float)
        scores, idx = similarity.topk(k=self.k, dim=1)
        pred = self.aggregate(idx)[:, self.k - 1]

        if self.database_labels is not None:
            pred = self.database_labels[pred]
        return pred

    def aggregate(self, predictions):
        """
        Aggregates array of nearest neighbours to single prediction for each k.
        If there is tie at given k, prediction from k-1 is used.

        Args:
            array of with shape [n_query, k] of nearest neighbours.
        Returns:
            array with predictions [n_query, k]. Column dimensions are predictions for [k=1,...,k=k]
        """

        results = defaultdict(list)
        for k in range(1, predictions.shape[1] + 1):
            for row in predictions[:, :k]:
                vals, counts = np.unique(row, return_counts=True)
                best = vals[np.argmax(counts)]

                counts_sorted = sorted(counts)
                if (len(counts_sorted)) > 1 and (counts_sorted[0] == counts_sorted[1]):
                    best = None
                results[k].append(best)

        results = pd.DataFrame(results).T.fillna(method="ffill").T
        return results.values


class KnnMatcher:
    """
    Find nearest match to query in existing database of features.
    Combines CosineSimilarity and KnnClassifier.
    """

    def __init__(self, database, k=1):
        self.similarity = CosineSimilarity()
        self.database = database
        self.classifier = KnnClassifier(
            database_labels=self.database.labels_string, k=k
        )

    def __call__(self, query):
        if isinstance(query, list):
            query = np.concatenate(query)

        if not isinstance(query, np.ndarray):
            raise ValueError("Query should be array or list of features.")

        sim_matrix = self.similarity(query, self.database.features)["cosine"]
        return self.classifier(sim_matrix)
