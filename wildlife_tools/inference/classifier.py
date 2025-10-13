from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Tuple


class KnnClassifier:
    """
    Predict query label as k labels of nearest matches in the database.
    If there is a tie at a given k, the prediction with the best score is used.
    """

    def __init__(self, database_labels: np.ndarray, k: int = 1, return_scores: bool = False):
        """
        Args:
            database_labels (np.ndarray): Array containing the labels of the database.
            k (int, optional): The number of nearest neighbors to consider.
            return_scores (bool, optional): Indicates whether to return scores along with predictions.
        """
        self.k = k
        self.database_labels = database_labels
        self.return_scores = return_scores

    def __call__(self, similarity: np.ndarray) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the label for each query based on the k nearest matches in the database.

        Args:
            similarity (np.ndarray): A 2D similarity matrix with `n_query` x `n_database` shape.

        Returns:
            If `return_scores` is False:

                - preds (np.ndarray): Prediction for each query.

            If `return_scores` is True, tuple of two arrays:

                - preds (np.ndarray): Prediction for each query.
                - scores (np.ndarray): The similarity scores corresponding to the predictions (mean for k > 1).
        """

        # Get ranked predictions and scores.
        similarity = torch.tensor(similarity, dtype=torch.float32)
        scores, idx = similarity.topk(k=self.k, dim=1)
        preds = self.database_labels[idx]

        preds = np.array(preds)
        scores = scores.numpy()

        # Aggregate k nearest neighbors
        data = []
        for pred, score in zip(preds, scores):
            vals, counts = np.unique(pred, return_counts=True)
            winners = vals[counts.max() == counts]

            # Check for ties
            if len(winners) == 1:
                best_pred = winners[0]
                best_score = score[best_pred == pred].mean()
            else:
                is_winner = np.isin(pred, winners)
                ties = pd.Series(score[is_winner]).groupby(pred[is_winner]).mean()
                best_pred = ties.idxmax()
                best_score = ties.max()
            data.append([best_pred, best_score])

        preds, scores = list(zip(*data))
        preds = np.array(preds)
        scores = np.array(scores)

        if self.return_scores:
            return preds, scores
        else:
            return preds


class TopkClassifier:
    """
    Predict top k query labels given nearest matches in the database.
    """

    def __init__(self, database_labels: np.ndarray, k: int = 10, return_all: bool = False):
        """
        Args:
            database_labels (np.ndarray): Array containing the labels of the database.
            k (int): The number of top predictions to return.
            return_all (bool): Indicates whether to return scores along with predictions.
        """
        self.k = k
        self.database_labels = database_labels
        self.return_all = return_all

    def __call__(self, similarity: np.ndarray) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts the top k labels for each query based on the similarity matrix.

        Args:
            similarity (np.ndarray): A 2D similarity matrix with `n_query` x `n_database` shape

        Returns:
            If `return_all` is False, single 2D array of shape `n_query` x `k`

                - preds (np.ndarray): The top k predicted labels for each query.

            If `return_all` is True, tuple of three 2D arrays of shape `n_query` x `k`:

                - preds (np.ndarray): The top k predicted labels for each query.
                - scores (np.ndarray): The similarity scores corresponding to the top k predictions.
                - idx (np.ndarray): The indices of the database entries corresponding to the top k predictions.
        """

        # Get ranked predictions, scores, and database index.
        similarity = torch.tensor(similarity, dtype=torch.float32)
        scores, idx = similarity.topk(k=similarity.shape[1], dim=1)
        preds = self.database_labels[idx]

        preds = np.array(preds)
        scores = scores.numpy()
        idx = idx.numpy()

        # Collect data for first label occurrence
        data = []
        for j, row in enumerate(preds):
            data_row = []
            visited = set()
            for i, value in enumerate(row):
                if value not in visited:
                    visited.add(value)
                    data_row.append((value, scores[j, i], idx[j, i]))
            data.append(list(zip(*data_row)))

        preds, scores, idx = list(zip(*data))
        preds = np.array(preds)[:, : self.k]
        scores = np.array(scores)[:, : self.k]
        idx = np.array(idx)[:, : self.k]

        if self.return_all:
            return preds, scores, idx
        else:
            return preds
