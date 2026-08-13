from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from ..data import ImageDataset


class PairSelector(ABC):
    """Base class for strategies that select shortlisted pairs from a priority matrix."""

    @abstractmethod
    def __call__(
        self,
        similarity_priority: np.ndarray,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        B: int,
    ) -> np.ndarray:
        raise NotImplementedError


class TopkPairSelector(PairSelector):
    """Default pair selection strategy: top-B priority pairs per query."""

    def __call__(
        self,
        similarity_priority: np.ndarray,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        B: int,
    ) -> np.ndarray:
        _, idx1 = torch.topk(torch.tensor(similarity_priority), min(B, similarity_priority.shape[1]))
        idx0 = np.indices(idx1.numpy().shape)[0]
        idx_keep = similarity_priority[idx0.flatten(), idx1.flatten()] > -np.inf
        grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T[idx_keep]
        return grid_indices


class MetadataPairSelector(TopkPairSelector):
    """
    Top-B pair selection that additionally ignores pairs based on matching metadata columns.

    Pairs are ignored if any `cols_equal` column has the same value in both datasets, or if any
    `cols_unequal` column has different values in both datasets.
    """

    def __init__(
        self,
        cols_equal: list[str] | None = None,
        cols_unequal: list[str] | None = None,
        ignore_unknown: bool = True,
    ):
        self.cols_equal = cols_equal or []
        self.cols_unequal = cols_unequal or []
        self.ignore_unknown = ignore_unknown

    def get_ignore_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame) -> list[tuple[int, int]]:
        ignore_pairs = []

        # Ignores same values in cols_equal
        for col in self.cols_equal:
            if col not in df1.columns or col not in df2.columns:
                continue
            s1 = df1[col]
            s2 = df2[col]
            if self.ignore_unknown and col == "identity":
                s1 = s1[s1 != "unknown"]
                s2 = s2[s2 != "unknown"]

            groups2 = s2.groupby(s2).groups
            for val, idx1 in s1.groupby(s1).groups.items():
                if val not in groups2:
                    continue
                idx2 = groups2[val]

                i, j = np.meshgrid(idx1.to_numpy(), idx2.to_numpy(), indexing="ij")
                ignore_pairs.extend(zip(i.ravel(), j.ravel()))

        # Ignores different values in cols_unequal
        for col in self.cols_unequal:
            if col not in df1.columns or col not in df2.columns:
                continue
            s1 = df1[col]
            s2 = df2[col]
            if self.ignore_unknown and col == "identity":
                s1 = s1[s1 != "unknown"]
                s2 = s2[s2 != "unknown"]

            values1 = s1.to_numpy()
            values2 = s2.to_numpy()
            idx1 = s1.index.to_numpy()
            idx2 = s2.index.to_numpy()

            mask = values1[:, None] != values2[None, :]
            i, j = np.where(mask)
            ignore_pairs.extend(zip(idx1[i], idx2[j]))

        return ignore_pairs

    def __call__(
        self,
        similarity_priority: np.ndarray,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        B: int,
    ) -> np.ndarray:
        ignore_pairs = self.get_ignore_pairs(dataset0.metadata, dataset1.metadata)
        if ignore_pairs:
            ignore_idx = np.array(ignore_pairs)
            similarity_priority[ignore_idx[:, 0], ignore_idx[:, 1]] = -np.inf
        return super().__call__(similarity_priority, dataset0, dataset1, B)
