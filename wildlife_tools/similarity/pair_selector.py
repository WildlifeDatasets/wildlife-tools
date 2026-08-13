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

    def get_ignore_mask(self, df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
        ignore_mask = np.zeros((len(df1), len(df2)), dtype=bool)

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
                idx2 = groups2.get(val)
                if idx2 is None:
                    continue
                ignore_mask[np.ix_(idx1.to_numpy(), idx2.to_numpy())] = True

        # Ignores different values in cols_unequal. Instead of enumerating the (typically huge)
        # set of unequal pairs directly, ignore the whole column's index range and then restore
        # (un-ignore) the equal-value blocks, whose combined size is usually much smaller.
        for col in self.cols_unequal:
            if col not in df1.columns or col not in df2.columns:
                continue
            s1 = df1[col]
            s2 = df2[col]
            if self.ignore_unknown and col == "identity":
                s1 = s1[s1 != "unknown"]
                s2 = s2[s2 != "unknown"]

            col_ignore = np.zeros_like(ignore_mask)
            col_ignore[np.ix_(s1.index.to_numpy(), s2.index.to_numpy())] = True

            groups2 = s2.groupby(s2).groups
            for val, idx1 in s1.groupby(s1).groups.items():
                idx2 = groups2.get(val)
                if idx2 is None:
                    continue
                col_ignore[np.ix_(idx1.to_numpy(), idx2.to_numpy())] = False

            ignore_mask |= col_ignore

        return ignore_mask

    def __call__(
        self,
        similarity_priority: np.ndarray,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        B: int,
    ) -> np.ndarray:
        ignore_mask = self.get_ignore_mask(dataset0.metadata, dataset1.metadata)
        if not ignore_mask.any():
            return super().__call__(similarity_priority, dataset0, dataset1, B)

        # Mask ignored pairs, restoring the original values afterwards.
        original_values = similarity_priority[ignore_mask].copy()
        similarity_priority[ignore_mask] = -np.inf

        try:
            return super().__call__(similarity_priority, dataset0, dataset1, B)
        finally:
            similarity_priority[ignore_mask] = original_values
