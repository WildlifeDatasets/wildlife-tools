import numpy as np
import torch

from ..data import ImageDataset


class PairSelector:
    """Default pair selection strategy: top-B priority pairs per query, with optional ignore_pairs."""

    def __call__(
        self,
        similarity_priority: np.ndarray,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        B: int,
        ignore_pairs: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if ignore_pairs:
            ignore_pairs = np.array(ignore_pairs)
            similarity_priority[ignore_pairs[:, 0], ignore_pairs[:, 1]] = -np.inf
        _, idx1 = torch.topk(torch.tensor(similarity_priority), min(B, similarity_priority.shape[1]))
        idx0 = np.indices(idx1.numpy().shape)[0]
        idx_keep = similarity_priority[idx0.flatten(), idx1.flatten()] > -np.inf
        grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T[idx_keep]
        return grid_indices
