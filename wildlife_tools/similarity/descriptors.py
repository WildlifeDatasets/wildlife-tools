import itertools

import faiss
import numpy as np
from tqdm import tqdm

from wildlife_tools.similarity.base import Similarity


def get_faiss_index(d, device="cpu"):
    if device == "cuda":
        resource = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        return faiss.GpuIndexFlatL2(resource, d, config)
    elif device == "cpu":
        return faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Invalid device: {device}")


class MatchDescriptors(Similarity):
    """
    Calculate similarity between query and database as number of descriptors correspondences
    after filtering with Low ratio test.

    Args:
        descriptor_dim: dimensionality of descriptors. 128 for SIFT, 256 for SuperPoint.
        thresholds: iterable with ratio test thresholds. Should be in [0, 1] interval.
        device: Specifies device used for nearest neigbour search.

    Returns:
        dict: Values are 2D array with number of correspondences for each threshold.
    """

    def __init__(
        self,
        descriptor_dim: int,
        thresholds: tuple[float] = (0.5,),
        device: str = "cpu",
    ):

        self.descriptor_dim = descriptor_dim
        self.thresholds = thresholds
        self.device = device

    def __call__(self, query, database):
        iterator = itertools.product(enumerate(query), enumerate(database))
        iterator_size = len(query) * len(database)
        similarities = {
            t: np.full((len(query), len(database)), np.nan, dtype=np.float16)
            for t in self.thresholds
        }

        index = get_faiss_index(d=self.descriptor_dim, device=self.device)
        for pair in tqdm(iterator, total=iterator_size, mininterval=1, ncols=100):
            (q_idx, q_data), (d_idx, d_data) = pair

            if (q_data is None) or (d_data is None):
                for t in self.thresholds:
                    similarities[t][q_idx, d_idx] = 0

            else:
                index.reset()
                index.add(q_data)
                score, idx = index.search(d_data, k=2)
                with np.errstate(divide="ignore"):
                    ratio = score[:, 0] / score[:, 1]
                for t in self.thresholds:
                    similarities[t][q_idx, d_idx] = np.sum(ratio < t)

        return similarities
