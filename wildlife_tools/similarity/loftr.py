import itertools

import kornia.feature as KF
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from wildlife_tools.similarity.base import Similarity


def batched(iterable, n):
    """
    Batch data into tuples of length n. The last batch may be shorter.
    Example: batched('ABCDEFG', 3) --> ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class MatchLOFTR(Similarity):
    """
    Calculate similarity between query and database as number of descriptors correspondences
    after filtering with Low ratio test.

    Args:
        pretrained: LOFTR model used. `outdoor` or `indoor`.
        thresholds: Iterable with confidence thresholds. Should be in [0, 1] interval.
        batch_size: Batch size used for the inference.
        device: Specifies device used for the inference.
        silent: disable tqdm bar.

    Returns:
        dict: Values are 2D array with number of correspondences for each threshold.
    """

    def __init__(
        self,
        pretrained: str = "outdoor",
        thresholds: tuple[float] = (0.99,),
        batch_size: int = 128,
        device: str = "cuda",
        silent: bool = False,
    ):
        self.device = device
        self.matcher = KF.LoFTR(pretrained=pretrained).to(device)
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.silent = silent

    def __call__(self, query, database):
        iterator = batched(
            itertools.product(enumerate(query), enumerate(database)), self.batch_size
        )
        iterator_size = int(np.ceil(len(query) * len(database) / self.batch_size))
        similarities = {
            t: np.full((len(query), len(database)), np.nan, dtype=np.float16)
            for t in self.thresholds
        }

        for pair_batch in tqdm(
            iterator, total=iterator_size, mininterval=1, ncols=100, disable=self.silent
        ):
            q, d = zip(*pair_batch)
            q_idx, q_data = list(zip(*q))
            d_idx, d_data = list(zip(*d))
            input_dict = {
                "image0": torch.stack(q_data).to(self.device),
                "image1": torch.stack(d_data).to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            batch_idx = correspondences["batch_indexes"].cpu().numpy()
            confidence = correspondences["confidence"].cpu().numpy()
            for t in self.thresholds:
                series = pd.Series(confidence > t)
                for j, group in series.groupby(batch_idx):
                    similarities[t][q_idx[j], d_idx[j]] = group.sum()
        return similarities
