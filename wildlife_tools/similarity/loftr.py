import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import kornia.feature as KF
import itertools
from wildlife_tools.similarity.base import Similarity


def batched(iterable, n):
    '''
    Batch data into tuples of length n. The last batch may be shorter.
    Example: batched('ABCDEFG', 3) --> ABC DEF G
    '''
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class MatchLOFTR(Similarity):
    def __init__(
        self,
        device: str ='cuda',
        pretrained: str ='outdoor',
        thresholds: tuple[float] = (0.99, ),
        batch_size: int = 128,
    ):
        self.device = device
        self.matcher = KF.LoFTR(pretrained=pretrained).to(device)
        self.thresholds = thresholds
        self.batch_size = batch_size


    def __call__(self, query, database):
        iterator = batched(itertools.product(enumerate(query), enumerate(database)), self.batch_size)
        iterator_size = int(np.ceil(len(query)*len(database) / self.batch_size))
        similarities = {t: np.full((len(query), len(database)), np.nan, dtype=np.float16) for t in self.thresholds}

        for pair_batch in tqdm(iterator, total=iterator_size, mininterval=1, ncols=100):
            q, d = zip(*pair_batch)
            q_idx, q_data = list(zip(*q))
            d_idx, d_data = list(zip(*d))
            input_dict = {
                "image0": torch.stack(q_data).to(self.device),
                "image1": torch.stack(d_data).to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            batch_idx = correspondences['batch_indexes'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            for t in self.thresholds:
                series = pd.Series(confidence > t)
                for j, group in series.groupby(batch_idx):
                    similarities[t][q_idx[j], d_idx[j]] = group.sum()
        return similarities
