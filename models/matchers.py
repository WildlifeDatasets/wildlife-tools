import numpy as np
import pandas as pd
import itertools
import pandas as pd
import torch
import kornia.feature as KF
from data.dataset import WildlifeDataset
import math
import os


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


def prepare_pairs(query, database=None, batch_size=128, chunk=1, chunk_total=1):
    '''
    Prepares data pairs given query and optionally the database. 
    
    Returns an iterator that iterates through the batches. Optionally, it can
    be splited to chunks for better memory optimization and parallelization.
    '''
    if database:
        pair_total = len(query)*len(database)
        pair_iterator = itertools.product(enumerate(query), enumerate(database))
    else:
        pair_total = math.comb(len(query), 2)
        pair_iterator = itertools.combinations(enumerate(query), 2)

    batch_total = int(np.ceil(pair_total / batch_size))
    batch_iterator = batched(pair_iterator, batch_size)

    batches = np.array_split(np.arange(batch_total), chunk_total)[chunk-1]
    batch_min, batch_max = batches[0], batches[-1] + 1
    iterator = itertools.islice(batch_iterator, batch_min, batch_max)

    print(f'Total pairs     : {pair_total}')
    print(f'Total batches   : {batch_total}')
    print(f'Batches in chunk: {len(batches)}')
    return iterator



class LOFTRMatcher():
    def __init__(
        self,
        device: str ='cuda',
        pretrained: str ='outdoor',
        thresholds: tuple[float] = (0.99, ),
        batch_size: int = 128,
        chunk: int = 1,
        chunk_total: int = 1,
    ):
        self.device = device
        self.matcher = KF.LoFTR(pretrained=pretrained).to(device)
        self.thresholds = thresholds
        self.batch_size = batch_size
        if chunk > chunk_total:
            raise ValueError('Current chunk is larger that chunk total.')
        self.chunk = chunk
        self.chunk_total = chunk_total
        self.similarity = None


    def train(
        self,
        dataset_query: WildlifeDataset,
        dataset_database: WildlifeDataset | None = None,
        **kwargs,
    ):
        if dataset_database:
            print('Matching query with database')
            query = [i[0] for i in dataset_query]
            database = [i[0] for i in dataset_database]
            similarity = {t: np.zeros((len(query), len(database))) for t in self.thresholds}
        else:
            print('Matching all pairs in query')
            query = [i[0] for i in dataset_query]
            database = None
            similarity = {t: np.zeros((len(query), len(query))) for t in self.thresholds}

        # Prepare pairs
        iterator = prepare_pairs(
            query = query,
            database = database,
            batch_size = self.batch_size,
            chunk = self.chunk,
            chunk_total = self.chunk_total
        )

        # Iterate over all pairs in data
        data = []
        for i, pair_batch in enumerate(iterator):
            a, b = zip(*pair_batch)
            a_idx, a_data = list(zip(*a))
            b_idx, b_data = list(zip(*b))
            input_dict = {
                "image0": torch.stack(a_data).to(self.device),
                "image1": torch.stack(b_data).to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            batch_idx = correspondences['batch_indexes'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            for t in self.thresholds:
                series = pd.Series(confidence > t)
                #similarity[t][a_idx, b_idx] = series.groupby(batch_idx).sum().values
                for j, group in series.groupby(batch_idx):
                    similarity[t][a_idx[j], b_idx[j]] = group.sum()


            if i % 100 == 0:
                print(f'batch {i}')
        self.similarity = similarity
        

    def save(self, path, name='similarity.npy'):
        np.save(os.path.join(path, name), self.similarity)

    def load(self, path):
        self.similarity = np.load(path, allow_pickle='TRUE').item()