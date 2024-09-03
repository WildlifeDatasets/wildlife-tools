from .collectors import CollectAll, CollectCounts, CollectCountsRansac
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def visualise_matches(img0, keypoints0, img1, keypoints1):
    keypoints0 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints0]
    keypoints1 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints1]

    # Create dummy matches (DMatch objects)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]

    # Draw matches
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.show()


class PairDataset(torch.utils.data.IterableDataset):
    '''
    Create IterableDataset from two mapping style datasets. 
    By default, product is used - each item in dataset0 creates pair with each item in dataset1.
    Can iterate over some specific pairs if list of those pairs is provided.

    Iteration returns 4-tuple (idx0, <dataset0 data at idx0>, idx1, <dataset1 data at idx1>)

    Args:
        dataset0: Dataset for first of the pair.
        dataset1: Dataset for second of the pair.
        pairs: list of 2-tuples with indexes. If provided, iterate only over those pairs.
        load_all: If True, all elements from datasets are used. If False, only first element is used.

    Example:
        dataset = PairProductDataset(['x', 'y'], ['a', 'b'])
        iterator = iter(dataset)
        next(iterator)
        >>> (0, 'x', 0, 'a')
        next(iterator)
        >>> (0, 'x', 1, 'b')
    '''

    def __init__(self, dataset0, dataset1, pairs=None, load_all=False):
        super().__init__()
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.pairs = pairs
        self.load_all = load_all

    def __len__(self):
        if self.pairs is None:
            return len(self.dataset0) * len(self.dataset1)
        else:
            return len(self.pairs)


    @property
    def grid_shape(self):
        '''  Indicates max possible value of the idx0 and idx1 indexes.'''
        return len(self.dataset0), len(self.dataset1)


    def __iter__(self):
        if self.pairs is None:
            iterator = itertools.product(range(len(self.dataset0)), range(len(self.dataset1)))
        else:
            iterator = self.pairs

        # Get Worker specific iterator
        worker = torch.utils.data.get_worker_info()
        if worker:
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        for idx0, idx1 in iterator:
            if self.load_all:
                yield idx0, self.dataset0[idx0], idx1, self.dataset1[idx1]
            else:
                yield idx0, self.dataset0[idx0][0], idx1, self.dataset1[idx1][0]



class MatchPairs():
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        tqdm_silent: bool = False,
        collector = None,
    ):
        if collector is None:
            collector = CollectCounts(thresholds=[0.5])


        self.collector = collector
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tqdm_kwargs = { 'mininterval':1, 'ncols':100, 'disable': tqdm_silent}


    def __call__(self, dataset0, dataset1, pairs=None):
        dataset_pairs = PairDataset(dataset0, dataset1, pairs=pairs)
        return self.match_pairs(dataset_pairs)


    def match_pairs(self, dataset_pairs):
        loader_length = int(np.ceil(len(dataset_pairs) / self.batch_size))
        loader = torch.utils.data.DataLoader(
            dataset_pairs,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.collector.init_store(grid_shape=dataset_pairs.grid_shape)
        for batch in tqdm(loader, total=loader_length, **self.tqdm_kwargs):
            matches = self.get_matches(batch)
            self.collector.add(matches)

        results = self.collector.process_results()
        return results


    def get_matches(self, batch):
        ''' 
        Process batch and get matches of pairs.

        Input - batch from PairDataset
        Output - list of dictionaries with keys: i0, j0, score, kpts0, kpts1.
        '''
        raise NotImplementedError

