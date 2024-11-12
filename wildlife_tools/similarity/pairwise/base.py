import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ...data import FeatureDataset
from .collectors import CollectCounts


def visualise_matches(img0, keypoints0, img1, keypoints1):
    keypoints0 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints0]
    keypoints1 = [cv2.KeyPoint(int(x[0]), int(x[1]), 1) for x in keypoints1]

    # Create dummy matches (DMatch objects)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(keypoints0))]

    # Draw matches
    img_matches = cv2.drawMatches(
        img0,
        keypoints0,
        img1,
        keypoints1,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img_matches)
    plt.show()


class PairDataset(torch.utils.data.IterableDataset):
    """
    Create iterable style dataset from two mapping style datasets.
    By default, product is used - each item in dataset0 creates pair with each item in dataset1.
    Can iterate over some specific pairs if list of those pairs is provided.

    Each iteration returns 4-tuple (idx0, <dataset0 data at idx0>, idx1, <dataset1 data at idx1>)

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
    """

    def __init__(self, dataset0, dataset1, pairs=None, load_all=False):
        super().__init__()
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.pairs = pairs
        self.load_all = load_all

    def __len__(self):
        """Number of pairs in dataset."""
        if self.pairs is None:
            return len(self.dataset0) * len(self.dataset1)
        else:
            return len(self.pairs)

    @property
    def grid_shape(self):
        """Indicates max possible value of the idx0 and idx1 indexes."""
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


class MatchPairs:
    """
    Base class for matching pairs from two datasets.
    Any child class needs to implement `get_matches` method that implements processing of pair batches.
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        tqdm_silent: bool = False,
        collector=None,
    ):
        """
        Args:
            batch_size: Number of pairs processed in one batch.
            num_workers: Number of workers used for data loading.
            tqdm_silent: If True, progress bar is disabled.
            collector: Collector object used for storing results.
                By default, CollectCounts(thresholds=[0.5]) is used.
        """

        if collector is None:
            collector = CollectCounts(thresholds=[0.5])

        self.collector = collector
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tqdm_kwargs = {"mininterval": 1, "ncols": 100, "disable": tqdm_silent}

    def __call__(
        self, dataset0: FeatureDataset, dataset1: FeatureDataset, pairs: np.array | None = None
    ):
        """
        Match pairs of features from two feature datasets.
        Output for each pair is stored and processed using the collector.

        Args:
            dataset0: First dataset (e.g. query).
            dataset1: Second dataset (e.g. database).
            pairs: Numpy array with pairs of indexes. If None, all pairs are used.

        Returns:
            Exact output is determined by the used collector.

        """
        dataset_pairs = PairDataset(dataset0, dataset1, pairs=pairs)

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

    def get_matches(self, batch: tuple):
        """
        Process batch and get matches of pairs for the batch. Implemented in child classes.

        Args:
            batch: 4-tuple with indexes and data from PairDataset.

        Returns:
            list of standartized dictionaries with keys: idx0, idx1, score, kpts0, kpts1.
               Length of list is equal to batch size.
        """
        raise NotImplementedError
