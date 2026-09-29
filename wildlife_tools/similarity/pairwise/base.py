import itertools
import pickle
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ...data import FeatureDataset, ImageDataset
from ...data.cache import CacheMixin
from ..base import Matcher
from .collectors import CollectCounts, Collector


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


class MatchPairs(CacheMixin, Matcher[FeatureDataset | ImageDataset]):
    """
    Base class for matching pairs from two datasets.
    Any child class needs to implement `get_matches` method that implements processing of pair batches.
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        tqdm_silent: bool = False,
        collector: Collector | None = None,
        cache_path: str | None = None,
        config_tag: str | None = None,
        cache_scores_only: bool = False,
    ):
        """
        Args:
            batch_size (int, optional): Number of pairs processed in one batch.
            num_workers (int, optional): Number of workers used for data loading.
            tqdm_silent (bool, optional): If True, progress bar is disabled.
            collector (Collector | None, optional): Collector object used for storing results.
            cache_path (str, optional): Path for cached pair matches. No caching for None.
                Cache stores raw matches, so it is independent of the collector, but it must be
                unique for each matcher configuration and each feature extractor.
            config_tag (str, optional): Free-form tag stored in the cache config. Reusing cache_path with a
                different tag raises an error. Changes of the image transform (for LoFTR) or of the
                feature extractor (for LightGlue) are not detected automatically, so encode them in
                the tag (e.g. "resize224_gray" or "sift256_resize224").
            cache_scores_only (bool, optional): If True, only scores are cached (keypoints are dropped),
                which greatly reduces cache size. Collectors needing keypoints (e.g. CollectCountsRansac)
                cannot be used then.
        """

        super().__init__(cache_path=cache_path, config_tag=config_tag)
        self.cache_scores_only = cache_scores_only

        if collector is None:
            collector = CollectCounts()

        self.collector = collector
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tqdm_kwargs = {"mininterval": 1, "ncols": 100, "disable": tqdm_silent}

    def cache_config(self) -> dict:
        return super().cache_config() | {"scores_only": self.cache_scores_only}

    def __call__(
        self,
        query: FeatureDataset | ImageDataset,
        database: FeatureDataset | ImageDataset,
        pairs: np.ndarray | None = None,
    ):
        """
        Match pairs of features from two feature datasets.
        Output for each pair is stored and processed using the collector.

        Args:
            query: Query dataset.
            database: Database dataset.
            pairs: Numpy array with pairs of indexes. If None, all pairs are used.

        Returns:
            results (dict): Exact output is determined by the used collector.
        """

        if self.cache_path is not None:
            return self._call_with_cache(query, database, pairs)

        dataset_pairs = PairDataset(query, database, pairs=pairs)

        self.collector.init_store(grid_shape=dataset_pairs.grid_shape)
        for matches in self._iter_matches(dataset_pairs):
            self.collector.add(matches)

        results = self.collector.process_results()
        return results

    def _call_with_cache(
        self,
        query: FeatureDataset | ImageDataset,
        database: FeatureDataset | ImageDataset,
        pairs: np.ndarray | None = None,
    ) -> Any:
        assert self.cache_path is not None
        if pairs is None:
            pair_list = list(itertools.product(range(len(query)), range(len(database))))
        else:
            pair_list = [(int(i0), int(i1)) for i0, i1 in pairs]
        keys0 = [self.get_key(query, i) for i in range(len(query))]
        keys1 = [self.get_key(database, i) for i in range(len(database))]

        with self._open_env() as env:
            with env.begin() as txn:
                missing = [(i0, i1) for i0, i1 in pair_list if txn.get(self.get_pair_key(keys0[i0], keys1[i1])) is None]

            for matches in self._iter_matches(PairDataset(query, database, pairs=missing)):
                with env.begin(write=True) as txn:
                    for m in matches:
                        i0, i1 = m.pop("idx0"), m.pop("idx1")
                        if self.cache_scores_only:
                            m = {"scores": m["scores"]}
                        txn.put(
                            self.get_pair_key(keys0[i0], keys1[i1]), pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)
                        )

            self.collector.init_store(grid_shape=(len(query), len(database)))
            with env.begin() as txn:
                for i0, i1 in pair_list:
                    val = txn.get(self.get_pair_key(keys0[i0], keys1[i1]))
                    assert val is not None
                    m = pickle.loads(val)
                    self.collector.add([m | {"idx0": i0, "idx1": i1}])

        return self.collector.process_results()

    def get_pair_key(self, key0: str, key1: str) -> bytes:
        return f"{key0}\x00{key1}".encode()

    def _iter_matches(self, dataset_pairs: PairDataset) -> Iterator[list[dict]]:
        loader_length = int(np.ceil(len(dataset_pairs) / self.batch_size))
        loader = torch.utils.data.DataLoader(
            dataset_pairs,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        for batch in tqdm(loader, total=loader_length, **self.tqdm_kwargs):
            yield self.get_matches(batch)

    def get_matches(self, batch: tuple):
        """
        Process batch and get matches of pairs for the batch. Implemented in child classes.

        Args:
            batch: 4-tuple with indexes and data from PairDataset.

        Returns:
            results (List[dict]): list of standartized dictionaries with keys: idx0, idx1, score, kpts0, kpts1.
                Length of list is equal to batch size.
        """
        raise NotImplementedError
