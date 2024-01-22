""" Split metadata to a subset. """

from copy import deepcopy

import numpy as np

from wildlife_tools.tools import realize


class Split:
    pass


class SplitChunk(Split):
    """Returns split as given nth chunk of metadata from equally sized metadata chunks."""

    def __init__(self, chunk=1, chunk_total=1):
        self.chunk = chunk
        self.chunk_total = chunk_total

    def __call__(self, metadata):
        metadata_chunks = np.array_split(metadata, self.chunk_total)
        return metadata_chunks[self.chunk - 1]


class SplitWildlife(Split):
    """
    Returns split given wildlife splitter.
    A splitter is callable that returns list of (train_idx, test_idx) given metadata.
    Example:
        method: SplitWildlife
        splitter:
            method: closed_set
            seed: 0
            ratio_train: 0.3
        split: train
    """

    def __init__(self, splitter, split="train", split_map=None, repeat_idx=0):
        self.splitter = splitter
        self.split = split
        self.split_map = split_map
        self.repeat_idx = repeat_idx

    def __call__(self, metadata):
        if self.split_map is None:
            dataset_idx = {"train": 0, "test": 1}[self.split]
        else:
            dataset_idx = self.split_map[self.split]

        splits = self.splitter.split(metadata)
        idx = splits[self.repeat_idx][dataset_idx]
        return metadata.iloc[idx]

    @classmethod
    def from_config(cls, config):

        # TODO: add this to wildlife datasets library
        """
        class RandomProportion():
            def __init__(self, seed=1):
                self.splitter = splits.TimeProportionSplit(seed=seed)

            def split(self, df):
                final_splits = []
                for idx_train, idx_test in self.splitter.split(df):
                    final_splits.append(self.splitter.resplit_random(df, idx_train, idx_test))
                return final_splits
        """

        config = deepcopy(config)
        from wildlife_datasets import splits

        splitters = {
            "open": splits.OpenSetSplit,
            "closed": splits.ClosedSetSplit,
            "disjoint": splits.DisjointSetSplit,
            "time_proportion": splits.TimeProportionSplit,
            # "random_proportion": splits.RandomProportionSplit,
        }

        splitter_config = config.pop("splitter")
        method = splitter_config.pop("method")
        splitter = splitters[method](**splitter_config)
        return cls(splitter=splitter, **config)


class SplitMetadata(Split):
    """Returns split based on value in one of the metadata columns."""

    def __init__(self, col, value):
        self.col = col
        self.value = value

    def __call__(self, metadata):
        return metadata[metadata[self.col] == self.value]


class SplitChain(Split):
    """Returns split from sequence of splits"""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, metadata):
        for step in self.steps:
            metadata = step(metadata)
        return metadata

    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        steps = []
        for config_step in config["steps"]:
            steps.append(realize(config_step))
        return cls(steps=steps)
