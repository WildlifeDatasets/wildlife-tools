import numpy as np
import pandas as pd


class SplitChunk:
    """Returns split as given nth chunk of metadata from equally sized metadata chunks."""

    def __init__(self, chunk: int = 1, chunk_total: int = 1):
        self.chunk = chunk
        self.chunk_total = chunk_total

    def __call__(self, metadata: pd.DataFrame):
        metadata_chunks = np.array_split(metadata, self.chunk_total)
        return metadata_chunks[self.chunk - 1]
