import numpy as np


class SplitChunk:
    """Returns split as given nth chunk of metadata from equally sized metadata chunks."""

    def __init__(self, chunk=1, chunk_total=1):
        self.chunk = chunk
        self.chunk_total = chunk_total

    def __call__(self, metadata):
        metadata_chunks = np.array_split(metadata, self.chunk_total)
        return metadata_chunks[self.chunk - 1]
