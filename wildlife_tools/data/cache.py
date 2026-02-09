from abc import ABC, abstractmethod
import numpy as np
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

from .dataset import FeatureDataset, ImageDataset
from ..tools import check_dataset_output


class BatchRunner(ABC):
    @abstractmethod
    def process_batch(self, batch):
        pass

    def get_key(self, dataset, index):
        return dataset.metadata["image_id"][index]

    def make_loader(self, dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def run_batches(self, dataset, batch_size, num_workers):
        loader = self.make_loader(dataset, batch_size, num_workers)

        outputs = []
        for batch in tqdm(loader, mininterval=1, ncols=100):
            out = self.process_batch(batch)
            if out is not None:
                outputs.append(out)

        return outputs



class FeatureCacheMixin(BatchRunner):
    def __init__(self, cache_path=None):
        self.cache_path = Path(cache_path) if cache_path is not None else None

    @abstractmethod
    def forward_batch(self, batch):
        pass

    def __call__(self, dataset: ImageDataset) -> FeatureDataset:
        """
        Extract features from input dataset and return them as a new FeatureDataset.

        Args:
            dataset (ImageDataset): Extract features from this dataset.

        Returns:
            feature_dataset (FeatureDataset): A FeatureDataset containing the extracted features
        """

        check_dataset_output(dataset, check_label=False)
        self.model = self.model.to(self.device).eval()
        features = self.extract_with_cache(dataset, self.batch_size, self.num_workers)
        self.model = self.model.to("cpu")

        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label,
        )

    def _load_cache(self):
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache):
        if self.cache_path is not None:
            with open(self.cache_path, "wb") as f:
                pickle.dump(cache, f)

    def extract_with_cache(self, dataset, batch_size, num_workers):
        # Handle the case when cache is not required
        if self.cache_path is None:
            loader = self.make_loader(dataset, batch_size, num_workers)
            feats = []
            for batch in tqdm(loader, mininterval=1, ncols=100):
                feats.append(self.process_batch(batch))
            return torch.cat(feats).numpy()

        # Load the cache and determine the missing entries
        cache = self._load_cache()
        keys = [self.get_key(dataset, i) for i in range(len(dataset))]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        
        if missing:
            # Define loader on the missing entries
            subset = torch.utils.data.Subset(dataset, missing)
            loader = self.make_loader(subset, batch_size, num_workers)

            # Load the missing entries
            ptr = 0
            for batch in tqdm(loader, mininterval=1, ncols=100):
                feats = self.forward_batch(batch)

                for j in range(len(feats)):
                    cache[keys[missing[ptr]]] = feats[j]
                    ptr += 1

            # Save the cache including the missing entries
            self._save_cache(cache)

        # Remove potentially 
        return np.stack([cache[k] for k in keys])

    def process_batch(self, batch):
        return self.forward_batch(batch)

    def prune_cache(self, valid_keys):
        cache = self._load_cache()
        cache = {k: cache[k] for k in valid_keys}
        self._save_cache(cache)
