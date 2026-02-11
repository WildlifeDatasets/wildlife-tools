from typing import Generic, Iterable, Optional, TypeVar, Union
from collections.abc import Sequence

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from tqdm import tqdm

from ..tools import check_dataset_output
from .dataset import FeatureDataset, ImageDataset


TBatch = tuple[torch.Tensor, torch.Tensor]
TDict = TypeVar("TDict") # np.ndarray | dict
TFeature = TypeVar("TFeature", bound=Sequence) # np.ndarray | list[dict]
TModel = TypeVar("TModel", bound=Sequence) # torch.Tensor | list[dict]


class CacheMixin(ABC, Generic[TModel]):
    def __init__(
            self,
            batch_size: int = 128,
            num_workers: int = 1,
            device: Optional[str] = "cpu",
            cache_path: Optional[str] = None,
            ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.cache_path = Path(cache_path) if cache_path is not None else None

    @abstractmethod
    def process_batch(self, batch: TBatch) -> TModel:
        pass

    def _load_cache(self) -> dict:
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache: dict) -> None:
        if self.cache_path is not None:
            with open(self.cache_path, "wb") as f:
                pickle.dump(cache, f)

    def get_key(self, dataset: ImageDataset, index: int) -> Union[str, int]:
        return dataset.metadata["image_id"][index]

    def make_loader(
            self,
            dataset: ImageDataset
            ) -> torch.utils.data.DataLoader :
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class FeatureCacheMixin(CacheMixin, Generic[TDict, TFeature, TModel]):
    @abstractmethod
    def cat_features_dictionary(self, feats: list[TDict]) -> TFeature:
        pass

    @abstractmethod
    def cat_features_model(self, feats: list[TModel]) -> TFeature:
        pass

    @abstractmethod
    def forward_batch(self, batch: TBatch) -> TModel:
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
        features = self.extract_with_cache(dataset)
        self.model = self.model.to("cpu")

        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label,
        )

    def extract_with_cache(self, dataset: ImageDataset) -> TFeature:
        
        # Handle the case when cache is not required
        if self.cache_path is None:
            loader = self.make_loader(dataset)
            feats = []
            for batch in tqdm(loader, mininterval=1, ncols=100):
                feats.append(self.process_batch(batch))
            return self.cat_features_model(feats)

        # Load the cache and determine the missing entries
        cache = self._load_cache()
        keys = [self.get_key(dataset, i) for i in range(len(dataset))]
        missing = [i for i, k in enumerate(keys) if k not in cache]

        if missing:
            # Define loader on the missing entries
            subset = torch.utils.data.Subset(dataset, missing)
            loader = self.make_loader(subset)

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
        return self.cat_features_dictionary([cache[k] for k in keys])

    def process_batch(self, batch: TBatch) -> TModel:
        return self.forward_batch(batch)

    def prune_cache(self, valid_keys: Iterable) -> None:
        cache = self._load_cache()
        cache = {k: cache[k] for k in valid_keys}
        self._save_cache(cache)
