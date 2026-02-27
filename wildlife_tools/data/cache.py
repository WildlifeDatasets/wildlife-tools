import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, Iterable, Optional, TypeVar, Union

import lmdb
import torch
from tqdm import tqdm

from ..tools import check_dataset_output
from .dataset import FeatureDataset, ImageDataset

TBatch = tuple[torch.Tensor, torch.Tensor]
TDict = TypeVar("TDict")  # np.ndarray | dict
TFeature = TypeVar("TFeature", bound=Sequence)  # np.ndarray | list[dict]
TModel = TypeVar("TModel", bound=Sequence)  # torch.Tensor | list[dict]


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
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_key(self, dataset: ImageDataset, index: int) -> str:
        return str(dataset.metadata["image_id"][index])

    def make_loader(self, dataset: ImageDataset) -> torch.utils.data.DataLoader:

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

    def _open_env(self) -> lmdb.Environment:
        assert self.cache_path is not None
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        return lmdb.open(
            str(self.cache_path),
            map_size=1 << 40,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
        )

    def extract_with_cache(self, dataset: ImageDataset) -> TFeature:

        # Handle the case when cache is not required
        if self.cache_path is None:
            loader = self.make_loader(dataset)
            feats = []
            for batch in tqdm(loader, mininterval=1, ncols=100):
                feats.append(self.process_batch(batch))
            return self.cat_features_model(feats)

        # Load the cache
        env = self._open_env()
        keys = [self.get_key(dataset, i) for i in range(len(dataset))]

        # Determine missing entries
        missing = []
        with env.begin() as txn:
            for i, k in enumerate(keys):
                if txn.get(k.encode()) is None:
                    missing.append(i)

        if missing:
            # Define loader on the missing entries
            subset = torch.utils.data.Subset(dataset, missing)
            loader = self.make_loader(subset)

            # Load the missing entries
            ptr = 0
            for batch in tqdm(loader, mininterval=1, ncols=100):
                feats = self.forward_batch(batch)

                # Write the batch
                with env.begin(write=True) as txn:
                    for j in range(len(feats)):
                        key = keys[missing[ptr]].encode()
                        txn.put(key, pickle.dumps(feats[j], protocol=pickle.HIGHEST_PROTOCOL))
                        ptr += 1

        # Read all features back in order
        outputs = []
        with env.begin() as txn:
            for k in keys:
                val = txn.get(k.encode())
                outputs.append(pickle.loads(val))

        # Close the cache
        env.close()

        # Merge the extracted features
        return self.cat_features_dictionary(outputs)

    def process_batch(self, batch: TBatch) -> TModel:
        return self.forward_batch(batch)

    def prune_cache(self, valid_keys: Iterable) -> None:
        cache = self._load_cache()
        cache = {k: cache[k] for k in valid_keys}
        self._save_cache(cache)
