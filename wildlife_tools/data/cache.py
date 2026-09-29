import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar

import lmdb
import torch
from tqdm import tqdm

from ..tools import check_dataset_output
from .dataset import FeatureDataset, ImageDataset

TBatch = tuple[torch.Tensor, torch.Tensor]
TDict = TypeVar("TDict")  # np.ndarray | dict
TFeature = TypeVar("TFeature", bound=Sequence)  # np.ndarray | list[dict]
TModel = TypeVar("TModel", bound=Sequence)  # torch.Tensor | list[dict]

CONFIG_KEY = b"\x00__config__"


def open_lmdb(path: Path) -> lmdb.Environment:
    Path(path).mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(path),
        map_size=1 << 40,
        subdir=True,
        lock=True,
        readahead=False,
        meminit=False,
    )


def read_cache_config(env: lmdb.Environment, key: bytes) -> dict | None:
    with env.begin() as txn:
        val = txn.get(key)
    return pickle.loads(val) if val is not None else None


def write_cache_config(env: lmdb.Environment, key: bytes, config: dict) -> None:
    with env.begin(write=True) as txn:
        txn.put(key, pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL))


def check_cache_config(env: lmdb.Environment, key: bytes, config: dict) -> None:
    stored = read_cache_config(env, key)
    if stored is None:
        write_cache_config(env, key, config)
    elif stored != config:
        raise ValueError(
            f"Cache at {env.path()} was created with config {stored}, "
            f"but the current config is {config}. Use a different cache_path."
        )


class CacheMixin:
    def __init__(self, cache_path: str | None = None, config_tag: str | None = None):
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.config_tag = config_tag

    def cache_config(self) -> dict:
        return {"class": type(self).__name__, "tag": self.config_tag}

    def get_key(self, dataset: ImageDataset | FeatureDataset, index: int) -> str:
        return str(dataset.metadata["image_id"][index])

    def _open_env(self) -> lmdb.Environment:
        assert self.cache_path is not None
        env = open_lmdb(self.cache_path)
        try:
            check_cache_config(env, CONFIG_KEY, self.cache_config())
        except Exception:
            env.close()
            raise
        return env


class FeatureCacheMixin(CacheMixin, ABC, Generic[TDict, TFeature, TModel]):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str | None = "cpu",
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        super().__init__(cache_path=cache_path, config_tag=config_tag)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def _save_entry(self, txn: lmdb.Transaction, key: bytes, entry) -> None:
        txn.put(key, pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))

    def make_loader(self, dataset: ImageDataset) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

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

        keys = [self.get_key(dataset, i) for i in range(len(dataset))]

        # Load the cache (closed automatically, also on errors)
        with self._open_env() as env:
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
                            self._save_entry(txn, key, feats[j])
                            ptr += 1

            # Read all features back in order
            outputs = []
            with env.begin() as txn:
                for k in keys:
                    val = txn.get(k.encode())
                    outputs.append(pickle.loads(val))

        # Merge the extracted features
        return self.cat_features_dictionary(outputs)

    def process_batch(self, batch: TBatch) -> TModel:
        return self.forward_batch(batch)
