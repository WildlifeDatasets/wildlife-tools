import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
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
    """
    Validate that the cache was created with the same config.
    On the first use, the config is stored in the cache. On later uses, it is compared with the stored one.

    Args:
        env (lmdb.Environment): Opened LMDB cache.
        key (bytes): Key under which the config is stored.
        config (dict): Current config, typically from `CacheMixin.cache_config`.

    Raises:
        ValueError: If the stored config differs from the current one.
    """
    stored = read_cache_config(env, key)
    if stored is None:
        write_cache_config(env, key, config)
    elif stored != config:
        raise ValueError(
            f"Cache at {env.path()} was created with config {stored}, "
            f"but the current config is {config}. Use a different cache_path."
        )


class CacheMixin:
    """
    Base class for objects caching their results in an LMDB database at `cache_path`.

    Entries are indexed by keys derived from `image_id` in the dataset metadata (see `get_key`).
    The config returned by `cache_config` is stored in the cache on the first use and validated
    on every later use, so that a cache is not reused with a different model.
    """

    def __init__(self, cache_path: str | None = None, config_tag: str | None = None):
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.config_tag = config_tag

    def cache_config(self) -> dict:
        """
        Config identifying what is stored in the cache. Child classes extend it with model properties.
        Properties not covered by it (e.g. image transforms) should be encoded in `config_tag`.

        Returns:
            config (dict): Config stored in and validated against the cache.
        """
        return {"class": type(self).__name__, "tag": self.config_tag}

    def get_key(self, dataset: ImageDataset | FeatureDataset, index: int) -> str:
        """
        Cache key of a single image. Datasets with overlapping `image_id` must not share a cache.

        Args:
            dataset (ImageDataset | FeatureDataset): Dataset with `image_id` column in metadata.
            index (int): Positional index of the image in the dataset.

        Returns:
            key (str): Cache key of the image.
        """
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


class ModelMixin:
    device: str
    _model: torch.nn.Module | None = None
    _model_factory: Callable[[], torch.nn.Module] | None = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            if self._model_factory is None:
                raise RuntimeError(f"{type(self).__name__} has no model. Pass a model or use lazy_load.")
            self._model = self._model_factory()
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module | None) -> None:
        self._model = model

    @contextmanager
    def model_on_device(self) -> Generator[None, None, None]:
        self.model = self.model.to(self.device).eval()
        try:
            yield
        finally:
            self.model = self.model.to("cpu")


class FeatureCacheMixin(CacheMixin, ModelMixin, ABC, Generic[TDict, TFeature, TModel]):
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
        features = self.extract_with_cache(dataset)

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
            with self.model_on_device():
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
                with self.model_on_device():
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
