"""Regression tests for targeted maintenance fixes in datasets and caching."""

import pandas as pd
from PIL import Image

from wildlife_tools.data import FeatureDataset
from wildlife_tools.data.cache import CacheMixin
from wildlife_tools.data.dataset import WildlifeDataset


class DummyCache(CacheMixin):
    """Minimal cache implementation for exercising cache-key behavior."""

    def process_batch(self, batch):
        return batch


def test_feature_dataset_from_config(tmp_path):
    """Feature datasets should round-trip through the config-based loader."""
    metadata = pd.DataFrame({"identity": ["a"], "path": ["image.jpg"]})
    dataset = FeatureDataset(features=[[1, 2, 3]], metadata=metadata)
    path = tmp_path / "features.pkl"
    dataset.save(path)

    loaded = FeatureDataset.from_config({"path": path, "load_label": False})

    assert loaded.metadata.equals(dataset.metadata)
    assert loaded.features == dataset.features
    assert loaded.load_label is False


def test_cache_key_falls_back_to_path_when_image_id_missing():
    """Cache keys should fall back to the configured path column when `image_id` is absent."""
    metadata = pd.DataFrame({"identity": ["a"], "path": ["image.jpg"]})
    dataset = WildlifeDataset(metadata=metadata, root=None)
    cache = DummyCache()

    assert cache.get_key(dataset, 0) == "image.jpg"


def test_segmentation_string_uses_literal_eval_not_eval():
    """Segmentation strings should be parsed safely and still produce valid samples."""
    metadata = pd.DataFrame(
        {
            "identity": ["a"],
            "path": ["does-not-matter.jpg"],
            "segmentation": ["[0, 0, 1, 0, 1, 1, 0, 1]"],
        }
    )
    dataset = WildlifeDataset(metadata=metadata, root=None, img_load="full_mask")
    dataset.get_image = lambda _: Image.new("RGB", (2, 2))

    image, label = dataset[0]

    assert image.size == (2, 2)
    assert label == 0


def test_missing_image_raises_file_not_found():
    """Unreadable image paths should raise a clear file-not-found error."""
    metadata = pd.DataFrame({"identity": ["a"], "path": ["missing.jpg"]})
    dataset = WildlifeDataset(metadata=metadata, root=None)

    try:
        dataset[0]
    except FileNotFoundError as exc:
        assert "missing.jpg" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for unreadable image path.")


def test_crop_black_raises_on_empty_image():
    """Cropping black borders should fail clearly when the image is entirely empty."""
    metadata = pd.DataFrame({"identity": ["a"], "path": ["unused.jpg"]})
    dataset = WildlifeDataset(metadata=metadata, root=None, img_load="crop_black")
    dataset.get_image = lambda _: Image.new("RGB", (2, 2))

    try:
        dataset[0]
    except ValueError as exc:
        assert "empty crop" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty crop.")
