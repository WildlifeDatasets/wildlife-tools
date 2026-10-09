import numpy as np
import pytest
import torch

from wildlife_tools.data import FeatureDataset
from wildlife_tools.features import DataToMemory
from wildlife_tools.features.local_utils import pad_features


def test_clip_features(dataset, extractor_clip):
    output = extractor_clip(dataset)
    assert isinstance(output, FeatureDataset)
    assert len(output) == len(dataset)


def test_dino_features(dataset, extractor_dino):
    output = extractor_dino(dataset)
    assert isinstance(output, FeatureDataset)
    assert len(output) == len(dataset)


def check_local_features(features0, features1):
    assert len(features0) == len(features1)
    for f1, f2 in zip(features0, features1):
        assert np.array_equal(f1["descriptors"], f2["descriptors"])


def test_features_deep(dataset_deep, extractor):
    output = extractor(dataset_deep)
    assert isinstance(output, FeatureDataset)
    assert len(output) == len(dataset_deep)
    assert len(output[0][0]) == 768


def test_data_memory(dataset_deep):
    extractor = DataToMemory()
    output = extractor(dataset_deep)
    assert len(output) == len(dataset_deep)


def test_deep_features_cached_identity(dataset_deep, extractor, extractor_cached):
    features0 = extractor(dataset_deep)
    features1 = extractor_cached(dataset_deep)
    assert np.array_equal(features0.features, features1.features)


def test_deep_features_cached_split(wd_dataset_deep, extractor_cached):
    m = 1
    n = len(wd_dataset_deep)

    features_all = extractor_cached(wd_dataset_deep)
    dataset0 = wd_dataset_deep.get_subset(range(0, m))
    dataset1 = wd_dataset_deep.get_subset(range(m, n))
    features0 = extractor_cached(dataset0)
    features1 = extractor_cached(dataset1)
    assert np.array_equal(features0.features, features_all.features[:m])
    assert np.array_equal(features1.features, features_all.features[m:])


def test_local_features_cached_identity(dataset_deep, extractor_aliked, extractor_aliked_cached):
    features0 = extractor_aliked(dataset_deep)
    features1 = extractor_aliked_cached(dataset_deep)
    check_local_features(features0.features, features1.features)


def test_local_features_cached_split(wd_dataset_deep, extractor_aliked_cached):
    m = 1
    n = len(wd_dataset_deep)

    features_all = extractor_aliked_cached(wd_dataset_deep)
    dataset0 = wd_dataset_deep.get_subset(range(0, m))
    dataset1 = wd_dataset_deep.get_subset(range(m, n))
    features0 = extractor_aliked_cached(dataset0)
    features1 = extractor_aliked_cached(dataset1)
    check_local_features(features0.features, features_all.features[:m])
    check_local_features(features1.features, features_all.features[m:])


def test_features_cache_config_mismatch(dataset_deep, cache_dir, random_aliked_cls):
    cache_path = cache_dir / "features_mismatch"
    random_aliked_cls(device="cpu", cache_path=cache_path)(dataset_deep)
    random_aliked_cls(device="cpu", cache_path=cache_path)(dataset_deep)

    with pytest.raises(ValueError):
        random_aliked_cls(device="cpu", max_num_keypoints=100, cache_path=cache_path)(dataset_deep)
    with pytest.raises(ValueError):
        random_aliked_cls(device="cpu", config_tag="resize224", cache_path=cache_path)(dataset_deep)


@pytest.mark.parametrize("n", [0, 3])
def test_pad_features(n):
    image_size = torch.tensor([50, 30])
    features = {
        "keypoints": torch.rand(n, 2) * 10,
        "keypoint_scores": torch.rand(n),
        "descriptors": torch.rand(n, 8),
        "lafs": torch.rand(n, 2, 3),
    }
    padded = pad_features(features, 10, image_size)

    assert all(len(v) == 10 for v in padded.values())
    assert padded["lafs"].shape == (10, 2, 3)
    for key in features:
        assert torch.equal(padded[key][:n], features[key])
    assert torch.all(padded["descriptors"][n:] == 0)
    assert torch.all(padded["keypoint_scores"][n:] == 0)
    assert torch.all(padded["keypoints"][n:] >= 0)
    assert torch.all(padded["keypoints"][n:] <= image_size - 1)


def test_pad_features_enough_keypoints():
    features = {"keypoints": torch.rand(10, 2), "descriptors": torch.rand(10, 8)}
    assert pad_features(features, 5, torch.tensor([50, 30])) is features


# Compatibility with wildlife-datasets
def test_wildlife_datasets_features1(wd_dataset_deep, extractor):
    features = extractor(wd_dataset_deep)
    assert isinstance(features, FeatureDataset)
    assert len(features) == len(wd_dataset_deep)
    assert len(features[0][0]) == 768


def test_wildlife_datasets_features2(wd_dataset_deep_no_labels, extractor):
    with pytest.raises(ValueError):
        extractor(wd_dataset_deep_no_labels)
