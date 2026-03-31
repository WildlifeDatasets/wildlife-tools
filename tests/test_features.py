import pytest
import numpy as np
from wildlife_tools.data import FeatureDataset
from wildlife_tools.features import DataToMemory


def check_sift_features(features0, features1):
    assert len(features0) == len(features1)
    for f1, f2 in zip(features0, features1):
        np.array_equal(f1["descriptors"], f2["descriptors"])


def test_features_deep(dataset_deep, extractor):
    output = extractor(dataset_deep)
    assert type(output) == FeatureDataset
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


def test_sift_features_cached_identity(dataset_deep, extractor_sift, extractor_sift_cached):
    features0 = extractor_sift(dataset_deep)
    features1 = extractor_sift_cached(dataset_deep)
    check_sift_features(features0.features, features1.features)


def test_sift_features_cached_split(wd_dataset, extractor_sift_cached):
    m = 1
    n = len(wd_dataset)

    features_all = extractor_sift_cached(wd_dataset)
    dataset0 = wd_dataset.get_subset(range(0, m))
    dataset1 = wd_dataset.get_subset(range(m, n))
    features0 = extractor_sift_cached(dataset0)
    features1 = extractor_sift_cached(dataset1)
    check_sift_features(features0.features, features_all.features[:m])
    check_sift_features(features1.features, features_all.features[m:])


# Compatibility with wildlife-datasets
def test_wildlife_datasets_features1(wd_dataset_deep, extractor):
    features = extractor(wd_dataset_deep)
    assert type(features) == FeatureDataset
    assert len(features) == len(wd_dataset_deep)
    assert len(features[0][0]) == 768


def test_wildlife_datasets_features2(wd_dataset_deep_no_labels, extractor):
    with pytest.raises(ValueError):
        extractor(wd_dataset_deep_no_labels)
