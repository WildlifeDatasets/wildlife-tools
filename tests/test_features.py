import pytest
from wildlife_tools.data import FeatureDataset
from wildlife_tools.features import DataToMemory


def test_features_deep(dataset_deep, extractor):
    output = extractor(dataset_deep)
    assert type(output) == FeatureDataset
    assert len(output) == len(dataset_deep)
    assert len(output[0][0]) == 768


def test_data_memory(dataset_deep):
    extractor = DataToMemory()
    output = extractor(dataset_deep)
    assert len(output) == len(dataset_deep)


# Compatibility with wildlife-datasets
def test_wildlife_datasets_features1(wd_dataset_labels, extractor):
    features = extractor(wd_dataset_labels)
    assert type(features) == FeatureDataset
    assert len(features) == len(wd_dataset_labels)
    assert len(features[0][0]) == 768


def test_wildlife_datasets_features2(wd_dataset_no_labels, extractor):
    with pytest.raises(ValueError):
        extractor(wd_dataset_no_labels)
