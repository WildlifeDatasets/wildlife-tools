
import os
import pytest
from wildlife_tools.data import WildlifeDataset, FeatureDataset
from PIL.Image import Image
from torch import Tensor

load_options_ok = ['full', 'crop_black']
load_options_error = ['full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide']


@pytest.mark.parametrize("img_load", load_options_ok)
def test_wildlife_dataset_img_load_ok(metadata, img_load):
    dataset = WildlifeDataset(**metadata, img_load=img_load)
    assert len(dataset) == 4
    assert isinstance(dataset[0][0], Image)
    assert isinstance(dataset.num_classes, int)


@pytest.mark.parametrize("img_load", load_options_error)
def test_wildlife_dataset_img_load_error(metadata, img_load):
    with pytest.raises(ValueError):
        dataset = WildlifeDataset(**metadata, img_load=img_load)
        dataset[0]


def test_wildlife_dataset_no_label(metadata):
    dataset = WildlifeDataset(**metadata, load_label=False)
    assert len(dataset) == 4
    assert isinstance(dataset[0], Image)
    assert isinstance(dataset.num_classes, int)


def test_deep_feature_dataset(dataset, features_deep):
    feature_dataset = FeatureDataset(features_deep, metadata=dataset.metadata)
    assert len(feature_dataset) == 4
    assert all(dataset.labels_string == feature_dataset.labels_string)
    assert isinstance(dataset.num_classes, int)


def test_sift_feature_dataset_save_load(dataset, features_sift):
    a = FeatureDataset(features_sift, metadata=dataset.metadata)
    a.save('test.pkl')
    b = FeatureDataset.from_file('test.pkl')


    assert a.metadata.equals(b.metadata)
    assert len(a.features) == len(b.features)
    os.remove('test.pkl')


# Compatibility with wildlife-datasets
def test_wildlife_datasets_load1(wd_dataset_labels):
    assert len(wd_dataset_labels) == 4
    assert isinstance(wd_dataset_labels[0][0], Tensor)
    assert isinstance(wd_dataset_labels.num_classes, int)


def test_wildlife_datasets_load2(wd_dataset_no_labels):
    assert len(wd_dataset_no_labels) == 4
    assert isinstance(wd_dataset_no_labels[0], Tensor)