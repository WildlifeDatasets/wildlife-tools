
import os
import pytest
from wildlife_tools.data import WildlifeDataset, FeatureDataset
from PIL.Image import Image

load_options = ['full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide', 'crop_black']


@pytest.mark.parametrize("img_load", load_options)
def test_wildllife_dataset_img_load(metadata, img_load):
    dataset = WildlifeDataset(**metadata, img_load=img_load)
    assert len(dataset) == 3
    assert isinstance(dataset[0][0], Image)
    assert isinstance(dataset.num_classes, int)


def test_wildllife_dataset_no_label(metadata):
    dataset = WildlifeDataset(**metadata, load_label=False)
    assert len(dataset) == 3
    assert isinstance(dataset[0], Image)
    assert isinstance(dataset.num_classes, int)


def test_deep_feature_dataset(dataset, features_deep):
    feature_dataset = FeatureDataset(features_deep, metadata=dataset.metadata)
    assert len(feature_dataset) == 3
    assert all(dataset.labels_string == feature_dataset.labels_string)
    assert isinstance(dataset.num_classes, int)


def test_sift_feature_dataset_save_load(dataset, features_sift):
    a = FeatureDataset(features_sift, metadata=dataset.metadata)
    a.save('test.pkl')
    b = FeatureDataset.from_file('test.pkl')


    assert a.metadata.equals(b.metadata)
    assert len(a.features) == len(b.features)
    os.remove('test.pkl')
