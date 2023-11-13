import pandas as pd
from pandas.util.testing import assert_frame_equal

def test_sp_extractor(dataset):
    dataset.transform = T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=(224, 224)),
        T.Grayscale(),
        T.ToTensor(),
    ])
    extractor = SuperPointFeatures()
    features = extractor.get_features(dataset)

    assert_frame_equal(features.metadata, dataset.metadata)
    assert len(features.features) == len(features.metadata)
    assert [i.shape for i in features.features] == [(76, 256), (151, 256)]

test_sp_extractor(dataset)

def test_sift_extractor(dataset):
    dataset.transform = T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=(224, 224)),
        T.Grayscale(),
        ])
    extractor = SIFTFeatures()
    features = extractor.get_features(dataset)

    assert_frame_equal(features.metadata, dataset.metadata)
    assert len(features.features) == len(features.metadata)
    assert [i.shape for i in features.features] == [(209, 128), (427, 128)]

test_sift_extractor(dataset)
