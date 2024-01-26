import pytest
import torchvision.transforms as T
import pandas as pd
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures, SIFTFeatures
from wildlife_tools.similarity import CosineSimilarity, MatchDescriptors
import numpy as np
import timm


@pytest.fixture(scope="session")
def metadata():
    return {'metadata':  pd.read_csv('TestDataset/metadata.csv'), 'root': 'TestDataset'}


@pytest.fixture(scope="session")
def array():
    return np.array([[1.0]])


@pytest.fixture(scope="session")
def dataset(metadata):
    return WildlifeDataset(**metadata)


@pytest.fixture(scope="session")
def dataset_deep(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
    return WildlifeDataset(**metadata, transform=transform)


@pytest.fixture(scope="session")
def dataset_sift(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.Grayscale()])
    return WildlifeDataset(**metadata, transform=transform)


@pytest.fixture(scope="session")
def dataset_loftr(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.Grayscale(), T.ToTensor()])
    return WildlifeDataset(**metadata, transform=transform, load_label=False)


@pytest.fixture(scope="session")
def features_sift(dataset_sift):
    extractor = SIFTFeatures()
    return extractor(dataset_sift)


@pytest.fixture(scope="session")
def features_deep(dataset_deep):
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
    extractor = DeepFeatures(backbone)
    return extractor(dataset_deep)


@pytest.fixture(scope="session")
def similarity_deep(features_deep):
    similarity = CosineSimilarity()
    return similarity(features_deep, features_deep)['cosine']


@pytest.fixture(scope="session")
def similarity_sift(features_sift):
    similarity = MatchDescriptors(descriptor_dim=128, thresholds=[0.8])
    return similarity(features_sift, features_sift)[0.8]