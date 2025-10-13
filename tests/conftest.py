import os
import pytest
import torchvision.transforms as T
import pandas as pd
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures, SiftExtractor, SuperPointExtractor
from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
import numpy as np
import timm


@pytest.fixture(scope="session")
def metadata():
    path = os.path.dirname(__file__)
    csv_path = os.path.join(path, 'TestDataset', 'metadata.csv')
    return {'metadata': pd.read_csv(csv_path), 'root': os.path.join(path, 'TestDataset')}


@pytest.fixture(scope="session")
def array():
    return np.array([[1.0]])


@pytest.fixture(scope="session")
def dataset(metadata):
    return ImageDataset(**metadata)


@pytest.fixture(scope="session")
def dataset_deep(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
    return ImageDataset(**metadata, transform=transform)


@pytest.fixture(scope="session")
def dataset_lightglue(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
    return ImageDataset(**metadata, transform=transform)


@pytest.fixture(scope="session")
def dataset_loftr(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.Grayscale(), T.ToTensor()])
    return ImageDataset(**metadata, transform=transform, load_label=True)


@pytest.fixture(scope="session")
def features_sift(dataset_lightglue):
    extractor = SiftExtractor()
    return extractor(dataset_lightglue)


@pytest.fixture(scope="session")
def features_superpoint(dataset_lightglue):
    extractor = SuperPointExtractor()
    return extractor(dataset_lightglue)


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
    similarity = MatchLightGlue(features='sift', descriptor_dim=128, thresholds=[0.5])
    return similarity(features_sift, features_sift)


@pytest.fixture(scope="session")
def similarity_superpoint(features_superpoint):
    similarity = MatchLightGlue(features='sift', descriptor_dim=128, thresholds=[0.5])
    return similarity(features_superpoint, features_superpoint)