import multiprocessing as mp
import os

import kornia.feature as KF
import numpy as np
import pandas as pd
import pytest
import timm
import torch
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
from wildlife_datasets import datasets

from wildlife_tools.data import ImageDataset
from wildlife_tools.features import AlikedExtractor, ClipFeatures, DeepFeatures, DinoFeatures
from wildlife_tools.similarity import CosineSimilarity

mp.set_start_method("spawn", force=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-extra-models",
        action="store_true",
        default=False,
        help="Run tests that download additional models (DISK, ALIKED, LightGlue).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "extra_models: test downloads additional models, run with --run-extra-models")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-extra-models"):
        return
    skip = pytest.mark.skip(reason="needs --run-extra-models")
    for item in items:
        if "extra_models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def metadata():
    path = os.path.dirname(__file__)
    csv_path = os.path.join(path, "TestDataset", "metadata.csv")
    return {"metadata": pd.read_csv(csv_path), "root": os.path.join(path, "TestDataset")}


@pytest.fixture(scope="session")
def array():
    return np.array([[1.0]])


@pytest.fixture(scope="session")
def backbone():
    return timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True)


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
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="session")
def extractor(backbone):
    return DeepFeatures(backbone)


@pytest.fixture(scope="session")
def extractor_cached(backbone, cache_dir):
    cache_path = cache_dir / "features_deep.pkl"
    return DeepFeatures(backbone, cache_path=cache_path)


@pytest.fixture(scope="session")
def extractor_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return ClipFeatures(model=model, processor=processor)


@pytest.fixture(scope="session")
def extractor_dino():
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    return DinoFeatures(model=model, processor=processor)


class RandomAlikedExtractor(AlikedExtractor):
    def build_model(self) -> torch.nn.Module:
        torch.manual_seed(0)
        return KF.ALIKED(
            self.model_name,
            max_num_keypoints=self.max_num_keypoints,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius,
        )


@pytest.fixture(scope="session")
def random_aliked_cls():
    return RandomAlikedExtractor


@pytest.fixture(scope="session")
def extractor_aliked():
    return RandomAlikedExtractor(device="cpu")


@pytest.fixture(scope="session")
def extractor_aliked_cached(cache_dir):
    return RandomAlikedExtractor(device="cpu", cache_path=cache_dir / "features_aliked")


@pytest.fixture(scope="session")
def features_aliked(dataset_lightglue, extractor_aliked):
    return extractor_aliked(dataset_lightglue)


@pytest.fixture(scope="session")
def features_deep(dataset_deep, extractor):
    return extractor(dataset_deep)


@pytest.fixture(scope="session")
def similarity_deep(features_deep):
    similarity = CosineSimilarity()
    return similarity(features_deep, features_deep)["cosine"]


@pytest.fixture(scope="session")
def wd_dataset(metadata):
    return datasets.WildlifeDataset(metadata["root"], metadata["metadata"], load_label=True, factorize_label=True)


@pytest.fixture(scope="session")
def wd_dataset_deep(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
    return datasets.WildlifeDataset(
        metadata["root"], metadata["metadata"], transform=transform, load_label=True, factorize_label=True
    )


@pytest.fixture(scope="session")
def wd_dataset_deep_no_labels(metadata):
    transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
    return datasets.WildlifeDataset(metadata["root"], metadata["metadata"], transform=transform)
