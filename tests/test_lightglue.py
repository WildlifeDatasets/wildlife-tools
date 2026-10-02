import numpy as np
import pytest
import torch

from wildlife_tools.features import AlikedExtractor, DiskExtractor, SiftExtractor, SuperPointExtractor
from wildlife_tools.similarity import CollectCountsRansac, MatchLightGlue

pytestmark = pytest.mark.extra_models

EXTRACTORS = {
    "sift": SiftExtractor,
    "superpoint": SuperPointExtractor,
    "disk": DiskExtractor,
    "aliked": AlikedExtractor,
}


def check_local_features(features0: list[dict], features1: list[dict]) -> None:
    assert len(features0) == len(features1)
    for f0, f1 in zip(features0, features1):
        for key in ["keypoints", "keypoint_scores", "descriptors", "image_size"]:
            assert torch.equal(f0[key], f1[key])


@pytest.mark.parametrize("name", list(EXTRACTORS))
def test_local_features_cached(name, dataset_lightglue, cache_dir):
    extractor_cls = EXTRACTORS[name]
    torch.manual_seed(0)
    features = extractor_cls(device="cpu")(dataset_lightglue)
    assert len(features) == len(dataset_lightglue)

    extractor_cached = extractor_cls(device="cpu", cache_path=cache_dir / f"features_{name}")
    torch.manual_seed(0)
    features_cached = extractor_cached(dataset_lightglue)
    features_cached_again = extractor_cached(dataset_lightglue)

    check_local_features(features.features, features_cached.features)
    check_local_features(features.features, features_cached_again.features)


@pytest.mark.parametrize("name", list(EXTRACTORS))
def test_match_lightglue_cached(name, dataset_lightglue, cache_dir):
    features = EXTRACTORS[name](device="cpu")(dataset_lightglue)
    n = len(features)

    output = MatchLightGlue(features=name, batch_size=2, device="cpu")(features, features)
    assert output.shape == (n, n)

    similarity = MatchLightGlue(
        features=name, batch_size=2, device="cpu", cache_path=str(cache_dir / f"matches_lightglue_{name}")
    )
    output_cached = similarity(features, features)
    output_cached_again = similarity(features, features)

    assert np.array_equal(output, output_cached, equal_nan=True)
    assert np.array_equal(output, output_cached_again, equal_nan=True)

    similarity.collector = CollectCountsRansac()
    output_ransac = similarity(features, features)
    assert output_ransac.shape == (n, n)
