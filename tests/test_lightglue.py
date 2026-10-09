from functools import partial

import numpy as np
import pytest
import torch

from wildlife_tools.features import (
    AlikedExtractor,
    DeDoDeExtractor,
    DiskExtractor,
    DoGHardNetExtractor,
    KeyNetAffNetHardNetExtractor,
    XFeatExtractor,
)
from wildlife_tools.similarity import CollectCountsRansac, MatchLightGlue

pytestmark = pytest.mark.extra_models

EXTRACTORS = {
    "disk": DiskExtractor,
    "aliked": AlikedExtractor,
    "xfeat": XFeatExtractor,
    "dedodeb": partial(DeDoDeExtractor, descriptor="B"),
    "doghardnet": DoGHardNetExtractor,
    "dog_affnet_hardnet": partial(DoGHardNetExtractor, affnet=True),
    "keynet_affnet_hardnet": KeyNetAffNetHardNetExtractor,
}


def check_local_features(features0: list[dict], features1: list[dict]) -> None:
    assert len(features0) == len(features1)
    for f0, f1 in zip(features0, features1):
        assert f0.keys() == f1.keys()
        for key in f0:
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


@pytest.mark.parametrize("name", list(EXTRACTORS))
def test_local_features_batched(name, dataset_lightglue):
    features = EXTRACTORS[name](device="cpu", batch_size=1)(dataset_lightglue)
    features_batched = EXTRACTORS[name](device="cpu", batch_size=2)(dataset_lightglue)

    assert len(features_batched) == len(features)
    for f0, f1 in zip(features.features, features_batched.features):
        assert f0.keys() == f1.keys()
        for key in f0:
            assert f0[key].shape == f1[key].shape
        assert torch.equal(f0["image_size"], f1["image_size"])


@pytest.mark.parametrize("name", ["xfeat", "doghardnet"])
def test_local_features_not_forced(name, dataset_lightglue):
    max_num_keypoints = 2048
    features = EXTRACTORS[name](device="cpu", force_num_keypoints=False, max_num_keypoints=max_num_keypoints)(
        dataset_lightglue
    )
    for f in features.features:
        n = len(f["keypoints"])
        assert 0 < n <= max_num_keypoints
        assert all(len(v) == n for k, v in f.items() if k != "image_size")

    output = MatchLightGlue(features=name, batch_size=1, device="cpu")(features, features)
    assert output.shape == (len(features), len(features))
