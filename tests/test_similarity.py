import numpy as np
import pytest

from wildlife_tools.data import ImageDataset
from wildlife_tools.similarity import CollectCountsRansac, CosineSimilarity, MatchLOFTR
from wildlife_tools.similarity.pairwise.collectors import Collector


class CollectMatches(Collector):
    def init_store(self, grid_shape: tuple | None = None) -> None:
        self.matches = []

    def add(self, results_list: list[dict]) -> None:
        self.matches.extend(results_list)

    def process_results(self) -> list[dict]:
        return sorted(self.matches, key=lambda m: (m["idx0"], m["idx1"]))


def test_cosine_similarity(features_deep):
    method = CosineSimilarity()
    output = method(features_deep, features_deep)
    assert output.shape == (4, 4)


def test_match_loftr(dataset_loftr):
    similarity = MatchLOFTR(device="cpu")
    output = similarity(dataset_loftr, dataset_loftr)
    assert output.shape == (4, 4)


def test_match_loftr_cached(dataset_loftr, cache_dir):
    output = MatchLOFTR(batch_size=2, device="cpu")(dataset_loftr, dataset_loftr)

    similarity = MatchLOFTR(batch_size=2, device="cpu", cache_path=str(cache_dir / "matches_loftr"))
    output_pairs = similarity(dataset_loftr, dataset_loftr, pairs=np.array([[0, 1], [1, 0]]))
    output_cached = similarity(dataset_loftr, dataset_loftr)
    output_cached_again = similarity(dataset_loftr, dataset_loftr)

    assert np.array_equal(output, output_cached, equal_nan=True)
    assert np.array_equal(output, output_cached_again, equal_nan=True)
    assert output_pairs[0, 1] == output[0, 1]
    assert np.isnan(output_pairs[0, 0])

    order = [1, 2, 3, 0]
    dataset_reordered = ImageDataset(
        metadata=dataset_loftr.metadata.iloc[order].reset_index(drop=True),
        root=dataset_loftr.root,
        transform=dataset_loftr.transform,
    )
    output_reordered = similarity(dataset_loftr, dataset_reordered)
    assert np.array_equal(output[:, order], output_reordered, equal_nan=True)

    similarity.collector = CollectCountsRansac()
    output_ransac = similarity(dataset_loftr, dataset_loftr)
    assert output_ransac.shape == (4, 4)


def test_match_loftr_cache_config_mismatch(dataset_loftr, cache_dir):
    cache_path = str(cache_dir / "matches_loftr_mismatch")
    pairs = np.array([[0, 1]])
    MatchLOFTR(batch_size=1, device="cpu", init_threshold=0.2, cache_path=cache_path)(
        dataset_loftr, dataset_loftr, pairs=pairs
    )
    MatchLOFTR(batch_size=1, device="cpu", init_threshold=0.2, cache_path=cache_path)(
        dataset_loftr, dataset_loftr, pairs=pairs
    )

    similarity = MatchLOFTR(batch_size=1, device="cpu", init_threshold=0.3, cache_path=cache_path)
    with pytest.raises(ValueError):
        similarity(dataset_loftr, dataset_loftr, pairs=pairs)


def test_match_loftr_apply_fine(dataset_loftr):
    coarse = MatchLOFTR(batch_size=2, device="cpu", apply_fine=False, collector=CollectMatches())(
        dataset_loftr, dataset_loftr
    )
    fine = MatchLOFTR(batch_size=2, device="cpu", apply_fine=True, collector=CollectMatches())(
        dataset_loftr, dataset_loftr
    )

    assert len(coarse) == len(fine)
    for c, f in zip(coarse, fine):
        np.testing.assert_allclose(c["scores"], f["scores"], atol=1e-5)
        np.testing.assert_allclose(c["kpts0"], f["kpts0"])
    keypoints_differ = [not np.allclose(c["kpts1"], f["kpts1"]) for c, f in zip(coarse, fine) if len(c["kpts1"])]
    assert any(keypoints_differ)


# Compatibility with wildlife-datasets
def test_wildlife_datasets_similarity(wd_dataset_deep, extractor):
    features = extractor(wd_dataset_deep)
    method = CosineSimilarity()
    output = method(features, features)
    assert output.shape == (4, 4)
