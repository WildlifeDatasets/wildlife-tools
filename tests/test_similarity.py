import numpy as np

from wildlife_tools.data import ImageDataset
from wildlife_tools.similarity import CollectCountsRansac, CosineSimilarity, MatchLOFTR


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


# Compatibility with wildlife-datasets
def test_wildlife_datasets_similarity(wd_dataset_deep, extractor):
    features = extractor(wd_dataset_deep)
    method = CosineSimilarity()
    output = method(features, features)
    assert output.shape == (4, 4)
