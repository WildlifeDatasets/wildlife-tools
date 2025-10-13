
from wildlife_tools.similarity import CosineSimilarity, MatchLOFTR


def test_cosine_similarity(features_deep):
    method = CosineSimilarity()
    output = method(features_deep, features_deep)
    assert output.shape == (4, 4)


def test_match_loftr(dataset_loftr):
    similarity = MatchLOFTR(device='cpu')
    output = similarity(dataset_loftr, dataset_loftr)
    assert output.shape == (4, 4)


# Compatibility with wildlife-datasets
def test_wildlife_datasets_similarity(wd_dataset_labels, extractor):
    features = extractor(wd_dataset_labels)
    method = CosineSimilarity()
    output = method(features, features)
    assert output.shape == (4, 4)