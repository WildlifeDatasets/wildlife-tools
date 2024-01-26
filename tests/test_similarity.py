
from wildlife_tools.similarity import CosineSimilarity, MatchDescriptors, MatchLOFTR


def test_cosine_similarity(features_deep):
    method = CosineSimilarity()
    output = method(features_deep, features_deep)
    assert 'cosine' in output
    assert output['cosine'].shape == (3, 3)


def test_match_descriptors(features_sift):
    similarity = MatchDescriptors(descriptor_dim=128, thresholds=[0.8])
    output = similarity(features_sift, features_sift)
    assert 0.8 in output
    assert output[0.8].shape == (3, 3)


def test_match_loftr(dataset_loftr):
    similarity = MatchLOFTR(device='cpu', thresholds=[0.8])
    output = similarity(dataset_loftr, dataset_loftr)
    assert 0.8 in output
    assert output[0.8].shape == (3, 3)