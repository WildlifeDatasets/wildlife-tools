
from wildlife_tools.inference import KnnMatcher, KnnClassifier
from wildlife_tools.data import FeatureDataset


def test_knn_classifier_deep(similarity_deep, dataset):
    classifier = KnnClassifier(k=1, database_labels=dataset.labels_string)
    output = classifier(similarity_deep)
    assert len(output) == len(dataset)


def test_knn_classifier_sift(similarity_sift, dataset):
    classifier = KnnClassifier(k=1, database_labels=dataset.labels_string)
    output = classifier(similarity_sift)
    assert len(output) == len(dataset)


def test_knn_matcher(features_deep, dataset):
    feature_dataset = FeatureDataset(features_deep, metadata=dataset.metadata)
    matcher = KnnMatcher(feature_dataset)
    output = matcher(features_deep)
    assert len(output) == len(features_deep)


