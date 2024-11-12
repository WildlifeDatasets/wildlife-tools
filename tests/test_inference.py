
import pytest
import numpy as np
from wildlife_tools.inference import TopkClassifier, KnnClassifier
    

@pytest.fixture
def database_labels():
    return np.array(['A', 'B', 'C', 'D', 'E'])


@pytest.fixture
def similarity():
    return np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5],  # query 1
        [0.1, 0.4, 0.5, 0.2, 0.3],  # query 2
        [0.3, 0.6, 0.1, 0.4, 0.2],  # query 3
        [0.5, 0.6, 0.7, 0.6, 0.5]   # query 4
    ])


def test_knn_without_scores_k1(database_labels, similarity):
    classifier = KnnClassifier(database_labels, k=1, return_scores=False)
    preds = classifier(similarity)
    expected_preds = np.array(['A', 'C', 'B', 'C'])
    np.testing.assert_array_equal(preds, expected_preds)


def test_knn_with_scores_k1(database_labels, similarity):
    classifier = KnnClassifier(database_labels, k=1, return_scores=True)
    preds, scores = classifier(similarity)
    expected_preds = np.array(['A', 'C', 'B', 'C'])
    expected_scores = np.array([0.9, 0.5, 0.6, 0.7])
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_almost_equal(scores, expected_scores)


def test_knn_without_scores_k2(database_labels, similarity):
    classifier = KnnClassifier(database_labels, k=2, return_scores=False)
    preds = classifier(similarity)
    expected_preds = np.array(['A', 'C', 'B', 'C'])
    np.testing.assert_array_equal(preds, expected_preds)


def test_knn_with_scores_k2(database_labels, similarity):
    classifier = KnnClassifier(database_labels, k=2, return_scores=True)
    preds, scores = classifier(similarity)
    expected_preds = np.array(['A', 'C', 'B', 'C'])
    expected_scores = np.array([0.9, 0.5, 0.6, 0.7])
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_almost_equal(scores, expected_scores)


def test_knn_with_ties_k3():
    database_labels = np.array(['A', 'A', 'B', 'B'])
    similarity = np.array([[0.9, 0.9, 0.8, 0.8]])

    classifier = KnnClassifier(database_labels, k=4, return_scores=True)
    preds, scores = classifier(similarity)
    expected_preds = np.array(['A'])  # 'A' is selected because of mean score
    expected_scores = np.array([0.9])
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_almost_equal(scores, expected_scores)


def test_topk_without_scores_k1(database_labels, similarity):
    classifier = TopkClassifier(database_labels, k=1, return_all=False)
    preds = classifier(similarity)
    expected_preds = np.array([['A'], ['C'], ['B'], ['C']])
    np.testing.assert_array_equal(preds, expected_preds)


def test_topk_with_scores_k1(database_labels, similarity):
    classifier = TopkClassifier(database_labels, k=1, return_all=True)
    preds, scores, _ = classifier(similarity)
    expected_preds = np.array([['A'], ['C'], ['B'], ['C']])
    expected_scores = np.array([[0.9], [0.5], [0.6], [0.7]])
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_almost_equal(scores, expected_scores)


def test_topk_with_scores_k3(database_labels, similarity):
    classifier = TopkClassifier(database_labels, k=3, return_all=True)
    preds, scores, _ = classifier(similarity)
    expected_preds = np.array([
        ['A', 'B', 'C'],
        ['C', 'B', 'E'],
        ['B', 'D', 'A'],
        ['C', 'B', 'D'],
    ])
    expected_scores = np.array([
        [0.9, 0.8, 0.7],
        [0.5, 0.4, 0.3],
        [0.6, 0.4, 0.3],
        [0.7, 0.6, 0.6]
    ])
    np.testing.assert_array_equal(preds, expected_preds)
    np.testing.assert_array_almost_equal(scores, expected_scores)


def test_topk_large_k(database_labels, similarity):
    classifier = TopkClassifier(database_labels, k=10, return_all=False)
    preds = classifier(similarity)
    expected_preds = np.array([
        ['A', 'B', 'C', 'D', 'E'],
        ['C', 'B', 'E', 'D', 'A'],
        ['B', 'D', 'A', 'E', 'C'],
        ['C', 'B', 'D', 'A', 'E']
    ])
    np.testing.assert_array_equal(preds, expected_preds)
