from typing import Callable

import numpy as np
import torch

from ..data import FeatureDataset, ImageDataset


def get_hits(dataset0, dataset1):
    """Return grid of label correspondences given two labeled datasets."""

    gt0 = dataset0.labels_string
    gt1 = dataset1.labels_string
    gt_grid0 = np.tile(gt0, (len(gt1), 1)).T
    gt_grid1 = np.tile(gt1, (len(gt0), 1))
    return gt_grid0 == gt_grid1


class SimilarityPipeline:
    """
    Implements pipeline for matching and calculating similarity scores between two image datasets.

    Given two (query and database) image datasets, the pipeline consists of the following steps:

        1. Apply image transforms.
        2. Extract features for both datasets.
        3. Compute similarity scores between query and database images.
        4. Calibrate similarity scores.
    """

    def __init__(
        self,
        matcher: Callable,
        extractor: Callable | None = None,
        calibration: Callable | None = None,
        transform: Callable | None = None,
    ):
        """
        Args:
            matcher (callable): A matcher that computes scores between two feature datasets.
            extractor (callable, optional): A function to extract features from the image datasets.
                Not needed for some matchers.
            calibration (callable, optional): A calibration model to refine similarity scores.
            transform (callable, optional): Image transformation function applied before feature
                extraction.
        """

        self.matcher = matcher
        self.calibration = calibration
        self.calibration_done = False
        self.extractor = extractor
        self.transform = transform

    def get_feature_dataset(self, dataset: ImageDataset) -> FeatureDataset:
        """Apply transformations and extract features from the image dataset."""

        if self.transform is not None:
            dataset.transform = self.transform
        if self.extractor is not None:
            return self.extractor(dataset)
        else:
            return dataset

    def fit_calibration(self, dataset0: ImageDataset, dataset1: ImageDataset):
        """
        Fit the calibration model using given two image datasets.
        Fitting the calibration model uses all possible pairs of images from the two datasets.
        Input scores are similarity scores calculated by the matcher.
        Binary input labels are based on ground truth labels (identity is the same or not).

        Args:
            dataset0 (ImageDataset): The first dataset (e.g., part of training set).
            dataset1 (ImageDataset): The second dataset (e.g., part of training set).

        """

        if self.calibration is None:
            raise ValueError("Calibration method is not assigned.")

        dataset0 = self.get_feature_dataset(dataset0)
        dataset1 = self.get_feature_dataset(dataset1)
        score = self.matcher(dataset0, dataset1)

        hits = get_hits(dataset0, dataset1)
        self.calibration.fit(score.flatten(), hits.flatten())
        self.calibration_done = True

    def __call__(self, dataset0: ImageDataset, dataset1: ImageDataset, pairs: list | None = None) -> np.ndarray:
        """
        Compute similarity scores between two image datasets, with optional calibration.

        Args:
            dataset0 (ImageDataset): The first dataset (e.g., query set).
            dataset1 (ImageDataset): The second dataset (e.g., database set).
            pairs (list of tuples, optional): Specific pairs of images to compute similarity scores.
                If None, compute similarity scores for all pairs.

        Returns:
            np.ndarray: 2D array of similarity scores between the query and database images.
                If `calibration` is provided, return the calibrated similarity scores.
        """

        if not self.calibration_done and (self.calibration is not None):
            raise ValueError("Calibration is not fitted. Use fit_calibration method.")

        dataset0 = self.get_feature_dataset(dataset0)
        dataset1 = self.get_feature_dataset(dataset1)
        score = self.matcher(dataset0, dataset1, pairs=pairs)

        if self.calibration is not None:
            if pairs is not None:
                pairs = np.array(pairs)
                idx0 = pairs[:, 0]
                idx1 = pairs[:, 1]
                score[idx0, idx1] = self.calibration.predict(score[idx0, idx1])
            else:
                score = self.calibration.predict(score.flatten()).reshape(score.shape)
        return score


class WildFusion:
    """
    `WildFusion` uses the mean of multiple calibrated `SimilarityPipeline` to calculate fused scores.

    Since many local feature matching models require deep neural network inference for each query and
    database pair, the computation quickly becomes infeasible even for moderately sized datasets.

    WildFusion can be used with a limited computational budget by applying it only B times per query
    image. It uses a fast-to-compute similarity score (e.g., cosine similarity of deep features) provided
    by the priority_pipeline to construct a shortlist of the most promising matches for a given query.
    Final ranking is then based on WildFusion scores calculated for the pairs in the shortlist.
    """

    def __init__(
        self,
        calibrated_pipelines: list[SimilarityPipeline],
        priority_pipeline: SimilarityPipeline | None = None,
        weight_computation: Callable | None = None,
    ):
        """
        Args:
            calibrated_pipelines (list[SimilarityPipeline]): List of SimilarityPipeline objects.
            priority_pipeline (SimilarityPipeline, optional): Fast-to-compute similarity matcher
                used for shortlisting.
        """

        self.calibrated_pipelines = calibrated_pipelines
        self.priority_pipeline = priority_pipeline
        self.weight_computation = weight_computation

    def fit_calibration(self, dataset0: ImageDataset, dataset1: ImageDataset):
        """
        Fit the all calibration models for all matchers in `calibrated_pipelines`.

        Args:
            dataset0 (ImageDataset): The first dataset (e.g., part of training set).
            dataset1 (ImageDataset): The second dataset (e.g., part of training set).
        """

        for matcher in self.calibrated_pipelines:
            matcher.fit_calibration(dataset0, dataset1)

        if (self.priority_pipeline is not None) and (self.priority_pipeline.calibration is not None):
            self.priority_pipeline.fit_calibration(dataset0, dataset1)

    def get_priority_pairs(self, dataset0: ImageDataset, dataset1: ImageDataset, B: int) -> np.ndarray:
        """Implements shortlisting strategy for selection of most relevant pairs."""

        if self.priority_pipeline is None:
            raise ValueError("Priority matcher is not assigned.")

        priority = self.priority_pipeline(dataset0, dataset1)
        _, idx1 = torch.topk(torch.tensor(priority), min(B, priority.shape[1]))
        idx0 = np.indices(idx1.numpy().shape)[0]
        grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T
        return grid_indices

    def __call__(
        self,
        dataset0: ImageDataset,
        dataset1: ImageDataset,
        pairs: list | None = None,
        B: int = None,
    ):
        """
        Compute fused similarity scores between two images datasets using multiple calibrated
        matchers. WildFusion score is is calculated as mean of calibrated similarity scores.

        Optionally, to limit the number of pairs to compute, shortlist strategy to select the most
        promising pairs can be used.

        Args:
            dataset0 (ImageDataset): The first dataset (e.g., query set).
            dataset1 (ImageDataset): The second dataset (e.g., database set).
            pairs (list of tuples, optional): Specific pairs of images to compute similarity scores.
                                              If None, compute similarity scores for all pairs.
                                              Is ignored if `B` is provided.
            B (int, optional): Number of pairs to compute similarity scores for. Required `priority_pipeline` to be assigned.
                                              If None, compute similarity scores for all pairs.

        Returns:
            score_combined (np.ndarray): 2D array of similarity scores between the query and database images.
                        If `calibration` is provided, returns the calibrated similarity scores.
        """

        if B is not None:
            pairs = self.get_priority_pairs(dataset0, dataset1, B=B)

        scores = []
        for matcher in self.calibrated_pipelines:
            scores.append(matcher(dataset0, dataset1, pairs=pairs))

        if self.weight_computation is None:
            scores = np.where(np.isnan(scores), -np.inf, np.array(scores))
            score_combined = np.mean(scores, axis=0)
        else:
            scores = np.where(np.isnan(scores), 0, np.array(scores))
            weights = self.weight_computation(scores, dataset0, dataset1)
            score_combined = np.mean(scores*weights, axis=0)
        return score_combined
