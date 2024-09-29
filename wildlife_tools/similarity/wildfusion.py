from typing import Callable
import numpy as np
import pandas as pd
import torch
from wildlife_tools.data import WildlifeDataset, FeatureDataset

def get_hits(dataset0, dataset1):
    '''Return grid of label correspondences given two labeled datasets.'''

    gt0 = dataset0.labels_string
    gt1 = dataset1.labels_string
    gt_grid0 = np.tile(gt0, (len(gt1), 1)).T
    gt_grid1 = np.tile(gt1, (len(gt0), 1))
    return (gt_grid0 == gt_grid1)


class SimilarityPipeline():
    '''
    Pipeline for similarity matching. 

    Implements funcion Image dataset x Image dataset -> Similarity scores.
    Couples query and database datasets with transforms, 
    feature extraction, matching, and calibration.

    Given two (query and database) image datasets:
        1. Apply image transforms (optional).
        2. Extract features for both datasets (optional).
        3. Compute similarity scores between query and database images.
        4. Calibrate similarity scores (optional).
    
    Args:
        matcher (callable): The similarity matcher that computes scores between two feature datasets.
        extractor (callable, optional): A function to extract features from the image datasets.
        calibration (callable, optional): A calibration model to refine similarity scores.
        transform (callable, optional): Image transformation function applied before feature extraction.
        calibration_done (bool): A flag indicating if the calibration model has been fitted.


    Returns:
        2D array of similarity scores.

        
    Example Usage:
        >>> pipeline = SimilarityPipeline(matcher=some_matcher, calibration=IsotonicCalibration())
        >>> pipeline.fit_calibration(database_set, database_set)
        >>> scores = pipeline(query_set, database_set)
    '''
    def __init__(
            self,
            matcher: Callable | None = None,
            extractor: Callable | None = None,
            calibration: Callable | None = None,
            transform: Callable | None = None
        ):
        self.matcher = matcher
        self.calibration = calibration
        self.calibration_done = False
        self.extractor = extractor
        self.transform = transform


    def get_feature_dataset(self, dataset: WildlifeDataset) -> FeatureDataset:
        ''' Apply transformations and extract features from the image dataset. '''

        if self.transform is not None:
            dataset.transform = self.transform
        if self.extractor is not None:
            return self.extractor(dataset)
        else:
            return dataset


    def fit_calibration(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset):
        '''
        Fit the calibration model using similarity scores and ground truth labels from two datasets.
        Updates the calibration model with the fitted parameters and sets `calibration_done` to True.
        '''

        if self.calibration is None:
            raise ValueError('Calibration method is not assigned.')

        dataset0 = self.get_feature_dataset(dataset0)
        dataset1 = self.get_feature_dataset(dataset1)
        score = self.matcher(dataset0, dataset1)

        hits = get_hits(dataset0, dataset1)
        self.calibration.fit(score.flatten(), hits.flatten())
        self.calibration_done = True


    def __call__(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset, pairs: list | None = None) -> np.ndarray:
        '''
        Compute similarity scores between two images datasets, with optional calibration.

        Args:
            dataset0 (WildlifeDataset): The first dataset (e.g., query set).
            dataset1 (WildlifeDataset): The second dataset (e.g., database set).
            pairs (list of tuples, optional): Specific pairs of images to compute similarity scores for.
                                              If None, compute similarity scores for all pairs.

        Returns:
            np.ndarray: 2D array of similarity scores between the query and database images.
                        If `calibration` is provided, returns the calibrated similarity scores.


        '''
        if not self.calibration_done and (self.calibration is not None):
            raise ValueError('Calibration is not fitted. Use fit_calibration method.')

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
    def __init__(
        self,
        calibrated_matchers: list[SimilarityPipeline],
        priority_matcher: SimilarityPipeline | None = None
    ):
        self.calibrated_matchers = calibrated_matchers
        self.priority_matcher = priority_matcher


    def fit_calibration(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset):
        for matcher in self.calibrated_matchers:
            matcher.fit_calibration(dataset0, dataset1)

        if self.priority_matcher is not None:
            self.priority_matcher.fit_calibration(dataset0, dataset1)


    def get_priority_pairs(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset, B):
        ''' Shortlisting strategy for selection of most relevant pairs.'''

        if self.priority_matcher is None:
            raise ValueError('Priority matcher is not assigned.')

        priority = self.priority_matcher(dataset0, dataset1)
        _, idx1 = torch.topk(torch.tensor(priority), min(B, priority.shape[1]))
        idx0 = np.indices(idx1.numpy().shape)[0]
        grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T
        return grid_indices


    def __call__(self, dataset0, dataset1, pairs=None, B=None):            
        if B is not None:
            pairs = self.get_priority_pairs(dataset0, dataset1, B=B)

        scores = []
        for matcher in self.calibrated_matchers:
            scores.append(matcher(dataset0, dataset1, pairs=pairs))

        score_combined = np.mean(scores, axis=0)
        score_combined = np.where(np.isnan(score_combined), -np.inf, score_combined)
        return score_combined
