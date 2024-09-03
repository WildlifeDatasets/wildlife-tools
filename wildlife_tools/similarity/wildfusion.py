import numpy as np
import pandas as pd
import torch
from .calibration import IsotonicCalibration
from wildlife_tools.data import WildlifeDataset, FeatureDataset

def get_hits(dataset0, dataset1):
    '''Return grid of label correspondences given two labeled datasets.'''

    gt0 = dataset0.labels_string
    gt1 = dataset1.labels_string
    gt_grid0 = np.tile(gt0, (len(gt1), 1)).T
    gt_grid1 = np.tile(gt1, (len(gt0), 1))
    return (gt_grid0 == gt_grid1)


class SimilarityPipeline():

    def __init__(self, matcher, extractor=None, calibration=None, transform=None):
        self.matcher = matcher
        self.calibration = calibration
        self.calibration_done = False
        self.extractor = extractor
        self.transform = transform


    def get_feature_dataset(self, dataset: WildlifeDataset) -> FeatureDataset:
        if self.transform is not None:
            dataset.transform = self.transform
        if self.extractor is not None:
            return self.extractor(dataset)
        else:
            return dataset


    def fit_calibration(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset):
        '''
        Fit calibration using scores from given two datasets (eg. training and validation datasets)
        Both datasets must have labels.
        '''
        if self.calibration is None:
            raise ValueError('Calibration method is not assigned.')

        dataset0 = self.get_feature_dataset(dataset0)
        dataset1 = self.get_feature_dataset(dataset1)
        score = self.matcher(dataset0, dataset1)

        hits = get_hits(dataset0, dataset1)
        self.calibration.fit(score.flatten(), hits.flatten())
        self.calibration_done = True


    def __call__(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset, pairs = None):
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
