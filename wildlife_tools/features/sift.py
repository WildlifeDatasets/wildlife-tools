import cv2
import numpy as np
from tqdm import tqdm

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features.base import FeatureExtractor


class SIFTFeatures(FeatureExtractor):
    """
    Extracts SIFT descriptors for each image in the dataset.

    Args:
        max_keypoints: Limit number of extracted keypoints / descriptors.

    Returns:
        list of arrays, each corresponds to an input image and have shape `[n_descriptors x 128]`.
    """

    descriptor_dim: int = 128

    def __init__(self, max_keypoints: int | None = None):
        self.max_keypoints = max_keypoints

    def __call__(self, dataset: WildlifeDataset):
        if self.max_keypoints:
            sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        else:
            sift = cv2.SIFT_create()

        descriptors = []
        for img, y in tqdm(dataset, mininterval=1, ncols=100):
            keypoint, d = sift.detectAndCompute(np.array(img), None)
            if len(keypoint) <= 1:
                descriptors.append(None)
            else:
                descriptors.append(d)
        return descriptors
