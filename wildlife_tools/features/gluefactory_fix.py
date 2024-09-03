'''
Fix SIFT extractor bugs in Gluefactory.
- Fixes forcing number of keypoints - Pending PR https://github.com/cvg/glue-factory/pull/50 
- Fixes situation with no keypoint and descriptors for OpenCV SIFT.
'''

import numpy as np
import torch
from packaging import version

try:
    import pycolmap
except ImportError:
    pycolmap = None

from gluefactory.models.utils.misc import pad_to_length
from gluefactory.models.extractors.sift import filter_dog_point, run_opencv_sift


def extract_single_image_fix(self, image: torch.Tensor):
    image_np = image.cpu().numpy().squeeze(0)
    if self.conf.backend.startswith("pycolmap"):
        if version.parse(pycolmap.__version__) >= version.parse("0.5.0"):
            detections, descriptors = self.sift.extract(image_np)
            scores = None  # Scores are not exposed by COLMAP anymore.
        else:
            detections, scores, descriptors = self.sift.extract(image_np)
        keypoints = detections[:, :2]  # Keep only (x, y).
        scales, angles = detections[:, -2:].T
        if scores is not None and (
            self.conf.backend == "pycolmap_cpu" or not pycolmap.has_cuda
        ):
            # Set the scores as a combination of abs. response and scale.
            scores = np.abs(scores) * scales
    elif self.conf.backend == "opencv":
        # TODO: Check if opencv keypoints are already in corner convention
        keypoints, scores, scales, angles, descriptors = run_opencv_sift(
            self.sift, (image_np * 255.0).astype(np.uint8)
        )

    if (descriptors is None) or (descriptors.size == 0):
        descriptors = np.empty(shape=(0, 128), dtype=np.float32)

    if (keypoints is None) or (keypoints.size == 0):
        keypoints = np.empty(shape=(0, 2), dtype=np.float32)

    pred = {
        "keypoints": keypoints,
        "scales": scales,
        "oris": angles,
        "descriptors": descriptors,
    }
    if scores is not None:
        pred["keypoint_scores"] = scores

    # sometimes pycolmap returns points outside the image. We remove them
    if self.conf.backend.startswith("pycolmap"):
        is_inside = (
            pred["keypoints"] + 0.5 < np.array([image_np.shape[-2:][::-1]])
        ).all(-1)
        pred = {k: v[is_inside] for k, v in pred.items()}

    if self.conf.nms_radius is not None:
        keep = filter_dog_point(
            pred["keypoints"],
            pred["scales"],
            pred["oris"],
            image_np.shape,
            self.conf.nms_radius,
            pred["keypoint_scores"],
        )
        pred = {k: v[keep] for k, v in pred.items()}

    pred = {k: torch.from_numpy(v) for k, v in pred.items()}
    if scores is not None:
        # Keep the k keypoints with highest score
        num_points = self.conf.max_num_keypoints
        if num_points is not None and len(pred["keypoints"]) > num_points:
            indices = torch.topk(pred["keypoint_scores"], num_points).indices
            pred = {k: v[indices] for k, v in pred.items()}

    if self.conf.force_num_keypoints:
        num_points = max(self.conf.max_num_keypoints, len(pred["keypoints"]))
        pred["keypoints"] = pad_to_length(
            pred["keypoints"],
            num_points,
            -2,
            mode="random_c",
            bounds=(0, min(image.shape[1:])),
        )
        pred["scales"] = pad_to_length(pred["scales"], num_points, -1, mode="zeros")
        pred["oris"] = pad_to_length(pred["oris"], num_points, -1, mode="zeros")
        pred["descriptors"] = pad_to_length(
            pred["descriptors"], num_points, -2, mode="zeros"
        )
        if pred["keypoint_scores"] is not None:
            pred["keypoint_scores"] = pad_to_length(
                pred["keypoint_scores"], num_points, -1, mode="zeros"
            )
    return pred