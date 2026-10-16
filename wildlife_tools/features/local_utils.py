import math

import kornia.feature as KF
import torch
from kornia.color import rgb_to_grayscale


def pad_features(features: dict[str, torch.Tensor], num: int, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Pad local features of a single image to `num` keypoints.

    Missing keypoints get random locations inside the image, all other entries (descriptors,
    scores, ...) are padded with zeros. Features with at least `num` keypoints are returned unchanged.

    Args:
        features (dict[str, torch.Tensor]): Features of one image. Each entry has the number of keypoints
            as its first dimension.
        num (int): Required number of keypoints.
        image_size (torch.Tensor): Image size as (width, height).

    Returns:
        features (dict[str, torch.Tensor]): Features with exactly `num` keypoints.
    """
    missing = num - len(features["keypoints"])
    if missing <= 0:
        return features

    padded = {}
    for key, value in features.items():
        if key == "keypoints":
            pad = torch.rand(missing, 2, dtype=value.dtype) * (image_size.to(value) - 1)
        else:
            pad = value.new_zeros((missing, *value.shape[1:]))
        padded[key] = torch.cat([value, pad])
    return padded


def to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """Convert a batch of RGB images (B, 3, H, W) to grayscale. Other images are returned unchanged."""
    return rgb_to_grayscale(images) if images.shape[1] == 3 else images


def laf_features(
    lafs: torch.Tensor, responses: torch.Tensor, descriptors: torch.Tensor
) -> list[dict[str, torch.Tensor]]:
    """
    Convert the output of Kornia LAF-based extractors to a list of feature dictionaries, one per image.

    Keypoints are the centers of the local affine frames (LAFs). Scales and orientations (in radians,
    in [0, 2*pi)) are computed from the frames, as needed by LightGlue.

    Args:
        lafs (torch.Tensor): Local affine frames of shape (B, N, 2, 3).
        responses (torch.Tensor): Detection scores of shape (B, N).
        descriptors (torch.Tensor): Descriptors of shape (B, N, D).

    Returns:
        features (list[dict[str, torch.Tensor]]): For each image, a dictionary with keypoints,
            keypoint_scores, descriptors, scales, oris and lafs.
    """
    keypoints = KF.get_laf_center(lafs)
    scales = KF.get_laf_scale(lafs).flatten(1)
    oris = torch.deg2rad(KF.get_laf_orientation(lafs).flatten(1)) % (2 * math.pi)
    return [
        {
            "keypoints": keypoints[i],
            "keypoint_scores": responses[i],
            "descriptors": descriptors[i],
            "scales": scales[i],
            "oris": oris[i],
            "lafs": lafs[i],
        }
        for i in range(len(lafs))
    ]
