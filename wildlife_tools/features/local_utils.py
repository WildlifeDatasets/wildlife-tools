import math

import kornia.feature as KF
import torch
from kornia.color import rgb_to_grayscale


def pad_features(features: dict[str, torch.Tensor], num: int, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
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
    return rgb_to_grayscale(images) if images.shape[1] == 3 else images


def laf_features(lafs: torch.Tensor, responses: torch.Tensor, descriptors: torch.Tensor) -> list[dict[str, torch.Tensor]]:
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
