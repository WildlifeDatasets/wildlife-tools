import math

import kornia.feature as KF
import torch
from kornia.color import rgb_to_grayscale

from ..data import FeatureCacheMixin
from .base import FeatureExtractor


def pad_features(features: dict[str, torch.Tensor], num: int, image_size: torch.Tensor) -> dict[str, torch.Tensor]:
    missing = num - len(features["keypoints"])
    if missing <= 0:
        return features

    padded = {}
    for key, value in features.items():
        if key == "keypoints":
            pad = torch.rand(missing, 2, dtype=value.dtype) * (image_size.flip(0).to(value) - 1)
        else:
            pad = value.new_zeros((missing, *value.shape[1:]))
        padded[key] = torch.cat([value, pad])
    return padded


class LocalFeatureExtractor(FeatureCacheMixin, FeatureExtractor):
    """
    Base class for local feature extractors.

    Common configuration of extractors:

        1. max_num_keypoints: Maximum number of keypoints to return.
        1. detection_threshold: Threshold for keypoints detection (use 0.0 if force_num_keypoints = True).
        1. force_num_keypoints: Force to return exactly max_num_keypoints keypoints. Missing keypoints
            are padded with random locations and zero descriptors.

    Each extracted feature is a dictionary with keypoints, keypoint_scores, descriptors and image_size.
    """

    def __init__(
        self,
        max_num_keypoints: int = 256,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        """
        Args:
            max_num_keypoints (int, optional): Maximum number of keypoints to return.
            detection_threshold (float, optional): Threshold for keypoints detection.
            force_num_keypoints (bool, optional): Force to return exactly max_num_keypoints keypoints.
            device (str | None, optional): Select between cuda and cpu devices.
            num_workers (int, optional): Number of workers used for data loading.
            cache_path (str, optional): Path for cached results. No caching for None.
            config_tag (str, optional): Free-form tag stored in the cache config. Reusing cache_path with a
                different tag raises an error. Changes of the image transform are not
                detected automatically, so encode them in the tag (e.g. "resize224").
        """

        super().__init__(
            batch_size=1,
            num_workers=num_workers,
            device=device,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.force_num_keypoints = force_num_keypoints
        self._model_factory = self.build_model

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def model_config(self) -> dict:
        return {
            "max_num_keypoints": self.max_num_keypoints,
            "detection_threshold": self.detection_threshold,
            "force_num_keypoints": self.force_num_keypoints,
        }

    def cache_config(self) -> dict:
        return super().cache_config() | {"model": self.model_config()}

    def cat_features_dictionary(self, feats: list[dict]) -> list[dict]:
        return feats

    def cat_features_model(self, feats: list[list[dict]]) -> list[dict]:
        return [x for sub in feats for x in sub]

    def forward_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[dict]:
        # Batch has always size 1
        image, _ = batch
        image = image.to(self.device)
        image_size = torch.tensor(image.shape[2:])
        with torch.inference_mode():
            features = {k: v.cpu() for k, v in self.extract(image).items()}
        if self.force_num_keypoints:
            features = pad_features(features, self.max_num_keypoints, image_size)
        features["image_size"] = image_size
        return [features]


class DiskExtractor(LocalFeatureExtractor):
    """
    DISK keypoints and descriptors.

    - Paper: DISK: learning local features with policy gradient
    - Link: https://arxiv.org/abs/2006.13566
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        window_size: int = 5,
        checkpoint: str = "depth",
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        self.window_size = window_size
        self.checkpoint = checkpoint

    def build_model(self) -> torch.nn.Module:
        return KF.DISK.from_pretrained(self.checkpoint)

    def model_config(self) -> dict:
        return super().model_config() | {"window_size": self.window_size, "checkpoint": self.checkpoint}

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model(
            image,
            n=self.max_num_keypoints,
            window_size=self.window_size,
            score_threshold=self.detection_threshold,
            pad_if_not_divisible=True,
        )[0]
        return {
            "keypoints": features.keypoints,
            "keypoint_scores": features.detection_scores,
            "descriptors": features.descriptors,
        }


class AlikedExtractor(LocalFeatureExtractor):
    """
    ALIKED keypoints and descriptors.

    - Paper: ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation
    - Link: https://arxiv.org/abs/2304.03608
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        model_name: str = "aliked-n16",
        nms_radius: int = 2,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        self.model_name = model_name
        self.nms_radius = nms_radius

    def build_model(self) -> torch.nn.Module:
        return KF.ALIKED.from_pretrained(
            self.model_name,
            max_num_keypoints=self.max_num_keypoints,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius,
        )

    def model_config(self) -> dict:
        return super().model_config() | {"model_name": self.model_name, "nms_radius": self.nms_radius}

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model(image)[0]
        return {
            "keypoints": features.keypoints,
            "keypoint_scores": features.keypoint_scores,
            "descriptors": features.descriptors,
        }


class SiftExtractor(LocalFeatureExtractor):
    """SIFT keypoints and descriptors."""

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        upright: bool = False,
        rootsift: bool = True,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        self.upright = upright
        self.rootsift = rootsift

    def build_model(self) -> torch.nn.Module:
        return KF.SIFTFeature(
            num_features=self.max_num_keypoints,
            upright=self.upright,
            rootsift=self.rootsift,
            score_threshold=self.detection_threshold,
        )

    def model_config(self) -> dict:
        return super().model_config() | {"upright": self.upright, "rootsift": self.rootsift}

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        lafs, responses, descriptors = self.model(image)
        return {
            "keypoints": KF.get_laf_center(lafs)[0],
            "keypoint_scores": responses[0],
            "descriptors": descriptors[0],
            "scales": KF.get_laf_scale(lafs)[0].reshape(-1),
            "oris": torch.deg2rad(KF.get_laf_orientation(lafs)[0].reshape(-1)) % (2 * math.pi),
        }
