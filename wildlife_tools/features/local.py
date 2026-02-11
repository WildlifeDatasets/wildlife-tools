import types
from typing import Optional

import torch
from gluefactory.models import get_model
from omegaconf import OmegaConf

from ..data import FeatureCacheMixin
from .gluefactory_fix import extract_single_image_fix  # https://github.com/cvg/glue-factory/pull/50


class GlueFactoryExtractor(FeatureCacheMixin):
    """
    Base class for Gluefactory extractors.

    Common configuration of extractors:

        1. max_num_keypoints: Maximum number of keypoints to return.
        1. detection_threshold: Threshold for keypoints detection (use 0.0 if force_num_keypoints = True).
        1. force_num_keypoints: Force to return exactly max_num_keypoints keypoints.

    """

    def __init__(
        self,
        config: dict,
        device: Optional[str] = None,
        num_workers: int = 1,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            config (dict): Configuration dictionary for the model.
            device (str | None, optional): Select between cuda and cpu devices.
            num_workers (int, optional): Number of workers used for data loading.
            cache_path (str, optional): Path for cached results. No caching for None.
        """

        super().__init__(
            batch_size=1,
            num_workers=num_workers,
            device=device,
            cache_path=cache_path,
            )

        config = OmegaConf.create(config)
        self.model = get_model(config.name)(config)

    def cat_features_dictionary(self, feats: list[dict]) -> list[dict]:
        return feats

    def cat_features_model(self, feats: list[list[dict]]) -> list[dict]:
        return [x for sub in feats for x in sub]

    def forward_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[dict]:
        # Batch has always size 1
        image, _ = batch
        image = image.to(self.device)
        with torch.inference_mode():
            output = self.model({"image": image})
            output = {k: v.squeeze(0).cpu() for k, v in output.items()}
            output["image_size"] = torch.tensor(image.shape[2:])
        return [output]


class SuperPointExtractor(GlueFactoryExtractor):
    """
    Superpoint keypoints and descriptors.

    - Paper: SuperPoint: Self-Supervised Interest Point Detection and Description
    - Link: https://arxiv.org/abs/1712.07629
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        **model_config,
    ):
        config = {
            "name": "gluefactory_nonfree.superpoint",
            "nms_radius": 3,
            "detection_threshold": detection_threshold,
            "force_num_keypoints": force_num_keypoints,
            "max_num_keypoints": max_num_keypoints,
        } | model_config
        super().__init__(config, device=device, cache_path=cache_path)


class DiskExtractor(GlueFactoryExtractor):
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
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        **model_config,
    ):
        config = {
            "name": "extractors.disk_kornia",
            "detection_threshold": detection_threshold,
            "force_num_keypoints": force_num_keypoints,
            "max_num_keypoints": max_num_keypoints,
        } | model_config
        super().__init__(config, device=device, cache_path=cache_path)


class AlikedExtractor(GlueFactoryExtractor):
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
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        **model_config,
    ):

        config = {
            "name": "extractors.aliked",
            "detection_threshold": detection_threshold,
            "force_num_keypoints": force_num_keypoints,
            "max_num_keypoints": max_num_keypoints,
        } | model_config
        super().__init__(config, device=device, cache_path=cache_path)


class SiftExtractor(GlueFactoryExtractor):
    """SIFT keypoints and descriptors."""

    def __init__(
        self,
        backend: str = "opencv",
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
        **model_config,
    ):

        config = {
            "name": "extractors.sift",
            "backend": backend,
            "detection_threshold": detection_threshold,
            "force_num_keypoints": force_num_keypoints,
            "max_num_keypoints": max_num_keypoints,
        } | model_config
        super().__init__(config, cache_path=cache_path)

        # Fix extract_single_image method.
        self.model.extract_single_image = types.MethodType(extract_single_image_fix, self.model)
