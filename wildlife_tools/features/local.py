import kornia.feature as KF
import torch

from ..data import FeatureCacheMixin
from .base import FeatureExtractor
from .local_utils import laf_features, pad_features, to_grayscale


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
        batch_size: int = 1,
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
            batch_size (int, optional): Number of images processed at once. Images in a batch must have the same size.
        """

        super().__init__(
            batch_size=batch_size,
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

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
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
        images, _ = batch
        images = images.to(self.device)
        image_size = torch.tensor([images.shape[3], images.shape[2]])
        with torch.inference_mode():
            batch_features = [{k: v.cpu() for k, v in f.items()} for f in self.extract(images)]

        outputs = []
        for features in batch_features:
            if self.force_num_keypoints:
                features = pad_features(features, self.max_num_keypoints, image_size)
            features["image_size"] = image_size
            outputs.append(features)
        return outputs


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
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
        window_size: int = 5,
        checkpoint: str = "depth",
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            batch_size=batch_size,
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

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        batch_features = self.model(
            images,
            n=self.max_num_keypoints,
            window_size=self.window_size,
            score_threshold=self.detection_threshold,
            pad_if_not_divisible=True,
        )
        return [
            {
                "keypoints": f.keypoints,
                "keypoint_scores": f.detection_scores,
                "descriptors": f.descriptors,
            }
            for f in batch_features
        ]


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
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
        model_name: str = "aliked-n16",
        nms_radius: int = 2,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            batch_size=batch_size,
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

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "keypoints": f.keypoints,
                "keypoint_scores": f.keypoint_scores,
                "descriptors": f.descriptors,
            }
            for f in self.model(images)
        ]


class XFeatExtractor(LocalFeatureExtractor):
    """
    XFeat keypoints and descriptors. Match with `MatchLightGlue(features="xfeat")`.

    - Paper: XFeat: Accelerated Features for Lightweight Image Matching
    - Link: https://arxiv.org/abs/2404.19174
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
            batch_size=batch_size,
        )

    def build_model(self) -> torch.nn.Module:
        return KF.XFeat.from_pretrained(top_k=self.max_num_keypoints, detection_threshold=self.detection_threshold)

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "keypoints": f["keypoints"],
                "keypoint_scores": f["scores"],
                "descriptors": f["descriptors"],
            }
            for f in self.model.detectAndCompute(images)
        ]


class DeDoDeExtractor(LocalFeatureExtractor):
    """
    DeDoDe keypoints and descriptors. Match with `MatchLightGlue(features="dedodeb")` for
    descriptor="B" and `MatchLightGlue(features="dedodeg")` for descriptor="G".
    The detection_threshold is not used.

    - Paper: DeDoDe: Detect, Don't Describe -- Describe, Don't Detect for Local Feature Matching
    - Link: https://arxiv.org/abs/2308.08479
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
        descriptor: str = "B",
        detector_weights: str = "L-C4-v2",
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
            batch_size=batch_size,
        )
        self.descriptor = descriptor
        self.detector_weights = detector_weights

    def build_model(self) -> torch.nn.Module:
        amp_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        return KF.DeDoDe.from_pretrained(
            detector_weights=self.detector_weights,
            descriptor_weights=f"{self.descriptor}-upright",
            amp_dtype=amp_dtype,
        )

    def model_config(self) -> dict:
        return super().model_config() | {"descriptor": self.descriptor, "detector_weights": self.detector_weights}

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        keypoints, scores, descriptors = self.model(images, n=self.max_num_keypoints)
        return [
            {
                "keypoints": keypoints[i],
                "keypoint_scores": scores[i],
                "descriptors": descriptors[i],
            }
            for i in range(len(keypoints))
        ]


class DoGHardNetExtractor(LocalFeatureExtractor):
    """
    Difference of Gaussians keypoints with HardNet descriptors. Match with
    `MatchLightGlue(features="doghardnet")`, or `MatchLightGlue(features="dog_affnet_hardnet")` for affnet=True.

    - Paper (HardNet): Working hard to know your neighbor's margins: Local descriptor learning loss
    - Link: https://arxiv.org/abs/1705.10872
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
        affnet: bool = False,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
            batch_size=batch_size,
        )
        self.affnet = affnet

    def build_model(self) -> torch.nn.Module:
        detector = KF.SIFTFeature(num_features=self.max_num_keypoints, score_threshold=self.detection_threshold).detector
        if self.affnet:
            detector.aff = KF.LAFAffNetShapeEstimator(pretrained=True)
        descriptor = KF.LAFDescriptor(KF.HardNet(pretrained=True), patch_size=32, grayscale_descriptor=True)
        return KF.LocalFeature(detector, descriptor)

    def model_config(self) -> dict:
        return super().model_config() | {"affnet": self.affnet}

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [laf_features(*self.model(to_grayscale(image[None])))[0] for image in images]


class KeyNetAffNetHardNetExtractor(LocalFeatureExtractor):
    """
    KeyNet keypoints with AffNet shapes and HardNet descriptors. Match with
    `MatchLightGlue(features="keynet_affnet_hardnet")`.

    - Paper (KeyNet): Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters
    - Link: https://arxiv.org/abs/1904.00889
    """

    def __init__(
        self,
        detection_threshold: float = 0.0,
        force_num_keypoints: bool = True,
        max_num_keypoints: int = 256,
        device: str | None = None,
        num_workers: int = 1,
        cache_path: str | None = None,
        config_tag: str | None = None,
        batch_size: int = 1,
    ):
        super().__init__(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            force_num_keypoints=force_num_keypoints,
            device=device,
            num_workers=num_workers,
            cache_path=cache_path,
            config_tag=config_tag,
            batch_size=batch_size,
        )

    def build_model(self) -> torch.nn.Module:
        return KF.KeyNetAffNetHardNet(num_features=self.max_num_keypoints, score_threshold=self.detection_threshold)

    def extract(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [laf_features(*self.model(to_grayscale(image[None])))[0] for image in images]
