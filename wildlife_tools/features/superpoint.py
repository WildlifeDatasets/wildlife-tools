import torch
from tqdm import tqdm

from wildlife_tools.features.base import FeatureExtractor
from wildlife_tools.features.models.superpoint import SuperPoint


class SuperPointFeatures(FeatureExtractor):
    def __init__(
        self,
        descriptor_dim: int = 256,
        max_keypoints: int | None = None,
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005,
        remove_borders: int = 4,
        device: str = "cpu",
        num_workers: int = 1,
        batch_size: int = 128,
    ):
        self.descriptor_dim = descriptor_dim
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.remove_borders = remove_borders
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size

    def __call__(self, dataset):
        if not self.max_keypoints:
            max_keypoints = -1
        else:
            max_keypoints = self.max_keypoints

        model = SuperPoint(
            config={
                "descriptor_dim": self.descriptor_dim,
                "nms_radius": self.nms_radius,
                "keypoint_threshold": self.keypoint_threshold,
                "max_keypoints": max_keypoints,
                "remove_borders": self.remove_borders,
            }
        ).to(self.device)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

        descriptors = []
        for image, label in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = model({"image": image.to(self.device)})
            descriptors.extend(
                [d.permute(1, 0).cpu().numpy() for d in output["descriptors"]]
            )
        return descriptors
