import kornia.feature as KF
import torch

from .base import MatchPairs


class MatchLightGlue(MatchPairs):
    """
    Implements matching using LightGlue model correspondences.
    Introduced in: "LightGlue: Local Feature Matching at Light Speed"
    """

    def __init__(
        self,
        features: str,
        init_threshold: float = 0.1,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            features (str): Features used for matching. Options: 'aliked', 'disk', 'xfeat', 'dedodeb',
                'dedodeg', 'doghardnet', 'dog_affnet_hardnet', 'keynet_affnet_hardnet'.
                Must match extracted features from the dataset.
            init_threshold (float, optional): Keep matches only over this threshold. Matches with
                lower values are not passed to the collector.
            device (str, optional): Device used for inference. Defaults to None.
        """
        super().__init__(**kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features = features
        self.init_threshold = init_threshold
        self.device = device
        self._model_factory = self.build_model

    def build_model(self) -> torch.nn.Module:
        model = KF.LightGlue(self.features)
        model.conf.depth_confidence = -1
        model.conf.width_confidence = -1
        model.conf.filter_threshold = self.init_threshold
        return model

    def cache_config(self) -> dict:
        return super().cache_config() | {"model": {"features": self.features, "filter_threshold": self.init_threshold}}

    def get_matches(self, batch):
        idx0, data0, idx1, data1 = batch
        keys = ["keypoints", "descriptors", "image_size", "scales", "oris", "lafs"]
        data = {
            "image0": {k: data0[k].to(self.device) for k in keys if k in data0},
            "image1": {k: data1[k].to(self.device) for k in keys if k in data1},
        }

        with torch.inference_mode():
            output = self.model(data)

        results = []
        for i, (i0, i1, scores, matches) in enumerate(zip(idx0, idx1, output["scores"], output["matches"])):
            matches = matches.cpu()
            kpts0 = data0["keypoints"][i][matches[:, 0]].cpu().numpy()
            kpts1 = data1["keypoints"][i][matches[:, 1]].cpu().numpy()
            scores = scores.cpu().numpy()
            results.append(
                {
                    "idx0": i0.item(),
                    "idx1": i1.item(),
                    "kpts0": kpts0,
                    "kpts1": kpts1,
                    "scores": scores,
                }
            )
        return results
