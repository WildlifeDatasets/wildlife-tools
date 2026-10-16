from copy import deepcopy
from typing import Any

import kornia.feature as KF
import numpy as np
import torch
from kornia.feature.loftr.loftr import default_cfg

from .base import MatchPairs


def loftr_config(pretrained: str, thr: float) -> dict[str, Any]:
    config = deepcopy(default_cfg)
    config["match_coarse"]["thr"] = thr
    if pretrained == "indoor_new":
        config["coarse"]["temp_bug_fix"] = True
    return config


class SkipFinePreprocess(torch.nn.Module):
    def forward(self, feat_f0: torch.Tensor, feat_f1: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        return feat_f0.new_empty(0), feat_f1.new_empty(0)


class SkipFineMatching(torch.nn.Module):
    def forward(self, feat_f0: torch.Tensor, feat_f1: torch.Tensor, data: dict) -> None:
        data["mkpts0_f"] = data["mkpts0_c"]
        data["mkpts1_f"] = data["mkpts1_c"]


class MatchLOFTR(MatchPairs):
    """
    Implements matching pairs using LoFTR model correspondences.
    Introduced in: "LoFTR: Detector-Free Local Feature Matching with Transformers"

    """

    def __init__(
        self,
        pretrained: str = "outdoor",
        init_threshold: float = 0.2,
        device: str | None = None,
        apply_fine: bool = False,
        **kwargs,
    ):
        """
        Args:
            pretrained: LOFTR model used. `outdoor`, `indoor` or `indoor_new`.
            device: Specifies device used for the inference.
            init_threshold: Keep matches only over this threshold.
            apply_fine: Use LoFTR fine refinement of keypoints locations. Has no effect on
                confidence, but is faster without fine refinement. False by default.
        """

        super().__init__(**kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pretrained = pretrained
        self.init_threshold = init_threshold
        self.apply_fine = apply_fine
        self.device = device
        self._model_factory = self.build_model

    def build_model(self) -> torch.nn.Module:
        model = KF.LoFTR(pretrained=self.pretrained, config=loftr_config(self.pretrained, self.init_threshold))
        if not self.apply_fine:
            model.fine_preprocess = SkipFinePreprocess()
            model.fine_matching = SkipFineMatching()
        return model

    def cache_config(self) -> dict:
        return super().cache_config() | {
            "model": loftr_config(self.pretrained, self.init_threshold),
            "pretrained": self.pretrained,
            "apply_fine": self.apply_fine,
        }

    def get_matches(self, batch):
        idx0, data0, idx1, data1 = batch
        data = {
            "image0": data0.to(self.device),
            "image1": data1.to(self.device),
        }
        with torch.inference_mode():
            output = self.model(data)

        batch_idx = output["batch_indexes"].cpu().numpy()
        confidence = output["confidence"].cpu().numpy()
        kpts0 = output["keypoints0"].cpu().numpy()
        kpts1 = output["keypoints1"].cpu().numpy()

        data = []
        for b, (i0, i1) in enumerate(zip(idx0, idx1)):
            (current,) = np.where(batch_idx == b)
            data.append(
                {
                    "idx0": i0.item(),
                    "idx1": i1.item(),
                    "kpts0": kpts0[current],
                    "kpts1": kpts1[current],
                    "scores": confidence[current],
                }
            )
        return data
