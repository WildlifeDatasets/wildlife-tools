import itertools
import kornia.feature as KF
from kornia.feature.loftr.loftr import *
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .base import MatchPairs


# Modified Kornia LoFTR module that enables skiping finegrained refinement
class LoFTR(Module):
    r"""Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        config: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor'
                    on the ScanNet.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> loftr = LoFTR('outdoor')
        >>> out = loftr(input)
    """

    def __init__(
        self,
        pretrained: str = "outdoor",
        config: dict[str, Any] = default_cfg,
        apply_fine=True,
        thr: float = 0.2
        ) -> None:

        super().__init__()
        config = deepcopy(config)
        config['match_coarse']['thr'] = thr

        self.apply_fine = apply_fine
        # Misc
        self.config = config
        if pretrained == "indoor_new":
            self.config["coarse"]["temp_bug_fix"] = True
        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], temp_bug_fix=config["coarse"]["temp_bug_fix"]
        )
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.pretrained = pretrained
        if pretrained is not None:
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")

            pretrained_dict = torch.hub.load_state_dict_from_url(urls[pretrained], map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict["state_dict"])
        self.eval()

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            data: dictionary containing the input data in the following format:

        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        # 1. Local Feature CNN
        _data: dict[str, Tensor | int | torch.Size] = {
            "bs": data["image0"].size(0),
            "hw0_i": data["image0"].shape[2:],
            "hw1_i": data["image1"].shape[2:],
        }

        if _data["hw0_i"] == _data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data["image0"], data["image1"]], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(_data["bs"]), feats_f.split(_data["bs"])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data["image0"]), self.backbone(data["image1"])

        _data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = self.pos_encoding(feat_c0).permute(0, 2, 3, 1)
        n, h, w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        n1, h1, w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in _data:
            mask_c0 = resize(data["mask0"], _data["hw0_c"], interpolation="nearest").flatten(-2)
        if "mask1" in _data:
            mask_c1 = resize(data["mask1"], _data["hw1_c"], interpolation="nearest").flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, _data, mask_c0=mask_c0, mask_c1=mask_c1)

        # Make fine-level optional
        if self.apply_fine:
            # 4. fine-level refinement
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, _data)
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

            #5. match fine-level
            self.fine_matching(feat_f0_unfold, feat_f1_unfold, _data)
        if self.apply_fine:
            rename_keys: dict[str, str] = {
                "mkpts0_f": "keypoints0",
                "mkpts1_f": "keypoints1",
                "mconf": "confidence",
                "b_ids": "batch_indexes",
            }
        else:
            rename_keys: dict[str, str] = {
                "mkpts0_c": "keypoints0",
                "mkpts1_c": "keypoints1",
                "mconf": "confidence",
                "b_ids": "batch_indexes",
            }
        out: dict[str, Tensor] = {}
        for k, v in rename_keys.items():
            _d = _data[k]
            if isinstance(_d, Tensor):
                out[v] = _d
            else:
                raise TypeError(f"Expected Tensor for item `{k}`. Gotcha {type(_d)}")
        return out


class MatchLOFTR(MatchPairs):
    """
    Calculate similarity between query and database based on LoFTR correspondences.

    Args:
        pretrained: LOFTR model used. `outdoor` or `indoor`.
        device: Specifies device used for the inference.
        init_threshold: Keep matches only over this threshold.
        apply_fine: Use LoFTR fine refinement of keypoints locations. Has no effect on confidence.

    Returns:
        dict: Values are 2D array with number of correspondences for each threshold.
    """

    def __init__(
        self,
        pretrained: str = "outdoor",
        init_threshold: float = 0.2,
        device: str | None = None,
        apply_fine: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = LoFTR(pretrained=pretrained, apply_fine=apply_fine, thr=init_threshold).to(device)
        self.device = device


    def get_matches(self, batch):
        ''' Process single batch of LoFTR matches'''

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
            current, = np.where(batch_idx == b)
            data.append({
                'idx0': i0.item(),
                'idx1': i1.item(),
                'kpts0': kpts0[current],
                'kpts1': kpts1[current],
                'scores': confidence[current],
            })
        return data