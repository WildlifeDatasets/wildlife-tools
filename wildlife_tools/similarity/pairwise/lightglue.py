import cv2
import torch
import numpy as np
from tqdm import tqdm
from gluefactory.models import get_model
from omegaconf import OmegaConf
from collections import defaultdict
import torchvision.transforms as T
import itertools
import torch
import numpy as np
from .base import MatchPairs


class MatchLightGlue(MatchPairs):
    
    def __init__(
        self,
        features : str,
        init_threshold: float = 0.1,
        device: str | None = None,
        **kwargs,
        ):
        super().__init__(**kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup config with deaults
        config =  OmegaConf.create({
                    "name": "matchers.lightglue_pretrained",
                    "features": features,
                    "depth_confidence": -1,
                    "width_confidence": -1,
                    "filter_threshold": init_threshold,
                })

        self.model = get_model(config.name)(config)
        self.device = device


    def get_matches(self, batch):
        idx0, data0, idx1, data1 = batch
        data = {
            'keypoints0': data0['keypoints'].to(self.device),
            'descriptors0': data0['descriptors'].to(self.device),
            'view0': {'image_size': data0['image_size'].to(self.device)},

            'keypoints1': data1['keypoints'].to(self.device),
            'descriptors1': data1['descriptors'].to(self.device),
            'view1': {'image_size': data1['image_size'].to(self.device)},
        }

        if ('scales' in data1) and ('oris' in data1):
            data["scales1"] = data1['scales'].to(self.device)
            data["oris1"] = data1['oris'].to(self.device)

        if ('scales' in data0) and ('oris' in data0):
            data["scales0"] = data0['scales'].to(self.device)
            data["oris0"] = data0['oris'].to(self.device)


        with torch.inference_mode():
            output = self.model(data)

        data = []
        for i, (i0, i1, scores, matches) in enumerate(zip(idx0, idx1, output['scores'], output['matches'])):
            matches = matches.cpu()
            kpts0 = data0['keypoints'][i][matches[:, 0]].cpu().numpy()
            kpts1 = data1['keypoints'][i][matches[:, 1]].cpu().numpy()
            scores = scores.cpu().numpy()
            data.append({
                'idx0': i0.item(),
                'idx1': i1.item(),
                'kpts0': kpts0,
                'kpts1': kpts1,
                'scores': scores,
            })
        return data

    
