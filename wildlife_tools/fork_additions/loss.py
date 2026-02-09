import torch.nn.functional as F
import torch.nn as nn
from wildlife_tools.train import ArcFaceLoss


class ArcFaceWithCrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.arcface_loss = ArcFaceLoss(*args, **kwargs)
        self.cross_entropy_loss = F.cross_entropy

    def forward(self, x, y):
        features, logits = x
        return self.arcface_loss(features, y) + 0.1 * self.cross_entropy_loss(logits, y)
