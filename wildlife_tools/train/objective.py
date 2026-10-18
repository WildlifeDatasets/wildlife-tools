import math

import torch
import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.losses import (
    AngularLoss,
    CircleLoss,
    ContrastiveLoss,
    CosFaceLoss,
    DynamicSoftMarginLoss,
    FastAPLoss,
    GeneralizedLiftedStructureLoss,
    HistogramLoss,
    LiftedStructureLoss,
    MarginLoss,
    MultiSimilarityLoss,
    NCALoss,
    NormalizedSoftmaxLoss,
    NPairsLoss,
    NTXentLoss,
    PNPLoss,
    ProxyAnchorLoss,
    ProxyNCALoss,
    RankedListLoss,
    SoftTripleLoss,
    SphereFaceLoss,
    SupConLoss,
    ThresholdConsistentMarginLoss,
    TupletMarginLoss,
)

from .ranking import GSAPLoss, SigLIPLoss, SmoothAPLoss

__all__ = [
    "AngularLoss",
    "ArcFaceLoss",
    "CircleLoss",
    "CombinedLoss",
    "ContrastiveLoss",
    "CosFaceLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "GSAPLoss",
    "GeneralizedLiftedStructureLoss",
    "HistogramLoss",
    "LiftedStructureLoss",
    "MarginLoss",
    "MinedLoss",
    "MultiSimilarityLoss",
    "NCALoss",
    "NPairsLoss",
    "NormalizedSoftmaxLoss",
    "NTXentLoss",
    "PNPLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "RankedListLoss",
    "SigLIPLoss",
    "SmoothAPLoss",
    "SoftTripleLoss",
    "SoftmaxLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletLoss",
    "TupletMarginLoss",
]


class ArcFaceLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning ArcFaceLoss.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
        margin (int, optional): Margin for ArcFace loss (in radians).
        scale (int, optional): Scale parameter for ArcFace loss.
    """

    def __init__(self, num_classes: int, embedding_size: int, margin: int = 0.5, scale: int = 64):

        super().__init__()
        self.loss = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=57.3 * margin,
            scale=scale,
        )

    def forward(self, embeddings, y):
        return self.loss(embeddings, y)


class SubCenterArcFaceLoss(nn.Module):
    """ArcFace with multiple centers per class; margin is in radians.

    Args:
        num_classes: Number of training classes.
        embedding_size: Embedding dimension.
        margin: Angular margin in radians, matching the ArcFace wrapper.
        scale: Logit scale.
        sub_centers: Number of centers per class.
    """

    def __init__(
        self, num_classes: int, embedding_size: int, margin: float = 0.5, scale: float = 64, sub_centers: int = 3
    ):
        super().__init__()
        self.loss = losses.SubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=math.degrees(margin),
            scale=scale,
            sub_centers=sub_centers,
        )

    def forward(self, embeddings, y):
        return self.loss(embeddings, y)


class TripletLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning TripletMarginLoss.

    Args:
        margin (int, optional): Margin for triplet loss.
        mining (str, optional): Type of triplet mining. One of: 'all', 'hard', 'semihard'
        distance (str, optional): Distance metric for triplet loss. One of: 'cosine', 'l2', 'l2_squared'

    """

    def __init__(self, margin: int = 0.2, mining: str = "semihard", distance: str = "l2_squared"):

        super().__init__()
        if distance == "cosine":
            distance = distances.CosineSimilarity()
        elif distance == "l2":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        elif distance == "l2_squared":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
        else:
            raise ValueError(f"Invalid distance: {distance}")

        self.loss = losses.TripletMarginLoss(distance=distance, margin=margin)
        self.miner = miners.TripletMarginMiner(distance=distance, type_of_triplets=mining, margin=margin)

    def forward(self, embeddings, y):
        indices_tuple = self.miner(embeddings, y)
        return self.loss(embeddings, y, indices_tuple)


class SoftmaxLoss(nn.Module):
    """
    CE with single dense layer classification head.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
    """

    def __init__(self, num_classes: int, embedding_size: int):

        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x, y):
        return self.criterion(self.linear(x), y)


class MinedLoss(nn.Module):
    """Apply a PyTorch Metric Learning miner before a compatible backend loss.

    Both modules are registered so proxies and any trainable loss parameters
    remain available to the optimizer and checkpoint state.
    """

    def __init__(self, loss: nn.Module, miner: nn.Module):
        super().__init__()
        self.loss = loss
        self.miner = miner

    def forward(self, embeddings, labels):
        return self.loss(embeddings, labels, self.miner(embeddings, labels))


class CombinedLoss(nn.Module):
    """Weighted sum of named ``objective(embeddings, labels)`` modules.

    Weights are stored in checkpoint state. Include this module's parameters
    in the optimizer to train constituent classifiers, proxies, or scales.
    """

    def __init__(self, objectives: dict[str, nn.Module], weights: dict[str, float] | None = None):
        super().__init__()
        if not objectives:
            raise ValueError("objectives must not be empty")
        if weights is None:
            weights = dict.fromkeys(objectives, 1.0)
        if weights.keys() != objectives.keys():
            raise ValueError("weights must have exactly the objective names")
        values = [float(weights[name]) for name in objectives]
        if any(not math.isfinite(value) or value < 0 for value in values) or not any(values):
            raise ValueError("weights must be finite, nonnegative, and not all zero")
        self.objectives = nn.ModuleDict(objectives)
        self.register_buffer("weights", torch.tensor(values))

    def forward(self, embeddings, labels):
        return sum(
            weight * objective(embeddings, labels) for weight, objective in zip(self.weights, self.objectives.values())
        )
