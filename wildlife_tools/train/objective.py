import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners


class ArcFaceLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning ArcFaceLoss.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
        margin (int): Margin for ArcFace loss (in radians).
        scale (int): Scale parameter for ArcFace loss.
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


class TripletLoss(nn.Module):
    """
    Wraps Pytorch Metric Learning TripletMarginLoss.

    Args:
        margin (int): Margin for triplet loss.
        mining (str): Type of triplet mining. One of: 'all', 'hard', 'semihard'
        distance (str): Distance metric for triplet loss. One of: 'cosine', 'l2', 'l2_squared'

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
