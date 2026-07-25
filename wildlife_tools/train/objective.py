import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners


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


class PerInstanceTemperatureScalingLoss(nn.Module):
    """
    Cross-entropy loss with a learned per-sample temperature regularized by class frequency.

    This loss is adapted from the PITS method introduced in
    "Animal Identification with Independent Foreground and Background Modeling"
    by Picek, Neumann, and Matas.

    The model is expected to return a tuple `(logits, temperature)` where:
    - `logits` has shape `(batch_size, num_classes)`
    - `temperature` has shape `(batch_size,)` or `(batch_size, 1)`
    """

    def __init__(self, label_counts: np.ndarray | None = None, lambda_weight: float = 0.1):
        super().__init__()
        self.label_counts = label_counts
        self.lambda_weight = lambda_weight
        self.softplus = nn.Softplus()
        self._target_temperature: torch.Tensor | None = None

    def set_label_counts(self, label_counts: np.ndarray) -> None:
        """Set class counts used to derive target temperatures and reset the cached targets."""
        self.label_counts = label_counts
        self._target_temperature = None

    def _get_target_temperature(self, device: torch.device) -> torch.Tensor:
        """Compute or retrieve the cached per-class target temperatures on the requested device."""
        if self.label_counts is None:
            raise ValueError("`label_counts` must be provided for PerInstanceTemperatureScalingLoss.")

        if self._target_temperature is None:
            counts = torch.as_tensor(self.label_counts, dtype=torch.float32)
            max_count = counts.max()
            frac = counts / max_count
            self._target_temperature = 1 - torch.log(frac)

        return self._target_temperature.to(device)

    def forward(self, output: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Compute the PITS objective for a batch.

        Args:
            output: Tuple `(logits, temperature)` produced by the model.
            y: Integer class labels for the batch.

        Returns:
            Scalar loss combining calibrated cross-entropy and temperature regularization.
        """
        logits, temperature = output
        target_temperature = self._get_target_temperature(logits.device)[y]

        temperature = self.softplus(temperature.squeeze()) + 1.0
        logits = self.softplus(logits)
        calibrated_logits = logits / temperature[:, None]

        ce_loss = nn.functional.cross_entropy(calibrated_logits, y)
        temperature_loss = nn.functional.mse_loss(temperature, target_temperature)
        return ce_loss + self.lambda_weight * temperature_loss
