"""Differentiable per-query and global average precision objectives.

The sigmoid ranks follow Eqs. (3), (8)--(10) of Psomas et al.,
"Global Average Precision for Representation Learning", arXiv:2610.09863.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _similarity_inputs(similarities, positive_mask, valid_mask):
    """Validate pair matrices, align masks, and promote low-precision scores."""
    if similarities.ndim != 2 or not similarities.is_floating_point():
        raise ValueError("similarities must be a floating-point matrix")
    if positive_mask.shape != similarities.shape:
        raise ValueError("positive_mask must have the same shape as similarities")
    if positive_mask.dtype != torch.bool and not torch.all((positive_mask == 0) | (positive_mask == 1)):
        raise ValueError("positive_mask must contain only 0 and 1")
    positives = positive_mask.to(device=similarities.device, dtype=torch.bool)
    if valid_mask is None:
        valid = torch.ones_like(positives)
    else:
        if valid_mask.shape != similarities.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean matrix with the same shape as similarities")
        valid = valid_mask.to(similarities.device)
    # Low temperatures require more precision than fp16/bfloat16 provide.
    if similarities.dtype in (torch.float16, torch.bfloat16):
        similarities = similarities.float()
    return similarities, positives, valid


def _embedding_inputs(embeddings, labels):
    """Build batch cosine similarities and masks that exclude self-pairs."""
    if embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be a floating-point matrix")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have one entry per embedding")
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        embeddings = embeddings.float()
    labels = labels.to(embeddings.device)
    positives = labels[:, None] == labels[None, :]
    valid = ~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
    context = (
        torch.autocast(device_type=embeddings.device.type, enabled=False)
        if embeddings.device.type in ("cpu", "cuda")
        else nullcontext()
    )
    with context:
        features = F.normalize(embeddings, dim=1)
        similarities = features @ features.T
    return similarities, positives, valid


def _precision(positive_chunk, positives, negatives, temperature):
    """Compute sigmoid precision at each positive score in a chunk."""
    # Include each positive's self-comparison as sigmoid(0) = 0.5,
    # exactly as the paper's sigmoid substitution in Eq. (3) specifies.
    positive_rank = torch.sigmoid((positives[None, :] - positive_chunk[:, None]) / temperature).sum(dim=1)
    negative_rank = torch.sigmoid((negatives[None, :] - positive_chunk[:, None]) / temperature).sum(dim=1)
    return positive_rank / (positive_rank + negative_rank)


def _smooth_ap_loss(positives, negatives, temperature, chunk_size):
    """Compute smooth AP loss with chunked comparisons and backward recomputation."""
    if positives.numel() == 0 or negatives.numel() == 0:
        return (positives.sum() + negatives.sum()) * 0
    total = positives.new_zeros(())
    for positive_chunk in positives.split(chunk_size):
        if torch.is_grad_enabled() and (positives.requires_grad or negatives.requires_grad):
            # Chunking alone retains every comparison matrix until backward.
            # Recompute each chunk to also bound saved activation memory.
            precision = checkpoint(_precision, positive_chunk, positives, negatives, temperature, use_reentrant=False)
        else:
            precision = _precision(positive_chunk, positives, negatives, temperature)
        total = total + precision.sum()
    return 1 - total / positives.numel()


class SmoothAPLoss(nn.Module):
    """Per-query sigmoid AP surrogate (mSAP).

    ``forward(embeddings, labels)`` uses cosine similarities and excludes
    self-pairs. ``from_similarity(S, Y, valid_mask=None)`` accepts rectangular
    similarity/relevance matrices, including cross-modal diagonal positives.
    Queries without valid positives are omitted from the mean. Empty batches
    return a differentiable zero. Each rank includes sigmoid(0) = 0.5.

    Args:
        temperature: Positive sigmoid temperature.
        chunk_size: Number of positive ranks computed per checkpointed chunk.
    """

    def __init__(self, temperature: float = 0.01, chunk_size: int = 64):
        """Set the sigmoid temperature and positive-comparison chunk size."""
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(self, embeddings, labels):
        """Compute the AP surrogate from labeled embeddings, excluding self-pairs."""
        return self.from_similarity(*_embedding_inputs(embeddings, labels))

    def from_similarity(self, similarities, positive_mask, valid_mask=None):
        """Optimize the mean AP of rows with at least one valid positive."""
        similarities, positives, valid = _similarity_inputs(similarities, positive_mask, valid_mask)
        terms = []
        for scores, row_positives, row_valid in zip(similarities, positives, valid):
            pos = scores[row_positives & row_valid]
            if pos.numel():
                neg = scores[~row_positives & row_valid]
                terms.append(_smooth_ap_loss(pos, neg, self.temperature, self.chunk_size))
        return torch.stack(terms).mean() if terms else similarities.sum() * 0


class GSAPLoss(SmoothAPLoss):
    """Global sigmoid AP surrogate (gSAP), arXiv:2610.09863.

    All valid directed pairs are ranked together, including negatives from
    queries without positives. Optional negative dropping removes the lowest
    negative scores and preserves all positives. Defaults follow the paper's
    distractor-query recipe; set ``negative_drop_rate=0`` for the full loss.

    Args:
        temperature: Positive sigmoid temperature, default 0.001.
        negative_drop_rate: Fraction of lowest negatives to drop, in [0, 1].
        chunk_size: Number of positive ranks computed per checkpointed chunk.
    """

    def __init__(self, temperature: float = 0.001, negative_drop_rate: float = 0.9, chunk_size: int = 64):
        """Configure global ranking temperature, negative dropping, and chunk size."""
        super().__init__(temperature=temperature, chunk_size=chunk_size)
        if not math.isfinite(negative_drop_rate) or not 0 <= negative_drop_rate <= 1:
            raise ValueError("negative_drop_rate must be in [0, 1]")
        self.negative_drop_rate = negative_drop_rate

    def from_similarity(self, similarities, positive_mask, valid_mask=None):
        """Optimize AP over the flattened set of valid pairs."""
        similarities, positives, valid = _similarity_inputs(similarities, positive_mask, valid_mask)
        pos = similarities[positives & valid]
        neg = similarities[~positives & valid]
        if self.negative_drop_rate and neg.numel():
            keep = neg.numel() - int(self.negative_drop_rate * neg.numel())
            neg = neg.topk(keep, sorted=False).values
        return _smooth_ap_loss(pos, neg, self.temperature, self.chunk_size)


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise objective with a shared learnable scale and bias.

    Uses cosine similarities and excludes self-pairs for embedding batches.
    ``from_similarity`` also supports rectangular/cross-modal matrices. The
    signed logistic loss is summed over valid pairs and divided by the number
    of queries, as in SigLIP; it is not averaged over all pairs. Negative-only
    batches still train. No-positive batches behave differently from AP losses.
    """

    def __init__(self, scale: float = 10.0, bias: float = -10.0, learnable: bool = True):
        """Initialize the logit scale and bias, optionally making them trainable."""
        super().__init__()
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(bias):
            raise ValueError("scale must be finite and positive, and bias must be finite")
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)), requires_grad=learnable)
        self.bias = nn.Parameter(torch.tensor(float(bias)), requires_grad=learnable)

    def forward(self, embeddings, labels):
        """Compute sigmoid pair loss from labeled embeddings, excluding self-pairs."""
        return self.from_similarity(*_embedding_inputs(embeddings, labels))

    def from_similarity(self, similarities, positive_mask, valid_mask=None):
        """Sum signed logistic losses over valid pairs and divide by query count."""
        similarities, positives, valid = _similarity_inputs(similarities, positive_mask, valid_mask)
        logits = similarities * self.log_scale.exp() + self.bias
        if not valid.any():
            return logits.sum() * 0
        signed_logits = torch.where(positives[valid], logits[valid], -logits[valid])
        return F.softplus(-signed_logits).sum() / similarities.shape[0]
