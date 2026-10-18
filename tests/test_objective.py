"""Offline objective tests: paper equations, gradients, and trainer integration."""

from itertools import chain

import pytest
import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners
from torch import nn
from torch.utils.data import TensorDataset

from wildlife_tools import train


def test_backend_loss_accepts_labeled_embeddings():
    """Check that an exported backend loss returns a scalar with finite embedding gradients."""
    torch.manual_seed(17)
    embeddings = torch.randn(10, 6, requires_grad=True)
    labels = torch.arange(5).repeat_interleave(2)
    loss = train.SupConLoss()(embeddings, labels)
    assert loss.shape == () and torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def _paper_ap_loss(scores, relevance, temperature):
    """Literal scalar implementation of Eqs. (3), (8), and (10)."""
    positive_indices = torch.where(relevance)[0].tolist()
    if not positive_indices:
        return scores.sum() * 0
    precisions = []
    for i in positive_indices:
        rank = sum(torch.sigmoid((scores[j] - scores[i]) / temperature) for j in range(len(scores)))
        positive_rank = sum(torch.sigmoid((scores[j] - scores[i]) / temperature) for j in positive_indices)
        precisions.append(positive_rank / rank)
    return 1 - torch.stack(precisions).mean()


@pytest.mark.parametrize("global_ranking", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 64])
def test_ap_values_and_gradients_match_paper(global_ranking, chunk_size):
    """Match AP values and gradients to the paper equations with single and multiple chunks."""
    scores = torch.tensor([[0.82, 0.41, -0.2], [0.6, 0.52, 0.1]], dtype=torch.double, requires_grad=True)
    relevance = torch.tensor([[True, False, True], [False, True, False]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    temperature = 0.07
    if global_ranking:
        objective = train.GSAPLoss(temperature, negative_drop_rate=0, chunk_size=chunk_size)
        expected = _paper_ap_loss(scores[valid], relevance[valid], temperature)
    else:
        objective = train.SmoothAPLoss(temperature, chunk_size=chunk_size)
        expected = torch.stack(
            [_paper_ap_loss(row[mask], pos[mask], temperature) for row, pos, mask in zip(scores, relevance, valid)]
        ).mean()
    actual = objective.from_similarity(scores, relevance, valid)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[0, 2] == 0  # Invalid pairs cannot affect training.


def test_global_ap_gradcheck():
    """Validate checkpointed gSAP derivatives against finite differences."""
    scores = torch.tensor([[0.7, 0.2, 0.1], [-0.1, 0.4, 0.3]], dtype=torch.double, requires_grad=True)
    relevance = torch.tensor([[True, False, False], [False, True, True]])
    objective = train.GSAPLoss(temperature=0.2, negative_drop_rate=0, chunk_size=1)
    assert torch.autograd.gradcheck(lambda x: objective.from_similarity(x, relevance), (scores,))


def test_global_ap_negative_dropping_keeps_all_positives_and_hardest_negatives():
    """Ensure pruning removes easy negatives while preserving every positive and its gradient."""
    scores = torch.tensor([[0.7, 0.9, 0.1], [0.5, 0.3, -0.2]], requires_grad=True)
    relevance = torch.tensor([[True, False, False], [True, False, False]])
    # Drop the lowest two of four negatives; both positives must remain.
    expected = _paper_ap_loss(scores.flatten()[[0, 3, 1, 4]], torch.tensor([True, True, False, False]), 0.1)
    actual = train.GSAPLoss(0.1, negative_drop_rate=0.5).from_similarity(scores, relevance)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad[0, 2] == scores.grad[1, 2] == 0
    assert scores.grad[0, 0] != 0 and scores.grad[1, 0] != 0


def test_global_ap_couples_queries_while_mean_ap_is_offset_invariant():
    """Verify that query score offsets affect gSAP while leaving per-query AP unchanged."""
    scores = torch.tensor([[0.8, 0.7], [0.4, 0.2]], dtype=torch.double)
    relevance = torch.tensor([[True, False], [True, False]])
    shifted = scores + torch.tensor([[0.0], [0.5]], dtype=torch.double)
    mean = train.SmoothAPLoss(temperature=0.05)
    glob = train.GSAPLoss(temperature=0.05, negative_drop_rate=0)
    torch.testing.assert_close(mean.from_similarity(scores, relevance), mean.from_similarity(shifted, relevance))
    assert glob.from_similarity(shifted, relevance) < glob.from_similarity(scores, relevance)


def test_global_ap_includes_negatives_from_queries_without_positives():
    """Check that distractor queries contribute negatives to global AP."""
    scores = torch.tensor([[0.8, 0.2]])
    relevance = torch.tensor([[True, False]])
    extended_scores = torch.cat([scores, torch.tensor([[0.9, 0.7]])])
    extended_relevance = torch.cat([relevance, torch.zeros_like(relevance)])
    glob = train.GSAPLoss(temperature=0.05, negative_drop_rate=0)
    mean = train.SmoothAPLoss(temperature=0.05)
    assert glob.from_similarity(extended_scores, extended_relevance) > glob.from_similarity(scores, relevance)
    torch.testing.assert_close(
        mean.from_similarity(extended_scores, extended_relevance), mean.from_similarity(scores, relevance)
    )


@pytest.mark.parametrize("cls", [train.SmoothAPLoss, train.GSAPLoss, train.SigLIPLoss])
def test_embedding_api_excludes_self_pairs(cls):
    """Exclude embedding self-matches while retaining explicit matrix diagonal positives."""
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([8, 8, 20])
    features = F.normalize(embeddings, dim=1)
    valid = ~torch.eye(3, dtype=torch.bool)
    kwargs = {"temperature": 0.1} if cls is not train.SigLIPLoss else {}
    if cls is train.GSAPLoss:
        kwargs["negative_drop_rate"] = 0
    objective = cls(**kwargs)
    expected = objective.from_similarity(features @ features.T, labels[:, None] == labels[None, :], valid)
    torch.testing.assert_close(objective(embeddings, labels), expected)
    # A cross-modal relevance matrix retains its diagonal when no mask is supplied.
    rectangular = torch.tensor([[0.8, 0.1]], requires_grad=True)
    loss = objective.from_similarity(rectangular, torch.tensor([[True, False]]))
    loss.backward()
    assert rectangular.grad[0, 0] != 0


@pytest.mark.parametrize("cls", [train.SmoothAPLoss, train.GSAPLoss])
@pytest.mark.parametrize("labels", [[1, 2, 3], [1, 1, 1]])
def test_ap_degenerate_batches_return_differentiable_zero(cls, labels):
    """Keep backward valid with zero loss for batches missing positives or negatives."""
    embeddings = torch.randn(len(labels), 3, requires_grad=True)
    loss = cls()(embeddings, torch.tensor(labels, dtype=torch.long))
    assert loss == 0
    loss.backward()
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


@pytest.mark.parametrize("cls", [train.SmoothAPLoss, train.GSAPLoss, train.SigLIPLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_losses_support_mixed_precision(cls, dtype):
    """Promote low-precision scores to float32 and retain finite gradients."""
    scores = torch.tensor([[0.501, 0.5], [0.501, 0.499]], dtype=dtype, requires_grad=True)
    loss = cls().from_similarity(scores, torch.eye(2, dtype=torch.bool))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("cls", [train.SmoothAPLoss, train.GSAPLoss, train.SigLIPLoss])
def test_autocast_does_not_reduce_embedding_similarity_precision(cls):
    """Keep cosine similarity computation at full precision under CPU autocast."""
    torch.manual_seed(9)
    embeddings = torch.randn(8, 5, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(2)
    objective = cls()
    expected = objective(embeddings, labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = objective(embeddings, labels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_subcenter_arcface_margin_is_in_radians():
    """Ensure the public margin in radians is converted correctly for the backend."""
    objective = train.SubCenterArcFaceLoss(4, 3, margin=0.5)
    assert objective.loss.margin == pytest.approx(0.5)


def test_siglip_matches_signed_logistic_formula_and_trains_negative_only_batches():
    """Match the signed logistic formula and retain learning from batches containing only negatives."""
    scores = torch.tensor([[0.7, -0.3], [0.2, 0.6]], dtype=torch.double, requires_grad=True)
    relevance = torch.eye(2, dtype=torch.bool)
    objective = train.SigLIPLoss(scale=2, bias=-0.5).double()
    logits = 2 * scores - 0.5
    expected = -(F.logsigmoid(logits[relevance]).sum() + F.logsigmoid(-logits[~relevance]).sum()) / 2
    torch.testing.assert_close(objective.from_similarity(scores, relevance), expected)
    negative_loss = objective.from_similarity(scores, torch.zeros_like(relevance))
    assert negative_loss > 0
    negative_loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    assert objective.log_scale.grad is not None and objective.bias.grad is not None


def test_miner_adapter_uses_selected_pairs():
    """Pass the miner's selected pairs to the loss and preserve gradient flow."""
    torch.manual_seed(42)
    embeddings = torch.randn(8, 4, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(2)
    loss = train.MultiSimilarityLoss()
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    objective = train.MinedLoss(loss, miner)
    expected = loss(embeddings, labels, miner(embeddings, labels))
    torch.testing.assert_close(objective(embeddings, labels), expected)
    objective(embeddings, labels).backward()
    assert torch.isfinite(embeddings.grad).all()


def test_combined_loss_updates_parameters_and_restores_weights(tmp_path):
    """Verify joint model/proxy updates and checkpoint restoration of parameters and loss weights."""
    torch.manual_seed(5)
    features = torch.randn(8, 5)
    labels = torch.arange(4).repeat_interleave(2)
    model = nn.Linear(5, 4)
    objective = train.CombinedLoss(
        {
            "classification": train.SoftmaxLoss(4, 4),
            "metric": train.ProxyAnchorLoss(4, 4),
            "ranking": train.GSAPLoss(temperature=0.1, negative_drop_rate=0.5, chunk_size=2),
        },
        {"classification": 0.25, "metric": 0.5, "ranking": 0.25},
    )
    before_model = model.weight.detach().clone()
    before = {name: p.detach().clone() for name, p in objective.named_parameters()}
    optimizer = torch.optim.SGD(chain(model.parameters(), objective.parameters()), lr=0.01)
    trainer = train.BasicTrainer(
        TensorDataset(features, labels),
        model,
        objective,
        optimizer,
        epochs=1,
        device="cpu",
        batch_size=8,
        num_workers=0,
    )
    trainer.train()
    assert not torch.equal(before_model, model.weight)
    for name, parameter in objective.named_parameters():
        assert not torch.equal(before[name], parameter)
    trainer.save(tmp_path, save_rng=False)
    saved_model = {name: p.detach().clone() for name, p in model.named_parameters()}
    saved_objective = {name: p.detach().clone() for name, p in objective.named_parameters()}
    with torch.no_grad():
        for parameter in chain(model.parameters(), objective.parameters()):
            parameter.zero_()
    objective.weights.fill_(1)
    trainer.load(tmp_path / "checkpoint.pth", load_rng=False)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, saved_model[name])
    for name, parameter in objective.named_parameters():
        torch.testing.assert_close(parameter, saved_objective[name])
    torch.testing.assert_close(objective.weights, torch.tensor([0.25, 0.5, 0.25]))
    assert trainer.epoch == 1
