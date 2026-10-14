# Training ML models

We provide a simple trainer for `WildlifeDataset` instances and a suite of metric learning objectives. Every objective
accepts `objective(embeddings, labels)` and returns a scalar loss, so it can be used directly with `BasicTrainer`.

## Replicability

The model can be trained with a specified seed to ensure replicable results by calling the `set_seed` function at the beginning of the training process. If the trainer is saved into checkpoint, the seed is stored as well, allowing for its later use in restarting the model and maintaining replicability throughout the restart.


## Examples

We load the dataset as in the [feature extraction](./inference.md) section.

```python
from wildlife_datasets.datasets import MacaqueFaces 
import torchvision.transforms as T

root = "data/MacaqueFaces"
transform = T.Compose([
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

MacaqueFaces.get_data(root)
dataset = MacaqueFaces(
    root,
    transform=transform,
    load_label=True,
    factorize_label=True,
)
```

Then we can finetune or train a model as follows.

```Python
import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from wildlife_tools.train import set_seed

# Download MegaDescriptor-L backbone from HuggingFace Hub
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', num_classes=0, pretrained=True)

# Arcface loss - needs backbone output size and number of classes.
embedding_size = backbone(dataset[0][0].unsqueeze(0)).size(1)
objective = ArcFaceLoss(
    num_classes=dataset.num_classes,
    embedding_size=embedding_size,
    margin=0.5,
    scale=64
    )

params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=1,
    batch_size=8,
    device='cuda',
)

trainer.train()
```

## Metric learning objectives

The established objectives use the existing [PyTorch Metric Learning backend](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/)
(version 2.9.0 or newer). Its constructors, distances, reducers, and additional keyword arguments apply to the directly
exported classes. ArcFace, Sub-center ArcFace, and Triplet use the wildlife-tools wrappers described below. The AP
surrogates and SigLIP are implemented in `wildlife_tools.train.ranking`.

| Family | Available classes | Key parameters / conventions |
| --- | --- | --- |
| Pairwise | `ContrastiveLoss` | `pos_margin=0`, `neg_margin=1`; normalized Euclidean distance by default |
| Triplet / margin | `TripletLoss`, `MarginLoss` | Triplet: `margin=0.2`, `mining="semihard"`, `distance="l2_squared"`; Margin: `margin`, `beta`, `learn_beta` |
| Adaptive / threshold margins | `DynamicSoftMarginLoss`, `ThresholdConsistentMarginLoss` | Dynamic margin: `min_val`, `num_bins`, `momentum`; threshold margin: `margin_plus`, `margin_minus`, `lambda_plus`, `lambda_minus` |
| Pair weighting | `MultiSimilarityLoss`, `CircleLoss` | Multi-Similarity: `alpha=2`, `beta=50`, `base=0.5`; Circle: `m=0.4`, `gamma=80` |
| Contrastive / neighborhood | `SupConLoss`, `NTXentLoss`, `NPairsLoss`, `NCALoss` | SupCon and NT-Xent: `temperature`; NCA: `softmax_scale`; NT-Xent is the InfoNCE-style per-positive objective |
| Angular / tuplet | `AngularLoss`, `TupletMarginLoss` | `alpha` and tuplet `margin` are in **degrees**, following the backend |
| Lifted structure | `LiftedStructureLoss`, `GeneralizedLiftedStructureLoss` | `pos_margin`, `neg_margin` |
| Histogram ranking | `HistogramLoss`, `FastAPLoss` | Histogram: `n_bins` or `delta`; FastAP: `num_bins` |
| Neighborhood ranking | `PNPLoss`, `RankedListLoss` | PNP: `variant`, `anneal`, `alpha`, `b`; Ranked List requires `margin` and `Tn`, for example `RankedListLoss(margin=0.4, Tn=5)` |
| Sigmoid ranking | `SmoothAPLoss`, `GSAPLoss` | `temperature`, `chunk_size`; gSAP also has `negative_drop_rate` |
| Shared pair threshold | `SigLIPLoss` | `scale=10`, `bias=-10`, `learnable=True` |
| Learned proxies | `ProxyAnchorLoss`, `ProxyNCALoss` | Both need `num_classes`, `embedding_size`; Proxy Anchor: `margin=0.1`, `alpha=32` |
| Multiple centers | `SoftTripleLoss`, `SubCenterArcFaceLoss` | Both need class count and embedding size; `centers_per_class` or `sub_centers` |
| Classifier margins | `ArcFaceLoss`, `CosFaceLoss`, `SphereFaceLoss` | ArcFace and Sub-center ArcFace margins are in **radians**; CosFace uses a cosine margin; SphereFace uses an integer multiplicative angular margin |
| Classifier reference | `NormalizedSoftmaxLoss`, `SoftmaxLoss` | Both need `num_classes`, `embedding_size`; normalized softmax also has `temperature` |

For example, replace the objective in the training recipe:

```python
from wildlife_tools.train import GSAPLoss, ProxyAnchorLoss, SupConLoss

objective = SupConLoss(temperature=0.1)
# Or a trainable proxy objective:
objective = ProxyAnchorLoss(num_classes=dataset.num_classes, embedding_size=embedding_size)
# Or the global AP objective from the linked paper:
objective = GSAPLoss(temperature=0.001, negative_drop_rate=0.9, chunk_size=64)
```

Move the backbone and objective to the training device **before** constructing the optimizer, and include both sets
of parameters. Proxy objectives, classifier heads, trainable MarginLoss beta, and SigLIP scale/bias need optimizer updates.
`BasicTrainer` includes objective state in its checkpoints.

```python
backbone = backbone.to("cuda")
objective = objective.to("cuda")
optimizer = SGD(itertools.chain(backbone.parameters(), objective.parameters()), lr=0.001)
```

### gSAP and SmoothAP

`GSAPLoss` implements the sigmoid gAP surrogate in [Global Average Precision for Representation Learning](https://arxiv.org/abs/2610.09863).
It ranks all valid directed batch pairs together. `SmoothAPLoss` computes the per-query surrogate and averages queries
with positives. The formulas include each positive's sigmoid self-comparison, `sigmoid(0) = 0.5`, as in the paper's
Eqs. (3), (8)--(10). Embedding self-pairs are excluded from the candidate set in both losses.

gSAP defaults to dropping the lowest 90% of negative scores. Set `negative_drop_rate=0` to optimize the full surrogate.
All positives are retained. Queries without positives still contribute negatives to gSAP; SmoothAP omits those queries
from its mean. Both return a differentiable zero if there are no positives. A batch containing only positives has zero AP loss.

Both losses checkpoint chunks of positive-versus-candidate comparisons, recomputing them during backward. Chunking
bounds saved activation memory but does not remove the comparison cost or the batch similarity matrix. Low-precision
inputs are promoted to float32. Large batches may still require negative dropping and a smaller chunk size.
The embedding API computes cosine similarities outside CPU/CUDA autocast. For the matrix API, compute similarities
in float32 before supplying them; casting an already rounded similarity matrix cannot restore its precision.

For existing similarities, local descriptors, or cross-modal training, use the matrix API:

```python
import torch
from wildlife_tools.train import GSAPLoss

queries = torch.nn.functional.normalize(query_embeddings, dim=1)
references = torch.nn.functional.normalize(reference_embeddings, dim=1)
similarities = queries @ references.T
positives = query_labels[:, None] == reference_labels[None, :]
loss = GSAPLoss(negative_drop_rate=0).from_similarity(similarities, positives)
```

`from_similarity` accepts a rectangular floating-point matrix, a boolean or binary positive mask, and an optional
boolean `valid_mask`. It never removes the diagonal automatically: diagonal pairs can be genuine cross-modal positives.
When the two axes contain the same observations, pass a mask excluding their self-pairs. Similarities must be larger
for better matches; convert distances accordingly.

### SigLIP

`SigLIPLoss` adapts the [SigLIP signed logistic objective](https://arxiv.org/abs/2303.15343) to labeled embedding batches.
The sum over valid pairs is divided by the query count, rather than the pair count. It learns a shared positive scale
and bias by default and supports the same matrix API. Unlike the AP losses, negative-only batches still produce a loss.

### Mining

Use a backend miner with a compatible backend objective through `MinedLoss`. For Multi-Similarity this also exposes
the pair-selection stage explicitly, instead of silently assuming that loss weighting performs mining.

```python
from pytorch_metric_learning import miners
from wildlife_tools.train import MinedLoss, MultiSimilarityLoss

objective = MinedLoss(MultiSimilarityLoss(), miners.MultiSimilarityMiner(epsilon=0.1))
```

`TripletLoss` already includes its triplet miner. gSAP and SmoothAP use their valid pair masks, rather than a tuple miner.

### Combining objectives

`CombinedLoss` registers all constituent modules and stores the weights in checkpoint state:

```python
from wildlife_tools.train import CombinedLoss, ProxyAnchorLoss, SoftmaxLoss

objective = CombinedLoss(
    objectives={
        "classification": SoftmaxLoss(dataset.num_classes, embedding_size),
        "metric": ProxyAnchorLoss(dataset.num_classes, embedding_size),
    },
    weights={"classification": 0.5, "metric": 0.5},
)
```

This combines categorical and metric supervision on the supplied embeddings. A nonlinear adaptation head, and a
particular categorical objective such as Seesaw, remain separate choices when constructing the HEAda model.

### Batch requirements for comparisons

Pair, triplet, contrastive, and AP losses need multiple images of at least some identities in each batch. Identity-balanced
sampling is usually appropriate. Proxy losses can learn from singleton identities. Backend objectives differ in their
mining and reduction rules; the table lists families, not an assertion that their numerical loss scales are comparable.

`BasicTrainer` currently shuffles images. Use a custom loader with `trainer.train_epoch(loader)` when comparing losses
under controlled identity-balanced batches. Gradient accumulation does not add cross-microbatch comparisons: the AP
and contrastive candidate sets are those of each forward pass.
