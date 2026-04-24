# Training ML models

We provide a simple trainer class for training on `WildlifeDataset` instances as well as wrappers for ArcFace, Triplet, Softmax, and Per-Instance Temperature Scaling (PITS) losses.

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

## Per-Instance Temperature Scaling

The `PerInstanceTemperatureScalingLoss` is adapted from the method proposed in *Animal Identification with Independent Foreground and Background Modeling* by Picek, Neumann, and Matas. It is intended for classification models that output both class logits and a per-sample temperature. Compared to a standard cross-entropy objective, it regularizes each sample-specific temperature using class frequency, which can help calibration and make the model less overconfident for rare identities.

The loss expects model output in the form `(logits, temperature)`. The trainer automatically passes `dataset.label_counts` into the loss when the dataset provides them.

```python
import torch
import torch.nn as nn
from torch.optim import AdamW

from wildlife_tools.train import BasicTrainer, PerInstanceTemperatureScalingLoss


class ClassificationWithTemperature(nn.Module):
    def __init__(self, backbone: nn.Module, embedding_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embedding_size, num_classes + 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        return logits, temperature


embedding_size = backbone(dataset[0][0].unsqueeze(0)).size(1)
model = ClassificationWithTemperature(backbone, embedding_size, dataset.num_classes)
objective = PerInstanceTemperatureScalingLoss(lambda_weight=0.1)
optimizer = AdamW(model.parameters(), lr=0.001)

trainer = BasicTrainer(
    dataset=dataset,
    model=model,
    objective=objective,
    optimizer=optimizer,
    epochs=1,
    batch_size=8,
    device="cuda",
)

trainer.train()
```

This feature is most useful when:

- the task is treated as closed-set classification over known identities
- the dataset exposes `label_counts`
- you want better-calibrated identity probabilities for later combination with metadata priors

Reference:

- arXiv: [2408.12930](https://arxiv.org/abs/2408.12930)
- Springer DOI: [10.1007/978-3-031-85181-0_16](https://doi.org/10.1007/978-3-031-85181-0_16)

```bibtex
@inproceedings{Picek2025ForegroundBackground,
  author = {Picek, Luk{\'a}{\v{s}} and Neumann, Luk{\'a}{\v{s}} and Matas, Ji{\v{r}}{\'i}},
  title = {Animal Identification with Independent Foreground and Background Modeling},
  booktitle = {Pattern Recognition - 46th DAGM German Conference, DAGM GCPR 2024, Proceedings},
  year = {2025},
  doi = {10.1007/978-3-031-85181-0_16},
  url = {https://arxiv.org/abs/2408.12930}
}
```
