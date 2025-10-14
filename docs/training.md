# Training ML models

We provide simple trainer class for training on `WildlifeDataset` instances as well as wrappers for ArcFace and Triplet losses.

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

