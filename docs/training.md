# Training
We provide simple trainer class for training on `WildlifeDataset` instances as well as wrappers for ArcFace and Triplet losses.


## Replicability
The model can be trained with a specified seed to ensure replicable results by calling the `set_seed` function at the beginning of the training process. If the trainer is saved into checkpoint, the seed is stored as well, allowing for its later use in restarting the model and maintaining replicability throughout the restart.


::: train.trainer
    options:
      show_root_heading: true
      heading_level: 2


::: train.objective
    options:
      show_root_heading: true
      heading_level: 2


## Examples
Fine-tuning MegaDescriptor-T from HuggingFace Hub

```Python
import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer
from wildlife_tools.train import set_seed

# Download MegaDescriptor-T backbone from HuggingFace Hub
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

# Arcface loss - needs backbone output size and number of classes.
objective = ArcFaceLoss(
    num_classes=dataset.num_classes,
    embedding_size=768,
    margin=0.5,
    scale=64
    )

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

set_seed(0)
trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=20,
    device='cpu',
)

trainer.train()

```

