# Training
We provide simple trainer class for training on `WildlifeDataset` instances as well as wrappers for ArcFace and Triplet losses.

## Training Example
Fine-tuning MegaDescriptor-T from HuggingFace Hub

```Python
import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

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


## Reference
::: train.trainer.BasicTrainer
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false


## Replicability
The model can be trained with a specified seed to ensure replicable results by calling the `set_seed` function at the beginning of the training process. If the trainer is saved into checkpoint, the seed is stored as well, allowing for its later use in restarting the model and maintaining replicability throughout the restart.

### Example

```Python
from wildlife_tools.train import set_seed, BasicTrainer


set_seed(0)
trainer = BasicTrainer(<add trainer parameters here>)
trainer.train()

```