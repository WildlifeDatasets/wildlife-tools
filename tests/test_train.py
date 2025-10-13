import pytest
from wildlife_tools.train import BasicTrainer, ArcFaceLoss
from itertools import chain
from torch.optim import SGD



def test_basic_trainer(dataset_deep, backbone):
    objective = ArcFaceLoss(num_classes=dataset_deep.num_classes, embedding_size=768, margin=0.5, scale=64)
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    trainer = BasicTrainer(
        dataset=dataset_deep,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=2,
        device='cpu',
    )
    trainer.train()


# Compatibility with wildlife-datasets
def test_wildlife_datasets_train1(wd_dataset_labels, backbone):
    objective = ArcFaceLoss(num_classes=wd_dataset_labels.num_classes, embedding_size=768, margin=0.5, scale=64)
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    trainer = BasicTrainer(
        dataset=wd_dataset_labels,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=2,
        device='cpu',
    )
    trainer.train()


def test_wildlife_datasets_train2(wd_dataset_no_labels, backbone):
    objective = ArcFaceLoss(num_classes=wd_dataset_no_labels.num_classes, embedding_size=768, margin=0.5, scale=64)
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)

    trainer = BasicTrainer(
        dataset=wd_dataset_no_labels,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=2,
        device='cpu',
    )
    with pytest.raises(ValueError):
        trainer.train()
