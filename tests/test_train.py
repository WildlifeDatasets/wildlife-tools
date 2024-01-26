import timm
from wildlife_tools.train import BasicTrainer, ArcFaceLoss
from itertools import chain
import torchvision.transforms as T
from torch.optim import SGD



def test_basic_trainer(dataset_deep):
    backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
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

