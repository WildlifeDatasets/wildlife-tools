import torch
import os
import numpy as np
from tqdm import tqdm 
from wildlife_tools.features.base import FeatureExtractor
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.tools import realize


class DeepFeatures(FeatureExtractor):
    '''
    Extracts features using forward pass of pytorch model.

    Args:
        model: Pytorch model used for the feature extraction.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.

    '''

    def __init__(
        self,
        model,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = 'cpu',
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model


    def __call__(self, dataset: WildlifeDataset):
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        outputs = []
        for image, label in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(image.to(self.device))
                outputs.append(output.cpu())
        return torch.cat(outputs).numpy()


    @classmethod
    def from_config(cls, config):
        model = realize(config.pop('model'))
        return cls(model=model, **config)