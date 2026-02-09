from typing import Optional

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from ..data import FeatureCacheMixin


class DeepFeatures(FeatureCacheMixin):
    """
    Extracts features using forward pass of pytorch model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            model (torch.nn.Module): Pytorch model used for the feature extraction.
            batch_size (int, optional): Batch size used for the feature extraction.
            num_workers (int, optional): Number of workers used for data loading.
            device (str, optional): Select between cuda and cpu devices.
            cache_path (str, optional): Path for cached results. No caching for None.
        """

        super().__init__(cache_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model

    def cat_features(self, feats):
        return torch.cat(feats).numpy()
    
    def forward_batch(self, batch):
        with torch.no_grad():
            images, _ = batch
            return self.model(images.to(self.device)).cpu()


class ClipFeatures:
    """
    Extract features using CLIP model (https://arxiv.org/pdf/2103.00020.pdf).
    Uses raw images of input ImageDataset (i.e. dataset.transform = None)
    """

    def __init__(
        self,
        model: Optional[CLIPModel] = None,
        processor: Optional[CLIPProcessor] = None,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            model (CLIPModel, optional): Uses VIT-L backbone by default.
            processor: (CLIPProcessor, optional). Uses VIT-L processor by default.
            batch_size (int, optional): Batch size used for the feature extraction.
            num_workers (int, optional): Number of workers used for data loading.
            device (str, optional): Select between cuda and cpu devices.
            cache_path (str, optional): Path for cached results. No caching for None.
        """

        super().__init__(cache_path)
        if model is None:
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

        if processor is None:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.transform = lambda x: processor(images=x, return_tensors="pt")["pixel_values"]

    def cat_features(self, feats):
        return torch.cat(feats).numpy()
    
    def forward_batch(self, batch):
        with torch.no_grad():
            images, _ = batch
            return self.model(self.transform(images).to(self.device)).pooler_output.cpu()

    def make_loader(self, dataset, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x,
        )
