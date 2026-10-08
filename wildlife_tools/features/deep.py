from collections.abc import Callable

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from ..data import FeatureCacheMixin
from .base import FeatureExtractor


def collate_fn(batch):
    images, labels = zip(*batch)  # tuple of PIL images
    return list(images), labels


class DeepFeatures(FeatureCacheMixin, FeatureExtractor):
    """
    Extracts features using forward pass of pytorch model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        """
        Args:
            model (torch.nn.Module): Pytorch model used for the feature extraction.
            batch_size (int, optional): Batch size used for the feature extraction.
            num_workers (int, optional): Number of workers used for data loading.
            device (str, optional): Select between cuda and cpu devices.
            cache_path (str, optional): Path for cached results. No caching for None.
            config_tag (str, optional): Free-form tag stored in the cache config. Reusing cache_path with a
                different tag raises an error. Changes of the model or of the image transform are not
                detected automatically, so encode them in the tag (e.g. "megadescriptor-L-384_resize384").
        """

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        self.model = model

    @classmethod
    def lazy_load(cls, model_factory: Callable[[], torch.nn.Module], **kwargs) -> "DeepFeatures":
        """
        Create the extractor without loading the model. The model is created by `model_factory`
        on its first use, so it is not loaded at all when all features are already cached.

        Example:
            ```python
            import functools
            import timm

            extractor = DeepFeatures.lazy_load(
                functools.partial(timm.create_model, "hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True),
                device="cuda",
                cache_path="cache/MacaqueFaces/megadescriptor",
            )
            ```

        Args:
            model_factory (Callable[[], torch.nn.Module]): Function without arguments returning the model.
                Use `functools.partial` instead of `lambda` if the extractor needs to be pickled.
            **kwargs: Remaining arguments of the class constructor (e.g. `batch_size`, `device`,
                `cache_path`, or `processor` for `ClipFeatures` and `DinoFeatures`).

        Returns:
            extractor (DeepFeatures): Extractor of the class on which the method is called.
        """
        extractor = cls(None, **kwargs)  # type: ignore[arg-type]
        extractor._model_factory = model_factory
        return extractor

    def cat_features_dictionary(self, feats: list[np.ndarray]) -> np.ndarray:
        return np.stack(feats, axis=0)

    def cat_features_model(self, feats: list[torch.Tensor]) -> np.ndarray:
        return torch.cat(feats).numpy()

    def forward_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            images, _ = batch
            return self.model(images.to(self.device)).cpu()


class ClipFeatures(DeepFeatures):
    """
    Extract features using CLIP model (https://arxiv.org/pdf/2103.00020.pdf).
    Uses raw images of input ImageDataset (i.e. dataset.transform = None)
    """

    def __init__(
        self,
        model: CLIPModel | None = None,
        processor: CLIPProcessor | None = None,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):
        """
        Args:
            model (CLIPModel, optional): Uses VIT-L backbone by default.
            processor: (CLIPProcessor, optional). Uses VIT-L processor by default.
            batch_size (int, optional): Batch size used for the feature extraction.
            num_workers (int, optional): Number of workers used for data loading.
            device (str, optional): Select between cuda and cpu devices.
            cache_path (str, optional): Path for cached results. No caching for None.
            config_tag (str, optional): Free-form tag stored in the cache config. Reusing cache_path with a
                different tag raises an error. Changes of the model or of the image transform are not
                detected automatically, so encode them in the tag (e.g. "clip-vit-large-patch14").
        """
        if processor is None:
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        super().__init__(
            model,  # type: ignore[arg-type]
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            cache_path=cache_path,
            config_tag=config_tag,
        )
        if model is None:
            self._model_factory = lambda: CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
        self.processor = processor
        self.transform = lambda x: processor(images=x, return_tensors="pt")["pixel_values"]

    def forward_batch(self, batch):
        with torch.no_grad():
            images, _ = batch
            return self.model(self.transform(images).to(self.device)).pooler_output.cpu()

    def make_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )


class DinoFeatures(DeepFeatures):
    def __init__(
        self,
        model,
        processor,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
        cache_path: str | None = None,
        config_tag: str | None = None,
    ):

        super().__init__(
            model,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            cache_path=cache_path,
            config_tag=config_tag,
        )

        self.processor = processor
        self.transform = lambda x: processor(images=x, return_tensors="pt")["pixel_values"]

    def forward_batch(self, batch):
        with torch.no_grad():
            images, _ = batch
            return self.model(self.transform(images).to(self.device)).pooler_output.cpu()

    def make_loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
