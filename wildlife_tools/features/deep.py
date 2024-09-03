import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from wildlife_tools.data import WildlifeDataset, FeatureDataset
from wildlife_tools.features.base import FeatureExtractor


class DeepFeatures(FeatureExtractor):
    """
    Extracts features using forward pass of pytorch model.

    Args:
        model: Pytorch model used for the feature extraction.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 128,
        num_workers: int = 1,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model

    def __call__(self, dataset: WildlifeDataset) -> FeatureDataset:
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        outputs = []
        for image, _ in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(image.to(self.device))
                outputs.append(output.cpu())

        self.model = self.model.to('cpu')
        features = torch.cat(outputs).numpy()

        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label
        )


class ClipFeatures:
    """
    Extract features using CLIP model (https://arxiv.org/pdf/2103.00020.pdf).
    Uses raw images of input WildlifeDataset (i.e. dataset.transform = None)

    Args:
        model: transformer.CLIPModel. Uses VIT-L backbone by default.
        processor: transformer.CLIPProcessor. Uses VIT-L processor by default.
        batch_size: Batch size used for the feature extraction.
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.
    """

    def __init__(
        self,
        model=None,
        processor=None,
        batch_size=128,
        num_workers=1,
        device="cpu",
    ):
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

    def __call__(self, dataset: WildlifeDataset) -> FeatureDataset:
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        dataset.transforms = None  # Reset transforms.

        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        outputs = []
        for image in tqdm(loader, mininterval=1, ncols=100):
            with torch.no_grad():
                output = self.model(self.transform(image).to(self.device)).pooler_output
                outputs.append(output.cpu())
        features = torch.cat(outputs).numpy()

        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label
        )