import torch
import torch.nn as nn
import timm
import os
from abc import ABC, abstractmethod
from transformers import CLIPModel, CLIPProcessor


class BaseBackbone(ABC):
    @abstractmethod
    def create_backbone(self, pretrained: bool) -> nn.Module:
        """Create and return the backbone model."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension of the backbone."""
        pass

    @abstractmethod
    def get_processor(self):
        """Return the processor/transform for this backbone (if any)."""
        pass

    @abstractmethod
    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone given input tensor."""
        pass


class MegaDescriptor(BaseBackbone):
    def __init__(self, model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224"):
        self.model_name = model_name

    def create_backbone(self, pretrained: bool) -> nn.Module:
        return timm.create_model(self.model_name, num_classes=0, pretrained=pretrained)

    def get_embedding_dim(self) -> int:
        return 768

    def get_processor(self):
        return None

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return backbone(x)


class CLIP(BaseBackbone):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self._processor = None

    def create_backbone(self, pretrained: bool) -> nn.Module:
        if pretrained:
            clip_model = CLIPModel.from_pretrained(self.model_name)
        else:
            from transformers import CLIPConfig

            config = CLIPConfig.from_pretrained(self.model_name)
            clip_model = CLIPModel(config)
        return clip_model.vision_model

    def get_embedding_dim(self) -> int:
        return 1024

    def get_processor(self):
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        outputs = backbone(x)
        return outputs.pooler_output


BACKBONE = {
    "megadescriptor": MegaDescriptor,
    "clip": CLIP,
}


class PtReIDModel(nn.Module):
    def __init__(self, config, checkpoint=None, pretrained=False):
        super().__init__()
        backbone_name = config.backbone_name
        if backbone_name.lower() not in BACKBONE:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE.keys())}")

        self.strategy = BACKBONE[backbone_name.lower()]()
        self.backbone = self.strategy.create_backbone(pretrained)
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.reduction_layer = nn.Linear(self.strategy.get_embedding_dim(), config.n_output_embd)
        self.head = ClassifierHead(config)

        if checkpoint is not None:
            assert os.path.isfile(checkpoint), f"The provided checkpoint: '{checkpoint}' does not exists."

            state_dict = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            self.load_state_dict(state_dict["model"])

        self._return_features = True
        self._return_logits = True

    @property
    def return_features(self):
        return self._return_features

    @return_features.setter
    def return_features(self, activate):
        assert isinstance(activate, bool)
        self._return_features = activate

    @property
    def return_logits(self):
        return self._return_logits

    @return_logits.setter
    def return_logits(self, activate):
        assert isinstance(activate, bool)
        self._return_logits = activate

    def get_processor(self):
        """Return the processor for the current backbone strategy."""
        return self.strategy.get_processor()

    def forward(self, x):
        features = self.strategy.extract_features(self.backbone, x)
        reduced_features = self.reduction_layer(features)
        logits = self.head(reduced_features)

        if self._return_logits and self._return_features:
            return reduced_features, logits
        elif self._return_features:
            return reduced_features
        elif self._return_logits:
            return logits


class ClassifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = [MLP(config) for _ in range(config.n_layers)]
        self.blocks.append(nn.Linear(config.n_output_embd, config.n_classes))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_output_embd, 4 * config.n_output_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_output_embd, config.n_output_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
