import torch
import torch.nn as nn
import timm
import os


class PtReIDModel(nn.Module):
    def __init__(self, config, checkpoint=None, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=pretrained)
        self.reduction_layer = nn.Linear(config.n_embd, config.n_output_embd)
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

    def forward(self, x):
        features = self.backbone(x)
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
