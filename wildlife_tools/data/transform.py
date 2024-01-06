import timm
import torchvision.transforms as T


class TransformTimm:
    """
    Pytorch transform function from timm library.
    Example configs in YAML file:

    transform:
        method: TransformTimm
        input_size: 224
        is_training: True
        auto_augment: 'rand-m10-n2-mstd1'
    """

    @classmethod
    def from_config(cls, config):
        return timm.data.transforms_factory.create_transform(**config)


class TransformTorchvision:
    """
    Pytorch transform function from torchvision library.
    Example configs in YAML file:

    transform:
        method: TransformTorchvision
        compose:
            - 'Resize(size=256)'
            - 'ToTensor()'
    """

    @classmethod
    def from_config(cls, config):
        compose = config["compose"]
        return T.Compose([eval(s, globals(), T.__dict__) for s in compose])
