import timm


class TimmBackbone:
    """
    Example config in YAML file:

    backbone:
        method: TimmBackbone
        model_name: 'efficientnet_b0'
        pretrained: True
    """

    @classmethod
    def from_config(cls, config, output_size=None):
        if output_size is None:
            output_size = 0
        return timm.create_model(num_classes=output_size, **config)
