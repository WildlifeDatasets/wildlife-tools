import torch


class OptimizerAdam:
    """
    Example config in YAML file:
    - Needs model params for initialization.

    optimizer:
        method: 'OptimizerAdam'
        lr: 1e-3
    """

    @classmethod
    def from_config(cls, config, params):
        return torch.optim.Adam(params=params, **config)


class OptimizerSGD:
    """
    Example config in YAML file:
    - Needs model params for initialization.

    optimizer:
        method: 'OptimizerSGD'
        lr: 1e-3
    """

    @classmethod
    def from_config(cls, config, params):
        return torch.optim.SGD(params=params, **config)


class SchedulerCosine:
    """
    Example config in YAML file:
    - Needs optimizer and number of epochs for initialization.

    scheduler:
        method: SchedulerCosine
    """

    @classmethod
    def from_config(cls, config, optimizer, epochs):
        lr = optimizer.defaults.get("lr")
        lr_min = lr * 1e-3 if lr is not None else 1e-5
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min, **config
        )
