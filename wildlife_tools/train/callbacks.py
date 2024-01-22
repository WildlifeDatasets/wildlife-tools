import os
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from wildlife_tools.tools import realize


class EpochCheckpoint:
    def __init__(self, folder=".", save_step=1):
        self.folder = folder
        self.save_step = save_step

    def __call__(self, trainer, **kwargs):
        os.makedirs(self.folder, exist_ok=True)
        if trainer.epoch % self.save_step == 0:
            trainer.save(
                folder=self.folder, file_name=f"checkpoint-{trainer.epoch}.pth"
            )


class EpochLog:
    def __init__(self, folder="."):
        self.folder = folder
        self.writer = SummaryWriter(log_dir=folder)

    def __call__(self, trainer, epoch_data: dict[str, int], **kwargs):
        if trainer.scheduler is not None:
            self.writer.add_scalar(
                "lr", trainer.scheduler.get_last_lr()[0], trainer.epoch
            )

        for key, value in epoch_data.items():
            self.writer.add_scalar(key, value, trainer.epoch)


class EpochCallbacks:
    """Returns sequence of epoch callbacks."""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, trainer, **kwargs):
        for step in self.steps:
            step(trainer=trainer, **kwargs)

    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        steps = []
        for config_step in config["steps"]:
            steps.append(realize(config_step))
        return cls(steps=steps)
