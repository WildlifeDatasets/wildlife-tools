import os
from torch.utils.tensorboard import SummaryWriter
from typing import List
from .trainer import BasicTrainer


class EpochCheckpoint:
    """Save trainer checkpoint after epoch."""

    def __init__(self, folder: str = ".", save_step: int = 1):
        self.folder = folder
        self.save_step = save_step

    def __call__(self, trainer: BasicTrainer, **kwargs):
        os.makedirs(self.folder, exist_ok=True)
        if trainer.epoch % self.save_step == 0:
            trainer.save(folder=self.folder, file_name=f"checkpoint-{trainer.epoch}.pth")


class EpochLog:
    """Log epoch training data into tensorboad."""

    def __init__(self, folder: str = ".", writer: SummaryWriter = None):
        if writer is None:
            writer = SummaryWriter(log_dir=folder)
        self.writer = writer

    def __call__(self, trainer: BasicTrainer, epoch_data: dict[str, int], **kwargs):
        if trainer.scheduler is not None:
            self.writer.add_scalar("lr", trainer.scheduler.get_last_lr()[0], trainer.epoch)

        for key, value in epoch_data.items():
            self.writer.add_scalar(key, value, trainer.epoch)


class EpochCallbacks:
    """Returns sequence of epoch callbacks."""

    def __init__(self, steps: List[int]):
        self.steps = steps

    def __call__(self, trainer: BasicTrainer, **kwargs):
        for step in self.steps:
            step(trainer=trainer, **kwargs)
