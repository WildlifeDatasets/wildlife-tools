import os

from torch.utils.tensorboard import SummaryWriter

from .trainer import BasicTrainer, EpochCallback


class EpochCheckpoint(EpochCallback):
    """Save trainer checkpoint after epoch."""

    def __init__(self, folder: str = ".", save_step: int = 1) -> None:
        self.folder = folder
        self.save_step = save_step

    def __call__(self, trainer: BasicTrainer, epoch_data: dict[str, float], **kwargs) -> None:
        os.makedirs(self.folder, exist_ok=True)
        if trainer.epoch % self.save_step == 0:
            trainer.save(folder=self.folder, file_name=f"checkpoint-{trainer.epoch}.pth")


class EpochLog(EpochCallback):
    """Log epoch training data into tensorboad."""

    def __init__(self, folder: str = ".", writer: SummaryWriter | None = None) -> None:
        if writer is None:
            writer = SummaryWriter(log_dir=folder)
        self.writer = writer

    def __call__(self, trainer: BasicTrainer, epoch_data: dict[str, float], **kwargs) -> None:
        if trainer.scheduler is not None:
            self.writer.add_scalar("lr", trainer.scheduler.get_last_lr()[0], trainer.epoch)

        for key, value in epoch_data.items():
            self.writer.add_scalar(key, value, trainer.epoch)


class EpochCallbacks(EpochCallback):
    """Returns sequence of epoch callbacks."""

    def __init__(self, steps: list[EpochCallback]) -> None:
        self.steps = steps

    def __call__(self, trainer: BasicTrainer, epoch_data: dict[str, float], **kwargs) -> None:
        for step in self.steps:
            step(trainer=trainer, epoch_data=epoch_data, **kwargs)
