import os
import random

import numpy as np
import torch
import torch.backends.cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_states(states):
    if "os_rng_state" in states and states["os_rng_state"]:
        os.environ["PYTHONHASHSEED"] = states["os_rng_state"]
    if "random_rng_state" in states:
        random.setstate(states["random_rng_state"])
    if "numpy_rng_state" in states:
        np.random.set_state(states["numpy_rng_state"])
    if "torch_rng_state" in states:
        torch.set_rng_state(states["torch_rng_state"])
    if "torch_cuda_rng_state" in states:
        torch.cuda.set_rng_state(states["torch_cuda_rng_state"])
    if "torch_cuda_rng_state_all" in states:
        torch.cuda.set_rng_state_all(states["torch_cuda_rng_state_all"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_states():
    """Gives dictionary of random states for reproducibility."""
    states = {}
    states["os_rng_state"] = os.environ.get("PYTHONHASHSEED")
    states["random_rng_state"] = random.getstate()
    states["numpy_rng_state"] = np.random.get_state()
    states["torch_rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
        states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return states


class BasicTrainer:
    """
    Implements basic training loop for Pytorch models.
    Checkpoints includes random states - any restarts from checkpoint preservers reproducibility.

    Args:
        dataset ():
            Training dataset that gives (x, y) tensor pairs.
        model (dict):
            Pytorch nn.Module for model / backbone.
        objective (dict):
            Pytorch nn.Module for objective / loss function.
        optimizer:
            Pytorch optimizer.
        scheduler (optional):
            Pytorch scheduler.
        epochs (int):
            Number of training epochs.
        device (str, default: 'cuda'):
            Device to be used for training.
        batch_size (int, default: 128):
            Training batch size.
        num_workers (int, default: 1):
            Number of data loading workers in torch DataLoader.
        accumulation_steps (int, default: 1):
            Number of gradient accumulation steps.
        epoch_callback:
            Callback function to be called after each epoch.
        writer (SummaryWriter, optional):
            TensorBoard SummaryWriter instance. If None, TensorBoard logging is disabled.
        log_interval (int, default: 3):
            Interval (in batches) at which to log training metrics to TensorBoard.
    """

    def __init__(
        self,
        dataset,
        model,
        objective,
        optimizer,
        epochs,
        scheduler=None,
        device="cuda",
        batch_size=128,
        num_workers=1,
        accumulation_steps=1,
        epoch_callback=None,
        writer=None,
        log_interval=3,
    ):
        self.dataset = dataset
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.epoch = 0
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accumulation_steps = accumulation_steps
        self.epoch_callback = epoch_callback
        self.log_interval = log_interval
        self.writer = writer
        self.global_step = 0

    def train(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

        for e in range(self.epochs):
            epoch_data = self.train_epoch(loader)
            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)

    def train_epoch(self, loader):
        model = self.model.train()
        losses = []
        correct = 0
        total = 0
        train_accuracy_epoch_avg = 0
        with tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1, ncols=100) as pbarT:
            for i, batch in enumerate(pbarT):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                out = model(x)
                loss = self.objective(out, y)
                loss.backward()
                if (i - 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                losses.append(loss.detach().cpu())

                _, predicted = out.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                dis_loss = loss.item()
                accuracy = 100. * correct / total

                pbarT.set_postfix(loss=f"{dis_loss:.4f}", accuracy=f"{accuracy:.2f}%")


                if self.writer and i % self.log_interval == 0:
                    self.global_step += 1
                    self.writer.add_scalar('Training/Batch_Loss', dis_loss, self.global_step)
                    self.writer.add_scalar('Training/Batch_Accuracy', accuracy, self.global_step)

        train_accuracy_epoch_avg = 100. * correct / total
        train_loss_epoch_avg = np.mean(losses)


        if self.writer:
            self.writer.add_scalar('Training/Epoch_Loss', train_loss_epoch_avg, self.epoch)
            self.writer.add_scalar('Training/Epoch_Accuracy', train_accuracy_epoch_avg, self.epoch)

        if self.scheduler:
            self.scheduler.step()

        return {
            "train_loss_epoch_avg": train_loss_epoch_avg,
            "train_accuracy_epoch_avg": train_accuracy_epoch_avg
        }

    def save(self, folder, file_name="checkpoint.pth", save_rng=True, **kwargs):
        if not os.path.exists(folder):
            os.makedirs(folder)

        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["objective"] = self.objective.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["epoch"] = self.epoch
        if save_rng:
            checkpoint["rng_states"] = get_random_states()
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, os.path.join(folder, file_name))

    def load(self, path, load_rng=True):
        checkpoint = torch.load(path, map_location=torch.device(self.device))

        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "objective" in checkpoint:
            self.objective.load_state_dict(checkpoint["objective"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "rng_states" in checkpoint and load_rng:
            set_random_states(checkpoint["rng_states"])
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
