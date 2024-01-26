import os
import random
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
import torch.backends.cudnn
from tqdm import tqdm

from wildlife_tools.tools import realize


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_states(states):
    if 'os_rng_state' in states and states["os_rng_state"]:
        os.environ["PYTHONHASHSEED"] = states["os_rng_state"]
    if 'random_rng_state' in states:
        random.setstate(states["random_rng_state"])
    if 'numpy_rng_state' in states:
        np.random.set_state(states["numpy_rng_state"])
    if 'torch_rng_state' in states:
        torch.set_rng_state(states["torch_rng_state"])
    if 'torch_cuda_rng_state' in states:
        torch.cuda.set_rng_state(states["torch_cuda_rng_state"])
    if 'torch_cuda_rng_state_all' in states:
        torch.cuda.set_rng_state_all(states["torch_cuda_rng_state_all"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_states():
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
        for i, batch in enumerate(
            tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1, ncols=100)
        ):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            out = model(x)
            loss = self.objective(out, y)
            loss.backward()
            if (i - 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses.append(loss.detach().cpu())

        if self.scheduler:
            self.scheduler.step()

        return {"train_loss_epoch_avg": np.mean(losses)}

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


class ClassifierTrainer:
    @classmethod
    def from_config(cls, config):
        """
        Use config dict to setup BasicTrainer for training classifier.

        Config keys:
            dataset (dict):
                Config dictionary of the training dataset.
            backbone (dict):
                Config dictionary of the backbone.
            objective (dict):
                Config dictionary of the objective.
            scheduler (dict | None, default: None):
                Config dictionary of the scheduler (no scheduler is used by default).
            epochs (int):
                Number of training epochs.
            device (str, default: 'cuda'):
                Device to be used.
            batch_size (int, default: 128):
                Training batch size.
            num_workers (int, default: 1):
                Number of data loading workers in torch DataLoader.
            accumulation_steps (int, default: 1):
                Number of gradient accumulation steps.

        Returns:
            Ready to use BasicTrainer

        """

        config = deepcopy(config)

        dataset = realize(
            config=config.pop("dataset"),
        )
        model = realize(
            config=config.pop("backbone"),
            output_size=dataset.num_classes,
        )
        objective = realize(
            config=config.pop("objective"),
        )
        optimizer = realize(
            config=config.pop("optimizer"),
            params=model.parameters(),
        )
        scheduler = realize(
            config=config.pop("scheduler", None),
            epochs=config.get("epochs"),
        )
        epoch_callback = realize(
            config=config.pop("epoch_callback", None),
        )

        return BasicTrainer(
            model=model,
            dataset=dataset,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_callback=epoch_callback,
            **config,
        )


class EmbeddingTrainer:
    @classmethod
    def from_config(cls, config):
        """Use config dict to setup BasicTrainer for training embedder.

        Config keys:
            dataset (dict):
                Config dictionary of the training dataset.
            backbone (dict):
                Config dictionary of the backbone.
            objective (dict):
                Config dictionary of the objective.
            scheduler (dict | None, default: None):
                Config dictionary of the scheduler (no scheduler is used by default).
            embedding_size (int | None, default: None):
                Adds a linear layer after the backbone with the target embedding size.
                By default, embedding size is inferred from backbone (e.g., num_classes=0 in TIMM).
            epochs (int):
                Number of training epochs.
            device (str, default: 'cuda'):
                Device to be used.
            batch_size (int, default: 128):
                Training batch size.
            num_workers (int, default: 1):
                Number of data loading workers in torch DataLoader.
            accumulation_steps (int, default: 1):
                Number of gradient accumulation steps.

        Returns:
            Instance of BasicTrainer

        """

        config = deepcopy(config)
        embedding_size = config.pop("embedding_size", None)

        dataset = realize(config=config.pop("dataset"))
        backbone = realize(config=config.pop("backbone"), output_size=embedding_size)

        if embedding_size is None:  # Infer embedding size
            with torch.no_grad():
                x = dataset[0][0].unsqueeze(0)
                embedding_size = backbone(x).shape[1]

        objective = realize(
            config=config.pop("objective"),
            embedding_size=embedding_size,
            num_classes=dataset.num_classes,
        )
        optimizer = realize(
            config=config.pop("optimizer"),
            params=chain(backbone.parameters(), objective.parameters()),
        )
        scheduler = realize(
            optimizer=optimizer,
            config=config.pop("scheduler", None),
            epochs=config.get("epochs"),
        )
        epoch_callback = realize(
            config=config.pop("epoch_callback", None),
        )
        return BasicTrainer(
            model=backbone,
            dataset=dataset,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_callback=epoch_callback,
            **config,
        )
