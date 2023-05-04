import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

def prepare_batch(batch, device='cpu'):
    if isinstance(batch, list):
        x, y = batch
    else:
        raise NotImplementedError('Batch needs to be (x, y) tuple.')

    x = x.to(device)
    y = y.to(device)
    return x, y


class ClassifierTrainer():
    def __init__(
        self,
        model,
        objective,
        optimizer,
        scheduler=None,
        epochs=100,
        device='cuda',
        batch_size=128,
        num_workers=1,
        **kwargs,
    ):
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = 0

    def predict(self, dataset):
        model = self.model.eval()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        outputs = []
        for batch in tqdm(loader, mininterval=1):
            x, _ = prepare_batch(batch, device=self.device)
            with torch.no_grad():
                outputs.append(model(x).cpu())
        return torch.cat(outputs)


    def train(self, dataset_train, evaluation=None, epoch_eval=1, **kwargs):
        loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

        for e in range(self.epochs):
            self.train_epoch(loader)
            self.epoch += 1

            if evaluation:
                evaluation.evaluate(self)

    def save(self, folder, name='checkpoint.pth'):
        checkpoint = { # Always save to cpu or cuda
            'model': self.model.state_dict(),
            'objective': self.objective.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(checkpoint, os.path.join(folder, name))


    def load(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model'])
        self.objective.load_state_dict(checkpoint['objective'])
        self.epoch = checkpoint['epoch']


    def train_epoch(self, loader):
        model = self.model.train()
        for batch in tqdm(loader, desc=f'Epoch {self.epoch}: ', mininterval=1):
            x, y = prepare_batch(batch, device=self.device)

            self.optimizer.zero_grad()
            out = model(x)
            loss = self.objective(out, y)
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()


class EmbeddingTrainer(ClassifierTrainer):
    def __init__(self, *args, mining_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mining_func = mining_func


    def train_epoch(self, loader):
        model = self.model.train()
        for batch in tqdm(loader, desc=f'Epoch {self.epoch}: ', mininterval=1):
            x, y = prepare_batch(batch, device=self.device)

            self.optimizer.zero_grad()
            embeddings = model(x)
            if self.mining_func is not None:
                indices_tuple = self.mining_func(embeddings, y)
                loss = self.objective(embeddings, y, indices_tuple)
            else:
                loss = self.objective(embeddings, y)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
