import torch
from collections import defaultdict
import os
from tqdm import tqdm
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.train import BasicTrainer
from wildlife_tools.features import DeepFeatures

from .utils import print_info


class BasicTrainerWithValidation(BasicTrainer):
    def __init__(
        self,
        dataset,
        val_dataset,
        save_dir,
        checkpoint_name,
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
        val_interval=5,
    ):
        super().__init__(
            dataset,
            model,
            objective,
            optimizer,
            epochs,
            scheduler,
            device,
            batch_size,
            num_workers,
            accumulation_steps,
            epoch_callback,
        )
        self.val_dataset = val_dataset
        self.val_interval = val_interval
        self.best_score = 0.0
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name

    def train(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        results = defaultdict(list)

        for e in tqdm(range(self.epochs), desc="INFO: Training progress", mininterval=1, ncols=100):
            epoch_data = self.train_epoch(loader)
            results["losses"].append(epoch_data["train_loss_epoch_avg"])
            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)

            if e % self.val_interval == 0:
                self.val()
        return results

    def val(self):
        extractor = DeepFeatures(self.model, device=self.device)
        matcher = CosineSimilarity()
        similarity = matcher(query=extractor(self.val_dataset), database=extractor(self.dataset))
        preds = KnnClassifier(k=1, database_labels=self.dataset.labels_string)(similarity)
        acc = sum(preds == self.val_dataset.labels_string) / len(preds)
        msg = f"Validation score={acc:.3f}."
        if acc > self.best_score:
            self.save(
                self.save_dir,
                file_name=self.checkpoint_name,
            )
            self.best_score = acc
            msg += f" This is a new highscore, updating the '{os.path.abspath(os.path.join(self.save_dir, self.checkpoint_name))}' checkpoint..."
        print_info(msg)
        self.model = self.model.to(self.device)
