import torch
import torch.nn.functional as F
from collections import defaultdict
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from wildlife_tools.similarity import cosine_similarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.train import BasicTrainer
from wildlife_tools.features import DeepFeatures

from .utils import print_info


class TrainerWithValidation(BasicTrainer):
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

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.dataset.sample_weights, num_samples=len(self.dataset), replacement=True
        )
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        results = defaultdict(list)

        for e in tqdm(range(self.epochs), desc="INFO: Training progress", mininterval=1, ncols=100):
            epoch_data = self.train_epoch(loader)
            results["losses"].append(epoch_data["train_loss_epoch_avg"])
            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)

            if e % self.val_interval == 0 and e > 0:
                self.val()
        return results

    def val(self):
        extractor = DeepFeatures(self.model, device=self.device)
        query = extractor(self.val_dataset)
        database = extractor(self.dataset)
        similarity = cosine_similarity(query.features, database.features, to_numpy=False)

        preds = KnnClassifier(k=1, database_labels=self.dataset.labels_string)(similarity.numpy())
        f1 = f1_score(query.labels_string, preds, average="weighted")

        msg = f"\nValidation F1 Score: {f1:.3f}"
        if f1 > self.best_score:
            self.save(
                self.save_dir,
                file_name=self.checkpoint_name,
            )
            self.best_score = f1
            msg += f" This is a new highscore, updating the '{os.path.abspath(os.path.join(self.save_dir, self.checkpoint_name))}' checkpoint..."
        print_info(msg)
        self.model = self.model.to(self.device)


class ClassificationTrainerWithValidation(TrainerWithValidation):

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
        val_interval=1,
    ):

        super().__init__(
            dataset,
            val_dataset,
            save_dir,
            checkpoint_name,
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
            val_interval,
        )

    def val(self):
        self.model.eval()
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        all_preds = []
        all_labels = []
        total_ce = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y, _ in val_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                _, logits = self.model(x)
                total_ce += self.objective.cross_entropy_loss(logits, y).item()
                num_batches += 1

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        f1 = f1_score(all_labels, all_preds, average="weighted")
        mean_ce = total_ce / num_batches

        self.model.train()

        msg = f"\nValidation F1 Score: {f1:.3f} | Cross-Entropy: {mean_ce:.4f}"
        if f1 > self.best_score:
            self.save(
                self.save_dir,
                file_name=self.checkpoint_name,
            )
            self.best_score = f1
            msg += f" This is a new highscore, updating the '{os.path.abspath(os.path.join(self.save_dir, self.checkpoint_name))}' checkpoint..."
        print_info(msg)
        self.model = self.model.to(self.device)

    def save(self, folder, file_name="checkpoint.pth"):
        if not os.path.exists(folder):
            os.makedirs(folder)

        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["epoch"] = self.epoch
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, os.path.join(folder, file_name))
