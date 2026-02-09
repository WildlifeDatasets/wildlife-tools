import torch
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score

from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.fork_additions import (
    warn_confused_pairs,
    print_info,
    NumpyDataset,
)


def test_metrics(config, model, metadata):
    model.eval()
    model.return_logits = False
    model.return_features = True

    extractor = DeepFeatures(model, device=config.device, batch_size=30)

    metadata = pd.read_csv(config.labels_path, index_col=0)

    database = NumpyDataset(
        metadata=metadata.query('split == "train"'),
        img_size=config.img_size,
        root=config.dataset_directory,
        transform=config.test_transforms,
    )

    query = NumpyDataset(
        metadata=metadata.query('split == "val"'),
        img_size=config.img_size,
        root=config.dataset_directory,
        transform=config.test_transforms,
    )

    matcher = CosineSimilarity()
    similarity = matcher(query=extractor(query), database=extractor(database))
    preds = KnnClassifier(k=1, database_labels=database.labels_string)(similarity)

    unique_labels = sorted(set(query.labels_string) | set(preds))
    conf_matrix = confusion_matrix(query.labels_string, preds, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)

    f1 = f1_score(query.labels_string, preds, average="weighted")
    print_info(f"Metric: F1 Score (weighted): {f1:.4f}")

    warn_confused_pairs(conf_matrix, unique_labels)
    out_path = os.path.abspath(os.path.join(config.save_directory, "metrics_confusion_matrix.csv"))
    conf_matrix_df.to_csv(out_path)
    print_info(f"Confusion matrix saved at: {out_path}")


def test_classification(config, model, metadata):
    model = model.to(config.device)
    model.eval()
    model.return_logits = True
    model.return_features = False

    test_dataset = NumpyDataset(
        metadata=metadata.query('split == "val"'),
        img_size=config.img_size,
        root=config.dataset_directory,
        transform=config.test_transforms,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
    )

    all_preds = []
    num_batches = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(config.device), y.to(config.device)

            out = model(x)
            num_batches += 1

            preds = out.argmax(dim=1)
            all_preds.extend([test_dataset.labels_map[pred] for pred in preds.tolist()])

    unique_labels = sorted(set(test_dataset.labels_string) | set(all_preds))
    conf_matrix = confusion_matrix(test_dataset.labels_string, all_preds, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)

    f1 = f1_score(test_dataset.labels_string, all_preds, average="weighted")
    print_info(f"Classification: F1 Score (weighted): {f1:.4f}")

    warn_confused_pairs(conf_matrix, unique_labels)
    out_path = os.path.abspath(os.path.join(config.save_directory, "classification_confusion_matrix.csv"))
    conf_matrix_df.to_csv(out_path)
    print_info(f"Confusion matrix saved at: {out_path}")
