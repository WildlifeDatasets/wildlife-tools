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


def test_metrics(config, model):
    model.eval()
    model.return_logits = False
    model.return_features = True

    extractor = DeepFeatures(model, device=config.device, batch_size=30)

    database = NumpyDataset(
        phase="train",
        metadata=config.metadata,
        root=config.dataset_directory,
        transform=config.test_transforms,
        img_size=config.img_size,
        max_length=2000,
        select_every=10,
    )

    query = NumpyDataset(
        phase="val",
        metadata=config.metadata,
        root=config.dataset_directory,
        transform=config.test_transforms,
        img_size=config.img_size,
        max_length=2000,
        select_every=10,
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


def test_classification(config, model):
    model = model.to(config.device)
    model.eval()
    model.return_logits = True
    model.return_features = False

    test_dataset = NumpyDataset(
        metadata=config.metadata,
        img_size=config.img_size,
        root=config.dataset_directory,
        transform=config.test_transforms,
        max_length=2000,
        select_every=10,
        phase="val",
        return_isolation=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
    )

    all_preds = []
    all_labels = []
    all_isolated = []
    num_batches = 0

    with torch.no_grad():
        for x, y, isolated in test_dataloader:
            x, y = x.to(config.device), y.to(config.device)

            out = model(x)
            num_batches += 1

            preds = out.argmax(dim=1)
            all_preds.extend([str(test_dataset.labels_map[pred]) for pred in preds.tolist()])
            all_labels.extend([str(test_dataset.labels_map[label]) for label in y.tolist()])
            all_isolated.extend(isolated.tolist())

    unique_labels = sorted(set(all_labels) | set(all_preds))

    # Calculate metrics for isolated subjects only
    isolated_preds = [p for p, iso in zip(all_preds, all_isolated) if iso]
    isolated_labels = [lbl for lbl, iso in zip(all_labels, all_isolated) if iso]
    if isolated_labels:
        isolated_f1 = f1_score(isolated_labels, isolated_preds, average="weighted")
        print_info(
            f"Classification (isolated only): F1 Score (weighted): {isolated_f1:.4f} ({len(isolated_labels)} samples)"
        )
        isolated_conf_matrix = confusion_matrix(isolated_labels, isolated_preds, labels=unique_labels)
        isolated_conf_matrix_df = pd.DataFrame(isolated_conf_matrix, index=unique_labels, columns=unique_labels)
        out_path = os.path.abspath(os.path.join(config.save_directory, "isolated_classification_confusion_matrix.csv"))
        isolated_conf_matrix_df.to_csv(out_path)
        print_info(f"Isolated confusion matrix saved at: {out_path}")

    conf_matrix = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)

    f1 = f1_score(all_labels, all_preds, average="weighted")
    print_info(f"Classification: F1 Score (weighted): {f1:.4f} ({len(all_labels)} samples)")

    warn_confused_pairs(conf_matrix, unique_labels)
    out_path = os.path.abspath(os.path.join(config.save_directory, "classification_confusion_matrix.csv"))
    conf_matrix_df.to_csv(out_path)
    print_info(f"Confusion matrix saved at: {out_path}")
