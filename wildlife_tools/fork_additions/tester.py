import torch
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score

from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.fork_additions import (
    warn_confused_pairs,
    print_info,
    NumpyDataset,
    ForkedDeepFeatures,
)


def _scenarios(detector_checkpoint):
    """Return [(label, detector_ckpt)] for each test scenario."""
    scenarios = [("all", None)]
    if detector_checkpoint:
        scenarios.append(("conf_only", detector_checkpoint))
    return scenarios


def test_metrics(config, model):
    model.eval()
    model.return_logits = False
    model.return_features = True

    extractor = ForkedDeepFeatures(model, device=config.device, batch_size=30)

    f1_scores = {}
    all_scenario_cm = None  # confusion matrix data from the "all" scenario

    for scenario_label, detector_ckpt in _scenarios(config.detector_checkpoint):
        database = extractor(
            NumpyDataset(
                metadata=config.metadata,
                img_size=config.img_size,
                root=config.dataset_directory,
                transform=config.test_transforms,
                max_length=20000,
                select_every=10,
                phase="train",
                return_isolation=True,
                detector_checkpoint=detector_ckpt,
            )
        )

        query = extractor(
            NumpyDataset(
                metadata=config.metadata,
                img_size=config.img_size,
                root=config.dataset_directory,
                transform=config.test_transforms,
                max_length=2000,
                select_every=10,
                phase="val",
                return_isolation=True,
                detector_checkpoint=detector_ckpt,
            )
        )

        matcher = CosineSimilarity()
        similarity = matcher(query=query, database=database)
        preds = KnnClassifier(k=1, database_labels=database.labels_string)(similarity)

        f1_all = f1_score(query.labels_string, preds, average="weighted")
        f1_scores[f"{scenario_label}_all"] = f1_all
        print_info(f"Metric ({scenario_label}, all): F1 Score (weighted): {f1_all:.4f}")

        isolated_similarities = similarity[query.isolations]
        isolated_preds = KnnClassifier(k=1, database_labels=database.labels_string)(isolated_similarities)
        isolated_labels_string = query.labels_string[query.isolations]

        f1_isolated = f1_score(isolated_labels_string, isolated_preds, average="weighted")
        f1_scores[f"{scenario_label}_isolated"] = f1_isolated
        print_info(f"Metric ({scenario_label}, isolated): F1 Score (weighted): {f1_isolated:.4f}")

        if scenario_label == "all":
            unique_labels = sorted(set(query.labels_string) | set(preds))
            all_scenario_cm = (query.labels_string, preds, unique_labels)

    q_labels, q_preds, unique_labels = all_scenario_cm
    conf_matrix = confusion_matrix(q_labels, q_preds, labels=unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
    warn_confused_pairs(conf_matrix, unique_labels)
    out_path = os.path.abspath(os.path.join(config.save_directory, "metrics_confusion_matrix.csv"))
    conf_matrix_df.to_csv(out_path)
    print_info(f"Confusion matrix saved at: {out_path}")

    print_info("=== Metric F1 Summary ===")
    if "conf_only_all" in f1_scores:
        print_info(f"  conf only (all):      {f1_scores['conf_only_all']:.4f}")
        print_info(f"  conf only (isolated): {f1_scores['conf_only_isolated']:.4f}")
    print_info(f"  all (all):            {f1_scores['all_all']:.4f}")
    print_info(f"  all (isolated):       {f1_scores['all_isolated']:.4f}")

    return f1_scores


def test_classification(config, model):
    model = model.to(config.device)
    model.eval()
    model.return_logits = True
    model.return_features = False

    f1_scores = {}
    all_scenario_data = None  # (labels, preds, unique_labels, isolated) from "all" scenario

    for scenario_label, detector_ckpt in _scenarios(config.detector_checkpoint):
        test_dataset = NumpyDataset(
            metadata=config.metadata,
            img_size=config.img_size,
            root=config.dataset_directory,
            transform=config.test_transforms,
            max_length=2000,
            select_every=10,
            phase="val",
            return_isolation=True,
            detector_checkpoint=detector_ckpt,
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

        with torch.no_grad():
            for x, y, isolated in test_dataloader:
                x, y = x.to(config.device), y.to(config.device)
                out = model(x)
                preds = out.argmax(dim=1)
                all_preds.extend([str(test_dataset.labels_map[pred]) for pred in preds.tolist()])
                all_labels.extend([str(test_dataset.labels_map[label]) for label in y.tolist()])
                all_isolated.extend(isolated.tolist())

        n = len(all_labels)
        f1_all = f1_score(all_labels, all_preds, average="weighted")
        f1_scores[f"{scenario_label}_all"] = f1_all
        print_info(f"Classification ({scenario_label}, all): F1 Score (weighted): {f1_all:.4f} ({n} samples)")

        isolated_preds_list = [p for p, iso in zip(all_preds, all_isolated) if iso]
        isolated_labels_list = [lbl for lbl, iso in zip(all_labels, all_isolated) if iso]
        if isolated_labels_list:
            f1_isolated = f1_score(isolated_labels_list, isolated_preds_list, average="weighted")
            f1_scores[f"{scenario_label}_isolated"] = f1_isolated
            print_info(
                f"Classification ({scenario_label}, isolated): F1 Score (weighted): {f1_isolated:.4f} ({len(isolated_labels_list)} samples)"
            )
        else:
            f1_scores[f"{scenario_label}_isolated"] = float("nan")

        if scenario_label == "all":
            unique_labels = sorted(set(all_labels) | set(all_preds))
            all_scenario_data = (all_labels, all_preds, unique_labels, all_isolated)

    s_labels, s_preds, s_unique_labels, s_isolated = all_scenario_data

    conf_matrix = confusion_matrix(s_labels, s_preds, labels=s_unique_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=s_unique_labels, columns=s_unique_labels)
    warn_confused_pairs(conf_matrix, s_unique_labels)
    out_path = os.path.abspath(os.path.join(config.save_directory, "classification_confusion_matrix.csv"))
    conf_matrix_df.to_csv(out_path)
    print_info(f"Confusion matrix saved at: {out_path}")

    iso_preds = [p for p, iso in zip(s_preds, s_isolated) if iso]
    iso_labels = [lbl for lbl, iso in zip(s_labels, s_isolated) if iso]
    if iso_labels:
        isolated_conf_matrix = confusion_matrix(iso_labels, iso_preds, labels=s_unique_labels)
        isolated_conf_matrix_df = pd.DataFrame(isolated_conf_matrix, index=s_unique_labels, columns=s_unique_labels)
        out_path = os.path.abspath(os.path.join(config.save_directory, "isolated_classification_confusion_matrix.csv"))
        isolated_conf_matrix_df.to_csv(out_path)
        print_info(f"Isolated confusion matrix saved at: {out_path}")

    print_info("=== Classification F1 Summary ===")
    if "conf_only_all" in f1_scores:
        print_info(f"  conf only (all):      {f1_scores['conf_only_all']:.4f}")
        print_info(f"  conf only (isolated): {f1_scores['conf_only_isolated']:.4f}")
    print_info(f"  all (all):            {f1_scores['all_all']:.4f}")
    print_info(f"  all (isolated):       {f1_scores['all_isolated']:.4f}")

    return f1_scores
