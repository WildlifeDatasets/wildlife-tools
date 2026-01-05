import torch
import numpy as np


def print_info(msg):
    print(f"INFO: {msg}")


def print_warning(msg):
    print(f"WARNING: {msg}")


def cuda_available():
    return torch.cuda.is_available() and torch.version.cuda is not None


def warn_confused_pairs(confusion_matrix: np.ndarray, class_names=None, threshold=0.25):
    """
    Analyze a multi-class confusion matrix and warn about pairs with high mutual confusion.

    Args:
        confusion_matrix: numpy array of shape (n_classes, n_classes)
                         where element [i, j] represents true class i predicted as class j
        class_names: list of class names (optional). If None, uses class indices.
        threshold: confusion rate threshold above which to warn.

    Returns:
        list: List of tuples (class_i, class_j)
    """
    n_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError(f"Number of class names ({len(class_names)}) must match matrix size ({n_classes})")

    warned_pairs = []
    msg = "Some identifiants are often confused by the model.\n"

    for i in range(n_classes):
        nb_true_positives_i = confusion_matrix[i, i]
        total_i = confusion_matrix[i, :].sum()
        for j in range(i + 1, n_classes):

            nb_true_positives_j = confusion_matrix[j, j]

            i_predicted_as_j = confusion_matrix[i, j]
            j_predicted_as_i = confusion_matrix[j, i]

            total_j = confusion_matrix[j, :].sum()

            if total_i > 0 and total_j > 0:
                total_confusion = i_predicted_as_j + j_predicted_as_i
                total_correct = nb_true_positives_i + nb_true_positives_j
                confusion_rate = total_confusion / total_correct

                i_to_j_rate = i_predicted_as_j / nb_true_positives_i
                j_to_i_rate = j_predicted_as_i / nb_true_positives_j

                if confusion_rate > threshold:
                    msg += f"\t-'{class_names[i]}' is confused as '{class_names[j]}' {i_to_j_rate:.1%} of the time, while '{class_names[j]}' is confused as '{class_names[i]}' {j_to_i_rate:.1%} of the time.\n"
                    warned_pairs.append((i, j))

    if warned_pairs:
        msg += """\n Your options are the following:
            \t1) If you can visually differienciate between those subjects yourself, you need to add nore data to your dataset.
            \t2) If you can not visually differienciate between those subjects yourself, you need to change your re-identification strategy."""
        print_warning(msg)

    return warned_pairs
