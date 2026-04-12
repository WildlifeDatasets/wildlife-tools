import torch
import numpy as np
import plotext as plt
from scipy.optimize import linear_sum_assignment

from .detector import ONNXDetector


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

                i_to_j_rate = i_predicted_as_j / total_correct + i_predicted_as_j
                j_to_i_rate = j_predicted_as_i / total_correct + j_predicted_as_i

                if confusion_rate > threshold:
                    msg += f"\t-'{class_names[i]}' is confused as '{class_names[j]}' {i_to_j_rate:.1%} of the time, while '{class_names[j]}' is confused as '{class_names[i]}' {j_to_i_rate:.1%} of the time.\n"
                    warned_pairs.append((i, j))

    if warned_pairs:
        msg += """\n Your options are the following:
            \t1) If you can visually differienciate between those subjects yourself, you need to add nore data to your dataset.
            \t2) If you can not visually differienciate between those subjects yourself, you need to change your re-identification strategy."""
        print_warning(msg)

    return warned_pairs


def print_counts(counts: list, labels: list, phase: str = ""):
    plt.bar([str(i) for i in labels], [int(c) for c in counts])
    plt.xlabel("Identity")
    plt.ylabel("Count")
    plt.title(f"Identity Distribution ({phase})")
    plt.show()
    plt.clear_figure()


def calculate_overlaps(b1, b2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes in xywh format.

    Args:
        b1: tuple/list (x, y, w, h) - first bounding box
        b2: tuple/list (x, y, w, h) - second bounding box

    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    b1_x2, b1_y2 = x1 + w1, y1 + h1
    b2_x2, b2_y2 = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def filter_matches(x, y, cost_matrix, thresh):
    num_matches = 0
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] <= thresh:
            num_matches += 1

    if num_matches == 0:
        return np.empty(0), np.empty(0)

    thr_x = np.empty(num_matches, dtype=np.float64)
    thr_y = np.empty(num_matches, dtype=np.float64)
    index = 0
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] <= thresh:
            thr_x[index] = x[i]
            thr_y[index] = y[i]
            index += 1

    return thr_x, thr_y


def linear_assignment(cost_matrix, thresh=None):
    if cost_matrix.size == 0:
        return (
            np.empty(0),
            np.empty(0),
        )
    x, y = linear_sum_assignment(cost_matrix)
    if thresh is None:
        return x, y
    elif isinstance(thresh, float):
        return filter_matches(x, y, cost_matrix, thresh)
    else:
        raise TypeError(f"thresh must be one of [NoneType, float, np.ndarray]. Not, {type(thresh)}.")


class DetectionFilter:
    def __init__(
        self,
        model: dict,
        iou_thresh=0.1,
        sample_every=30,
        conf_thresh=0.75,
    ):
        self.model = ONNXDetector(**model)
        self.iou_thresh = iou_thresh
        self.sample_every = sample_every
        self.conf_thresh = conf_thresh
        self.last_sampled = {}  # track_id -> last frame number
        self.current_frame = 0

    def __call__(self, frame, detections):
        self.current_frame += 1

        if len(detections) == 0:
            return detections

        predictions = self.model(frame)[0]["pred_instances"]

        bboxes = predictions["bboxes"]
        scores = predictions["scores"]

        if len(bboxes) == 0:
            return []

        num_gts = len(detections)
        num_preds = len(bboxes)
        cost_matrix = np.zeros((num_gts, num_preds))

        for i, (track_id, x, y, w, h) in enumerate(detections):
            for j, pred in enumerate(bboxes):
                iou = calculate_overlaps((x, y, w, h), (pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item()))
                cost_matrix[i, j] = 1 - iou

        matched_gts, matched_preds = linear_assignment(cost_matrix, thresh=1 - self.iou_thresh)

        filtered_detections = []
        for gt_idx, pred_idx in zip(matched_gts, matched_preds):
            track_id, x, y, w, h = detections[int(gt_idx)]
            pred_conf = scores[int(pred_idx)]

            if pred_conf < self.conf_thresh:
                continue

            if track_id in self.last_sampled:
                frames_since_last = self.current_frame - self.last_sampled[track_id]
                if frames_since_last < self.sample_every:
                    continue

            filtered_detections.append((track_id, x, y, w, h))
            self.last_sampled[track_id] = self.current_frame

        return filtered_detections
