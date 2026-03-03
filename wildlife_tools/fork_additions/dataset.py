from typing import Callable, Tuple
import torch
import pandas as pd
import cv2
import os
import json
import random
import numpy as np

from .utils import print_info, print_counts, calculate_overlaps


class BalancedImageDataset:
    BBOX_COLUMNS = ["frame_id", "class_id", "instance_id", "x", "y", "w", "h", "score"]
    SUPPORTED_PHASES = ["train", "val", "test"]

    def __init__(
        self,
        metadata: str,
        root: str,
        phase: str,
        transform: Callable | None = None,
        max_length: int = 1000,
        select_every: int = 5,
        overlap_thr: float = 0.8,
        return_isolation: bool = False,
    ):
        self.return_isolation = bool(return_isolation)

        assert phase in self.SUPPORTED_PHASES
        self.phase = phase
        self.select_every = select_every
        self.overlap_thr = overlap_thr

        self.mapping = None
        with open(os.path.join(root, metadata), "r") as f:
            self.mapping = json.load(f)

        assert os.path.isdir(root), f"The provided dataset directory: '{root}' does not exists."
        bboxes_dir = os.path.join(root, "bboxes", self.phase)
        assert os.path.isdir(bboxes_dir), f"The '{bboxes_dir}' does not exists."

        self.sequences = dict()
        for file in os.listdir(bboxes_dir):
            name, ext = os.path.splitext(file)
            if ext == ".csv":
                seq_path = os.path.abspath(os.path.join(bboxes_dir, file))
                sequence = pd.read_csv(seq_path)
                assert (
                    sequence.columns.tolist() == self.BBOX_COLUMNS
                ), f"The bounding boxes ground truth file: '{seq_path}' columns must be {self.BBOX_COLUMNS}, reveived {sequence.columns}."
                self.sequences[name] = sequence
                for row in sequence[["class_id", "instance_id"]].itertuples(index=False):
                    idx = self._find_instance_in_mapping(name, row.class_id, row.instance_id)
                    if idx is not None:
                        global_identity = self.mapping["classes"][idx]

        videos_dir = os.path.join(root, "videos", self.phase)
        self.crops_dir = os.path.join(root, "crops", self.phase)

        labels_csv_path = os.path.abspath(os.path.join(self.crops_dir, "labels.csv"))

        self.labels_map = self.mapping.get("classes")
        assert self.labels_map is not None, f"The '{metadata}' file must contain classes."
        identity_crops = {cls_: [] for cls_ in self.labels_map}

        if os.path.isfile(labels_csv_path):
            print_info(f"Loading labels from '{labels_csv_path}'...")
            labels_df = pd.read_csv(labels_csv_path)
            has_isolation = "isolated" in labels_df.columns
            for row in labels_df.itertuples(index=False):
                isolated = row.isolated if has_isolation else True
                identity_crops[row.identity].append((row.path, isolated))
            self.sequences = identity_crops
        elif os.path.isdir(videos_dir):
            os.makedirs(self.crops_dir, exist_ok=True)
            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

            for video_file in os.listdir(videos_dir):
                if video_file.lower().endswith(video_extensions):
                    print_info(
                        f"Extracting crops from '{os.path.abspath(os.path.join(videos_dir, video_file))}' to '{os.path.abspath(self.crops_dir)}'..."
                    )
                    video_path = os.path.join(videos_dir, video_file)
                    video_name = os.path.splitext(video_file)[0]

                    if video_name not in self.sequences:
                        continue

                    sequence = self.sequences[video_name]
                    del self.sequences[video_name]

                    last_seen_bbox = {}  # Track last bbox per identity in this video
                    cap = cv2.VideoCapture(video_path)
                    frame_id = 1
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_id % self.select_every != 0:
                            frame_id += 1
                            continue

                        frame_bboxes = sequence[sequence["frame_id"] == frame_id]

                        frame_identity_counts = {}
                        for row in frame_bboxes.itertuples(index=False):
                            idx = self._find_instance_in_mapping(video_name, row.class_id, row.instance_id)
                            if idx is not None:
                                global_identity = self.mapping["classes"][idx]
                                frame_identity_counts[global_identity] = (
                                    frame_identity_counts.get(global_identity, 0) + 1
                                )

                        for row in frame_bboxes.itertuples(index=False):
                            idx = self._find_instance_in_mapping(video_name, row.class_id, row.instance_id)
                            if idx is None:
                                continue

                            global_identity = self.mapping["classes"][idx]

                            if frame_identity_counts[global_identity] > 1:  # Filter out duplicate labels (errors)
                                continue

                            # Filter out redundant crops (immobile subjects)
                            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
                            current_bbox = (x, y, w, h)
                            if global_identity in last_seen_bbox:
                                overlap = calculate_overlaps(last_seen_bbox[global_identity], current_bbox)
                                if overlap > self.overlap_thr:
                                    continue
                            crop = frame[y : y + h, x : x + w]

                            if crop.size == 0:
                                continue

                            output_dir = os.path.join(self.crops_dir, global_identity)
                            os.makedirs(output_dir, exist_ok=True)

                            output_path = os.path.join(
                                output_dir, f"{video_name}_{frame_id}_{row.class_id}_{row.instance_id}.jpg"
                            )

                            cv2.imwrite(output_path, crop)

                            isolation_status = True
                            if self.return_isolation:
                                # Enlarge current bbox by 10%
                                enlarged_w = w * 1.1
                                enlarged_h = h * 1.1
                                enlarged_x = x - (enlarged_w - w) / 2
                                enlarged_y = y - (enlarged_h - h) / 2
                                enlarged_bbox = (enlarged_x, enlarged_y, enlarged_w, enlarged_h)

                                # Check overlap with all other bboxes in the frame
                                for other_row in frame_bboxes.itertuples(index=False):
                                    if other_row.instance_id == row.instance_id and other_row.class_id == row.class_id:
                                        continue
                                    other_bbox = (
                                        int(other_row.x),
                                        int(other_row.y),
                                        int(other_row.w),
                                        int(other_row.h),
                                    )
                                    if calculate_overlaps(enlarged_bbox, other_bbox) > 0:
                                        isolation_status = False
                                        break

                            identity_crops[global_identity].append((os.path.abspath(output_path), isolation_status))
                            last_seen_bbox[global_identity] = current_bbox

                        frame_id += 1
                    cap.release()

            self.sequences = identity_crops

            if identity_crops:
                labels_data = []
                for identity, items in identity_crops.items():
                    for path, isolated in items:
                        labels_data.append({"identity": identity, "path": path, "isolated": isolated})
                labels_df = pd.DataFrame(labels_data)
                labels_df.to_csv(labels_csv_path, index=False)
                print_info(f"Saved labels to '{labels_csv_path}'")

        self.root = root
        self.transform = transform

        self.ids_2_idx = {id_: i for i, id_ in enumerate(self.labels_map)}

        length = sum([len(self.sequences[s]) for s in self.sequences])
        max_length = int(max_length)
        if max_length > 0:
            length = min(max_length, length)

        self.length = length

        self._compute_sample_weights()

    def __len__(self):
        return self.length

    @property
    def num_classes(self):
        return len(self.labels_map)

    @property
    def labels_string(self):
        """Returns string labels for each sample in iteration order."""
        labels = []
        for idx in range(len(self)):
            rng = random.Random(idx)
            global_identity = rng.choice(self.labels_map)
            labels.append(str(global_identity))
        return np.array(labels)

    @property
    def sample_weights(self):
        return [1.0] * self.length

    def _find_instance_in_mapping(self, name: str, class_id: int, instance_id: int) -> int | None:
        sequence_mapping = self.mapping.get(name, [])
        assert sequence_mapping, f"{sequence_mapping} does not have an identity mapping."
        for idx, pair in enumerate(sequence_mapping):
            if pair[0] == class_id and pair[1] == instance_id:
                return idx
        return None

    def _compute_sample_weights(self):
        counts = torch.tensor([len(self.sequences.get(i, [])) for i in self.labels_map], dtype=torch.float32)
        print_counts(counts=counts.tolist(), labels=self.labels_map, phase=self.phase)
        self.sample_weights_dict = {identity: 1.0 / count for identity, count in zip(self.labels_map, counts.tolist())}

    def __getitem__(self, idx: int):
        rng = random.Random(idx)
        if not self.labels_map:
            raise ValueError(f"No labels to {self.phase} on.")
        global_identity = rng.choice(self.labels_map)

        # Select a random sample from the identity's crops
        identity_samples = self.sequences.get(global_identity, [])
        if not identity_samples:
            return self.__getitem__(idx + self.length)  # Use a different seed for retry

        img_path, isolation_status = rng.choice(identity_samples)

        img = self.get_image(img_path)
        if img is None:
            return self.__getitem__(idx + self.length)  # Use a different seed for retry

        if self.transform:
            img = self.transform(img)

        if self.return_isolation:
            return img, self.ids_2_idx[global_identity], isolation_status

        return img, self.ids_2_idx[global_identity]

    def get_image(self, path: str):
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class NumpyDataset(BalancedImageDataset):
    def __init__(
        self,
        metadata: str,
        img_size: Tuple,
        root: str,
        phase: str,
        transform: Callable | None = None,
        max_length: int = 1000,
        select_every: int = 5,
        return_isolation: bool = False,
    ):
        super().__init__(
            metadata=metadata,
            root=root,
            transform=transform,
            phase=phase,
            max_length=max_length,
            select_every=select_every,
            return_isolation=return_isolation,
        )
        self.img_size = img_size

    def get_image(self, path: str):
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return img
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _compute_sample_weights(self):
        counts = torch.tensor([len(self.sequences.get(i, [])) for i in self.labels_map], dtype=torch.float32)
        print_counts(counts=counts.tolist(), labels=self.labels_map, phase=self.phase)
        self.sample_weights_dict = {identity: 1.0 for identity in self.labels_map}
