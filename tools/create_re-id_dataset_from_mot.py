import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd


def enlarge_bbox(x: float, y: float, w: float, h: float, factor: float = 0.1) -> Tuple[float, float, float, float]:
    w_enlarge = w * factor
    h_enlarge = h * factor

    new_x = x - w_enlarge / 2
    new_y = y - h_enlarge / 2
    new_w = w + w_enlarge
    new_h = h + h_enlarge

    return new_x, new_y, new_w, new_h


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def filter_non_overlapping(
    detections: List[Tuple[int, float, float, float, float]], enlarge_factor: float = 0.1
) -> List[Tuple[int, float, float, float, float]]:
    if len(detections) <= 1:
        return detections

    # Enlarge all bounding boxes
    enlarged = []
    for track_id, x, y, w, h in detections:
        enlarged_bbox = enlarge_bbox(x, y, w, h, enlarge_factor)
        enlarged.append((track_id, x, y, w, h, enlarged_bbox))

    # Check for overlaps
    non_overlapping = []
    for i, (track_id, x, y, w, h, enlarged_bbox) in enumerate(enlarged):
        has_overlap = False
        for j, (_, _, _, _, _, other_bbox) in enumerate(enlarged):
            if i != j:
                iou = compute_iou(enlarged_bbox, other_bbox)
                if iou > 0:
                    has_overlap = True
                    break

        if not has_overlap:
            non_overlapping.append((track_id, x, y, w, h))

    return non_overlapping


def crop_bbox(frame: np.ndarray, x: float, y: float, w: float, h: float, enlarge_factor: float = 0.1) -> np.ndarray:
    x_e, y_e, w_e, h_e = enlarge_bbox(x, y, w, h, enlarge_factor)

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, int(x_e))
    y1 = max(0, int(y_e))
    x2 = min(frame_w, int(x_e + w_e))
    y2 = min(frame_h, int(y_e + h_e))

    crop = frame[y1:y2, x1:x2]
    return crop


def load_label_mapping(mapping_path: str) -> Dict[str, List]:
    """Load and validate a label mapping JSON file.

    The JSON must contain a "classes" key with a list of class names, and one or more
    video keys each mapping instance IDs to classes by list index.

    Supported formats per video key:
      - 2D list: [[class_id, instance_id], ...] — one entry per class.
      - 1D list: [instance_id, ...]             — one entry per class.

    All keys (except "classes") must have the same length as "classes".
    """
    with open(mapping_path) as f:
        mapping = json.load(f)

    if "classes" not in mapping:
        raise ValueError(f"Mapping file '{mapping_path}' must contain a 'classes' key.")

    classes = mapping["classes"]
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"'classes' must be a list of strings, got: {type(classes).__name__}")

    n_classes = len(classes)
    for key, value in mapping.items():
        if key == "classes":
            continue
        if not isinstance(value, list):
            raise ValueError(f"Mapping key '{key}' must be a list, got {type(value).__name__}.")
        if len(value) != n_classes:
            raise ValueError(
                f"Mapping key '{key}' has {len(value)} entries but 'classes' has {n_classes}. "
                f"All video keys must have the same number of entries as 'classes'."
            )

    return mapping


def build_instance_to_class(mapping: Dict[str, List], video_stem: str) -> Optional[Dict[str, str]]:
    """Build a dict mapping instance track IDs (as they appear in the dataframe) to class names.

    Returns None if the video_stem is not found in the mapping.
    """
    if video_stem not in mapping:
        return None

    classes = mapping["classes"]
    entries = mapping[video_stem]

    instance_to_class: Dict[str, str] = {}
    is_2d = isinstance(entries[0], list)

    for i, entry in enumerate(entries):
        if is_2d:
            class_id, instance_id = entry
            instance_key = f"{class_id}_{instance_id}"
        else:
            instance_key = str(entry)
        instance_to_class[instance_key] = classes[i]

    return instance_to_class


def extract_reid_crops(
    video_path: str,
    mot_path: str,
    output_dir: str,
    sample_interval: int = 10,
    enlarge_factor: float = 0.1,
    acknowledge_classes: bool = True,
    instance_to_class: Optional[Dict[str, str]] = None,
):
    tracks_df = pd.read_csv(mot_path)

    if tracks_df.columns[3] == "cx" and tracks_df.columns[4] == "cy":
        tracks_df["x"] = tracks_df["cx"] - tracks_df["w"] / 2
        tracks_df["y"] = tracks_df["cy"] - tracks_df["h"] / 2
        tracks_df.drop(["cx", "cy"], axis=1, inplace=True)
    elif tracks_df.columns[3] != "x" or tracks_df.columns[4] != "y":
        raise ValueError(
            f"Invalid bbox format in CSV. Expected columns 'x', 'y' (top-left) or 'cx', 'cy' (center), "
            f"but got '{tracks_df.columns[3]}', '{tracks_df.columns[4]}'. "
            f"CSV columns should be: frame_id, class_id, instance_id, x, y, w, h (or cx, cy instead of x, y)"
        )

    if acknowledge_classes:
        tracks_df["instance"] = tracks_df["class_id"].astype(str) + "_" + tracks_df["instance_id"].astype(str)
    else:
        tracks_df["instance"] = tracks_df["instance_id"].astype(str)
    tracks_df.drop(["class_id", "instance_id"], inplace=True, axis=1)

    video_name = Path(video_path).stem

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    tracks_grouped = tracks_df.groupby("frame_id")

    frame_count = 0
    sampled_frames = 0
    saved_crops = 0
    crops_per_id = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if (frame_count - 1) % sample_interval != 0:
            continue

        sampled_frames += 1

        if frame_count not in tracks_grouped.groups:
            continue

        frame_detections = tracks_grouped.get_group(frame_count)
        detections = list(frame_detections[["instance", "x", "y", "w", "h"]].itertuples(index=False, name=None))

        non_overlapping = filter_non_overlapping(detections, enlarge_factor)

        for track_id, x, y, w, h in non_overlapping:
            if instance_to_class is not None:
                if str(track_id) not in instance_to_class:
                    continue
                dir_name = instance_to_class[str(track_id)]
            else:
                dir_name = str(track_id)
            id_dir = output_root / dir_name
            id_dir.mkdir(exist_ok=True)

            crop = crop_bbox(frame, x, y, w, h, enlarge_factor)

            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_filename = f"frame_{video_name}_{frame_count:06d}.png"
            crop_path = id_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            saved_crops += 1
            crops_per_id[track_id] = crops_per_id.get(track_id, 0) + 1

    cap.release()

    print(f"Extraction of video:{video_name} complete! The results are saved at: '{output_dir}'")
    print(f"Total frames processed: {frame_count}")
    print(f"Sampled frames: {sampled_frames}")
    print(f"Total crops saved: {saved_crops}")
    print(f"Number of identities: {len(crops_per_id)}")
    print("\nCrops per identity:")
    for track_id in sorted(crops_per_id.keys()):
        if instance_to_class is not None:
            if str(track_id) not in instance_to_class:
                continue
            track_id = instance_to_class[str(track_id)]
        else:
            track_id = str(track_id)
        print(f"  ID {track_id}: {crops_per_id[track_id]} crops")


def find_matching_pairs(input_dir: str) -> List[Tuple[Path, Path]]:
    """Find matching video/bbox pairs from a directory with 'videos/' and 'bboxes/' subdirectories.

    Files are matched by stem name (e.g. videos/clip1.mp4 <-> bboxes/clip1.csv).
    """
    input_path = Path(input_dir)
    videos_dir = input_path / "videos"
    bboxes_dir = input_path / "bboxes"

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Expected 'videos' subdirectory at: {videos_dir}")
    if not bboxes_dir.is_dir():
        raise FileNotFoundError(f"Expected 'bboxes' subdirectory at: {bboxes_dir}")

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    videos_by_stem = {}
    for f in videos_dir.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            videos_by_stem[f.stem] = f

    bboxes_by_stem = {}
    for f in bboxes_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".csv":
            bboxes_by_stem[f.stem] = f

    matched = []
    unmatched_videos = set(videos_by_stem.keys()) - set(bboxes_by_stem.keys())
    unmatched_bboxes = set(bboxes_by_stem.keys()) - set(videos_by_stem.keys())

    for stem in sorted(set(videos_by_stem.keys()) & set(bboxes_by_stem.keys())):
        matched.append((videos_by_stem[stem], bboxes_by_stem[stem]))

    if unmatched_videos:
        print(f"Warning: {len(unmatched_videos)} video(s) without matching CSV: {sorted(unmatched_videos)}")
    if unmatched_bboxes:
        print(f"Warning: {len(unmatched_bboxes)} CSV(s) without matching video: {sorted(unmatched_bboxes)}")

    if not matched:
        raise FileNotFoundError(
            f"No matching video/CSV pairs found in {input_dir}. "
            "Ensure files in 'videos/' and 'bboxes/' share the same stem name."
        )

    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Extract re-ID crops from MOT tracking data", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing 'videos/' and 'bboxes/' subdirectories "
        "with matching file stems, OR a path to a single video file (requires --mot-path)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./reid_dataset", help="Output directory for re-ID dataset"
    )
    parser.add_argument("-n", "--sample-interval", type=int, default=30, help="Sample every N frames")
    parser.add_argument(
        "-e", "--enlarge-factor", type=float, default=0.0, help="Factor by which to enlarge bboxes (0.1 = 10%%)"
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=str,
        default=None,
        help="Path to a JSON file mapping MOT instance IDs to class names. "
        "Must contain a 'classes' key and one key per video stem.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    mapping = None
    if args.mapping is not None:
        mapping = load_label_mapping(args.mapping)

    assert input_dir.is_dir()
    pairs = find_matching_pairs(args.input_dir)
    print(f"Found {len(pairs)} matching video/CSV pair(s)\n")
    for i, (video_path, mot_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] Processing: {video_path.name}")

        # Determine acknowledge_classes and build instance mapping from the JSON
        instance_map = None
        acknowledge_classes = False
        if mapping is not None:
            instance_map = build_instance_to_class(mapping, video_path.stem)
            if instance_map is None:
                print(f"  Warning: no mapping entry for video '{video_path.stem}', skipping class mapping.")
            else:
                # Detect 2D format (acknowledge_classes) by inspecting the raw mapping entry
                sample_entry = mapping[video_path.stem][0]
                acknowledge_classes = isinstance(sample_entry, list)

        extract_reid_crops(
            video_path=str(video_path),
            mot_path=str(mot_path),
            output_dir=args.output_dir,
            sample_interval=args.sample_interval,
            enlarge_factor=args.enlarge_factor,
            acknowledge_classes=acknowledge_classes,
            instance_to_class=instance_map,
        )


if __name__ == "__main__":
    main()
