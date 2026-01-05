import os
import pandas as pd
import random


def check_if_image(file_path: str) -> bool:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions


def create_labels_file(dataset_directory: str, val_split: float, seed: int = 42):
    """
    Create a labels CSV file for your dataset, given that it is organized as desbribed in the README.md file.

    Args:
        dataset_directory: Path to the dataset directory containing identity folders
        val_split: Fraction of data to use for validation (e.g., 0.2 for 20%)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)

    data = {
        "image_id": [],
        "identity": [],
        "path": [],
        "split": [],
    }

    img_id = 0

    for dataset_dir_blob in os.listdir(dataset_directory):
        dir_path = os.path.join(dataset_directory, dataset_dir_blob)
        if os.path.isdir(dir_path):
            identity = dataset_dir_blob

            for maybe_image in os.listdir(dir_path):
                image_path = os.path.abspath(os.path.join(dir_path, maybe_image))

                if check_if_image(image_path):
                    is_train = random.random() > val_split

                    data["image_id"].append(img_id)
                    img_id += 1
                    data["identity"].append(identity)
                    data["path"].append(image_path)
                    data["split"].append("train" if is_train else "val")

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dataset_directory, "labels.csv"), index=False)

    print(f"Created labels.csv with {len(df)} images")
    print(f"Train: {(df['split'] == 'train').sum()}, Val: {(df['split'] == 'val').sum()}")
    print(f"Unique identities: {df['identity'].nunique()}")
