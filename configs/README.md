# User Configuration Guide

This README explains all configuration options available in `user_configs.yaml` for the Precision Track Re-Identification pipeline.

## Table of Contents

- [General On/Off Options](#general-onoff-options)
- [General Directories and Paths](#general-directories-and-paths)
- [Training Parameters](#training-parameters)
- [Pipeline Overview](#pipeline-overview)

## General On/Off Options

### `train`

- **Type**: Boolean (`true` or `false`)
- **Default**: `true`
- **Description**: Controls whether to execute the training phase. When enabled, the pipeline will:
  - Create or use an existing `labels.csv` file in the dataset directory
  - Split the dataset into training and validation sets
  - Train the MegaDescriptor-T-224 model using ArcFace loss
  - Save the trained model checkpoint as `precision_track_re-identificator.pth`

**Note**: Set as `false` if you have already trained a network and you only want ot test and/or deploy it.

### `test`

- **Type**: Boolean (`true` or `false`)
- **Default**: `true`
- **Description**: Controls whether to execute the testing phase. When enabled, the pipeline will:
  - Load the trained model checkpoint
  - Extract features from validation set images
  - Perform re-identification using k-NN classification (k=1) with cosine similarity
  - Generate a confusion matrix saved as `re-identification_confusion_matrix.csv`
  - Display warnings about frequently confused identity pairs

### `deploy`

- **Type**: Boolean (`true` or `false`)
- **Default**: `true`
- **Description**: Controls whether to execute the deployment phase. When enabled, the pipeline will:
  - Export the trained model to ONNX format (`precision_track_re-identificator_DEPLOYED.onnx`)
  - Convert to FP16 precision if supported by GPU hardware (compute capability >= 7.0)
  - Optionally create a TensorRT engine if CUDA is available and TensorRT is installed

## Classification parameters

### `num_classes`

- **Type**: Int
- **Default**: `5`
- **Description**: The number of distinct subjects you wish the model to learn to re-identify

## General Directories and Paths

### `dataset_directory`

- **Type**: String (path)
- **Default**: `"../../datasets/MICE/re-id/v3/"`
- **Description**: Root directory containing your MOT dataset. This directory should:
  - Contain a `./bboxes` and a `./videos` sub directory.
  - Your `metadata` .json file.

**Note**: The pipeline will automatically create a crops and saved them in the `<dataset_directory>/crops/` directory. The system will also save a `labels.csv` for each phase (train and val) **if one doesn't exist**. If you want to regenerate your dataset, simply delete the `<dataset_directory>/crops/` directory.

### `metadata`

- **Type**: String (path)
- **Default**: `"re_id_mapping.json"`
- **Description**: Path to your metadata `.json` file that maps identities between each of yours MOT bounding box files to your defined class identifiants. Here's an example:

```json
{
	"classes": ["Horizontal", "Vertical", "Top", "Top_twice", "Full"],
	"gt_mosaic": [
		[0, 4],
		[0, 1],
		[0, 3],
		[0, 5],
		[0, 2]
	],
	"mosaic_2025-12-22T09_08_33": [
		[0, 2],
		[0, -1], // NOTE set to -1 if absent from the .csv file
		[0, 6],
		[0, 5],
		[0, 4]
	]
}
```

**Structure:**

- **`classes`**: A list of your dataset's identity names (e.g., individual animal names or IDs).
- **Video entries** (e.g., `gt_mosaic`, `mosaic_2025-12-22T09_08_33`): Each key corresponds to a video name and contains mappings between the MOT bounding box file columns:
  - The first value in each pair corresponds to the **label column** (2nd column in the MOT `bboxes.csv` files)
  - The second value corresponds to the **instance ID column** (3rd column in the MOT `bboxes.csv` files)

This mapping allows the network to associate each unique IDS (a combination of the label and the instance ID of each subjects) of your MOT bounding box files with your defined identities in the `classes` list.

### `save_directory`

- **Type**: String (path)
- **Default**: `"../work_dir/training_runs/"`
- **Description**: The training runs root directory, this is were your `deployed checkpoints` and your `confusion matrix` will be saved.

## Crop confidence filter (optional)

- **`detector_checkpoint`**: Path to your PrecisionTrack's checkpoint. This option is optionnal, meaning you can leave a `null` value to disable it. If enable, the dataset creation pipeline will be able to access a confidence level for each crop of your dataset. Knowing the confidence level of your crop will then enable the testing pipeline to calculate a F1 score specifically for confident crops (confidence > 0.75).

## Training Parameters

### `batch_size`

- **Type**: Integer
- **Default**: `80`
- **Description**: Number of images processed in each training batch. This value affects:
  - GPU memory usage (higher values require more memory)
  - Training speed (larger batches are more efficient but may reduce gradient noise)
  - Gradient accumulation steps (calculated as `160 / batch_size`)

**Recommendations**:

- Reduce if encountering out-of-memory errors
- Typical values: 32, 64, 80, 128 (depending on GPU memory)

### `epochs`

- **Type**: Integer
- **Default**: `100`
- **Description**: Total number of training epochs. The training schedule includes:
  - Warmup phase: First 2/3 of epochs with linear learning rate increase
  - Cosine annealing: Final 1/3 of epochs with cosine decay to minimum learning rate

**Note**: The model checkpoint is saved periodically during training and at completion.

### `val_split`

- **Type**: Float (0.0 to 1.0)
- **Default**: `0.2`
- **Description**: Proportion of the dataset reserved for validation. For example:
  - `0.2` means 20% validation, 80% training
  - `0.3` means 30% validation, 70% training

**Important**: This value is only used when creating a new `labels.csv` file. If a labels file already exists, this parameter is ignored.

### `seed`

- **Type**: Integer
- **Default**: `42`
- **Description**: Random seed for reproducible train/validation splits. Using the same seed ensures:
  - Consistent dataset splits across runs
  - Reproducible experimental results

**Important**: This value is only used when creating a new `labels.csv` file. If a labels file already exists, this parameter is ignored.
