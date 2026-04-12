<p align="center">
<img src="./docs/resources/precision_track.png" alt="PrecisionTrack" width="200">
<span style="font-size: 32px; margin: 0 20px; vertical-align: top;">+</span>
<img src="https://github.com/WildlifeDatasets/wildlife-tools/raw/main/docs/resources/tools-logo.png" alt="Wildlife tools" width="150">
</p>

<div align="center">
  <p align="center"><span style="font-size: 16;">A Python toolkit for training custom animal re-identification models. Seamlessly integrate with PrecisionTrack to enable appearance-based identity tracking for any species in your behavioral studies.</span></p>

<p align="center"><span style="font-size: 16;"><a href="https://wildlifedatasets.github.io/wildlife-tools/">Documentation</a></span></p>

</div>

## Introduction

**PrecisionTrack ReID** bridges the gap between [wildlife-tools](https://wildlifedatasets.github.io/wildlife-tools/) and [PrecisionTrack](https://github.com/VincentCoulombe/precision_track/tree/main), enabling researchers to train species-specific re-identification models and deploy them for automated identity tracking in behavioral studies.

### Key Features

- **End-to-end workflow**: Train, validate, and deploy custom re-identification models
- **Species-agnostic**: Works with any visually distinguishable animal species
- **Seamless integration**: Trained models can be directly imported into PrecisionTrack's tracking pipeline
- **Built on wildlife-tools**: Leverages state-of-the-art models like [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-T-224) and [CLIP](https://github.com/openai/CLIP)
- **Dataset compatibility**: Works seamlessly with [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets)

More information can be found in the [documentation](https://wildlifedatasets.github.io/wildlife-tools/).

## Installation

1. Create a python virtual environment

```script
conda create -n precision_track_reid python==3.11
```

2. Activate your python virtual environment

```script
conda activate precision_track_reid
```

3. Install using `pip`

```script
pip install git+https://github.com/VincentCoulombe/precision_track-ReID
```

To install with CUDA support (includes TensorRT and PyCUDA):

```script
pip install "git+https://github.com/VincentCoulombe/precision_track-ReID[cuda]"
```

3. Or clone the repository using `git` and install it.

```script
git clone https://github.com/VincentCoulombe/precision_track-ReID.git

cd precision_track-ReID
pip install -e .
```

To install with CUDA support:

```script
pip install -e ".[cuda]"
```

## How to use

### 1. Create a MOT dataset

The first step is to have a MOT dataset, meaning [MOT-styled annonotations](https://motchallenge.net/).

For reference, you can check the [MICE sequential dataset](https://drive.google.com/drive/folders/1WcDkX-92X6SCgZPAZXFyDc6EGUzU0Onq?usp=drive_link) which have MOT-styled annonotations under its `./bboxes/*` directories.

**Note** The MOT-styled annonotations are used inside the dataset creation pipeline. It will indicate where to crop the images. As such, we recommend the bounding boxes to be as precise as possible.

### 2. Create a dataset metadata file

You will also need a dataset metadata file (identities between each of yours MOT bounding box files to your defined class identifiants). This file will be a `.json`. Here's an example of a valid file:

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

**Note** You will need to register the path to your dataset metadata file inside your `./configs/user_configs.yaml` file. You can refer to our [Config guide](https://github.com/VincentCoulombe/precision_track-ReID/tree/main/configs) for more details.

### 3. Format your dataset's file Tree

Format your `dataset_directory` so it has the following structure. You will then need to register your `dataset_directory` inside your `./configs/user_configs.yaml` file.

```bash
<Your dataset root directory>/
  ├── bboxes/
  │ ├── video1.csv # NOTE: This is your MOT annotations (your bounding bboxes). They can have any name
  │ ├── video2.csv
  │ ├── etc...
  ├── videos/
  │ ├── video1.mp4 # NOTE: Your videos must match their correspondig MOT bboxes files.
  │ ├── video2.avi
  │ ├── etc...
  ├── <your metadata file>/ # NOTE This is the metadata file from step 1.
```

**Note**: The pipeline will automatically create a crops and saved them in the `<dataset_directory>/crops/` directory. The system will also save a `labels.csv` for each phase (train and val) **if one doesn't exist**. If you want to regenerate your dataset, simply delete the `<dataset_directory>/crops/` directory.

### 3. Train, test and deploy your re-identification model

You can now launch the training, testing and deployment processes with the following commands:

```script
cd ./tools
python train_test_deploy.py
```

### 4. Move your deployed checkpoints to your PrecisionTrack deployment directory

Once done, move the newly generated deployed ONNX or TensorRT checkpoints to your PrecisionTrack's [deploying_directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs). Then, follow PrecisionTrack's documentation to operationalize your checkpoint.

## Citation

```
@InProceedings{Cermak_2024_WACV,
    author    = {\v{C}erm\'ak, Vojt\v{e}ch and Picek, Luk\'a\v{s} and Adam, Luk\'a\v{s} and Papafitsoros, Kostas},
    title     = {{WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification}},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5953-5963}
}
```

```latex
@misc{precision_track2025,
    title={PrecisionTrack: A Platform for Automated Long-Term Social Behavior Analysis in Naturalized Environments},
    author={Coulombe & al},
    year={2025}
}
```
