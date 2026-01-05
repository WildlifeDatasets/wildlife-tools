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

### 1. Format your dataset's file Tree

```bash
<Your dataset root directory>/
  ├── <Your first identifiant>/
  │ ├── image1.png # NOTE: Your images can have any name and extensions
  │ ├── image2.png
  │ ├── etc...
  ├── <Your second identifiant>/
  │ ├── image1.png # NOTE: Your images can have any name and extensions
  │ ├── image2.png
  │ ├── etc...
  ├── <Etc...>/
```

### 2. Train, test and deploy your re-identification model

You can now launch the training, testing and deployment processes with the following commands:

```script
cd ./tools
python train_test_deploy.py
```

### 3 Move your deployed checkpoints to your PrecisionTrack deployment directory

Once done, simply move the newly generated deployed checkpoints to your PrecisionTrack's [deploying_directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs)

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
