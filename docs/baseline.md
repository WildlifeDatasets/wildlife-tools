# Baseline results

This section shows how the toolkit can be used in realistic pipelines for preparing data, training models and extracting features. Specifically, we present guidelines on how to replicate the main results of the accompanying paper and provide baseline results. This includes training and inference with MegaDescriptor flavours.


## Prepare datasets

Preparing includes resizing images, cropping bounding boxes, and cropping black backgrounds of segmented images.
If multiple identities exist in one image (e.g. ATRW dataset), we crop them and split them into two images. More details about preparing datasets, resizing and splits can be found in this [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/data/prepare_datasets.ipynb).


**We save and use two sets of images:**

- For inference with 518x518 images: CLIP, DINOv2, and MegaDescriptor-L-384
- For inference with 256x256 images: MegaDescriptor-L/B/S/T-224


**Datasets splits:**

- Observations are approximately split into 80% training and 20% test sets.
- Each class is present in both the training set and the test set. Images with unknown identities are discarded.
- Training sets are aggregated into a single dataset and used for training MegaDescriptors.
- Test set for each dataset is used for evaluation.


## Training
Metadata for aggregated dataset used for training can be found [here](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/data/metadata/combined/combined_all.csv).



## Inference
In general, we use `DeepFeatures` feature extractor, cosine similarity and 1-NN `KnnClassifier`. We provide [metadata for each dataset](https://github.com/WildlifeDatasets/wildlife-tools/tree/main/baselines/data/metadata/datasets) and [results for each model](https://github.com/WildlifeDatasets/wildlife-tools/tree/main/baselines/inference/results)



## Notebooks and weights


| model | Training  | Inference  | Weights |
| ----- | -------           |     -------        |    ---- |
| MegaDescriptor-L-384 | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-L-384.ipynb) | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/inference/MegaDescriptor-L-384.ipynb) | [HuggingFace Hub](https://huggingface.co/BVRA/MegaDescriptor-L-384) |
| MegaDescriptor-L-224 | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-L-224.ipynb) | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/inference/MegaDescriptor-L-224.ipynb) | [HuggingFace Hub](https://huggingface.co/BVRA/MegaDescriptor-L-224)  |
| MegaDescriptor-B-224 | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-B-224.ipynb) | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/inference/MegaDescriptor-B-224.ipynb) | [HuggingFace Hub](https://huggingface.co/BVRA/MegaDescriptor-B-224) |
| MegaDescriptor-S-224 | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-S-224.ipynb) | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/inference/MegaDescriptor-S-224.ipynb) | [HuggingFace Hub](https://huggingface.co/BVRA/MegaDescriptor-S-224) |
| MegaDescriptor-T-224 | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/training/MegaDescriptor-T-224.ipynb) | [notebook](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/baselines/inference/MegaDescriptor-T-224.ipynb) | [HuggingFace Hub](https://huggingface.co/BVRA/MegaDescriptor-T-224) |


