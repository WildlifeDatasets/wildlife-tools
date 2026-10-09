# Introduction

The `wildlife-tools` library offers a simple interface for various tasks in the wildlife re-identification domain. Its main features are:

- It covers use cases such as training, feature extraction, similarity calculation, image retrieval, and classification.
- It provides traning codes and usage examples for our models [MegaDescriptor](./megadescriptor.md) and [WildFusion](./wildfusion.md).
- It supports [caching](./caching.md) of extracted features and matching scores, so that repeated runs compute only what is missing.
- It complements the [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets) library, which acts as dataset repository.


## Installation

`wildlife-tools` requires Python 3.11 or newer. Install it using `pip`

```script
pip install git+https://github.com/WildlifeDatasets/wildlife-tools
```

or clone the repository using `git` and install it.

```script
git clone git@github.com:WildlifeDatasets/wildlife-tools.git

cd wildlife-tools
pip install -e .
```


## Modules in the in the `wildlife-tools`

- The `data` module provides tools for creating instances of datasets.
- The `train` module offers tools for fine-tuning feature extractors.
- The `features` module provides tools for extracting features using various extractors.
- The `similarity` module provides tools for constructing a similarity matrix from query and database features.
- The `inference` module offers tools for creating predictions using the similarity matrix.



## Relations between modules

```mermaid
  graph TD;
      A[Data]-->|ImageDataset|B[Features]
      A-->|ImageDataset|C;
      C[Train]-->|finetuned extractor|B;
      B-->|query and database features|D[Similarity]
      D-->|similarity matrix|E[Inference]
```

## Citation

If you like our package, please cite us.

```
@InProceedings{Cermak_2024_WACV,
    author    = {{\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Picek, Lukas and Adam, Luk{\'a}{\v{s}} and Papafitsoros, Kostas},
    title     = {{WildlifeDatasets: An open-source toolkit for animal re-identification}},
    booktitle = {2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    pages     = {5941--5951},
    year      = {2024},
    organization={IEEE}
}
```

```
@inproceedings{cermak2024wildfusion,
  title={Wildfusion: Individual animal identification with calibrated similarity fusion},
  author={Cermak, Vojt{\v{e}}ch and Picek, Lukas and Adam, Luk{\'a}{\v{s}} and Neumann, Luk{\'a}{\v{s}} and Matas, Ji{\v{r}}{\'\i}},
  booktitle={European Conference on Computer Vision},
  pages={18--36},
  year={2024},
  organization={Springer}
}
```
