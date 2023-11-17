## Data splits

Utilities for creating subset of the datasets from WildlifeDataset, by subseting its metadata dataframe.

- splits provided wildlife-datasets SplitWildlife

- Splits can be chained using SplitChain.

For example I want to parallelize matching SIFT descriptors over multiple jobs. This requires splitting wildlifeDataset to query and databse subsets. query subsets can be subsequently split into additional equally sized subsets.

Example: Split dataset

## Transforms

Simple wrapper that provides unified interface to popular image transform methods such as timm.create_transforms and Torchvision Compose

## Database

Storage of extracted features.


## Example of usage:

Which can be constructed from Yaml file using tools.realize:

For hands on example, please refer to examples/data.ipynb