# Inference for ML models

Feature extractors extract feature vectors from images. These features are used for inference, that is predicting the identity in a query image from a database of known individuals.

## Example of inference

### Create dataset

The simplest way is to use some dataset from the [WildlifeDataset](https://github.com/WildlifeDatasets/wildlife-datasets) library. Here, we use the [MacaqueFaces](https://github.com/clwitham/MacaqueFaces/) dataset. We specify that we want to apply transform `transform`. It is also possible to [load bounding boxes](https://wildlifedatasets.github.io/wildlife-datasets/wildlife_dataset/) or segmentation masks if provided.

```Python
from wildlife_datasets.datasets import MacaqueFaces 
import torchvision.transforms as T

root = "data/MacaqueFaces"
transform = T.Compose([
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

MacaqueFaces.get_data(root)
dataset = MacaqueFaces(
    root,
    transform=transform,
    load_label=True,
    factorize_label=True,
)
```

Optionally, split the dataset into the database and query sets. The following split is performed so that both sets contains two individuals, each with ten images.

```Python
idx_train = list(range(10)) + list(range(190,200))
idx_test = list(range(10,20)) + list(range(200,210))
dataset_database = dataset.get_subset(idx_train)
dataset_query = dataset.get_subset(idx_test)
```

### Extract features

Extract features using the [MegaDescriptor](./megadescriptor.md) model.

```Python
import timm
from wildlife_tools.features import DeepFeatures

model_name = "hf-hub:BVRA/MegaDescriptor-L-384"
backbone = timm.create_model(model_name, num_classes=0, pretrained=True)
extractor = DeepFeatures(backbone, batch_size=4, device='cuda')

query, database = extractor(dataset_query), extractor(dataset_database)
```

Both `query` and `database` are of shape 20xn, where n depends on the model and is the size of the feature (embedding) vector.

### Calculate similarity

Calculate cosine similarity between query and database deep features.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)
```

The `similarity` is a 20x20 matrix, where each row corresponds to one query image and the entries are the scores of matches between the query image and the images in the database. The highest the score, the more probable it is that the images show the same individual.

### Evaluate

Use the cosine similarity in nearest neigbour classifier and get predictions.

```Python
import numpy as np
from wildlife_tools.inference import KnnClassifier

classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
predictions = classifier(similarity)
accuracy = np.mean(dataset_query.labels_string == predictions)
```

## Feature extractors

There are additional ways to extract features from images. Feature extractors, implemented as classes, can be created with specific arguments that define the extraction properties. After instantiation, the extractor functions as a callable, requiring only a single argument — the `WildlifeDataset` instance. The specific output type and shape vary based on the chosen feature extractor.

### Deep features

The features may be extracted by any model, for example by [MiewID-msv3](https://huggingface.co/conservationxlabs/miewid-msv3), which is another model for animal re-identification.

```Python
from transformers import AutoModel

backbone = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
extractor = DeepFeatures(backbone, device='cuda')
features = extractor(dataset)
```

If the model should be loaded only when it is needed (for example when the features are [cached](./caching.md)), create the extractor by `lazy_load` with a function creating the model.

```Python
import functools

extractor = DeepFeatures.lazy_load(
    functools.partial(AutoModel.from_pretrained, 'conservationxlabs/miewid-msv3', trust_remote_code=True),
    device='cuda',
)
```

### Local features

Local feature extractors detect keypoints and compute their descriptors. They are based on [Kornia](https://kornia.readthedocs.io/) and their features are matched by `MatchLightGlue` with the corresponding `features` argument.

| Extractor | `MatchLightGlue(features=...)` |
|---|---|
| `AlikedExtractor` | `'aliked'` |
| `DiskExtractor` | `'disk'` |
| `XFeatExtractor` | `'xfeat'` |
| `DeDoDeExtractor(descriptor='B')` | `'dedodeb'` |
| `DeDoDeExtractor(descriptor='G')` | `'dedodeg'` |
| `DoGHardNetExtractor()` | `'doghardnet'` |
| `DoGHardNetExtractor(affnet=True)` | `'dog_affnet_hardnet'` |
| `KeyNetAffNetHardNetExtractor` | `'keynet_affnet_hardnet'` |

```Python
from wildlife_tools.features import AlikedExtractor, DiskExtractor, DoGHardNetExtractor, XFeatExtractor

device = 'cuda'
num_workers = 4

extractor = AlikedExtractor(device=device, num_workers=num_workers)
features = extractor(dataset)

extractor = DiskExtractor(device=device, num_workers=num_workers)
features = extractor(dataset)

extractor = DoGHardNetExtractor(device=device, num_workers=num_workers)
features = extractor(dataset)

extractor = XFeatExtractor(device=device, num_workers=num_workers)
features = extractor(dataset)
```

Image loading is often the bottleneck of local feature extraction. Using multiple workers (`num_workers`) for data loading speeds up the extraction significantly. Images can also be processed in batches (`batch_size`), but only if all images have the same size, for example after resizing.

For possible keywords, look at their [definitions](https://github.com/WildlifeDatasets/wildlife-tools/blob/main/wildlife_tools/features/local.py).



## Similarity scores



### Global similarity score

The `similarity.cosine` module provides tools for calculating global similarity scores between query and database. This is done using cosine similarity between fixed-length embeddings extracted by neural networks.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity = CosineSimilarity()
sim = similarity(query, database)
```

### Matching-based similarity scores

The `similarity.pairwise` module provides tools and methods for calculating pairwise matching similarity scores. At its core, the `MatchPairs` base class offers pairwise matching with support for batch processing, making it essential for neural network-based matching. Specific implementations of of `MatchPairs` are:

- `MatchLightGlue`: It uses the LightGlue model, a lightweight neural matching that uses keypoints and descriptors from the [local feature extractors](#local-features).
- `MatchLOFTR`: It uses the LOFTR (Local Feature TRansformer) model, which performs descriptor-free matching using directly pair of images. Pretrained models `'outdoor'`, `'indoor'` and `'indoor_new'` are available.

Outputs from the matchers, such as confidence scores for local matches and keypoints, are processed using collectors from `similarity.pairwise.collectors`. In particular, the `CollectCounts` collector calculates  matching similarity scores by counting significant matches based on given confidence thresholds.

The following example matches all pairs by the LightGlue matcher with ALIKED features and calculates similarity scores based on the count of significant matches at confidence thresholds of 0.25, 0.5, and 0.75.

```python
from wildlife_tools.features import AlikedExtractor
from wildlife_tools.similarity import MatchLightGlue, CollectCounts

transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
dataset_query.transform, dataset_database.transform = transform, transform
extractor = AlikedExtractor()
matcher = MatchLightGlue(features='aliked', collector=CollectCounts(thresholds=[0.25, 0.5, 0.75]))
output = matcher(extractor(dataset_query), extractor(dataset_database))
```

The following example matches all pairs by the LOFTR matcher and calculates similarity scores based on the count of significant matches at confidence thresholds of 0.25, 0.5, and 0.75. Note that LOFTR operates directly on image pairs and requires no feature extraction.

```python
from wildlife_tools.similarity import MatchLOFTR, CollectCounts

transform = T.Compose([T.Resize([224, 224]), T.Grayscale(), T.ToTensor()])
dataset_query.transform, dataset_database.transform = transform, transform
matcher = MatchLOFTR(collector=CollectCounts(thresholds=[0.25, 0.5, 0.75]))
output = matcher(dataset_query, dataset_database)
```



### Similarity scores calibration

The `similarity.calibration` module offers tools to improve the interpretability and utility of similarity scores by calibrating them. Calibration allows similarity scores to be interpreted as probabilities, making them suitable for confidence assessments and thresholding. This also enables the effective ensemble of multiple scores by mapping them onto a common probabilistic scale. We implemented calibration methods

  - `LogisticCalibration`: Uses logistic regression to map similarity scores to probabilities, providing a smooth and parametric approach to calibration.
  - `IsotonicCalibration`: A non-parametric approach that fits isotonic regression. Conceptually similar to score binning.

The `reliability_diagram` function allows for visual comparison of calibrated and uncalibrated scores. This tool is helpful for assessing calibration quality, as it visualizes how well the predicted probabilities align with observed outcomes.

```python
from wildlife_tools.similarity.calibration import IsotonicCalibration

calibration = IsotonicCalibration()
calibration.fit([0, 0.5, 1], [0, 1, 1])
calibration.predict([0, 0.25, 0.8])
```

For additional examples on similarity, see the provided [Jupyter notebooks](https://github.com/WildlifeDatasets/wildlife-tools/tree/main/examples).