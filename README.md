# Introduction
The `wildlife-tools` library offers a simple interface for various tasks in the Wildlife Re-Identification domain. It covers use cases such as training, feature extraction, similarity calculation, image retrieval, and classification. It complements the `wildlife-datasets` library, which acts as dataset repository. All datasets there can be used in combination with `WildlifeDataset` component, which serves for loading extracting images and image tensors other tasks. 

More information can be found in [Documentation](https://wildlifedatasets.github.io/wildlife-tools/)


## Modules in the in the `wildlife-tools`

- The `data` module provides tools for creating instances of the `WildlifeDataset`.
- The `train` module offers tools for fine-tuning feature extractors on the `WildlifeDataset`.
- The `features` module provides tools for extracting features from the `WildlifeDataset` using various extractors.
- The `similarity` module provides tools for constructing a similarity matrix from query and database features.
- The `evaluation` module offers tools for creating and evaluating predictions using the similarity matrix.



## Relations between modules:

```mermaid
  graph TD;
      A[Data]-->|WildlifeDataset|B[Features]
      A-->|WildlifeDataset|C;
      C[Train]-->|finetuned extractor|B;
      B-->|query and database features|D[Similarity]
      D-->|similarity matrix|E[Evaluation]
```



## Example
### 1. Create `WildlifeDataset` 
Using metadata from `wildlife-datasets`, create `WildlifeDataset` object for the StripeSpotter dataset.

```Python
from wildlife_datasets.datasets import StripeSpotter
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T

metadata = StripeSpotter('datasets/StripeSpotter')
transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
dataset = WildlifeDataset(metadata.df, metadata.root, transform=transform)
```

Optionally, split metadata into subsets. In this example, query is first 100 images and rest are in database.

```Python
database = WildlifeDataset(metadata.df.iloc[100:,:], metadata.root, transform=transform)
query = WildlifeDataset(metadata.df.iloc[:100,:], metadata.root, transform=transform)
```

### 2. Extract features
Extract features using MegaDescriptor Tiny, downloaded from HuggingFace hub.

```Python
from wildlife_tools.features import DeepFeatures

name = 'hf-hub:BVRA/MegaDescriptor-T-224'
extractor = DeepFeatures(timm.create_model(name, num_classes=0, pretrained=True))
query, database = extractor(query), extractor(database)
```

### 3. Calculate similarity
Calculate cosine similarity between query and database deep features.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)
```


### 4. Evaluate
Use the cosine similarity in nearest neigbour classifier and get predictions.

```Python
classifier = KnnClassifier(k=1, database_labels=database.labels_string)
predictions = classifier(similarity['cosine'])
```


## Installation

```shell
pip install wildlife-tools
```


Following dependencies were used:

```
torch==2.0.1
pytorch-metric-learning==1.6.0
faiss-gpu==1.7.2
pycocotools==2.0.4
timm==0.9.2
```
