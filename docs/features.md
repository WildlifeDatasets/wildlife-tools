# Feature extraction
Feature extractors offers a standardized way to extract features from instances of the `WildlifeDataset`.

Feature extractors, implemented as classes, can be created with specific arguments that define the extraction properties. After instantiation, the extractor functions as a callable, requiring only a single argument—the `WildlifeDataset` instance. The specific output type and shape vary based on the chosen feature extractor. In general, the output is iterable, with the first dimension corresponding to the size of the `WildlifeDataset` input.

## Deep features


The `DeepFeatures` extractor operates by extracting features through the forward pass of a PyTorch model. The output is a 2D array, where the rows represent images, and the columns correspond to the embedding dimensions. The size of the columns is determined by the output size of the model performing the feature extraction.

### Example
The term `dataset` refers to any instance of WildlifeDataset with transforms that convert it into a tensor with the appropriate shape.

```Python
import timm
from wildlife_tools.features import DeepFeatures

backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
extractor = DeepFeatures(backbone, device='cuda')
features = extractor(dataset)
```

### Reference
::: features.deep.DeepFeatures
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false



## SIFT features
The `SIFTFeatures` extractor retrieves a set of SIFT descriptors for each provided image. The output is a list with a length of `n_inputs`, containing arrays. These arrays are 2D with a shape of `n_descriptors` x `128`, where the value of `n_descriptors` depends on the number of SIFT descriptors extracted for the specific image. If one or less descriptors are extracted, the value is None.  The SIFT implementation from OpenCV is used.

### Example
The term `dataset` refers to any instance of WildlifeDataset with transforms that convert it into grayscale PIL image.

```Python
from wildlife_tools.features import SIFTFeatures

extractor = SIFTFeatures()
features = extractor(dataset)
```


### Reference
::: features.sift.SIFTFeatures
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false



## Data to memory

The `DataToMemory` extractor loads the `WildlifeDataset` into memory. This is particularly usefull for the `LoftrMatcher`, which operates directly with image tensors. While it is feasible to directly use the `WildlifeDataset` and load images from storage dynamically, the `LoftrMatcher` lacks a loading buffer. Consequently, loading images on the fly could become a significant bottleneck, especially when matching all query-database pairs, involving `n_query` x `n_database` image loads.

::: features.memory.DataToMemory
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false