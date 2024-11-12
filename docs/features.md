# Feature extraction
Feature extractors offers a standardized way to extract features from instances of the `WildlifeDataset`.

Feature extractors, implemented as classes, can be created with specific arguments that define the extraction properties. After instantiation, the extractor functions as a callable, requiring only a single argument—the `WildlifeDataset` instance. The specific output type and shape vary based on the chosen feature extractor. Output is `FeatureDataset` instance.


::: features.deep
    options:
      show_root_heading: true
      heading_level: 2


::: features.local
    options:
      show_root_heading: true
      heading_level: 2


::: features.memory
    options:
      show_root_heading: true
      heading_level: 2


## Examples

### Example - SuperPoint features

```Python
from wildlife_tools.features.local import SuperPointExtractor

extractor = SuperPointExtractor(backend='opencv', detection_threshold=0.0, force_num_keypoints=True, max_num_keypoints=256)
features = extractor(dataset)
```


### Example - Deep features

```Python
import timm
from wildlife_tools.features.deep import DeepFeatures

backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
extractor = DeepFeatures(backbone, device='cuda')
features = extractor(dataset)
```