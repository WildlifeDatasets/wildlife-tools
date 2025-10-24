# WildFusion - calibrated score fusion

The `similarity.wildfusion` module provides a tools to combine any set of similarity scores using score calibration. For example, cosine similarity between deep features outputs scores in the [-1, 1] interval, while scores obtained using local feature matching range from 0 to infinity. To combine them, calibration is used to convert any raw similarity score into a probability that two images represent the same identity.

This functionality is implemented using `WildFusion` class, which uses multiple `SimilarityPipeline` objects as building blocks, which implements pipeline of matching and calculating calibrated similarity scores. In addition, `WildFusion` class allows significant speeds up of calculation of matching scores.






## Examples
### Example - SimilarityPipeline

We use LightGlue matching with SuperPoint descriptors and keypoints extracted from images resized to
 512x512. The scores are calibrated using isotonic regression.

```Python
import timm
import torchvision.transforms as T
from wildlife_tools.features import SuperPointExtractor
from wildlife_tools.similarity import MatchLightGlue

from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.similarity.calibration import IsotonicCalibration


pipeline = SimilarityPipeline(
  matcher = MatchLightGlue(features='superpoint'),
  extractor = SuperPointExtractor(),
  transform = T.Compose([
      T.Resize([512, 512]),
      T.ToTensor()
  ]),
  calibration = IsotonicCalibration()
),
pipeline.fit_calibration(calibration_dataset1, calibration_dataset2)
scores = pipeline(query, database)
```




### Example - WildFusion


```python
import timm
import torchvision.transforms as T
from wildlife_tools.features import *
from wildlife_tools.similarity import CosineSimilarity, MatchLOFTR, MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.calibration import IsotonicCalibration


matchers = [

    SimilarityPipeline(
        matcher = MatchLightGlue(features='superpoint'),
        extractor = SuperPointExtractor(),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = IsotonicCalibration()
    ),

    SimilarityPipeline(
        matcher = MatchLightGlue(features='aliked'),
        extractor = AlikedExtractor(),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = IsotonicCalibration()
    ),

    SimilarityPipeline(
        matcher = MatchLightGlue(features='disk'),
        extractor = DiskExtractor(),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = IsotonicCalibration()
    ),

    SimilarityPipeline(
        matcher = MatchLightGlue(features='sift'),
        extractor = SiftExtractor(),
        transform = T.Compose([
            T.Resize([512, 512]),
            T.ToTensor()
        ]),
        calibration = IsotonicCalibration()
    ),

    SimilarityPipeline(
        matcher = MatchLOFTR(pretrained='outdoor'),
        extractor = None,
        transform = T.Compose([
            T.Resize([512, 512]),
            T.Grayscale(),
            T.ToTensor(),
        ]),
        calibration = IsotonicCalibration()
    ),

    SimilarityPipeline(
        matcher = CosineSimilarity(),
        extractor = DeepFeatures(
            model = timm.create_model(
              'hf-hub:BVRA/wildlife-mega-L-384',
              num_classes=0,
              pretrained=True
              )
        ),
        transform = T.Compose([
            T.Resize(size=(384, 384)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        calibration = IsotonicCalibration()
    ),
]

wildfusion = WildFusion(calibrated_matchers = matchers)
wildfusion.fit_calibration(calibration_dataset1, calibration_dataset2)
similarity = wildfusion(query, database)
```




### Example - WildFusion with shortlist
Cosine similarity of MegaDescriptor features is used to construct the shortlist. Then, after 
calibration, WildFusion is run with a budget of 100 score calculations per query image.


```python
priority_matcher =  SimilarityPipeline(
    matcher = CosineSimilarity(),
    extractor = DeepFeatures(
        model = timm.create_model(
        'hf-hub:BVRA/wildlife-mega-L-384',
        num_classes=0,
        pretrained=True
        )
    ),
    transform = T.Compose([
        T.Resize(size=(384, 384)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]),
)

wildfusion = WildFusion(calibrated_matchers = matchers)
wildfusion.fit_calibration(calibration_dataset1, calibration_dataset2)
similarity = wildfusion(query, database, B=100)

```


