# WildFusion - calibrated score fusion

`WildFusion` offers a way to combine any set of similarity scores using the mean of calibrated scores. For example, cosine similarity between deep features outputs scores in the [-1, 1] interval, while scores obtained using local feature matching range from 0 to infinity. To combine them, `WildFusion` uses calibration to convert any raw similarity score into a probability that two images represent the same identity.


## SimilarityPipeline
`SimilarityPipeline` serves as a building block for using similarity scores in `WildFusion`. 

Specifically, given two image datasets (query and database), it performs:

1. Apply image transforms.
2. Extract features for both datasets.
3. Compute similarity scores between query and database images.
4. Calibrate similarity scores.

### Reference
::: similarity.wildfusion.SimilarityPipeline
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false
      members:
        - __call__
        - fit_calibration



### Example

We use LightGlue matching with SuperPoint descriptors and keypoints extracted from images resized to
 512x512. The scores are calibrated using isotonic regression.

```Python
import timm
import torchvision.transforms as T
from features.local import SuperPointExtractor
from similarity.pairwise.lightglue import MatchLightGlue

from similarity.wildfusion import SimilarityPipeline
from similarity.calibration import IsotonicCalibration


pipeline = SimilarityPipeline(
  matcher = MatchLightGlue(features='superpoint'),
  extractor = SuperPointExtractor(),
  transform = T.Compose([
      T.Resize([512, 512]),
      T.ToTensor()
  ]),
  calibration = IsotonicCalibration()
),
pipeline.fit_calibration(calibration_dataset0, calibration_dataset0)
scores = pipeline(query, database)
```


## WildFusion

`WildFusion` uses mean of multiple calibrated `SimilarityPipeline` to calculate fused scores. 
Since many local feature matching models require deep neural network inference for each query and 
database pair, the computation quickly becomes infeasible even for moderately sized datasets.

WildFusion can be used with a limited computational budget by applying it only B times per query 
image. It uses a fast-to-compute similarity score (e.g., cosine similarity of deep features) provided 
by the priority_matcher to construct a shortlist of the most promising matches for a given query. 
Final ranking is then based on WildFusion scores calculated for the pairs in the shortlist.


### Reference
::: similarity.wildfusion.WildFusion
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false
      members:
        - __call__


### Example


```python
import timm
import torchvision.transforms as T
from features.deep import DeepFeatures
from features.local import *
from similarity.cosine import CosineSimilarity
from similarity.pairwise.loftr import MatchLOFTR
from similarity.pairwise.lightglue import MatchLightGlue

from similarity.wildfusion import SimilarityPipeline, WildFusion
from similarity.calibration import IsotonicCalibration


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




### Example - Shortlist
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


