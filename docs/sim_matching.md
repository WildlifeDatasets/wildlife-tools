# Matching based similarity scores
The `similarity.pairwise` module provides tools and methods for calculating pairwise matching similarity scores. At its core, the `MatchPairs` base class offers pairwise matching with support for batch processing, making it essential for neural network-based matching.

Specific implementations of of `MatchPairs` are:

- `MatchLightGlue`: This class uses the LightGlue model, a lightweight neural matching that uses extracted SIFT, DISK, ALIKED or SuperPoint keypoints and descriptors.
-  `MatchLOFTR`: This class uses the LOFTR (Local Feature TRansformer) model, which performs descriptor-free matching using directly pair of images.


Outputs from the matchers, such as confidence scores for local matches and keypoints, are processed using collectors from `similarity.pairwise.collectors`. In particular, the `CollectCounts` collector calculates  matching similarity scores by counting significant matches based on given confidence thresholds.

Performance consideration: For the best performance LOFTR requires higher image resolution such as 512x512. However, this makes it about 5x slower than Lighglue with default values (256 keypoints and descriptors per image). 



::: similarity.pairwise.base
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^_[^_]"
        - "!PairDataset"


::: similarity.pairwise.lightglue
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^_[^_]"


::: similarity.pairwise.loftr
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^_[^_]"
        - "!LoFTR"


::: similarity.pairwise.collectors
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^_[^_]"


## Examples


### Example - SuperGlue matching
Matches all pairs using the SuperGlue matcher with SuperPoint features and calculates similarity scores based on the count of significant matches at confidence thresholds of 0.25, 0.5, and 0.75.

```python
import torchvision.transforms as T
from wildlife_tools.features.local import SuperPointExtractor
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.pairwise.collectors import CollectCounts
from wildlife_tools.data.dataset import WildlifeDataset

transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
dataset_query = WildlifeDataset(metadata_query, transform=transform)
dataset_database = WildlifeDataset(metadata_database, transform=transform)

extractor = SuperPointExtractor()
matcher = MatchLightGlue(features='superpoint', collector=CollectCounts(thresholds=[0.25, 0.5, 0.75]))
output = matcher(extractor(dataset_query), extractor(dataset_database))

```


### Example - LOFTR matching
Matches all pairs using the LOFTR matcher and calculates similarity scores based on the count of significant matches at confidence thresholds of 0.25, 0.5, and 0.75. Note that LOFTR operates directly on image pairs and requires no feature extraction.

```python
from wildlife_tools.similarity.pairwise.loftr import MatchLOFTR
from wildlife_tools.similarity.pairwise.collectors import CollectCounts
from wildlife_tools.data.dataset import WildlifeDataset

transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
dataset_query = WildlifeDataset(metadata_query, transform=transform)
dataset_database = WildlifeDataset(metadata_database, transform=transform)

extractor = SuperPointExtractor()
matcher = MatchLightGlue(features='superpoint', collector=CollectCounts(thresholds=[0.25, 0.5, 0.75]))
output = matcher(dataset_query, dataset_database)

```