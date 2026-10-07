# Caching

Feature extraction and pairwise matching are often the slowest parts of the pipeline. Feature extractors and pairwise matchers can therefore store their results in a cache on disk. When they are run again, results already in the cache are loaded and only the missing ones are computed. This makes it possible to rerun experiments cheaply, to resume interrupted runs and to extend the dataset with new images without recomputing the old ones.

Caching is supported by:

- deep feature extractors: `DeepFeatures`, `ClipFeatures`, `DinoFeatures`,
- local feature extractors: `AlikedExtractor`, `DiskExtractor`, `SiftExtractor`, `SuperPointExtractor`,
- pairwise matchers: `MatchLightGlue`, `MatchLOFTR`.

Caching is disabled by default. It is enabled by setting `cache_path`.

## How it works

- **Storage.** The cache is an [LMDB](https://lmdb.readthedocs.io/) database stored in the directory `cache_path`. It is created on the first use. To clear the cache, delete the directory.
- **Keys.** Features are stored under the `image_id` column of the dataset metadata. Matches are stored under the pair of `image_id`s of the query and database images. The order matters, so matching `(query, database)` and `(database, query)` are cached separately.
- **Config check.** On the first use, the cache stores a config describing what it contains (the class, the model configuration and `config_tag`). On every later use, the current config is compared with the stored one, and a `ValueError` is raised if they differ. This prevents mixing results of different models in one cache.

## Rules for using the cache

- **The metadata must contain the `image_id` column.** Datasets from the [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets) library have it.
- **Use a separate cache for each dataset.** Entries are identified only by `image_id`, which is unique within a dataset but not across datasets. Subsets of the same dataset (such as query and database) can share a cache. A simple way to follow this rule is to include the dataset name in `cache_path` (for example `cache/<dataset>/<model>`).
- **Use one `cache_path` for each configuration.** For example, SuperPoint and ALIKED features need separate caches, as do LightGlue matchers for different features.
- **Encode everything not detected automatically in `config_tag`.** The config check covers the model, but not the image transform. For matchers, it also does not cover the feature extractor that produced the input features. Use `config_tag` to describe them (for example `"resize512"` or `"superpoint_resize512"`). Changing the transform or the extractor then requires changing the tag, which in turn requires a new `cache_path`.

## Examples

### Feature extraction

```Python
import timm
from wildlife_tools.features import DeepFeatures, SuperPointExtractor

backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True)
extractor = DeepFeatures(
    backbone,
    device="cuda",
    cache_path="cache/MacaqueFaces/megadescriptor",
    config_tag="resize384",
)
features = extractor(dataset)  # computes all features
features = extractor(dataset)  # loads all features from the cache

extractor = SuperPointExtractor(
    num_workers=4,
    cache_path="cache/MacaqueFaces/superpoint",
    config_tag="resize512",
)
features = extractor(dataset)
```

### Matching

```Python
from wildlife_tools.similarity import MatchLightGlue

matcher = MatchLightGlue(
    features="superpoint",
    cache_path="cache/MacaqueFaces/lightglue_superpoint",
    config_tag="superpoint_resize512",
)
scores = matcher(query, database)
```

The matcher caches raw matches, so the same cache can be used with different collectors. If only the scores are needed, `cache_scores_only=True` drops the keypoints and makes the cache much smaller. Collectors that need keypoints (such as `CollectCountsRansac`) cannot be used with such a cache.

### WildFusion

`WildFusion` does not have its own cache, but each `SimilarityPipeline` uses the caches of its extractor and matcher. Since `WildFusion` with a shortlist computes only the selected pairs, later runs with a larger budget compute only the newly selected ones.

```Python
import torchvision.transforms as T
from wildlife_tools.features import SuperPointExtractor
from wildlife_tools.similarity import MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.wildfusion import SimilarityPipeline

pipeline = SimilarityPipeline(
    matcher=MatchLightGlue(
        features="superpoint",
        cache_path="cache/MacaqueFaces/lightglue_superpoint",
        config_tag="superpoint_resize512",
    ),
    extractor=SuperPointExtractor(
        num_workers=4,
        cache_path="cache/MacaqueFaces/superpoint",
        config_tag="resize512",
    ),
    transform=T.Compose([
        T.Resize([512, 512]),
        T.ToTensor(),
    ]),
    calibration=IsotonicCalibration(),
)
```
