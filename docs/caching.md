# Caching

Feature extraction and pairwise matching are often the slowest parts of the pipeline. Feature extractors and pairwise matchers can therefore store their results in a cache on disk. When they are run again, results already in the cache are loaded and only the missing ones are computed. This makes it possible to rerun experiments cheaply, to resume interrupted runs and to extend the dataset with new images without recomputing the old ones.

Caching is supported by:

- deep feature extractors: `DeepFeatures`, `ClipFeatures`, `DinoFeatures`,
- local feature extractors: `AlikedExtractor`, `DiskExtractor`, `XFeatExtractor`, `DeDoDeExtractor`, `DoGHardNetExtractor`, `KeyNetAffNetHardNetExtractor`,
- pairwise matchers: `MatchLightGlue`, `MatchLOFTR`.

Caching is disabled by default. It is enabled by setting `cache_path`.

## How it works

- **Storage.** The cache is an [LMDB](https://lmdb.readthedocs.io/) database stored in the directory `cache_path`. It is created on the first use. To clear the cache, delete the directory.
- **Keys.** Features are stored under the `image_id` column of the dataset metadata. Matches are stored under the pair of `image_id`s of the query and database images. The order matters, so matching `(query, database)` and `(database, query)` are cached separately.
- **Config check.** On the first use, the cache stores a config describing what it contains. On every later use, the current config is compared with the stored one, and a `ValueError` is raised if they differ. The config always contains the class name and `config_tag`. In addition, it contains the parameters of local feature extractors (such as `max_num_keypoints`) and the model settings of matchers (such as the LightGlue features or the LoFTR weights). For deep feature extractors, the model is not part of the config.
- **Models are loaded only when needed.** If all results are already in the cache, the model is not loaded at all. Models are moved to the GPU only while they compute missing results.

## Rules for using the cache

- **The metadata must contain the `image_id` column.** Datasets from the [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets) library have it.
- **Use a separate cache for each dataset.** Entries are identified only by `image_id`, which is unique within a dataset but not across datasets. Subsets of the same dataset (such as query and database) can share a cache. A simple way to follow this rule is to include the dataset name in `cache_path` (for example `cache/<dataset>/<model>`).
- **Use one `cache_path` for each configuration.** For example, ALIKED and DISK features need separate caches, as do LightGlue matchers for different features.
- **Encode everything not detected automatically in `config_tag`.** The config check does not cover the image transform. For deep feature extractors, it also does not cover the model, and for matchers, it does not cover the feature extractor that produced the input features. Use `config_tag` to describe them (for example `"resize512"`, `"megadescriptor-L-384_resize384"` or `"aliked_resize512"`). Changing them then requires changing the tag, which in turn requires a new `cache_path`.

## Examples

### Feature extraction

```Python
import timm
from wildlife_tools.features import AlikedExtractor, DeepFeatures

backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True)
extractor = DeepFeatures(
    backbone,
    device="cuda",
    cache_path="cache/MacaqueFaces/megadescriptor",
    config_tag="megadescriptor-L-384_resize384",
)
features = extractor(dataset)  # computes all features
features = extractor(dataset)  # loads all features from the cache

extractor = AlikedExtractor(
    num_workers=4,
    cache_path="cache/MacaqueFaces/aliked",
    config_tag="resize512",
)
features = extractor(dataset)
```

### Lazy loading of deep features

Local feature extractors and matchers create their models themselves and load them only when needed. `DeepFeatures` gets an already created model, which is therefore always loaded. To load the model only when it is needed, create the extractor by `lazy_load` with a function creating the model. When all features are in the cache, the model is then never loaded.

```Python
import functools
import timm
from wildlife_tools.features import DeepFeatures

extractor = DeepFeatures.lazy_load(
    functools.partial(timm.create_model, "hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True),
    device="cuda",
    cache_path="cache/MacaqueFaces/megadescriptor",
    config_tag="megadescriptor-L-384_resize384",
)
```

`lazy_load` works also for `ClipFeatures` and `DinoFeatures`, which additionally need the `processor` argument. A `lambda` can be used instead of `functools.partial`, but then the extractor cannot be pickled.

### Matching

```Python
from wildlife_tools.similarity import MatchLightGlue

matcher = MatchLightGlue(
    features="aliked",
    cache_path="cache/MacaqueFaces/lightglue_aliked",
    config_tag="aliked_resize512",
)
scores = matcher(query, database)
```

The matcher caches raw matches, so the same cache can be used with different collectors. If only the scores are needed, `cache_scores_only=True` drops the keypoints and makes the cache much smaller. Collectors that need keypoints (such as `CollectCountsRansac`) cannot be used with such a cache.

### WildFusion

`WildFusion` does not have its own cache, but each `SimilarityPipeline` uses the caches of its extractor and matcher. Since `WildFusion` with a shortlist computes only the selected pairs, later runs with a larger budget compute only the newly selected ones.

```Python
import torchvision.transforms as T
from wildlife_tools.features import AlikedExtractor
from wildlife_tools.similarity import MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.wildfusion import SimilarityPipeline

pipeline = SimilarityPipeline(
    matcher=MatchLightGlue(
        features="aliked",
        cache_path="cache/MacaqueFaces/lightglue_aliked",
        config_tag="aliked_resize512",
    ),
    extractor=AlikedExtractor(
        num_workers=4,
        cache_path="cache/MacaqueFaces/aliked",
        config_tag="resize512",
    ),
    transform=T.Compose([
        T.Resize([512, 512]),
        T.ToTensor(),
    ]),
    calibration=IsotonicCalibration(),
)
```
