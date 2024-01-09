# Wildlife dataset

WildlifeDataset is a class for creating pytorch style datasets by integration of datasets provided by wildlife-datasets library. It has implemented \_\_len\_\_ and \_\_getattr\_\_ methods, which allows using pytorch dataloaders for training and inference.


## Metadata dataframe
Integral part of WildlifeDataset is metadata dataframe, which includes all information about images in the dataset.
Typical dataset from the wildlife-dataset have following metadata table:


| image_id | identity |    path          | split |           bbox | segmentation |
| --------- | ------- |     -------      |  ---- |     -------    |     -------    |
| image_1   | a       | images/a/image_1 | train | `bbox` | `compressed rle` |
| image_2   | a       | images/a/image_2 | test  | `bbox` | `compressed rle` |
| image_3   | b       | images/b/image_3 | train | `bbox` | `compressed rle` |


Columns `image_id`, `identity`, `path` are required, other columns are optional. In the table above, `bbox` is bounding box in form [x, y, width, height], and can be stored both as list or string. `compressed rle` is segmentation mask in compressed RLE format as described by [pycocotools](https://pypi.org/project/pycocotools/)

## Loading methods
If metadata table have optional `bbox` or `segmentation` columns, additional alternative image loading methods can be used.

| Argument | Loading effect |
| --------- | ------- |
| `full`        | Full image |
| `full_mask`   | Full image with redacted background |
| `full_hide`   | Full image with redacted foreground |
| `bbox`        | Bounding box cropp |
| `bbox_mask`   | Bounding box cropp with redacted background |
| `bbox_hide`   | Bounding box cropp with redacted foreground |
| `crop_black`  | Black background cropp, if there is one |

![Image loading methods](figures/loading_methods.png)



## Example

```python
from wildlife_tools.data.dataset import WildlifeDataset
import pandas as pd

metadata = pd.read_csv('ExampleDataset/metadata.csv')
dataset = WildlifeDataset(metadata, 'ExampleDataset')

# View first image in the dataset.
image, label = dataset[0]

```


## Reference
::: data.dataset.WildlifeDataset
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false