# Global similarity scores

The `similarity.cosine` module provides tools for calculating global similarity scores between query and database. This is done using cosine similarity between fixed-length embeddings extracted by neural networks.


::: similarity.cosine
    options:
      show_root_heading: true
      heading_level: 2


## Example
`query` and `database` are instance of `FeatureDataset` with deep features.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity = CosineSimilarity()
sim = similarity(query, database)
```