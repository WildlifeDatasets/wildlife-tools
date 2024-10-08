

Cosine similarity between query and database deep features can be calculated using `CosineSimilarity` class. 

## CosineSimilarity

Then, query datasets features should be 2D array with shape `n_query` x `dim_embeddings` and database dataset features should be 2D array with shape`n_database` x `dim_embeddings`. Output is `n_query` x `n_database` 2D array of cosine similarity matrix.


### Reference
::: similarity.cosine.CosineSimilarity
    options:
      show_symbol_type_heading: true
      show_bases: false
      show_root_toc_entry: false


### Example
Here, `query` and `database` are instance of `FeatureDataset` with deep features.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity = CosineSimilarity()
sim = similarity(query, database)
```