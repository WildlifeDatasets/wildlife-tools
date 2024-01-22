# Similarity
`Similarity` subclasses provides a standardized method for calculating the similarity between two sets of features extracted by the same feature extractor.


Once instantiated, the `Similarity` object functions as a callable, expecting two arguments: `query_features` and `database_features`. The specific input type and shape depend on the chosen feature extractor.

The output of all `Similarity` objects is a dictionary. The keys represent properties of the similarity, such as the used threshold, while the values contain arrays with the shape `n_query` x `n_database`. In other words, each row in the array corresponds to one query image.


## CosineSimilarity

Calculates cosine similarity between query and database features. Query should be 2D array with shape `n_query` x `dim_embeddings` and database should be 2D array with shape`n_database` x `dim_embeddings`. Output is dictionary with `cosine` key and value that contains 2D array with cosine similarities.


### Example
In this context, query and database are 2D arrays of deep features.

```Python
from wildlife_tools.similarity import CosineSimilarity

similarity = CosineSimilarity()
sim = similarity(query, database)
```


### Reference
::: similarity.cosine.CosineSimilarity
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false

## MatchDescriptors

Calculates similarity between query and database as number of descriptor correspondences after filtering with ratio test.
For each descriptor in query, nearest two descriptors in database are found. If their ratio of their distance is lesser than threshold, they are considered as valid correspondence. Similarity is calculated as sum of all correspondences.

Output is dictionary with key for each threshold. Values contains 2D array with number of correspondences.


### Example
In this context, query and database are sets of SIFT descriptors with `descriptor_dim` = 128.

```Python
from wildlife_tools.similarity import MatchDescriptors

similarity = MatchDescriptors(descriptor_dim=128, thresholds=[0.8])
sim = similarity(query, database)
```


### Reference
::: similarity.descriptors.MatchDescriptors
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false




## MatchLOFTR

Uses LOFTR matching capabilities to calculate number of correspondences. Does not use descriptors and takes pair of greyscale image tensors instead. LOFTR implementation from Kornia is used.

Output is dictionary with key for each confidence threshold. Values contains corresponding 2D array with cosine similarities.


### Reference
::: similarity.loftr.MatchLOFTR
    options:
      show_symbol_type_heading: false
      show_bases: false
      show_root_toc_entry: false
