# Inference
This `inference` module uses similarity scores to perform predictions on unseen data. This includes
classification (`KnnClassifier` class) and ranking (`TopkClassifier` class) using using nearest neigbours.
Similarity scores are expected to be in the form of 2D array with shape `n_query` x `n_database`.

::: inference.classifier
    options:
      show_root_heading: true
      heading_level: 2