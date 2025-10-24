# Reference similarity

::: similarity.calibration
    options:
      show_root_heading: true
      show_bases: false
      show_root_toc_entry: false
      heading_level: 2

::: similarity.cosine
    options:
      show_root_heading: true
      heading_level: 2

::: similarity.wildfusion
    options:
      show_root_heading: true
      heading_level: 2
      filters:
        - "!^_[^_]"
        - "!get_feature_dataset"
        - "!get_hits"
        - "!get_priority_pairs"

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