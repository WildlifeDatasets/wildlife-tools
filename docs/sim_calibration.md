# Similarity scores calibration
The `similarity.calibration` module offers tools to improve the interpretability and utility of similarity scores by calibrating them. Calibration allows similarity scores to be interpreted as probabilities, making them suitable for confidence assessments and thresholding. This also enables the effective ensemble of multiple scores by mapping them onto a common probabilistic scale.

- Calibration Methods:
    - `LogisticCalibration`: Uses logistic regression to map similarity scores to probabilities, providing a smooth and parametric approach to calibration.
    - `IsotonicCalibration`: A non-parametric approach that fits isotonic regression. Conceptually similar to score binning.
- Visualization:
    - The `reliability_diagram` function allows for visual comparison of calibrated and uncalibrated scores. This tool is helpful for assessing calibration quality, as it visualizes how well the predicted probabilities align with observed outcomes.

 
::: similarity.calibration
    options:
      show_root_heading: true
      show_bases: false
      show_root_toc_entry: false
      heading_level: 2

## Examples
Example of isotonic regression calibration.

```python
from similarity.calibration import IsotonicCalibration

calibration = IsotonicCalibration()
calibration.fit([0, 0.5, 1], [0, 1, 1])
calibration.predict([0, 0.25, 0.8])
```