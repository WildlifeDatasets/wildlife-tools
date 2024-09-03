import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator


class LogisticCalibration:
    """Performs logistic regression calibration with optional interpolation. """

    def __init__(self):
        self.model = LogisticRegression()


    def fit(self, scores, hits):
        """Fit the logistic regression model to calibrate scores.

        Args:
            x (array-like): Raw uncalibrated scores.
            y (array-like): Ground truth binary labels.

        Returns:
            None
        """
        self.model.fit(np.atleast_2d(scores).T, hits)


    def predict(self, scores):
        """Predict new data using the calibrated model.

        Args:
            x (array-like): Raw uncalibrated scores.

        Returns:
            array-like: Calibrated scores.
        """
        return self.model.predict_proba(np.atleast_2d(scores).T)[:, 1]



class IsotonicCalibration:
    """Performs isotonic regression calibration with optional interpolation.
    
    Example:
        calibration = IsotonicCalibration()
        calibration.fit([0, 0.5, 1], [0, 1, 1])
        calibration.predict([0, 0.25, 0.8])
        >>> array([0. , 0.5, 1. ])
    """

    def __init__(self, interpolate=True, strict=True):
        """Initialize the IsotonicCalibration class.

        Args:
            interpolate (bool): If True, use spline interpolation for calibration. Defaults to True.
            strict (bool): If True, apply strict adjustment to predictions. 
        """
        self.calibration = IsotonicRegression(out_of_bounds='clip')
        self.spline = None
        self.strict = strict
        self.interpolate = interpolate
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None


    def fit(self, x, y):
        """Fit the isotonic regression model to calibrate the scores.

        Args:
            x (array-like): Raw uncalibrated scores.
            y (array-like): Ground truth binary labels.

        Returns:
            None
        """
        self.x_min, self.x_max = np.max(x), np.min(x)
        self.y_min, self.y_max = np.max(y), np.min(y)
        self.calibration.fit(x, y)

        if self.interpolate:
            bin_centers = self.calibration.f_.x
            x_new = (bin_centers[:-1] + bin_centers[1:]) / 2
            x_new = np.concatenate([x_new,  [self.x_min, self.x_max]])
            x_new = np.sort(x_new)
            y_new = self.calibration.predict(x_new)
            self.spline = PchipInterpolator(x_new, y_new)


    def predict(self, x):
        """Predict new data using the calibrated model.

        Args:
            x (array-like): Raw uncalibrated scores.

        Returns:
            array-like: Calibrated scores.
        """
        x = np.float64(x)

        if self.interpolate:
            y = self.spline(x)
            y = np.where(x < self.x_max, self.y_max, y)
            y = np.where(x > self.x_min, self.y_min, y)
        else:
            y = self.calibration.predict(x)

        if self.strict:
            y = y + x * np.finfo(np.float64).eps
        return y


def reliability_diagram(score, hits, ax=None, skip_plot=False, num_bins=10, title='Reliability Diagram'):
    """Plot reliability diagram for a given set of scores and hits."""

    df = pd.DataFrame({'score': score, 'hits': hits})
    bins = np.linspace(0, 1, num_bins+1)
    df_bins = pd.cut(df['score'], bins)
    results = df.groupby(df_bins)['hits'].mean()
    sizes = df.groupby(df_bins)['hits'].size()
    x_values = bins[:-1] + (bins[1] - bins[0]) / 2

    # Plot calibration diagram as a bar plot
    ece = np.sum(sizes * np.abs(results.values - x_values)) / np.sum(sizes)

    if skip_plot:
        return ece

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_values, results.values, align='center', width=0.08, color='skyblue')
    
    for bar, value, size in zip(bars, results.values, sizes):
        height = bar.get_height()
        if pd.isnull(height):
            height = 0
        if pd.isna(value):
            print_value = 'NaN'
        else:
            print_value = f'{value:.2f}'
        x_pos = bar.get_x() + bar.get_width() / 2
        ax.text(x_pos, height+0.05, print_value, ha='center', va='bottom')
        ax.text(x_pos, 0.03, size, ha='center', va='top')

    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return ece
