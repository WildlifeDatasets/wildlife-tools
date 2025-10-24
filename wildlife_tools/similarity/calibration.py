import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class LogisticCalibration:
    """Performs logistic regression calibration."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, scores: np.ndarray, hits: np.ndarray):
        """
        Fit the logistic regression model to calibrate raw scores.

        Args:
            scores (np.ndarray): Raw uncalibrated scores.
            hits (np.ndarray): Ground truth binary labels.
        """

        self.model.fit(np.atleast_2d(scores).T, hits)

    def predict(self, scores: np.ndarray):
        """
        Predict calibrated scores using a fitted calibration model.

        Args:
            scores (np.ndarray): Raw uncalibrated scores.

        Returns:
            prediction (np.ndarray): Calibrated scores.
        """

        return self.model.predict_proba(np.atleast_2d(scores).T)[:, 1]


class IsotonicCalibration:
    """
    Performs isotonic regression calibration for ranking.

    Compared to standard isotonic regression, this implementation uses spline interpolation
    to ensure that the calibration curve is strictly increasing, which is necessary for ranking.
    """

    def __init__(self, interpolate: bool = True, strict: bool = True):
        """
        Args:
            interpolate (bool): If True, use spline interpolation for calibration.
            strict (bool): If True, apply strict adjustment to predictions.
        """

        self.calibration = IsotonicRegression(out_of_bounds="clip")
        self.spline = None
        self.strict = strict
        self.interpolate = interpolate
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def fit(self, scores: np.ndarray, hits: np.ndarray):
        """Fit the isotonic regression model to calibrate the scores.

        Args:
            scores (np.ndarray): Raw uncalibrated scores.
            hits (np.ndarray): Ground truth binary labels.
        """

        x = scores
        y = hits
        self.x_min, self.x_max = np.max(x), np.min(x)
        self.y_min, self.y_max = np.max(y), np.min(y)
        self.calibration.fit(x, y)

        if self.interpolate:
            bin_centers = self.calibration.f_.x
            x_new = (bin_centers[:-1] + bin_centers[1:]) / 2
            x_new = np.concatenate([x_new, [self.x_min, self.x_max]])
            x_new = np.sort(x_new)
            y_new = self.calibration.predict(x_new)
            self.spline = PchipInterpolator(x_new, y_new)

    def predict(self, scores: np.ndarray):
        """
        Predict calibrated scores using a fitted calibration model.

        Args:
            scores (np.ndarray): Raw uncalibrated scores.

        Returns:
            calibrated_scores (np.ndarray): Calibrated scores.
        """
        x = np.float64(scores)

        if self.interpolate:
            y = self.spline(x)
            y = np.where(x < self.x_max, self.y_max, y)
            y = np.where(x > self.x_min, self.y_min, y)
        else:
            y = self.calibration.predict(x)

        if self.strict:
            y = y + x * np.finfo(np.float64).eps
        return y


def reliability_diagram(
    scores: np.ndarray,
    hits: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    skip_plot: bool = False,
    num_bins: int = 10,
    title: str = "Reliability Diagram",
):
    """
    Calculates ECE (Expected calibration error) and plots reliability diagram for a given set of
    scores and hits.

    Args:
        scores (np.ndarray): Raw uncalibrated scores.
        hits (np.ndarray): Ground truth binary labels.
        ax (matplotlib.axes.Axes, optional): Axes to plot the diagram on. If None, a new figure is created.
        skip_plot (bool, optional): If True, only return ECE value.
        num_bins (int, optional): Number of bins to divide the scores into.
        title (str, optional): Title of the plot.

    Returns:
        ece (float): Expected Calibration Error.
    """

    df = pd.DataFrame({"score": scores, "hits": hits})
    bins = np.linspace(0, 1, num_bins + 1)
    df_bins = pd.cut(df["score"], bins)
    results = df.groupby(df_bins)["hits"].mean()
    sizes = df.groupby(df_bins)["hits"].size()
    x_values = bins[:-1] + (bins[1] - bins[0]) / 2

    # Plot calibration diagram as a bar plot
    ece = np.sum(sizes * np.abs(results.values - x_values)) / np.sum(sizes)

    if skip_plot:
        return ece

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_values, results.values, align="center", width=0.08, color="skyblue")

    for bar, value, size in zip(bars, results.values, sizes):
        height = bar.get_height()
        if pd.isnull(height):
            height = 0
        if pd.isna(value):
            print_value = "NaN"
        else:
            print_value = f"{value:.2f}"
        x_pos = bar.get_x() + bar.get_width() / 2
        ax.text(x_pos, height + 0.05, print_value, ha="center", va="bottom")
        ax.text(x_pos, 0.03, size, ha="center", va="top")

    ax.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return ece
