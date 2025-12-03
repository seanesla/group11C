"""
Validation metrics for ML model evaluation.

Provides bootstrap confidence intervals and probability calibration utilities
for classifier and regressor models.
"""

import logging
from typing import Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any metric.

    Args:
        y_true: Ground truth values
        y_pred: Predictions (labels, probabilities, or continuous - passed directly to metric_fn)
        metric_fn: Function(y_true, y_pred) -> float. No assumptions made about y_pred type.
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    point_estimate = float(metric_fn(y_true, y_pred))

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, size=n_samples)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|bin_accuracy - bin_confidence| * bin_weight)

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        bin_count = np.sum(mask)
        if bin_count == 0:
            continue

        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(y_prob[mask])
        bin_weight = bin_count / len(y_true)

        ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, list]:
    """
    Compute data for reliability diagram visualization.

    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins

    Returns:
        Dict with keys: bin_edges, mean_predicted_prob, fraction_of_positives, bin_counts
        All values are plain Python lists (JSON-serializable).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        bin_count = int(np.sum(mask))
        counts.append(bin_count)

        if bin_count == 0:
            mean_predicted.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
            fraction_positive.append(0.0)
        else:
            mean_predicted.append(float(np.mean(y_prob[mask])))
            fraction_positive.append(float(np.mean(y_true[mask])))

    return {
        'bin_edges': [float(x) for x in bin_edges],
        'mean_predicted_prob': mean_predicted,
        'fraction_of_positives': fraction_positive,
        'bin_counts': counts
    }
