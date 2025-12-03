"""Tests for validation metrics utilities."""

import numpy as np
import pytest

# pytest runs from repo root where src is a package
from src.utils.validation_metrics import (
    bootstrap_confidence_interval,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestBootstrapCI:
    def test_returns_valid_interval(self):
        """CI lower < point < CI upper."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1])

        def accuracy(yt, yp):
            return np.mean(yt == yp)

        point, lower, upper = bootstrap_confidence_interval(y_true, y_pred, accuracy)

        assert lower <= point <= upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_reproducible_with_seed(self):
        """Same seed produces same results."""
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)

        def accuracy(yt, yp):
            return np.mean(yt == yp)

        result1 = bootstrap_confidence_interval(y_true, y_pred, accuracy, random_state=42)
        result2 = bootstrap_confidence_interval(y_true, y_pred, accuracy, random_state=42)

        assert result1 == result2


class TestECE:
    def test_perfect_calibration(self):
        """Perfect calibration should have ECE near 0."""
        # When predicted prob matches actual frequency
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 0.2  # Allow some binning noise

    def test_ece_in_valid_range(self):
        """ECE should be in [0, 1]."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1


class TestReliabilityDiagram:
    def test_returns_json_serializable(self):
        """All values should be plain Python types."""
        import json

        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        data = reliability_diagram_data(y_true, y_prob)

        # Should not raise
        json.dumps(data)

    def test_bin_counts_sum_to_n(self):
        """All samples should be assigned to exactly one bin."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        data = reliability_diagram_data(y_true, y_prob, n_bins=10)

        assert sum(data['bin_counts']) == 100
