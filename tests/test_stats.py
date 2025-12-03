"""
Test suite for stats.py module.

This module contains comprehensive pytest tests for stats.py.
The tests cover various scenarios including normal operations and edge cases.

Usage:
    Run all tests:
        $ pytest test_stats.py

    Run with verbose output:
        $ pytest -v test_stats.py

    Run specific test class:
        $ pytest test_stats.py::TestBasicMetrics

    Run specific test method:
        $ pytest test_stats.py::TestBasicMetrics::test_compute_mae_perfect_predictions
"""

import warnings

import numpy as np
import pytest

from cionic.stats import (
    coefficient_of_variation,
    compute_absolute_error,
    compute_absolute_percentage_error,
    compute_mae,
    compute_mape,
    compute_percentage_within_five_percent,
    compute_percentage_within_ten_percent,
    compute_percentage_within_threshold,
    compute_rmse,
)


class TestBasicMetrics:
    """Test basic error metrics (MAE, MAPE, RMSE)."""

    def test_compute_mae_perfect_predictions(self):
        """Test MAE with perfect predictions (should be 0)."""
        true = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])
        assert compute_mae(true, pred) == 0.0

    def test_compute_mae_known_values(self):
        """Test MAE with known values."""
        true = np.array([1, 2, 3, 4, 5])
        pred = np.array([2, 3, 4, 5, 6])
        expected = 1.0  # All predictions are off by 1
        assert compute_mae(true, pred) == expected

    def test_compute_mape_perfect_predictions(self):
        """Test MAPE with perfect predictions (should be 0)."""
        true = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])
        assert compute_mape(true, pred) == 0.0

    def test_compute_mape_known_values(self):
        """Test MAPE with known values."""
        true = np.array([100, 200])
        pred = np.array([110, 180])
        # Errors: 10/100 = 10%, 20/200 = 10%
        expected = 10.0
        assert compute_mape(true, pred) == expected

    def test_compute_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions (should be 0)."""
        true = np.array([1, 2, 3, 4, 5])
        pred = np.array([1, 2, 3, 4, 5])
        assert compute_rmse(true, pred) == 0.0

    def test_compute_rmse_known_values(self):
        """Test RMSE with known values."""
        true = np.array([1, 2, 3])
        pred = np.array([2, 3, 4])
        # Squared errors: [1, 1, 1], mean = 1, sqrt = 1
        expected = 1.0
        assert compute_rmse(true, pred) == expected

    def test_rmse_penalizes_large_errors(self):
        """Test that RMSE penalizes large errors more than MAE."""
        true = np.array([0, 0, 0])
        pred1 = np.array([1, 1, 1])  # Uniform errors
        pred2 = np.array([0, 0, 3])  # One large error

        mae1, mae2 = compute_mae(true, pred1), compute_mae(true, pred2)
        rmse1, rmse2 = compute_rmse(true, pred1), compute_rmse(true, pred2)

        # MAEs are equal
        assert mae1 == mae2 == 1.0
        # RMSE should be larger for pred2 due to large error
        assert rmse2 > rmse1


class TestElementWiseMetrics:
    """Test element-wise error metrics."""

    def test_compute_absolute_error(self):
        """Test absolute error computation."""
        true = np.array([1, 2, 3])
        pred = np.array([2, 1, 5])
        expected = np.array([1, 1, 2])
        result = compute_absolute_error(true, pred)
        np.testing.assert_array_equal(result, expected)

    def test_compute_absolute_percentage_error(self):
        """Test absolute percentage error computation."""
        true = np.array([100, 200])
        pred = np.array([110, 180])
        expected = np.array([10.0, 10.0])
        result = compute_absolute_percentage_error(true, pred)
        np.testing.assert_array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        true = np.array([])
        pred = np.array([])

        # Should handle empty arrays gracefully
        assert np.isnan(compute_mae(true, pred))
        assert np.isnan(compute_mape(true, pred))
        assert np.isnan(compute_rmse(true, pred))

    def test_mismatched_array_lengths(self):
        """Test error handling with mismatched array lengths."""
        true = np.array([1, 2, 3])
        pred = np.array([1, 2])

        # NumPy should handle broadcasting or raise error
        with pytest.raises((ValueError, IndexError)):
            compute_mae(true, pred)

    def test_division_by_zero_in_mape(self):
        """Test MAPE behavior when true values contain zeros."""
        true = np.array([0, 1, 2])
        pred = np.array([1, 2, 3])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            expected = 75.0  # Only the last two values contribute: (1/1 + 1/2)/2 * 100
            assert compute_mape(true, pred) == expected
            assert len(w) == 1
            assert "excludes" in str(w[0].message)
            assert "zero true values" in str(w[0].message)

    def test_single_value_arrays(self):
        """Test behavior with single-value arrays."""
        true = np.array([5])
        pred = np.array([6])

        assert compute_mae(true, pred) == 1.0
        assert compute_mape(true, pred) == 20.0
        assert compute_rmse(true, pred) == 1.0


class TestThresholdMetrics:
    """Test threshold-based metrics."""

    def test_compute_percentage_within_threshold_all_within(self):
        """Test when all predictions are within threshold."""
        true = np.array([100, 200, 300])
        pred = np.array([102, 198, 305])  # 2%, 1%, 1.67% errors
        result = compute_percentage_within_threshold(true, pred, 5.0)
        assert result == 100.0

    def test_compute_percentage_within_threshold_none_within(self):
        """Test when no predictions are within threshold."""
        true = np.array([100, 200])
        pred = np.array([120, 240])  # 20%, 20% errors
        result = compute_percentage_within_threshold(true, pred, 5.0)
        assert result == 0.0

    def test_compute_percentage_within_threshold_partial(self):
        """Test when some predictions are within threshold."""
        true = np.array([100, 100, 100, 100])
        pred = np.array([102, 108, 95, 85])  # 2%, 8%, 5%, 15% errors
        result = compute_percentage_within_threshold(true, pred, 5.0)
        assert result == 50.0  # 2 out of 4 within 5%

    def test_compute_percentage_within_five_percent(self):
        """Test convenience function for 5% threshold."""
        true = np.array([100, 100])
        pred = np.array([104, 110])  # 4%, 10% errors
        result = compute_percentage_within_five_percent(true, pred)
        assert result == 50.0

    def test_compute_percentage_within_ten_percent(self):
        """Test convenience function for 10% threshold."""
        true = np.array([100, 100])
        pred = np.array([104, 110])  # 4%, 10% errors
        result = compute_percentage_within_ten_percent(true, pred)
        assert result == 100.0


class TestCoefficientOfVariation:
    """Test coefficient of variation function with various mean values."""

    def test_coefficient_of_variation_positive_mean(self):
        """Test CV with positive mean."""
        sd = 2.0
        mean = 10.0
        result = coefficient_of_variation(sd, mean)
        assert result == 20.0  # (2/10) * 100

    def test_coefficient_of_variation_negative_mean(self):
        """Test CV with negative mean - should use absolute value."""
        sd = 2.0
        mean = -10.0
        result = coefficient_of_variation(sd, mean)
        assert result == 20.0  # (2/|-10|) * 100 = (2/10) * 100

    def test_coefficient_of_variation_zero_mean(self):
        """Test CV with zero mean - should return NaN."""
        sd = 2.0
        mean = 0.0
        result = coefficient_of_variation(sd, mean)
        assert np.isnan(result)

    def test_coefficient_of_variation_zero_sd(self):
        """Test CV with zero standard deviation."""
        sd = 0.0
        mean = 10.0
        result = coefficient_of_variation(sd, mean)
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
