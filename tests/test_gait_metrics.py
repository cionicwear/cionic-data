"""
Test suite for gait_metrics.py module.

This module contains comprehensive pytest tests for gait_metrics.py.
The tests cover various scenarios including normal operations and edge cases.

Usage:
    Run all tests:
        $ pytest test_gait_metrics.py

    Run with verbose output:
        $ pytest -v test_gait_metrics.py

    Run specific test class:
        $ pytest test_gait_metrics.py::TestComputePeakTroughDifference

    Run specific test method:
        $ pytest test_gait_metrics.py::TestComputePeakTroughDifference::\
            test_basic_functionality
"""

import numpy as np

from cionic.gait_metrics import compute_peak_trough_difference


class TestComputePeakTroughDifference:
    """Test compute_peak_trough_difference function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple sinusoidal data."""
        # Create a simple signal with clear peaks and troughs
        # Peaks at indices 10, 30, 50 with values 5, 5, 5
        # Troughs at indices 20, 40 with values -3, -3
        n_samples = 60
        elapsed_s = np.linspace(0, 10, n_samples)
        degrees = np.zeros(n_samples)

        # Create peaks and troughs
        degrees[10] = 5
        degrees[30] = 5
        degrees[50] = 5
        degrees[20] = -3
        degrees[40] = -3

        # Smooth the signal a bit
        for i in range(1, n_samples - 1):
            if degrees[i] == 0:
                degrees[i] = (degrees[i - 1] + degrees[i + 1]) / 2

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        result = compute_peak_trough_difference(data, component='degrees')

        # Average peaks: (5 + 5 + 5) / 3 = 5
        # Average troughs: (-3 + -3) / 2 = -3
        # Difference: 5 - (-3) = 8
        assert result is not None
        assert abs(result - 8.0) < 1.0  # Allow some tolerance due to peak detection

    def test_empty_data(self):
        """Test with empty data array (should return None)."""
        data = np.recarray((0,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])

        result = compute_peak_trough_difference(data, component='degrees')
        assert result is None

    def test_no_peaks_found(self):
        """Test when no peaks are found (should return None)."""
        # Create a monotonically decreasing signal (no peaks)
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)
        degrees = np.linspace(10, 0, n_samples)  # Decreasing signal

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        result = compute_peak_trough_difference(data, component='degrees')
        assert result is None

    def test_no_troughs_found(self):
        """Test when no troughs are found (should return None)."""
        # Create a monotonically increasing signal (no troughs)
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)
        degrees = np.linspace(0, 10, n_samples)  # Increasing signal

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        result = compute_peak_trough_difference(data, component='degrees')
        assert result is None

    def test_peak_kwargs_height(self):
        """Test with peak_kwargs height threshold too high (should return None)."""
        # Create signal with peaks at height 5 and troughs at height -3
        n_samples = 200
        elapsed_s = np.linspace(0, 20, n_samples)
        degrees = (
            np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 4
        )  # peak height 4, trough height -4
        degrees += 1  # shift: peaks at 5, troughs at -3

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        # Height threshold too high (peaks are at ~5, threshold is 10)
        peak_kwargs_too_high = {'height': 10}
        result = compute_peak_trough_difference(
            data, component='degrees', peak_kwargs=peak_kwargs_too_high
        )
        assert result is None, "Should return None when height threshold is too high"

    def test_peak_kwargs_distance(self):
        """Test with peak_kwargs distances."""
        # Create a signal with interspersed peaks and troughs (alternating)
        # to test distance parameter. This is more realistic and ensures
        # proper detection
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)

        # Peaks at indices 10, 20, 30, 40
        # Troughs at indices 15, 25, 35, 45
        peak_values = [5.0, 6.0, 7.0, 8.0]
        peak_indices = [10, 20, 30, 40]

        trough_values = [-3.0, -4.0, -5.0, -6.0]
        trough_indices = [15, 25, 35, 45]

        # Create smooth signal with simple linear interpolation
        # Start below first peak height so find_peaks can detect it as a local maximum
        all_points = [0]  # Start at index 0
        all_values = [0.0]  # Start below first peak (which is 5.0)

        # Combine peaks and troughs in alternating order
        for i in range(len(peak_indices)):
            all_points.append(peak_indices[i])
            all_values.append(peak_values[i])
            if i < len(trough_indices):
                all_points.append(trough_indices[i])
                all_values.append(trough_values[i])

        # End below last trough so it can be detected as a local minimum
        all_points.append(n_samples - 1)
        all_values.append(0.0)

        # Interpolate for all indices
        degrees = np.interp(np.arange(n_samples), all_points, all_values)

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        # First, test with small distance that allows all peaks/troughs
        # With distance=5, all 4 peaks and 4 troughs should be found
        result_small = compute_peak_trough_difference(
            data, component='degrees', peak_kwargs={'distance': 5}
        )
        assert result_small is not None
        # Average of 4 peaks: (5.0 + 6.0 + 7.0 + 8.0) / 4 = 6.5
        # Average of 4 troughs: (-3.0 + -4.0 + -5.0 + -6.0) / 4 = -4.5
        # Difference = 6.5 - (-4.5) = 11.0
        assert abs(result_small - 11.0) < 0.001

        # Now test with distance large: distance=15
        # Peaks are spaced 10 apart, so only every other peak will be kept
        # - higher magnituge preferred by find_peaks --> 6.0/8.0 to be kept.
        # Troughs are spaced 10 apart, so only every other trough will be kept
        # - higher magnitude preferred by find_peaks --> -4.0/-6.0 to be kept.
        result_large = compute_peak_trough_difference(
            data, component='degrees', peak_kwargs={'distance': 15}
        )
        assert (
            result_large is not None
        ), "Should return valid result with large distance"
        assert isinstance(result_large, float), "Result should be a float"
        # With (6.0, 8.0) and (-4.0, -6.0), avg_peaks=7.0, avg_troughs=-5.0, diff=12.0
        assert abs(result_large - 12.0) < 0.001, "Result should be close to 12.0"

    def test_different_component_name(self):
        """Test with different component name."""
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)
        radians = np.sin(np.linspace(0, 2 * np.pi, n_samples))

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('radians', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['radians'] = radians

        result = compute_peak_trough_difference(data, component='radians')
        assert result is not None
        assert isinstance(result, float)

    def test_known_peak_trough_values(self):
        """Test with known peak and trough values for exact calculation."""
        # Create a simple signal with exactly 2 peaks and 2 troughs
        n_samples = 200
        elapsed_s = np.linspace(0, 20, n_samples)

        # Set specific peak and trough values
        # Peaks at indices 50 and 150
        # Troughs at indices 100 and 180
        # Use np.interp for smooth transitions
        all_points = [0, 50, 100, 150, 180, n_samples - 1]
        all_values = [-2.0, 10.0, -5.0, 8.0, -3.0, 0.0]
        degrees = np.interp(np.arange(n_samples), all_points, all_values)

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        # Use distance that ensures we capture the peaks/troughs
        peak_kwargs = {'height': 0, 'distance': 40}
        result = compute_peak_trough_difference(
            data, component='degrees', peak_kwargs=peak_kwargs
        )

        assert result is not None
        # Average peaks: (10 + 8) / 2 = 9
        # Average troughs: (-5 + -3) / 2 = -4
        # Difference: 9 - (-4) = 13
        # Allow some tolerance for peak detection
        assert abs(result - 13.0) < 0.001

    def test_all_negative_values(self):
        """Test with all negative values.

        Peaks and troughs should still be detected.
        """
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)
        # Create signal that oscillates but stays negative
        degrees = -5 + np.sin(np.linspace(0, 4 * np.pi, n_samples)) * 2

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        result = compute_peak_trough_difference(data, component='degrees')

        assert result is not None
        assert isinstance(result, float)

    def test_single_peak_single_trough(self):
        """Test with exactly one peak and one trough."""
        n_samples = 100
        elapsed_s = np.linspace(0, 10, n_samples)
        degrees = np.zeros(n_samples)

        # Single peak at index 30, single trough at index 70
        degrees[30] = 10.0
        degrees[70] = -5.0

        # Create smooth transitions
        for i in range(n_samples):
            if i < 30:
                degrees[i] = -2.0 + (i / 30) * 12.0
            elif i < 70:
                degrees[i] = 10.0 - ((i - 30) / 40) * 15.0
            else:
                degrees[i] = -5.0 + ((i - 70) / 30) * 3.0

        data = np.recarray((n_samples,), dtype=[('elapsed_s', 'f8'), ('degrees', 'f8')])
        data['elapsed_s'] = elapsed_s
        data['degrees'] = degrees

        peak_kwargs = {'height': 0, 'distance': 30}
        result = compute_peak_trough_difference(
            data, component='degrees', peak_kwargs=peak_kwargs
        )

        assert result is not None
        # Should be approximately 10 - (-5) = 15
        assert abs(result - 15.0) < 0.001
