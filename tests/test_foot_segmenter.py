"""
Comprehensive test suite for foot_segmenter.py module.

This module provides full test coverage for all functions in foot_segmenter.py,
including helper functions, segmentation algorithms, and utility functions.

Usage:
    pytest tests/test_foot_segmenter.py -v
    pytest tests/test_foot_segmenter.py::TestHelperFunctions -v
    pytest tests/test_foot_segmenter.py --cov=cionic.foot_segmenter --cov-report=html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cionic.foot_segmenter import (
    _create_segments_from_ic_peaks,
    _detect_rest_phases,
    _filter_data_by_time_range,
    _find_events_in_segment,
    _find_gait_event,
    _find_signal_peaks,
    _remove_short_phases,
    _validate_component_inputs_jasiewicz,
    _validate_segment_events,
    event_idxs_to_times,
    get_required_columns,
    segment_by_peaks,
    segment_cionic,
    segment_footpod,
    segment_jasiewicz,
    segment_seel,
)


# Helper functions for test assertions
def validate_event_indices(indices, data, event_name="event"):
    """Validate that all event indices are within data bounds."""
    for idx in indices:
        assert 0 <= idx < len(data), f"Invalid {event_name} index: {idx}"


def validate_segment_structure(segment, data, segment_idx=None, required_columns=None):
    """Validate that a segment has correct structure and is within data bounds."""
    prefix = f"Segment {segment_idx}" if segment_idx is not None else "Segment"
    assert isinstance(segment, pd.DataFrame), f"{prefix} is not a DataFrame"
    assert len(segment) > 0, f"{prefix} is empty"
    assert 'elapsed_s' in segment.columns, f"{prefix} missing 'elapsed_s'"
    if required_columns:
        for col in required_columns:
            assert col in segment.columns, f"{prefix} missing '{col}'"
    assert segment['elapsed_s'].min() >= 0, f"{prefix} has negative time"
    assert (
        segment['elapsed_s'].max() <= data['elapsed_s'].max()
    ), f"{prefix} extends beyond data range"


def validate_segments(segments, data, required_columns=None):
    """Validate a list of segments."""
    for i, segment in enumerate(segments):
        validate_segment_structure(
            segment, data, segment_idx=i, required_columns=required_columns
        )


def validate_segments_in_time_range(segments, time_range):
    """Validate that all segments are within the specified time range."""
    t_min, t_max = time_range
    for segment in segments:
        assert (
            segment['elapsed_s'].min() >= t_min
        ), f"Segment starts before range: {segment['elapsed_s'].min()}"
        assert (
            segment['elapsed_s'].max() <= t_max
        ), f"Segment ends after range: {segment['elapsed_s'].max()}"


def validate_lengths_match(*lists, names=None):
    """Validate that multiple lists have matching lengths."""
    if not lists:
        return
    lengths = [len(lst) for lst in lists]
    if names:
        assert all(
            length == lengths[0] for length in lengths
        ), f"Mismatched lengths: {dict(zip(names, lengths))}"
    else:
        assert all(
            length == lengths[0] for length in lengths
        ), f"Mismatched lengths: {lengths}"


def validate_events_in_segments(segments, data, event_indices, event_name="event"):
    """Validate that events are within their corresponding segments."""
    for i, (segment, event_idx) in enumerate(zip(segments, event_indices)):
        segment_start_time = segment['elapsed_s'].iloc[0]
        segment_end_time = segment['elapsed_s'].iloc[-1]
        event_time = data['elapsed_s'].iloc[event_idx]
        assert (
            segment_start_time <= event_time <= segment_end_time
        ), f"{event_name} event {i} not within segment {i}"


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_filter_data_by_time_range(self):
        """Test filtering data by time range."""
        data = pd.DataFrame(
            {'elapsed_s': [0, 1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50, 60]}
        )

        # Test normal range
        result = _filter_data_by_time_range(data, (1, 4))
        assert len(result) == 4
        assert result['elapsed_s'].min() >= 1
        assert result['elapsed_s'].max() <= 4
        assert result.index.tolist() == [0, 1, 2, 3]  # Reset index

        # Test default range
        result = _filter_data_by_time_range(data)
        assert len(result) == len(data)

        # Test empty range
        result = _filter_data_by_time_range(data, (10, 20))
        assert len(result) == 0

    def test_find_events_in_segment(self):
        """Test finding events within a segment."""
        events = [5, 10, 15, 20, 25]

        # Test events in segment
        result = _find_events_in_segment(events, segment_start=8, segment_end=22)
        assert result == [10, 15, 20]

        # Test no events in segment
        result = _find_events_in_segment(events, segment_start=30, segment_end=40)
        assert result == []

        # Test boundary conditions (exclusive)
        result = _find_events_in_segment(events, segment_start=10, segment_end=20)
        assert result == [15]  # 10 and 20 are excluded

        # Test verbose mode
        result = _find_events_in_segment(
            events, segment_start=8, segment_end=22, verbose=True, segment_idx=0
        )
        assert result == [10, 15, 20]

    def test_validate_segment_events(self):
        """Test validating events within a segment."""
        events_dict = {'IC': [5, 15, 25], 'EC': [10, 20, 30]}

        # Test valid segment (one of each) - from t=11 to t=21, IC=15, EC=20
        assert _validate_segment_events(
            events_dict, 11, 21, verbose=False, segment_idx=0
        )

        # Test invalid segment (multiple EC) - from t=8 to t=22, IC=15, EC=[10, 20]
        assert not (
            _validate_segment_events(
                events_dict,
                segment_start=8,
                segment_end=22,
                verbose=False,
                segment_idx=0,
            )
        )

        # Test invalid segment (multiple IC) - from t=4 to t=16, IC=[5, 15], EC=10
        assert not (
            _validate_segment_events(
                events_dict,
                segment_start=4,
                segment_end=16,
                verbose=False,
                segment_idx=0,
            )
        )

        # Test invalid segment (no EC) - from t=5 to t=8, IC=5, no EC
        assert not (
            _validate_segment_events(
                events_dict,
                segment_start=5,
                segment_end=8,
                verbose=False,
                segment_idx=0,
            )
        )

        # Test verbose mode
        assert not _validate_segment_events(
            events_dict, segment_start=4, segment_end=16, verbose=True, segment_idx=0
        )

    def test_validate_component_inputs(self):
        """Test component input validation."""
        # Test valid inputs
        _validate_component_inputs_jasiewicz("accel_x", "accel_y", "max", "min")

        # Test invalid component
        with pytest.raises(ValueError, match="Component must be one of"):
            _validate_component_inputs_jasiewicz("invalid", "accel_y", "max", "min")

        # Test invalid peak type
        with pytest.raises(ValueError, match="Peak type must be"):
            _validate_component_inputs_jasiewicz("accel_x", "accel_y", "invalid", "min")

        with pytest.raises(ValueError, match="Peak type must be"):
            _validate_component_inputs_jasiewicz("accel_x", "accel_y", "max", "invalid")

    def test_find_signal_peaks(self):
        """Test finding peaks in a signal."""
        # Create signal with clear peaks and troughs, each with distinct prominences
        signal = np.array([0.0, 0.1, 0.7, 0.3, 0.15, 0.5, 0.25, 0.1, 0.8, 0.05])

        # Test max peaks
        # Prominence measures how much a peak stands out from its surrounding baseline.
        # It's calculated as: min(peak - left_base, peak - right_base)
        # where:
        #   - left_base: minimum value from signal start (or last higher peak) to peak
        #   - right_base: minimum value from peak to signal end (or next higher peak)
        peaks = _find_signal_peaks(signal, peak_type="max", distance=2)
        assert len(peaks) == 3
        assert peaks.tolist() == [
            8,  # val 0.8, prom 0.75 (l_base=0.0, r_base=0.05, min(0.8, 0.75)=0.75)
            2,  # val 0.7, prom 0.60 (l_base=0.0, r_base=0.1, min(0.7, 0.6)=0.6)
            5,  # val 0.5, prom 0.35 (l_base=0.15, r_base=0.1, min(0.35, 0.4)=0.35)
        ]  # Sorted by prominence descending (0.75 > 0.6 > 0.35)

        # Test min peaks (troughs)
        # For troughs, negate signal first, then calc prominences.
        # Prominence represents how deep the trough is relative to surrounding peaks.
        peaks = _find_signal_peaks(signal, peak_type="min", distance=2)
        assert len(peaks) == 2  # doesn't include start and end of signal
        assert peaks.tolist() == [
            7,  # val 0.1, prom 0.60 (l_base=0.7, r_base=0.8, deeper trough)
            4,  # val 0.15, prom 0.35 (l_base=0.7, r_base=0.5, shallower trough)
        ]  # Sorted by prominence descending (0.6 > 0.35)

        # Test with height threshold
        peaks = _find_signal_peaks(signal, peak_type="max", height=0.5, distance=2)
        assert all(signal[p] >= 0.5 for p in peaks)

        # Test with prominence threshold
        # Only peaks with prominence >= 0.2 will be detected.
        # This filters out small/noisy peaks that don't stand out significantly.
        peaks = _find_signal_peaks(signal, peak_type="max", prominence=0.4, distance=2)
        assert len(peaks) == 2
        assert peaks.tolist() == [
            8,  # val 0.8, prom 0.75 (l_base=0.0, r_base=0.05, min(0.8, 0.75)=0.75)
            2,  # val 0.7, prom 0.60 (l_base=0.0, r_base=0.1, min(0.7, 0.6)=0.6)
        ]  # Sorted by prominence descending (0.75 > 0.6)

        # Test empty signal
        peaks = _find_signal_peaks(np.array([]), peak_type="max")
        assert len(peaks) == 0

        # Test signal with no peaks
        peaks = _find_signal_peaks(np.ones(10), peak_type="max", distance=2)
        assert len(peaks) == 0

    def test_find_gait_event(self):
        """Test finding a gait event within a time window."""
        # Create deterministic signal with clear peaks at known locations
        t = np.linspace(0, 10, 100)

        # Signal with clear max peak at index 50
        accel_x = np.zeros(100)
        accel_x[50] = 1.0  # Peak at index 50
        peak_shape = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2])
        accel_x[45:56] = np.maximum(accel_x[45:56], peak_shape)

        # Signal with clear min peak (trough) at index 50
        accel_z = np.ones(100) * 0.5
        accel_z[50] = 0.0  # Trough at index 50
        trough_shape = np.array(
            [0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3]
        )
        accel_z[45:56] = np.minimum(accel_z[45:56], trough_shape)

        data = pd.DataFrame(
            {
                'elapsed_s': t,
                'accel_x': accel_x,
                'accel_z': accel_z,
            }
        )

        # Test finding max peak - should find exactly 1 event at index 50
        event = _find_gait_event(
            data,
            component='accel_x',
            peak_type='max',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=False,
        )
        assert event is not None, "Expected an event, got None"
        assert event == 50, f"Expected event at index 50, got {event}"

        # Test finding min peak (trough) - should find exactly 1 event at index 50
        event = _find_gait_event(
            data,
            component='accel_z',
            peak_type='min',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=False,
        )
        assert event is not None, "Expected an event, got None"
        assert event == 50, f"Expected event at index 50, got {event}"

        # Test empty window - should return None
        event = _find_gait_event(
            data,
            component='accel_x',
            peak_type='max',
            window_range=(100, 200),
            time_idx=50,
            verbose=False,
        )
        assert event is None, f"Expected None for empty window, got {event}"

        # Test signal with no peaks - should return None
        data_no_peak = pd.DataFrame(
            {
                'elapsed_s': t,
                'accel_x': np.ones(100) * 0.5,  # Flat signal, no peaks
            }
        )
        event = _find_gait_event(
            data_no_peak,
            component='accel_x',
            peak_type='max',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=False,
        )
        assert event is None, f"Expected None for flat signal, got {event}"

        # Test verbose mode - should still find exactly 1 event
        event = _find_gait_event(
            data,
            component='accel_x',
            peak_type='max',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=True,
        )
        assert event is not None, "Expected an event in verbose mode, got None"
        assert event == 50, f"Expected event at index 50, got {event}"

        # Test with multiple peaks in window - should return the most prominent one
        # Create signal with 2 peaks: smaller at index 48, larger at index 52
        accel_multi = np.zeros(100)
        # First peak at index 48 (smaller, prominence ~0.3)
        accel_multi[48] = 0.5
        accel_multi[46:51] = np.maximum(
            accel_multi[46:51], np.array([0.2, 0.3, 0.5, 0.3, 0.2])
        )
        # Second peak at index 52 (larger, prominence ~0.6)
        accel_multi[52] = 0.8
        accel_multi[50:55] = np.maximum(
            accel_multi[50:55], np.array([0.2, 0.4, 0.8, 0.4, 0.2])
        )

        data_multi = pd.DataFrame(
            {
                'elapsed_s': t,
                'accel_x': accel_multi,
            }
        )

        event = _find_gait_event(
            data_multi,
            component='accel_x',
            peak_type='max',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=False,
        )
        assert event is not None, "Expected an event (most prominent), got None"
        assert event == 52, f"Expected most prominent peak at index 52, got {event}"

        # Test with multiple troughs in window - should return the most prominent one
        accel_multi_trough = np.ones(100) * 0.5
        # First trough at index 48 (shallower, prominence ~0.3)
        accel_multi_trough[48] = 0.2
        accel_multi_trough[46:51] = np.minimum(
            accel_multi_trough[46:51], np.array([0.3, 0.25, 0.2, 0.25, 0.3])
        )
        # Second trough at index 52 (deeper, prominence ~0.5)
        accel_multi_trough[52] = 0.0
        accel_multi_trough[50:55] = np.minimum(
            accel_multi_trough[50:55], np.array([0.3, 0.1, 0.0, 0.1, 0.3])
        )

        data_multi_trough = pd.DataFrame(
            {
                'elapsed_s': t,
                'accel_z': accel_multi_trough,
            }
        )

        event = _find_gait_event(
            data_multi_trough,
            component='accel_z',
            peak_type='min',
            window_range=(-0.5, 0.5),
            time_idx=50,
            verbose=False,
        )
        assert event is not None, "Expected an event (most prominent), got None"
        assert event == 52, f"Expected most prominent trough at index 52, got {event}"

    def test_create_segments_from_ic_peaks(self):
        """Test creating segments from IC peaks."""
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 10, 100),
                'roll': np.sin(np.linspace(0, 4 * np.pi, 100)),
            }
        )

        ic_peaks = [10, 30, 50, 70]
        ec_peaks = [20, 40, 60, 80]

        # Test normal case
        # Segments are created btwn consecutive IC peaks
        # With N IC peaks, there are N-1 gaps between them, so N-1 segments
        final_ec, final_ic, segments = _create_segments_from_ic_peaks(
            data, ic_peaks, ec_peaks, False
        )
        assert len(final_ec) == len(final_ic) == len(segments)
        assert len(segments) == 3  # 4 IC peaks -> 3 segments (one less)

        # Test with invalid segments (no EC in segment)
        ic_peaks = [10, 30, 50]
        ec_peaks = [5, 35, 55]  # EC before first IC, after last IC
        final_ec, final_ic, segments = _create_segments_from_ic_peaks(
            data, ic_peaks, ec_peaks, False
        )
        assert len(segments) == 1  # Only ic 30 --> ic 50 is valid

        # Test verbose mode
        final_ec, final_ic, segments = _create_segments_from_ic_peaks(
            data, ic_peaks, ec_peaks, True
        )
        assert isinstance(segments, list)

    def test_detect_rest_phases(self):
        """Test detecting rest phases with hysteresis."""
        # Create signal with rest phases
        # Signal: [0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 0.5, 0.5]
        # Threshold: 1.0, hysteresis_factor: 0.5
        # - Lower threshold: (1 - 0.5) * 1.0 = 0.5 (rest)
        # - Upper threshold: (1 + 0.5) * 1.0 = 1.5 (active)
        # - Hysteresis zone: 0.5 <= signal <= 1.5 (maintains previous state)
        signal = np.array([0.3, 0.5, 0.5, 2.0, 2.0, 2.0, 0.5, 0.5, 0.3, 0.3])
        threshold = 1.0
        hysteresis_factor = 0.5

        forward_mask, backward_mask = _detect_rest_phases(
            signal, threshold, hysteresis_factor
        )

        assert len(forward_mask) == len(backward_mask) == len(signal)
        assert np.all(forward_mask >= 0)
        assert np.all(forward_mask <= 1)
        assert np.all(backward_mask >= 0)
        assert np.all(backward_mask <= 1)

        # Verify forward mask contents
        # Indices 0-2: signal < 1.0 threshold -> rest (0)
        # Indices 3-5: signal > 1.0 threshold -> active (1)
        # Indices 6-7: signal=0.5, in hysteresis zone -> maintains previous state (1)
        # Indices 8-9: signal < 0.5 outside hysteresis zone -> rest (0)
        expected_forward = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        assert np.allclose(
            forward_mask, expected_forward
        ), f"Forward mask mismatch: got {forward_mask}, expected {expected_forward}"

        # Verify backward mask contents
        # Same behavior as fwd, but start w/ last state of fwd mask (0) and work bwd
        expected_backward = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        assert np.allclose(
            backward_mask, expected_backward
        ), f"Backward mask mismatch: got {backward_mask}, expected {expected_backward}"

        # Test edge cases
        signal = np.array([0.5])
        forward_mask, backward_mask = _detect_rest_phases(
            signal, threshold, hysteresis_factor
        )
        assert len(forward_mask) == 1

        # Test all rest (below threshold)
        # Signal values 0.5 are below lower threshold (0.5), so should be rest (0)
        signal = np.array([0.5] * 10)
        forward_mask, backward_mask = _detect_rest_phases(
            signal, threshold, hysteresis_factor
        )
        expected_rest_forward = np.zeros(10)
        expected_rest_backward = np.zeros(10)
        assert np.allclose(
            forward_mask, expected_rest_forward
        ), f"All rest forward mask mismatch: got {forward_mask}"
        assert np.allclose(
            backward_mask, expected_rest_backward
        ), f"All rest backward mask mismatch: got {backward_mask}"

        # Test all active (above threshold)
        # Signal values 2.0 are above upper threshold (1.5), so should be active (1)
        # Note: forward_mask[0] is always 0 (initialized), forward_mask[1:] should be 1
        signal = np.array([2.0] * 10)
        forward_mask, backward_mask = _detect_rest_phases(
            signal, threshold, hysteresis_factor
        )
        expected_active_fwd = np.array([0.0] + [1.0] * 9)
        expected_active_bwd = np.ones(10)  # Backward extends active phase
        assert np.allclose(
            forward_mask, expected_active_fwd
        ), f"Fwd mask mismatch: got {forward_mask}, expected {expected_active_fwd}"
        assert np.allclose(
            backward_mask, expected_active_bwd
        ), f"Bwd mask mismatch: got {backward_mask}, expected {expected_active_bwd}"

    def test_remove_short_phases(self):
        """Test removing short phases from mask."""
        # Create mask with short phases (boolean array)
        # Mask: [True, True, False, False, False, True, True, True, False, False]
        # True regions: [0-1] (length 2), [5-7] (length 3)
        # False regions: [2-4] (length 3), [8-9] (length 2)
        mask = np.array(
            [True, True, False, False, False, True, True, True, False, False],
            dtype=bool,
        )
        min_samples = 3

        result = _remove_short_phases(
            mask.copy(), min_samples
        )  # Copy to avoid modifying original

        assert len(result) == len(mask)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool  # Should remain boolean

        # Verify actual output contents
        # Removes short True phases (set to False) and short False phases (set to True)
        expected_result = np.array(
            [False, False, False, False, False, True, True, True, True, True],
            dtype=bool,
        )
        assert np.array_equal(
            result, expected_result
        ), f"Mismatch: got {result.astype(int)}, expected {expected_result.astype(int)}"

        # Test with all True - no boundaries, so nothing processed
        mask = np.array([True] * 10, dtype=bool)
        result = _remove_short_phases(mask.copy(), 5)
        assert np.all(
            result
        ), f"All True mask should remain all True, got {result.astype(int)}"
        assert result.dtype == bool

        # Test with all False - no boundaries, so nothing processed
        mask = np.array([False] * 10, dtype=bool)
        result = _remove_short_phases(mask.copy(), 5)
        assert np.all(
            ~result
        ), f"All False mask should remain all False, got {result.astype(int)}"
        assert result.dtype == bool

        # Test empty mask
        mask = np.array([], dtype=bool)
        result = _remove_short_phases(mask.copy(), 5)
        assert len(result) == 0

        # Test with phases that are exactly min_samples length
        # Mask: [True, True, True, False, False, False, True, True, True]
        # All phases have length 3 --> no short phases removed
        mask_long = np.array(
            [True, True, True, False, False, False, True, True, True], dtype=bool
        )
        result_long = _remove_short_phases(mask_long.copy(), 3)
        assert np.array_equal(
            result_long, mask_long
        ), f"Mismatch: got {result_long.astype(int)}, expected {mask_long.astype(int)}"


class TestSegmentationAlgorithms:
    """Test suite for segmentation algorithms. Checks that the algorithms run without
    significant errors. See TestIntegrationRealData for tests on a real dataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample gait data."""
        t = np.linspace(0, 10, 1000)
        data = pd.DataFrame(
            {
                'elapsed_s': t,
                'roll': np.sin(2 * np.pi * t / 2) + 0.1 * np.random.randn(len(t)),
                'accel_x': np.sin(2 * np.pi * t / 2) + 0.1 * np.random.randn(len(t)),
                'accel_y': np.cos(2 * np.pi * t / 2) + 0.1 * np.random.randn(len(t)),
                'accel_z': np.sin(2 * np.pi * t / 2) + 0.1 * np.random.randn(len(t)),
                'gyro_x': 0.5 * np.sin(2 * np.pi * t / 2)
                + 0.05 * np.random.randn(len(t)),
                'gyro_y': 0.5 * np.cos(2 * np.pi * t / 2)
                + 0.05 * np.random.randn(len(t)),
                'gyro_z': 0.5 * np.sin(2 * np.pi * t / 2)
                + 0.05 * np.random.randn(len(t)),
            }
        )
        return data

    def test_segment_by_peaks(self, sample_data):
        """Test basic peak-based segmentation."""
        peaks, troughs, segments = segment_by_peaks(
            sample_data, signal_column='roll', verbose=False
        )

        assert isinstance(peaks, list)
        assert isinstance(troughs, list)
        assert isinstance(segments, list)
        validate_lengths_match(peaks, troughs, segments)

        validate_event_indices(peaks, sample_data, "peak")
        validate_event_indices(troughs, sample_data, "trough")
        validate_segments(segments, sample_data, required_columns=['roll'])

        # Verify segments are between consecutive peaks
        if len(segments) > 0:
            for i in range(len(peaks) - 1):
                segment_start = segments[i]['elapsed_s'].iloc[0]
                segment_end = segments[i]['elapsed_s'].iloc[-1]
                peak_time = sample_data['elapsed_s'].iloc[peaks[i]]
                next_peak_time = sample_data['elapsed_s'].iloc[peaks[i + 1]]
                assert (
                    segment_start == peak_time
                ), f"Segment {i} doesn't start at peak {i}"
                assert (
                    segment_end == next_peak_time
                ), f"Segment {i} doesn't end at peak {i+1}"

        # Test with custom parameters
        peaks, troughs, segments = segment_by_peaks(
            sample_data,
            signal_column='roll',
            peak_height=0.5,
            peak_distance=50,
            verbose=False,
        )
        assert isinstance(segments, list)
        # With higher threshold, should have fewer or equal segments
        assert (
            len(segments) <= len(sample_data) // 50
        ), f"Too many segments with high threshold: {len(segments)}"

        # Test with time range
        peaks, troughs, segments = segment_by_peaks(
            sample_data, signal_column='roll', segmentation_range=(2, 8), verbose=False
        )
        assert isinstance(segments, list)
        validate_segments_in_time_range(segments, (2, 8))

        # Test with different signal column
        peaks, troughs, segments = segment_by_peaks(
            sample_data, signal_column='accel_x', verbose=False
        )
        assert isinstance(segments, list)
        validate_segments(segments, sample_data, required_columns=['accel_x'])

    def test_segment_jasiewicz(self, sample_data):
        """Test Jasiewicz segmentation algorithm."""
        ec_peaks, ic_peaks, segments, maxima, troughs = segment_jasiewicz(
            sample_data, verbose=False
        )

        assert isinstance(ec_peaks, list)
        assert isinstance(ic_peaks, list)
        assert isinstance(segments, list)
        assert isinstance(maxima, list)
        assert hasattr(troughs, '__len__')  # Can be list or array

        validate_event_indices(ec_peaks, sample_data, "EC peak")
        validate_event_indices(ic_peaks, sample_data, "IC peak")
        validate_event_indices(maxima, sample_data, "maxima")

        # May have 0 segments if no valid cycles found
        if len(segments) > 0:
            validate_lengths_match(
                ec_peaks, ic_peaks, segments, names=["EC", "IC", "segments"]
            )
            validate_segments(segments, sample_data)
            validate_events_in_segments(segments, sample_data, ic_peaks, "IC")
            validate_events_in_segments(segments, sample_data, ec_peaks, "EC")

        # Test with custom parameters
        ec_peaks, ic_peaks, segments, maxima, troughs = segment_jasiewicz(
            sample_data,
            ic_component='accel_y',
            ec_component='accel_z',
            ic_peak_type='max',
            ec_peak_type='min',
            verbose=False,
        )
        assert isinstance(segments, list)
        if len(segments) > 0:
            validate_lengths_match(
                ec_peaks, ic_peaks, segments, names=["EC", "IC", "segments"]
            )

        # Test with time range
        ec_peaks, ic_peaks, segments, maxima, troughs = segment_jasiewicz(
            sample_data, segmentation_range=(1, 9), verbose=False
        )
        assert isinstance(segments, list)
        validate_segments_in_time_range(segments, (1, 9))

        # Test invalid component
        with pytest.raises(ValueError):
            segment_jasiewicz(sample_data, ic_component='invalid', verbose=False)

    def test_segment_cionic(self, sample_data):
        """Test Cionic segmentation algorithm."""
        ec_peaks, ic_peaks, segments = segment_cionic(sample_data, verbose=False)

        assert isinstance(ec_peaks, list)
        assert isinstance(ic_peaks, list)
        assert isinstance(segments, list)
        validate_lengths_match(
            ec_peaks, ic_peaks, segments, names=["EC", "IC", "segments"]
        )

        validate_event_indices(ec_peaks, sample_data, "EC peak")
        validate_event_indices(ic_peaks, sample_data, "IC peak")
        validate_segments(segments, sample_data)
        validate_events_in_segments(segments, sample_data, ic_peaks, "IC")
        validate_events_in_segments(segments, sample_data, ec_peaks, "EC")

        # Test with custom parameters
        ec_peaks, ic_peaks, segments = segment_cionic(
            sample_data,
            ic_marker_start_fraction=0.2,
            ic_marker_end_fraction=0.8,
            verbose=False,
        )
        assert isinstance(segments, list)
        if len(segments) > 0:
            validate_lengths_match(
                ec_peaks, ic_peaks, segments, names=["EC", "IC", "segments"]
            )

        # Test invalid fraction parameters
        with pytest.raises(ValueError):
            segment_cionic(
                sample_data,
                ic_marker_start_fraction=0.8,
                ic_marker_end_fraction=0.2,  # start > end
                verbose=False,
            )

        with pytest.raises(ValueError):
            segment_cionic(
                sample_data, ic_marker_start_fraction=-0.1, verbose=False  # < 0
            )

        # Test with time range
        ec_peaks, ic_peaks, segments = segment_cionic(
            sample_data, segmentation_range=(1, 9), verbose=False
        )
        assert isinstance(segments, list)
        validate_segments_in_time_range(segments, (1, 9))

    def test_segment_seel(self, sample_data):
        """Test Seel segmentation algorithm."""
        segments, tos, ics, ff_starts, ff_ends = segment_seel(
            sample_data, verbose=False
        )

        assert isinstance(segments, list)
        assert isinstance(tos, list)
        assert isinstance(ics, list)
        assert isinstance(ff_starts, list)
        assert isinstance(ff_ends, list)

        # Verify event indices are valid; segments are non-empty/havecorrect structure
        validate_event_indices(tos, sample_data, "TO")
        validate_event_indices(ics, sample_data, "IC")
        validate_event_indices(ff_starts, sample_data, "FF start")
        validate_event_indices(ff_ends, sample_data, "FF end")
        validate_segments(segments, sample_data)

        # Verify event counts match segment count
        if len(segments) > 0:
            validate_lengths_match(tos, segments, names=["TO", "segments"])
            validate_lengths_match(ics, segments, names=["IC", "segments"])

        # Test with custom parameters
        segments, tos, ics, ff_starts, ff_ends = segment_seel(
            sample_data, accel_threshold=1.5, gyro_threshold=0.5, verbose=False
        )
        assert isinstance(segments, list)
        if len(segments) > 0:
            validate_lengths_match(tos, ics, segments, names=["TO", "IC", "segments"])

        # Test invalid jerk_window_fraction
        with pytest.raises(ValueError):
            segment_seel(sample_data, jerk_window_fraction=1.5, verbose=False)

        with pytest.raises(ValueError):
            segment_seel(sample_data, jerk_window_fraction=-0.1, verbose=False)

        # Test with time range
        segments, tos, ics, ff_starts, ff_ends = segment_seel(
            sample_data, segmentation_range=(1, 9), verbose=False
        )
        assert isinstance(segments, list)
        validate_segments_in_time_range(segments, (1, 9))

        # Test with minimal data (should handle gracefully)
        minimal_data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 1, 50),
                'accel_x': np.random.randn(50),
                'accel_y': np.random.randn(50),
                'accel_z': np.random.randn(50),
                'gyro_x': np.random.randn(50),
                'gyro_y': np.random.randn(50),
                'gyro_z': np.random.randn(50),
            }
        )
        segments, tos, ics, ff_starts, ff_ends = segment_seel(
            minimal_data, verbose=False
        )
        assert isinstance(segments, list)
        if len(segments) > 0:
            validate_lengths_match(tos, ics, segments, names=["TO", "IC", "segments"])


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_event_idxs_to_times(self):
        """Test converting event indices to times."""
        data = pd.DataFrame({'elapsed_s': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]})
        events = [2, 5, 8]

        times = event_idxs_to_times(data, events)
        assert len(times) == 3
        assert np.allclose(times, [20, 50, 80])

        # Test with time range - events must be valid indices after filtering
        # Events 2, 5 map to indices 1, 4 in filtered data
        times = event_idxs_to_times(data, [1, 4], (10, 70))
        assert len(times) == 2
        assert np.allclose(times, [20, 50])

        # Test empty events
        times = event_idxs_to_times(data, [])
        assert len(times) == 0

    def test_get_required_columns(self):
        """Test getting required columns for segmentation methods."""
        # Test peak method
        cols = get_required_columns("peak")
        assert 'elapsed_s' in cols
        assert 'roll' in cols

        # Test seel method
        cols = get_required_columns("seel")
        assert 'elapsed_s' in cols
        assert 'accel_x' in cols
        assert 'gyro_x' in cols

        # Test jasiewicz/mod-jasiewicz methods
        cols = get_required_columns("jasiewicz")
        assert 'elapsed_s' in cols
        assert 'roll' in cols
        assert 'accel_x' in cols

    def test_segment_footpod(self):
        """Test segment_footpod convenience function."""
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 10, 1000),
                'roll': np.sin(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'accel_x': np.sin(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'accel_y': np.cos(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'accel_z': np.sin(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'gyro_x': 0.5 * np.sin(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'gyro_y': 0.5 * np.cos(2 * np.pi * np.linspace(0, 10, 1000) / 2),
                'gyro_z': 0.5 * np.sin(2 * np.pi * np.linspace(0, 10, 1000) / 2),
            }
        )

        # Test all segmentation methods
        methods = ["peak", "jasiewicz", "mod-jasiewicz", "seel"]
        for method in methods:
            ic_peaks, ec_peaks = segment_footpod(data, method=method)
            assert isinstance(ic_peaks, list)
            assert isinstance(ec_peaks, list)
            validate_lengths_match(ic_peaks, ec_peaks, names=["IC", "EC"])
            validate_event_indices(ic_peaks + ec_peaks, data, "event")

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            segment_footpod(data, method="invalid")

        # Test with time range
        ic_peaks, ec_peaks = segment_footpod(
            data, method="peak", segmentation_range=(1, 9)
        )
        assert isinstance(ic_peaks, list)
        assert isinstance(ec_peaks, list)
        validate_lengths_match(ic_peaks, ec_peaks, names=["IC", "EC"])
        # Verify all event times are within the specified range
        for idx in ic_peaks + ec_peaks:
            event_time = data['elapsed_s'].iloc[idx]
            assert (
                1 <= event_time <= 9
            ), f"Event at time {event_time} outside range (1, 9)"


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        # These should handle empty data gracefully
        with pytest.raises((KeyError, IndexError)):
            segment_by_peaks(empty_df, verbose=False)

        # Test with minimal valid structure but missing required column
        minimal_df = pd.DataFrame({'elapsed_s': []})
        with pytest.raises(KeyError):
            segment_by_peaks(minimal_df, verbose=False)

        # Test with valid structure but empty
        minimal_df = pd.DataFrame({'elapsed_s': [], 'roll': []})
        _, _, segments = segment_by_peaks(minimal_df, verbose=False)
        assert len(segments) == 0

    def test_single_peak(self):
        """Test handling of data with single peak."""
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 2, 100),
                'roll': np.sin(np.linspace(0, np.pi, 100)),
            }
        )

        _, _, segments = segment_by_peaks(data, verbose=False)
        # Should handle gracefully - may have 0 segments if only one peak
        assert isinstance(segments, list)

    def test_no_peaks_detected(self):
        """Test handling when no peaks are detected."""
        # Flat signal with no peaks
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 10, 100),
                'roll': np.ones(100) * 0.05,  # Below peak threshold
            }
        )

        _, _, segments = segment_by_peaks(data, peak_height=0.5, verbose=False)
        assert len(segments) == 0

    def test_missing_columns(self):
        """Test handling of missing required columns."""
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 10, 100),
                'roll': np.sin(np.linspace(0, 4 * np.pi, 100)),
            }
        )

        # Should work for peak method
        _, _, segments = segment_by_peaks(data, verbose=False)
        assert isinstance(segments, list)

        # Should fail for methods requiring more columns
        with pytest.raises((KeyError, ValueError)):
            segment_jasiewicz(data, verbose=False)

    def test_extreme_time_ranges(self):
        """Test handling of extreme time ranges."""
        data = pd.DataFrame(
            {
                'elapsed_s': np.linspace(0, 10, 100),
                'roll': np.sin(np.linspace(0, 4 * np.pi, 100)),
            }
        )

        # Test range before data
        _, _, segments = segment_by_peaks(
            data, segmentation_range=(-10, -5), verbose=False
        )
        assert len(segments) == 0

        # Test range after data
        _, _, segments = segment_by_peaks(
            data, segmentation_range=(20, 30), verbose=False
        )
        assert len(segments) == 0

        # Test infinite range - should process all data and produce segments
        # With 2 periods of sine wave, expect 1 valid segment (btwn 2 peaks w 1 trough)
        _, _, segments = segment_by_peaks(
            data, segmentation_range=(0, np.inf), verbose=False
        )
        assert isinstance(segments, list)
        assert (
            len(segments) == 1
        ), f"Expected 1 segment for 2-period sine wave, got {len(segments)}"

    def test_very_short_segments(self):
        """Test handling of very short data segments."""
        data = pd.DataFrame({'elapsed_s': [0, 0.1], 'roll': [0, 0.1]})

        _, _, segments = segment_by_peaks(data, verbose=False)
        assert isinstance(segments, list)
        # May have 0 segments due to distance requirements


class TestIntegrationRealData:
    """Integration tests using real-world data from collection 1274.

    These tests verify that segmentation algorithms work correctly on
    real data from the validation notebook. The data is pre-processed
    and stored as fixtures to minimize repository size.
    """

    @pytest.fixture(scope="class")
    def real_foot_data(self):
        """Load pre-processed foot IMU data from collection 1274."""
        # Get path to fixture file
        test_dir = Path(__file__).parent
        fixture_path = test_dir / "fixtures" / "foot_imu_1274_9_70s.npz"

        if not fixture_path.exists():
            pytest.skip(
                f"Fixture not found: {fixture_path}\n"
                "Run scripts/generate_test_fixtures.py to generate it."
            )

        # Load numpy compressed file
        loaded = np.load(fixture_path, allow_pickle=True)

        # Reconstruct DataFrame
        columns = loaded['_columns'].tolist()
        dtypes = loaded['_dtypes'].tolist()

        data_dict = {}
        for col in columns:
            if col.startswith('_'):  # Skip metadata columns
                continue
            data_dict[col] = loaded[col]

        df = pd.DataFrame(data_dict)
        # Convert dtypes where possible
        for col, dtype_str in zip(columns, dtypes):
            if col in df.columns and not col.startswith('_'):
                try:
                    if 'float' in dtype_str:
                        df[col] = df[col].astype(float)
                    elif 'int' in dtype_str:
                        df[col] = df[col].astype(int)
                except (ValueError, TypeError):
                    pass  # Keep original dtype if conversion fails

        return df

    def test_real_data_loaded(self, real_foot_data):
        """Verify that real data fixture loads correctly."""
        assert isinstance(real_foot_data, pd.DataFrame)
        assert len(real_foot_data) > 0
        assert 'elapsed_s' in real_foot_data.columns

        # Verify time range (should be ~9-70 seconds)
        assert real_foot_data['elapsed_s'].min() >= 9.0
        assert real_foot_data['elapsed_s'].max() <= 70.0

        # Verify required columns exist
        assert 'roll' in real_foot_data.columns
        assert 'accel_x' in real_foot_data.columns
        assert 'gyro_x' in real_foot_data.columns

    def test_segment_by_peaks_real_data(self, real_foot_data):
        """Test peak-based segmentation on real data."""
        peaks, troughs, segments = segment_by_peaks(
            real_foot_data, signal_column='roll', verbose=False
        )

        assert isinstance(peaks, list)
        assert isinstance(troughs, list)
        assert isinstance(segments, list)
        validate_lengths_match(peaks, troughs, segments)

        # Should detect multiple gait cycles (expect ~20-30 segments for 61 seconds)
        assert (
            len(segments) >= 15
        ), f"Expected at least 15 segments, got {len(segments)}"
        assert len(segments) <= 35, f"Expected at most 35 segments, got {len(segments)}"

        validate_segments(segments, real_foot_data)

    def test_segment_jasiewicz_real_data(self, real_foot_data):
        """Test Jasiewicz segmentation on real data."""
        ec_peaks, ic_peaks, segments, _, _ = segment_jasiewicz(
            real_foot_data, verbose=False
        )

        assert isinstance(ec_peaks, list)
        assert isinstance(ic_peaks, list)
        assert isinstance(segments, list)
        validate_lengths_match(ec_peaks, ic_peaks, segments)

        # Should detect multiple gait cycles
        assert (
            len(segments) >= 15
        ), f"Expected at least 15 segments, got {len(segments)}"
        assert len(segments) <= 35, f"Expected at most 35 segments, got {len(segments)}"

        validate_event_indices(ic_peaks, real_foot_data, "IC")
        validate_event_indices(ec_peaks, real_foot_data, "EC")

    def test_segment_cionic_real_data(self, real_foot_data):
        """Test Cionic (modified Jasiewicz) segmentation on real data."""
        ec_peaks, ic_peaks, segments = segment_cionic(real_foot_data, verbose=False)

        assert isinstance(ec_peaks, list)
        assert isinstance(ic_peaks, list)
        assert isinstance(segments, list)
        validate_lengths_match(ec_peaks, ic_peaks, segments)

        # Should detect multiple gait cycles
        assert (
            len(segments) >= 15
        ), f"Expected at least 15 segments, got {len(segments)}"
        assert len(segments) <= 35, f"Expected at most 35 segments, got {len(segments)}"

        validate_event_indices(ic_peaks, real_foot_data, "IC")
        validate_event_indices(ec_peaks, real_foot_data, "EC")

    def test_segment_seel_real_data(self, real_foot_data):
        """Test Seel segmentation on real data."""
        segments, tos, ics, ff_starts, ff_ends = segment_seel(
            real_foot_data, verbose=False
        )

        assert isinstance(segments, list)
        assert isinstance(tos, list)
        assert isinstance(ics, list)
        assert isinstance(ff_starts, list)
        assert isinstance(ff_ends, list)

        validate_lengths_match(segments, tos, ics, ff_starts, ff_ends)

        # Should detect multiple gait cycles
        assert (
            len(segments) >= 15
        ), f"Expected at least 15 segments, got {len(segments)}"
        assert len(segments) <= 35, f"Expected at most 35 segments, got {len(segments)}"

        for idx_list, name in zip(
            [tos, ics, ff_starts, ff_ends], ["TO", "IC", "FF_start", "FF_end"]
        ):
            validate_event_indices(idx_list, real_foot_data, name)

    def test_segment_footpod_all_methods_real_data(self, real_foot_data):
        """Test segment_footpod convenience function with all methods on real data."""
        methods = ["peak", "jasiewicz", "mod-jasiewicz", "seel"]

        for method in methods:
            ic_peaks, ec_peaks = segment_footpod(real_foot_data, method=method)

            assert isinstance(ic_peaks, list)
            assert isinstance(ec_peaks, list)

            # Should detect multiple events
            assert (
                len(ic_peaks) >= 15
            ), f"Method {method}: Expected at least 15 IC events"
            assert (
                len(ec_peaks) >= 15
            ), f"Method {method}: Expected at least 15 EC events"

            validate_event_indices(ic_peaks, real_foot_data, "IC")
            validate_event_indices(ec_peaks, real_foot_data, "EC")

    def test_segmentation_consistency_real_data(self, real_foot_data):
        """Test that different methods produce consistent results on real data."""
        # Run all methods
        _, _, segs_peak = segment_by_peaks(
            real_foot_data, signal_column='roll', verbose=False
        )
        _, _, segs_jas, _, _ = segment_jasiewicz(real_foot_data, verbose=False)
        _, _, segs_cion = segment_cionic(real_foot_data, verbose=False)
        segs_seel, _, _, _, _ = segment_seel(real_foot_data, verbose=False)

        # All methods should detect similar # of gait cycles (this is normative gait)
        segment_counts = [
            len(segs_peak),
            len(segs_jas),
            len(segs_cion),
            len(segs_seel),
        ]

        min_count = min(segment_counts)
        max_count = max(segment_counts)
        # Allow up to 10% variation in segment counts between methods
        # Also allow at least 5 to handle edge cases where min_count is small
        # Ex: min_count=20 --> difference up to max(6, 5)=6, so max_count can be 26
        # Ex: min_count=10 --> difference up to max(3, 5)=5, so max_count can be 15
        max_allowed_difference = max(min_count * 0.1, 5)
        assert (
            max_count - min_count <= max_allowed_difference
        ), f"Segment counts too different: {segment_counts}\
        (difference: {max_count - min_count}, allowed: {max_allowed_difference})"

        # All methods should detect at least some segments
        assert all(
            count >= 15 for count in segment_counts
        ), f"Too few segments: {segment_counts}"
