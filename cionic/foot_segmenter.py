"""Gait Cycle Segmentation and Event Detection for Footpod data.

This module provides functions for segmenting gait cycles and detecting gait events
using IMU data. It implements multiple segmentation algorithms:

1. Basic segmentation using angle peaks and troughs
2. Jasiewicz algorithm using acceleration and angle data
3. Cionic algorithm (modified Jasiewicz) - updates the Jasiewicz algorithm's window
detection to reduce reliance on roll for maxima (still uses roll for trough detection).
4. Seel algorithm for detailed gait phase detection - computes other gait events
such as swing foot flat start and end in addition to initial and end contact.

The module can detect various gait events:
- Initial Contact (IC) / Heel Strike
- End Contact (EC) / Toe Off
- Foot Flat phases (Seel only)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

#########################################
# Helper Functions                      #
#########################################


def _filter_data_by_time_range(
    data: pd.DataFrame, time_range: Tuple[float, float] = (0, np.inf)
) -> pd.DataFrame:
    """Filter data by time range and reset index.

    Args:
        data (pd.DataFrame): DataFrame with 'elapsed_s' column
        time_range (tuple[float, float], optional): Start and end times.
            Defaults to (0, np.inf).

    Returns:
        pd.DataFrame: Filtered data with reset index
    """
    filtered_data = data[
        (data["elapsed_s"] >= time_range[0]) & (data["elapsed_s"] <= time_range[1])
    ].copy()
    filtered_data.reset_index(drop=True, inplace=True)
    return filtered_data


def _find_events_in_segment(
    events: List[int],
    segment_start: int,
    segment_end: int,
    verbose: bool = False,
    segment_idx: Optional[int] = None,
) -> List[int]:
    """Find events within a segment.

    Args:
        events (list[int]): List of event indices
        segment_start (int): Start index of segment
        segment_end (int): End index of segment
        verbose (bool, optional): Whether to print detection info. Defaults to False.
        segment_idx (int, optional): Segment index for verbose output. Defaults to None.

    Returns:
        list[int]: Events found within the segment
    """
    events_in_segment = [
        event for event in events if segment_start < event < segment_end
    ]
    if verbose and segment_idx is not None:
        print(f"Found {len(events_in_segment)} events in segment {segment_idx}")
    return events_in_segment


def _validate_segment_events(
    events_dict: Dict[str, List[int]],
    segment_start: int,
    segment_end: int,
    verbose: bool = False,
    segment_idx: Optional[int] = None,
) -> bool:
    """Validate events within a segment.

    Checks if the segment has exactly one of each event type.

    Args:
        events_dict (dict[str, list[int]]): Dictionary mapping event names to indices
        segment_start (int): Start index of segment
        segment_end (int): End index of segment
        verbose (bool, optional): Whether to print validation info. Defaults to False.
        segment_idx (int, optional): Segment index for verbose output. Defaults to None.

    Returns:
        bool: True if segment has exactly one of each event type
    """
    for event_name, events in events_dict.items():
        events_in_segment = _find_events_in_segment(events, segment_start, segment_end)
        if len(events_in_segment) != 1:
            if verbose:
                print(
                    f"Skipping segment {segment_idx} due to {len(events_in_segment)} "
                    f"{event_name} events found."
                )
            return False
    return True


def _validate_component_inputs(
    ic_component: str, ec_component: str, ic_peak_type: str, ec_peak_type: str
) -> None:
    """Validate component and peak type inputs.

    Args:
        ic_component (str): Initial contact component name.
        ec_component (str): End contact component name.
        ic_peak_type (str): Initial contact peak type ('max' or 'min').
        ec_peak_type (str): End contact peak type ('max' or 'min').

    Raises:
        ValueError: If component or peak type is invalid.
    """
    valid_components = ["accel_x", "accel_y", "accel_z"]
    if ic_component not in valid_components or ec_component not in valid_components:
        raise ValueError(f"Component must be one of {valid_components}")
    if ic_peak_type not in ["max", "min"] or ec_peak_type not in ["max", "min"]:
        raise ValueError("Peak type must be 'max' or 'min'")


def _find_signal_peaks(
    signal: np.ndarray,
    peak_type: str = "max",
    height: Optional[float] = None,
    distance: int = 25,
    prominence: float = 0.1,
) -> np.ndarray:
    """Find peaks in a signal using scipy's find_peaks.

    Detects peaks with configurable parameters and sorts them by prominence.

    Args:
        signal (np.ndarray): The signal to find peaks in
        peak_type (str, optional): 'max' for maxima or 'min' for minima.
            Defaults to 'max'.
        height (float, optional): Minimum/maximum height for peak detection.
            Defaults to None.
        distance (int, optional): Minimum distance between peaks in samples.
            Defaults to 25.
        prominence (float, optional): Minimum prominence of peaks.
            Defaults to 0.1.

    Returns:
        np.ndarray: Indices of detected peaks, sorted by prominence
    """
    if peak_type == "min":
        signal = -signal
        if height is not None:
            height = -height

    peaks, properties = find_peaks(
        signal, height=height, distance=distance, prominence=prominence
    )

    if len(peaks) > 0:
        sorted_indices = np.argsort(properties["prominences"])[::-1]
        peaks = peaks[sorted_indices]

    return peaks


def _find_gait_events(
    data: pd.DataFrame,
    component: str,
    peak_type: str,
    window_range: Tuple[float, float],
    time_idx: int,
    verbose: bool = False,
) -> List[int]:
    """Find gait events within a time window.

    Args:
        data (pd.DataFrame): DataFrame with time and component data.
        component (str): Column name for event detection.
        peak_type (str): Type of peak to detect ('max' or 'min').
        window_range (tuple[float, float]): (pre, post) time window in seconds.
        time_idx (int): Index to center the window around.
        verbose (bool, optional): Whether to print detection info.
            Defaults to False.

    Returns:
        list[int]: Detected event indices. Empty list if no events found.
    """
    event_time = data["elapsed_s"].iloc[time_idx]
    pre_time = event_time + window_range[0]
    post_time = event_time + window_range[1]

    if verbose:
        print(f"Event search window: {pre_time} to {post_time}")

    segment = data[(data["elapsed_s"] >= pre_time) & (data["elapsed_s"] <= post_time)]
    if segment.empty:
        return []

    most_prominent_peaks = _find_signal_peaks(segment[component], peak_type=peak_type)
    if len(most_prominent_peaks) == 0:
        if peak_type == "min":
            most_prominent_peaks = _find_signal_peaks(
                segment[component], peak_type="max"
            )
        if len(most_prominent_peaks) == 0:
            return []

    return [most_prominent_peaks[0] + segment.index[0]]


def _create_segments_from_ic_peaks(
    data: pd.DataFrame, ic_peaks: List[int], ec_peaks: List[int], verbose: bool = False
) -> Tuple[List[int], List[int], List[pd.DataFrame]]:
    """Create segments from IC peaks with EC validation.

    Creates gait cycle segments from initial contact peaks, validating that each
    segment contains exactly one end contact event within the IC window.

    Args:
        data (pd.DataFrame): DataFrame containing 'elapsed_s' and sensor columns.
        ic_peaks (list[int]): List of initial contact peak indices.
        ec_peaks (list[int]): List of end contact peak indices.
        verbose (bool, optional): Whether to print segment information.
            Defaults to False.

    Returns:
        tuple[list[int], list[int], list[pd.DataFrame]]:
            - List of validated EC peak indices
            - List of validated IC peak indices
            - List of segmented DataFrames
    """
    segments = []
    final_ec_peaks = []
    final_ic_peaks = []

    for i in range(len(ic_peaks) - 1):
        segment = data.iloc[ic_peaks[i] : ic_peaks[i + 1]]

        events_dict = {"EC": ec_peaks}
        if not _validate_segment_events(
            events_dict, ic_peaks[i], ic_peaks[i + 1], verbose, i
        ):
            continue

        ec_in_segment = _find_events_in_segment(ec_peaks, ic_peaks[i], ic_peaks[i + 1])
        segments.append(segment)
        final_ec_peaks.append(ec_in_segment[0])
        final_ic_peaks.append(ic_peaks[i])

        if verbose:
            print(
                f"Segment {i}: Time {segment['elapsed_s'].min()} to "
                f"{segment['elapsed_s'].max()}"
            )

    return final_ec_peaks, final_ic_peaks, segments


def _detect_rest_phases(
    signal: np.ndarray, threshold: float, hysteresis_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect rest phases using forward-backward hysteresis.

    Helper function for Seel algorithm. Detects rest phases in a signal using
    hysteresis to avoid rapid switching between states.

    Args:
        signal (np.ndarray): Signal to analyze.
        threshold (float): Threshold for rest detection.
        hysteresis_factor (float): Factor to reduce threshold for phase exit.

    Returns:
        tuple[np.ndarray, np.ndarray]: (forward_mask, backward_mask) boolean arrays
            indicating rest phases.
    """
    N = len(signal)
    forward_mask = np.zeros(N)
    backward_mask = np.zeros(N)

    for k in range(1, N):
        if signal[k] > (1 + hysteresis_factor) * threshold:
            forward_mask[k] = 1
        elif signal[k] < (1 - hysteresis_factor) * threshold:
            forward_mask[k] = 0
        else:
            forward_mask[k] = forward_mask[k - 1]

    backward_mask[-1] = forward_mask[-1]
    for k in range(N - 2, -1, -1):
        if forward_mask[k] == 1:
            backward_mask[k] = 1
        elif signal[k] < (1 - hysteresis_factor) * threshold:
            backward_mask[k] = 0
        else:
            backward_mask[k] = backward_mask[k + 1]

    return forward_mask, backward_mask


def _remove_short_phases(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """Remove short phases in a boolean mask.

    Helper function for Seel algorithm. Removes phases that are shorter than
    the minimum duration to filter out noise.

    Args:
        mask (np.ndarray): Boolean mask to process.
        min_samples (int): Minimum number of samples for a phase to be kept.

    Returns:
        np.ndarray: Processed boolean mask with short phases removed.
    """
    zero_regions = np.where(mask)[0]
    if len(zero_regions) > 0:
        zero_boundaries = np.where(np.diff(zero_regions) > 1)[0]
        if len(zero_boundaries) > 0:
            zero_starts = np.concatenate(
                ([zero_regions[0]], zero_regions[zero_boundaries + 1])
            )
            zero_ends = np.concatenate(
                (zero_regions[zero_boundaries], [zero_regions[-1]])
            )

            for start, end in zip(zero_starts, zero_ends):
                if end - start < min_samples:
                    mask[start : end + 1] = False

    one_regions = np.where(~mask)[0]
    if len(one_regions) > 0:
        one_boundaries = np.where(np.diff(one_regions) > 1)[0]
        if len(one_boundaries) > 0:
            one_starts = np.concatenate(
                ([one_regions[0]], one_regions[one_boundaries + 1])
            )
            one_ends = np.concatenate((one_regions[one_boundaries], [one_regions[-1]]))

            for start, end in zip(one_starts, one_ends):
                if end - start < min_samples:
                    mask[start : end + 1] = True

    return mask


#########################################
# Plotting Helpers                      #
#########################################


def _plot_orientation_with_events(
    data: pd.DataFrame,
    signal_column: str = "roll",
    events_dict: Optional[Dict[str, Tuple[List[int], str]]] = None,
    ic_component: Optional[str] = None,
    ec_component: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot orientation signal with detected events.

    Args:
        data (pd.DataFrame): DataFrame with 'elapsed_s' and signal columns
        signal_column (str): Name of the signal column to plot. Defaults to 'roll'.
        events_dict (dict[str, tuple[list[int], str]], optional): Dictionary mapping
            event names to tuples of (indices, color). Defaults to None.
        ic_component (str, optional): Acceleration component to plot normalized for
            IC detection. Defaults to None.
        ec_component (str, optional): Acceleration component to plot normalized for
            EC detection. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data["elapsed_s"], data[signal_column], label=signal_column, color="b")

    if ic_component is not None:
        norm_ic = (data[ic_component] - data[ic_component].mean()) / data[
            ic_component
        ].max()
        plt.plot(
            data["elapsed_s"], norm_ic, label=f"{ic_component} (normalized)", color="r"
        )

    if ec_component is not None:
        norm_ec = (data[ec_component] - data[ec_component].mean()) / data[
            ec_component
        ].max()
        plt.plot(
            data["elapsed_s"], norm_ec, label=f"{ec_component} (normalized)", color="g"
        )

    if events_dict:
        for event_name, (indices, color) in events_dict.items():
            filtered_indices = [idx for idx in indices if idx < len(data)]
            if filtered_indices:
                plt.vlines(
                    data["elapsed_s"].iloc[filtered_indices],
                    ymin=-1,
                    ymax=1,
                    linestyle="--",
                    linewidth=2,
                    color=color,
                    label=event_name,
                )

    plt.title(title or f"{signal_column.capitalize()} with Events")
    plt.xlabel("Time")
    plt.ylabel(f"{signal_column.capitalize()} Angle (radians)")
    plt.legend()
    plt.grid()
    plt.show()


def _plot_tilt_rate_detection(
    times: List[float],
    tilt_rates: List[float],
    threshold_times: List[float],
    to_times: List[float],
    gamma_max: float,
) -> None:
    """Plot tilt rate detection results.

    Args:
        times (list[float]): List of time values
        tilt_rates (list[float]): List of tilt rate values
        threshold_times (list[float]): List of times where threshold was crossed
        to_times (list[float]): List of toe-off event times
        gamma_max (float): Maximum tilt rate value
    """
    plt.figure(figsize=(12, 6))
    plt.plot(times, tilt_rates, label="Tilt Rate")

    for t_time in threshold_times:
        plt.axvline(t_time, color="g", linestyle="--", alpha=0.5)
    if threshold_times:
        plt.axvline(
            threshold_times[-1], color="g", linestyle="--", label="Threshold Cross"
        )

    for to_time in to_times:
        plt.axvline(to_time, color="r", linestyle="--", alpha=0.5)
    if to_times:
        plt.axvline(to_times[-1], color="r", linestyle="--", label="TO")

    plt.axhline(
        0.5 * gamma_max,
        color="orange",
        linestyle="--",
        label="0.5*Γmax",
        xmin=0,
        xmax=1,
    )
    plt.axhline(0, color="k", linestyle="-", alpha=0.3)

    plt.grid(True)
    plt.legend()
    plt.title("Toe-Off Detection using Tilt Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Tilt Rate")
    plt.xlim(min(times), min(20, max(times)))
    plt.show()


def _plot_ic_detection(
    accel_times: List[float],
    accel_norms: List[float],
    jerk_times: List[float],
    jerk_norms: List[float],
    ic_times: List[float],
    threshold: float,
    jerk_threshold: float,
) -> None:
    """Plot IC detection results.

    Args:
        accel_times (list[float]): List of time values for acceleration
        accel_norms (list[float]): List of acceleration norm values
        jerk_times (list[float]): List of time values for jerk
        jerk_norms (list[float]): List of jerk norm values
        ic_times (list[float]): List of initial contact event times
        threshold (float): Threshold value used for detection
        jerk_threshold (float): Fraction of max jerk used for threshold
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(211)
    plt.plot(accel_times, accel_norms, label="Accel Norm")
    for ic_time in ic_times:
        plt.axvline(ic_time, color="r", linestyle="--", alpha=0.5)
    if ic_times:
        plt.axvline(ic_times[-1], color="r", linestyle="--", label="IC")
    plt.grid(True)
    plt.legend()
    plt.title("Acceleration Norm")
    plt.xlim(min(accel_times), min(20, max(accel_times)))

    plt.subplot(212)
    plt.plot(jerk_times, jerk_norms, label="Jerk Norm")
    plt.axhline(
        threshold,
        color="g",
        linestyle="--",
        label=f"Threshold ({jerk_threshold:.2f}*max)",
        xmin=0,
        xmax=1,
    )
    for ic_time in ic_times:
        plt.axvline(ic_time, color="r", linestyle="--", alpha=0.5)
    if ic_times:
        plt.axvline(ic_times[-1], color="r", linestyle="--", label="IC")
    plt.grid(True)
    plt.legend()
    plt.title("Jerk Norm with Detection Threshold")
    plt.xlabel("Time (s)")
    plt.xlim(min(jerk_times), min(20, max(jerk_times)))
    plt.tight_layout()
    plt.show()


def _plot_seel_detection(
    data: pd.DataFrame,
    accel_norm: np.ndarray,
    gyro_norm: np.ndarray,
    ra_backward: np.ndarray,
    rg_backward: np.ndarray,
    ff_mask: np.ndarray,
    ff_starts: np.ndarray,
    ff_ends: np.ndarray,
    accel_thresholds: List[float],
    gyro_thresholds: List[float],
    verbose: bool = False,
) -> None:
    """Plot Seel algorithm detection process.

    Creates visualization of the Seel algorithm's foot-flat detection process,
    showing acceleration, gyroscope, rest signals, and detected events.

    Args:
        data (pd.DataFrame): DataFrame with time and sensor data.
        accel_norm (np.ndarray): Acceleration norm signal.
        gyro_norm (np.ndarray): Gyroscope norm signal.
        ra_backward (np.ndarray): Rest signal from acceleration.
        rg_backward (np.ndarray): Rest signal from gyroscope.
        ff_mask (np.ndarray): Foot-flat mask.
        ff_starts (np.ndarray): Foot-flat start indices.
        ff_ends (np.ndarray): Foot-flat end indices.
        accel_thresholds (list[float]): Acceleration threshold values.
        gyro_thresholds (list[float]): Gyroscope threshold values.
        verbose (bool, optional): Whether to print detection info.
            Defaults to False.
    """
    mask_20s = data["elapsed_s"] <= 20
    data_20s = data[mask_20s]
    a_20s = accel_norm[mask_20s]
    gyro_norm_20s = gyro_norm[mask_20s]
    ra_backward_20s = ra_backward[mask_20s]
    rg_backward_20s = rg_backward[mask_20s]
    ff_mask_20s = ff_mask[mask_20s]

    ff_starts_20s = ff_starts[ff_starts < len(data_20s)]
    ff_ends_20s = ff_ends[ff_ends < len(data_20s)]

    plt.figure(figsize=(15, 12))

    plt.subplot(511)
    plt.plot(data_20s["elapsed_s"], a_20s - 9.81, label="|Accel - g|")
    plt.axhline(accel_thresholds[0], color="r", linestyle="--", label="Upper threshold")
    plt.axhline(accel_thresholds[1], color="g", linestyle="--", label="Lower threshold")
    plt.grid(True)
    plt.legend()
    plt.title("Acceleration Difference from Gravity")

    plt.subplot(512)
    plt.plot(data_20s["elapsed_s"], gyro_norm_20s, label="Gyro norm")
    plt.axhline(gyro_thresholds[0], color="r", linestyle="--", label="Upper threshold")
    plt.axhline(gyro_thresholds[1], color="g", linestyle="--", label="Lower threshold")
    plt.grid(True)
    plt.legend()
    plt.title("Gyroscope Norm")

    plt.subplot(513)
    plt.plot(data_20s["elapsed_s"], ra_backward_20s, label="Accel rest", alpha=0.7)
    plt.plot(data_20s["elapsed_s"], rg_backward_20s, label="Gyro rest", alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.title("Individual Rest Signals")

    plt.subplot(514)
    plt.plot(data_20s["elapsed_s"], ff_mask_20s.astype(int), label="FF mask")
    if len(ff_starts_20s) > 0:
        plt.vlines(
            data_20s["elapsed_s"].iloc[ff_starts_20s], 0, 1, "g", label="FF start"
        )
    if len(ff_ends_20s) > 0:
        plt.vlines(data_20s["elapsed_s"].iloc[ff_ends_20s], 0, 1, "r", label="FF end")
    plt.grid(True)
    plt.legend()
    plt.title("Final Foot-Flat Detection")

    plt.subplot(515)
    plt.plot(data_20s["elapsed_s"], data_20s["roll"], label="roll")
    plt.grid(True)
    plt.legend()
    plt.title("roll")

    plt.tight_layout()
    plt.show()

    if verbose:
        print("\nEvents in first 20 seconds:")
        print(f"FF starts: {ff_starts_20s}")
        print(f"FF ends: {ff_ends_20s}")


#########################################
# Segmentation Algorithms               #
#########################################


def segment_by_peaks(
    data: pd.DataFrame,
    signal_column: str = "roll",
    peak_height: float = 0.1,
    peak_distance: int = 10,
    trough_height: float = -0.1,
    segmentation_range: Tuple[float, float] = (0, np.inf),
    plot_peaks: bool = False,
    verbose: bool = True,
) -> Tuple[List[int], List[int], List[pd.DataFrame]]:
    """Segment gait cycles using peak detection on a signal column.

    Uses peak detection on the specified signal column to identify gait cycles.
    Each cycle is defined from peak to peak, with validation that each cycle
    contains exactly one trough.

    Args:
        data (pd.DataFrame): DataFrame containing 'elapsed_s' and signal column
        signal_column (str, optional): Column name to use for peak detection.
            Defaults to 'roll'.
        peak_height (float, optional): Minimum height for peak detection.
            Defaults to 0.2.
        peak_distance (int, optional): Minimum distance between peaks in samples.
            Defaults to 10.
        trough_height (float, optional): Minimum height for trough detection.
            Defaults to -0.1.
        segmentation_range (tuple[float, float], optional): Time range to segment.
            Defaults to (0, np.inf).
        plot_peaks (bool, optional): Whether to plot peaks and troughs.
            Defaults to False.
        verbose (bool, optional): Whether to print segment information.
            Defaults to True.

    Returns:
        tuple[list[int], list[int], list[pd.DataFrame]]:
            - List of peak indices
            - List of trough indices
            - List of segmented DataFrames
    """
    data = _filter_data_by_time_range(data, segmentation_range)

    # 1) Detect peaks + troughs
    peaks, _ = find_peaks(
        data[signal_column], height=peak_height, distance=peak_distance
    )
    troughs, _ = find_peaks(
        -data[signal_column], height=-trough_height, distance=peak_distance
    )

    if verbose:
        print(
            f"Found {len(peaks)} peaks and {len(troughs)} "
            f"troughs in {signal_column} data."
        )

    if plot_peaks:
        events = {"EC Peaks": (peaks, "r"), "IC Peaks": (troughs, "g")}
        _plot_orientation_with_events(
            data,
            signal_column=signal_column,
            events_dict=events,
            title="Kinematics with EC and IC Peaks",
        )

    # 2) Create segments from peak to peak, validating each segment has 1 trough
    segments = []
    final_peaks = []
    final_troughs = []

    for i in range(len(peaks) - 1):
        segment = data.iloc[peaks[i] : peaks[i + 1]]
        events_dict = {"troughs": troughs}
        if not _validate_segment_events(
            events_dict, peaks[i], peaks[i + 1], verbose, i
        ):
            continue

        troughs_in_segment = _find_events_in_segment(troughs, peaks[i], peaks[i + 1])
        segments.append(segment)
        final_peaks.append(peaks[i])
        final_troughs.append(troughs_in_segment[0])

        if verbose:
            print(
                f"Segment {i}: Time {segment['elapsed_s'].min()} to "
                f"{segment['elapsed_s'].max()}"
            )

    return final_peaks, final_troughs, segments


def segment_jasiewicz(
    data: pd.DataFrame,
    ic_component: str = "accel_z",
    ic_peak_type: str = "min",
    ec_component: str = "accel_x",
    ec_peak_type: str = "max",
    roll_trough_magnitude: float = 0.1,
    roll_distance: int = 10,
    ic_window: Tuple[float, float] = (-0.1, 0.1),
    ec_window: Tuple[float, float] = (-0.25, 0.05),
    plot_peaks: bool = False,
    verbose: bool = True,
    segmentation_range: Tuple[float, float] = (0, np.inf),
) -> Tuple[List[int], List[int], List[pd.DataFrame], List[int], List[int]]:
    """Segment gait cycles using roll and acceleration data.

    EC search windows should be around peak plantarflexion (trough in roll).
    IC search windows should be around peak dorsiflexion (peak in roll).

    Args:
        data (pd.DataFrame): DataFrame containing 'elapsed_s', 'roll', and accel
            columns.
        ic_component (str, optional): Column name for initial contact detection
            ('accel_x', 'accel_y', or 'accel_z'). Defaults to 'accel_z'.
        ic_peak_type (str, optional): Type of peak to detect for IC ('max' or 'min').
            Defaults to 'min'.
        ec_component (str, optional): Column name for end contact detection
            ('accel_x', 'accel_y', or 'accel_z'). Defaults to 'accel_x'.
        ec_peak_type (str, optional): Type of peak to detect for EC ('max' or 'min').
            Defaults to 'max'.
        roll_trough_magnitude (float, optional): Height threshold for roll peaks.
            Defaults to 0.1.
        roll_distance (int, optional): Minimum distance between roll peaks/troughs.
            Defaults to 10.
        ic_window (tuple[float, float], optional): Time window around trough for IC
            detection (pre, post) in seconds. Defaults to (-0.1, 0.1).
        ec_window (tuple[float, float], optional): Time window around maxima for EC
            detection (pre, post) in seconds. Defaults to (-0.25, 0.05).
        plot_peaks (bool, optional): Whether to plot the detected peaks.
            Defaults to False.
        verbose (bool, optional): Whether to print detection information.
            Defaults to True.
        segmentation_range (tuple[float, float], optional): Time range of the data to
            segment. Defaults to (0, np.inf).

    Returns:
        tuple[list[int], list[int], list[pd.DataFrame], list[int], list[int]]:
            - List of validated EC peak indices
            - List of validated IC peak indices
            - List of segmented DataFrames
            - List of roll maxima indices
            - List of roll trough indices
    """
    _validate_component_inputs(ic_component, ec_component, ic_peak_type, ec_peak_type)
    data = _filter_data_by_time_range(data, segmentation_range)

    # 1) troughs (peak plantarflexion) and peaks/maxima (peak dorsiflexion)
    troughs, _ = find_peaks(
        -data["roll"], height=roll_trough_magnitude, distance=roll_distance
    )
    if verbose:
        print(f"Found {len(troughs)} troughs in roll data.")

    maxima = []
    if len(troughs) > 1:
        for i in range(len(troughs) - 1):
            segment = data.iloc[troughs[i] : troughs[i + 1]]
            peak, _ = find_peaks(segment["roll"], distance=roll_distance)
            if len(peak) > 0:
                maxima.append(peak[0] + troughs[i])

    if verbose:
        print(f"Found {len(maxima)} maxima in roll data.")

    # 2) Detect EC events by searching for peaks in acceleration around roll troughs
    ec_peaks = []
    for trough_idx in troughs:
        ec_peaks.extend(
            _find_gait_events(
                data, ec_component, ec_peak_type, ec_window, trough_idx, verbose
            )
        )

    # 3) Detect IC events by searching for peaks in acceleration around roll maxima
    ic_peaks = []
    for max_idx in maxima:
        ic_peaks.extend(
            _find_gait_events(
                data, ic_component, ic_peak_type, ic_window, max_idx, verbose
            )
        )

    if plot_peaks:
        events = {"EC Peaks": (ec_peaks, "orange"), "IC Peaks": (ic_peaks, "purple")}
        _plot_orientation_with_events(
            data,
            signal_column="roll",
            events_dict=events,
            ic_component=ic_component,
            ec_component=ec_component,
            title="Roll with EC and IC Peaks",
        )

    final_ec_peaks, final_ic_peaks, segments = _create_segments_from_ic_peaks(
        data, ic_peaks, ec_peaks, verbose
    )

    return final_ec_peaks, final_ic_peaks, segments, maxima, troughs


def segment_cionic(
    data: pd.DataFrame,
    ic_component: str = "accel_z",
    ic_peak_type: str = "min",
    ec_component: str = "accel_x",
    ec_peak_type: str = "max",
    roll_trough_magnitude: float = 0.1,
    roll_distance: int = 10,
    ic_window: Tuple[float, float] = (-0.25, 0.05),
    ec_window: Tuple[float, float] = (-0.25, 0.05),
    plot_peaks: bool = False,
    verbose: bool = True,
    segmentation_range: Tuple[float, float] = (0, np.inf),
) -> Tuple[List[int], List[int], List[pd.DataFrame]]:
    """Segment gait cycles using roll and acceleration data with Cionic algorithm.

    Modified Jasiewicz algorithm that uses IC component markers instead of roll maxima.

    Args:
        data (pd.DataFrame): DataFrame with 'elapsed_s', 'roll', and accel columns.
        ic_component (str, optional): Column name for IC detection
            ('accel_x', 'accel_y', 'accel_z'). Defaults to 'accel_z'.
        ic_peak_type (str, optional): Type of peak to detect for IC ('max' or 'min').
            Defaults to 'min'.
        ec_component (str, optional): Column name for EC detection
            ('accel_x', 'accel_y', 'accel_z'). Defaults to 'accel_x'.
        ec_peak_type (str, optional): Type of peak to detect for EC ('max' or 'min').
            Defaults to 'max'.
        roll_trough_magnitude (float, optional): Height threshold for roll peaks.
            Defaults to 0.1.
        roll_distance (int, optional): Minimum distance between roll peaks/troughs.
            Defaults to 10.
        ic_window (tuple[float, float], optional): Time window around marker for IC
            detection (pre, post) in seconds. Defaults to (-0.25, 0.05).
        ec_window (tuple[float, float], optional): Time window around trough for EC
            detection (pre, post) in seconds. Defaults to (-0.25, 0.05).
        plot_peaks (bool, optional): Whether to plot the detected peaks.
            Defaults to False.
        verbose (bool, optional): Whether to print detection information.
            Defaults to True.
        segmentation_range (tuple[float, float], optional): Time range of the data to
            segment. Defaults to (0, np.inf).

    Returns:
        tuple[list[int], list[int], list[pd.DataFrame]]:
            - List of validated EC peak indices
            - List of validated IC peak indices
            - List of segmented DataFrames
    """
    _validate_component_inputs(ic_component, ec_component, ic_peak_type, ec_peak_type)
    data = _filter_data_by_time_range(data, segmentation_range)

    # 1) Find troughs (peak plantarflexion)
    troughs, _ = find_peaks(
        -data["roll"], height=roll_trough_magnitude, distance=roll_distance
    )
    if verbose:
        print(f"Found {len(troughs)} troughs in roll data.")

    # 2) Find IC markers in accel between troughs (instead of using maxima)
    ic_markers = []
    for i in range(len(troughs) - 1):
        segment = data.iloc[troughs[i] : troughs[i + 1]]
        start_idx = int(len(segment) * 0.15)
        end_idx = int(len(segment) * 0.85)
        segment = segment.iloc[start_idx:end_idx]

        peaks = _find_signal_peaks(segment[ic_component], peak_type="max")
        if len(peaks) > 0:
            ic_markers.append(peaks[0] + segment.index[0])

    if verbose:
        print(f"Found {len(ic_markers)} markers in {ic_component} data.")

    # 3) Detect EC events by searching for peaks in acceleration around roll troughs
    ec_peaks = []
    for trough_idx in troughs:
        ec_peaks.extend(
            _find_gait_events(
                data, ec_component, ec_peak_type, ec_window, trough_idx, verbose
            )
        )

    # 4) Detect IC events by searching for peaks in acceleration around IC markers
    ic_peaks = []
    for marker_idx in ic_markers:
        ic_peaks.extend(
            _find_gait_events(
                data, ic_component, ic_peak_type, ic_window, marker_idx, verbose
            )
        )

    if plot_peaks:
        events = {"EC Peaks": (ec_peaks, "orange"), "IC Peaks": (ic_peaks, "purple")}
        _plot_orientation_with_events(
            data,
            signal_column="roll",
            events_dict=events,
            ic_component=ic_component,
            ec_component=ec_component,
            title="Roll with EC and IC Peaks",
        )

    final_ec_peaks, final_ic_peaks, segments = _create_segments_from_ic_peaks(
        data, ic_peaks, ec_peaks, verbose
    )

    return final_ec_peaks, final_ic_peaks, segments


def segment_seel(
    data: pd.DataFrame,
    verbose: bool = False,
    accel_threshold: float = 2,  # m/s^2, expected during foot-flat
    gyro_threshold: float = 0.75,  # rad/s, expected during foot-flat
    hysteresis_factor: float = 0.5,  # for foot-flat detection
    min_phase_duration: float = 0.1,  # seconds
    jerk_threshold: float = 0.9,  # fraction of max jerk for IC detection
    segmentation_range: Tuple[float, float] = (0, np.inf),
) -> Tuple[List[pd.DataFrame], List[int], List[int], List[int], List[int]]:
    """Segment gait cycles using the Seel algorithm.

    This implementation is based on:
    Seel et al. "Calibration-Free Gait Assessment by Foot-Worn Inertial Sensors"

    The algorithm detects:
    - Foot-flat (FF): When foot is stationary on ground
    - Heel-rise (HR): End of foot-flat phase
    - Toe-off (TO): When foot leaves ground
    - Initial contact (IC): When foot hits ground

    Args:
        data (pd.DataFrame): DataFrame with columns:
            - time: timestamps in seconds
            - accel_x/y/z: acceleration in m/s^2
            - gyro_x/y/z: angular velocity in rad/s
        verbose (bool, optional): Whether to print detection information.
            Defaults to False.
        accel_threshold (float, optional): Expected acceleration during foot-flat
            (gravity). Defaults to 2.
        gyro_threshold (float, optional): Maximum angular velocity during foot-flat.
            Defaults to 0.75.
        hysteresis_factor (float, optional): Factor to reduce thresholds for phase
            exit. Defaults to 0.5.
        min_phase_duration (float, optional): Minimum duration of foot-flat phase
            in seconds. Defaults to 0.1.
        jerk_threshold (float, optional): Fraction of max jerk to use for IC
            detection. Defaults to 0.9.
        segmentation_range (tuple[float, float], optional): Time range to segment.
            Defaults to (0, np.inf).

    Returns:
        tuple[list[pd.DataFrame], list[int], list[int], list[int], list[int]]:
            - List of segmented DataFrames
            - List of toe-off indices
            - List of initial contact indices
            - List of foot-flat start indices
            - List of foot-flat end indices
    """
    data = _filter_data_by_time_range(data, segmentation_range)

    # 1) Calculate signal norms and detect foot-flat phases
    accel_norm = np.linalg.norm(data[["accel_x", "accel_y", "accel_z"]].values, axis=1)
    gyro_norm = np.linalg.norm(data[["gyro_x", "gyro_y", "gyro_z"]].values, axis=1)
    a_diff = np.abs(accel_norm - 9.81)

    # 2) Detect rest phases w hysteresis: foot flat when both accel/gyro indicate rest
    _, ra_backward = _detect_rest_phases(a_diff, accel_threshold, hysteresis_factor)
    _, rg_backward = _detect_rest_phases(gyro_norm, gyro_threshold, hysteresis_factor)

    ff_mask = ~((ra_backward == 1) | (rg_backward == 1))

    if verbose:
        print("\nDebug information:")
        print(f"Acceleration rest signal (first 10): {ra_backward[:10]}")
        print(f"Gyroscope rest signal (first 10): {rg_backward[:10]}")
        print(f"Combined FF mask (first 10): {ff_mask[:10]}")
        print(f"Number of FF samples: {np.sum(ff_mask)}")

    # 3) Remove short foot-flat phases and find transitions
    min_samples = int(
        min_phase_duration
        * (len(data) / (data["elapsed_s"].max() - data["elapsed_s"].min()))
    )
    ff_mask = _remove_short_phases(ff_mask, min_samples)

    ff_ends = np.where(np.diff(ff_mask.astype(int)) == -1)[0]
    ff_starts = np.where(np.diff(ff_mask.astype(int)) == 1)[0]

    if verbose:
        acc_thresholds = [
            (1 + hysteresis_factor) * accel_threshold,
            (1 - hysteresis_factor) * accel_threshold,
        ]
        gyro_thresholds = [
            (1 + hysteresis_factor) * gyro_threshold,
            (1 - hysteresis_factor) * gyro_threshold,
        ]
        print(
            f"\nFound {len(ff_ends)} foot-flat ends and {len(ff_starts)} "
            f"foot-flat starts."
        )
        _plot_seel_detection(
            data,
            accel_norm,
            gyro_norm,
            ra_backward,
            rg_backward,
            ff_mask,
            ff_starts,
            ff_ends,
            acc_thresholds,
            gyro_thresholds,
            verbose,
        )

    # 4) Detect toe-off (TO) events via tilt rate analysis between ff end and ff start
    tos = []
    ics = []
    all_tilt_rates = []
    all_times = []
    all_threshold_times = []
    all_to_times = []

    for i in range(len(ff_ends) - 1):
        hr_idx = ff_ends[i]
        next_ff_idx = ff_starts[i + 1] if i < len(ff_starts) - 1 else len(data)
        omega = data.iloc[hr_idx:next_ff_idx][["gyro_x", "gyro_y", "gyro_z"]].values

        if len(omega) < 10:
            continue

        # Calculate tilt rate: rate of change of foot orientation
        cum_omega = np.cumsum(omega, axis=0)
        tilt_rate = np.zeros(len(omega))

        for t in range(len(omega)):
            cum_omega_t = cum_omega[t]
            cum_omega_norm = np.linalg.norm(cum_omega_t)
            if cum_omega_norm > 0:
                tilt_rate[t] = np.dot(omega[t], cum_omega_t) / cum_omega_norm

        # TO occurs when tilt rate crosses threshold then crosses zero
        mid_idx = len(tilt_rate) // 2
        gamma_max = np.max(tilt_rate[:mid_idx])
        threshold_crossings = np.where(tilt_rate >= 0.5 * gamma_max)[0]

        if len(threshold_crossings) == 0:
            continue

        first_threshold_idx = threshold_crossings[0]
        zero_crossings = np.where(tilt_rate[first_threshold_idx:] <= 0)[0]

        if len(zero_crossings) == 0:
            continue

        to_idx = first_threshold_idx + zero_crossings[0]
        tos.append(hr_idx + to_idx)

        if data["elapsed_s"].iloc[hr_idx] <= 20:
            if verbose:
                window_time = data["elapsed_s"].iloc[hr_idx : hr_idx + len(tilt_rate)]
                all_tilt_rates.extend(tilt_rate)
                all_times.extend(window_time)
                if window_time.iloc[first_threshold_idx] <= 20:
                    all_threshold_times.append(window_time.iloc[first_threshold_idx])
                if window_time.iloc[to_idx] <= 20:
                    all_to_times.append(window_time.iloc[to_idx])

        if verbose and i == len(ff_ends) - 2:
            _plot_tilt_rate_detection(
                all_times, all_tilt_rates, all_threshold_times, all_to_times, gamma_max
            )

    if verbose:
        print(f"Number of TOs: {len(tos)}")

    # 5) Detect initial contact (IC) events using jerk between TO and next foot-flat
    all_accel_times = []
    all_accel_norms = []
    all_jerk_times = []
    all_jerk_norms = []
    all_ic_times = []

    j_win = 0.7
    for i in range(len(tos) - 1):
        to_idx = tos[i]
        next_ff_idx = ff_starts[i + 1] if i < len(ff_starts) - 1 else len(data)

        if next_ff_idx - to_idx < 10:
            continue

        # Search window starts at 70% of the way from TO to foot-flat
        t_win_idx = int(to_idx + j_win * (next_ff_idx - to_idx))
        window = data.iloc[t_win_idx:next_ff_idx]
        Ts = np.mean(np.diff(window["elapsed_s"]))

        # Calculate jerk (rate of change of acceleration) - IC causes large jerk spike
        accel = window[["accel_x", "accel_y", "accel_z"]].values
        jerk = (accel[1:] - accel[:-1]) / Ts
        jerk_norm = np.linalg.norm(jerk, axis=1)

        j_max = np.max(jerk_norm)
        threshold = jerk_threshold * j_max
        ic_candidates = np.where(jerk_norm >= threshold)[0]

        if len(ic_candidates) > 0:
            ic_idx = t_win_idx + ic_candidates[0] + 1
            ics.append(ic_idx)

            if data["elapsed_s"].iloc[t_win_idx] <= 20:
                if verbose:
                    accel_time = window["elapsed_s"]
                    all_accel_times.extend(accel_time)
                    all_accel_norms.extend(np.linalg.norm(accel, axis=1))

                    jerk_time = window["elapsed_s"].iloc[1:]
                    all_jerk_times.extend(jerk_time)
                    all_jerk_norms.extend(jerk_norm)

                    if accel_time.iloc[ic_candidates[0] + 1] <= 20:
                        all_ic_times.append(accel_time.iloc[ic_candidates[0] + 1])

            if verbose and i == len(tos) - 2:
                _plot_ic_detection(
                    all_accel_times,
                    all_accel_norms,
                    all_jerk_times,
                    all_jerk_norms,
                    all_ic_times,
                    threshold,
                    jerk_threshold,
                )

    if verbose:
        print(f"Number of ICs: {len(ics)}")

    # 6) Create segments from IC to IC, validating all events are present
    segments = []
    final_tos = []
    final_ics = []
    final_ff_starts = []
    final_ff_ends = []

    for i in range(len(ics) - 1):
        segment = data.iloc[ics[i] : ics[i + 1]]

        events_dict = {"TO": tos, "FF Start": ff_starts, "FF End": ff_ends}
        if not _validate_segment_events(events_dict, ics[i], ics[i + 1], verbose, i):
            continue

        seg_tos = _find_events_in_segment(tos, ics[i], ics[i + 1])
        seg_ff_starts = _find_events_in_segment(ff_starts, ics[i], ics[i + 1])
        seg_ff_ends = _find_events_in_segment(ff_ends, ics[i], ics[i + 1])

        segments.append(segment)
        final_ics.append(ics[i])
        final_tos.append(seg_tos[0])
        final_ff_starts.append(seg_ff_starts[0])
        final_ff_ends.append(seg_ff_ends[0])

        if verbose:
            print(
                f"Segment {i}: Time {segment['elapsed_s'].min()} to "
                f"{segment['elapsed_s'].max()}"
            )

    return segments, final_tos, final_ics, final_ff_starts, final_ff_ends


#########################################
# Gait Notebook Utilities               #
#########################################


def event_idxs_to_times(
    data: pd.DataFrame, events: List[int], time_range: Tuple[float, float] = (0, np.inf)
) -> np.ndarray:
    """Convert event indices to times.

    Args:
        data (pd.DataFrame): DataFrame containing 'elapsed_s' column.
        events (list[int]): List of event indices.
        time_range (tuple[float, float], optional): Time range to convert.
            Defaults to (0, np.inf).

    Returns:
        np.ndarray: Array of event times in seconds.
    """
    return _filter_data_by_time_range(data, time_range).iloc[events]['elapsed_s'].values


def plot_segment_analysis(
    segments: List[pd.DataFrame],
    tos: List[int],
    ics: List[int],
    ff_starts: Optional[List[int]] = None,
    ff_ends: Optional[List[int]] = None,
    num_samples: int = 100,
    plot_individual: bool = True,
    verbose: bool = False,
    main: str = "roll",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot and analyze gait segments with normalized time.

    Creates a visualization showing individual segments, average trajectory, and
    standard deviation. Also marks gait events (TO, IC, foot-flat) if provided.

    Args:
        segments (list[pd.DataFrame]): List of segmented DataFrames
        tos (list[int]): List of toe-off event indices
        ics (list[int]): List of initial contact event indices
        ff_starts (list[int], optional): List of foot-flat start indices.
            Defaults to None.
        ff_ends (list[int], optional): List of foot-flat end indices.
            Defaults to None.
        num_samples (int, optional): Number of samples for interpolation.
            Defaults to 100.
        plot_individual (bool, optional): Whether to plot individual segments.
            Defaults to True.
        verbose (bool, optional): Whether to print segment information.
            Defaults to False.
        main (str, optional): Column name for the main signal to plot.
            Defaults to 'roll'.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Mean signal values
            - Standard deviation of signal values
            - Uniform time points used for interpolation
    """
    if verbose:
        for i, segment in enumerate(segments):

            print(f"Segment {i} saved with time normalized to start at 0.")

    plt.figure(figsize=(12, 6))

    if plot_individual:
        to_times = []
        ic_times = []
        to_vals = []
        ic_vals = []

        ff_start_times = []
        ff_start_vals = []
        ff_end_times = []
        ff_end_vals = []

        for i, segment in enumerate(segments):
            seg_start = segment.index[0]
            seg_end = segment.index[-1]
            segment_time = segment["elapsed_s"] - segment["elapsed_s"].min()
            segment_time = segment_time / segment_time.max()

            plt.plot(segment_time, segment[main], alpha=0.5)

            seg_tos = [to for to in tos if seg_start <= to <= seg_end]
            if len(seg_tos) > 0:
                rel_to_idx = seg_tos[0] - seg_start
                to_times.append(segment_time.iloc[rel_to_idx])
                to_vals.append(segment[main].iloc[rel_to_idx])

            seg_ics = [ic for ic in ics if seg_start <= ic <= seg_end]
            if len(seg_ics) > 0:
                rel_ic_idx = seg_ics[0] - seg_start
                ic_times.append(segment_time.iloc[rel_ic_idx])
                ic_vals.append(segment[main].iloc[rel_ic_idx])

            if ff_starts is not None:
                seg_ff_starts = [
                    idx for idx in ff_starts if seg_start <= idx <= seg_end
                ]
                if len(seg_ff_starts) > 0:
                    rel_ff_start_idx = seg_ff_starts[0] - seg_start
                    ff_start_times.append(segment_time.iloc[rel_ff_start_idx])
                    ff_start_vals.append(segment[main].iloc[rel_ff_start_idx])

            if ff_ends is not None:
                seg_ff_ends = [idx for idx in ff_ends if seg_start <= idx <= seg_end]
                if len(seg_ff_ends) > 0:
                    rel_ff_end_idx = seg_ff_ends[0] - seg_start
                    ff_end_times.append(segment_time.iloc[rel_ff_end_idx])
                    ff_end_vals.append(segment[main].iloc[rel_ff_end_idx])

        plt.scatter(to_times, to_vals, color="orange", label="Toe-Off")
        plt.scatter(ic_times, ic_vals, color="purple", label="Initial Contact")
        if ff_starts is not None and len(ff_start_times) > 0:
            plt.scatter(
                ff_start_times, ff_start_vals, color="green", label="Foot-Flat Start"
            )
        if ff_ends is not None and len(ff_end_times) > 0:
            plt.scatter(ff_end_times, ff_end_vals, color="red", label="Foot-Flat End")

    interpolated_rolls = []
    for segment in segments:
        segment_filtered = segment.dropna(subset=[main])
        seg_time = segment_filtered["elapsed_s"] - segment_filtered["elapsed_s"].min()

        # Check if segment has valid time range
        if len(seg_time) < 2 or seg_time.max() == 0:
            continue  # Skip segments with insufficient data

        seg_time = seg_time / seg_time.max()
        seg_roll = segment_filtered[main].values

        interp_func = interp1d(
            seg_time, seg_roll, kind="linear", fill_value="extrapolate"
        )
        uniform_time = np.linspace(0, 1, num_samples)
        interp_roll = interp_func(uniform_time)

        # Ensure interp_roll is 1D array
        if np.isscalar(interp_roll):
            continue
        interp_roll = np.atleast_1d(interp_roll)

        interpolated_rolls.append(interp_roll)

    if len(interpolated_rolls) == 0:
        print("Warning: No valid segments for interpolation")
        return np.array([]), np.array([]), np.linspace(0, 1, num_samples)

    interpolated_rolls = np.array(interpolated_rolls)
    # Ensure correct shape: (n_segments, num_samples)
    if interpolated_rolls.ndim == 1:
        interpolated_rolls = interpolated_rolls.reshape(1, -1)

    mean_roll = np.mean(interpolated_rolls, axis=0)
    std_roll = np.std(interpolated_rolls, axis=0)

    uniform_time = np.linspace(0, 1, num_samples)
    plt.plot(uniform_time, mean_roll, color="k", label=f"Average {main}", linewidth=2)
    plt.fill_between(
        uniform_time,
        mean_roll - std_roll,
        mean_roll + std_roll,
        color="k",
        alpha=0.2,
        label="Std Dev",
    )

    plt.legend()
    plt.title("All Segments Over Time")
    plt.xlabel("Normalized Time")
    plt.ylabel(f"{main} Angle (radians)")
    plt.grid()
    plt.show()

    return mean_roll, std_roll, uniform_time


def segment_footpod(
    data: pd.DataFrame,
    method: str = "peak",
    segmentation_range: Tuple[float, float] = (0, np.inf),
    **segment_kwargs,
) -> Tuple[List[int], List[int]]:
    """Segment footpod data using specified method and return IC and EC peaks.

    This is a convenience function that wraps the different segmentation algorithms
    and returns a consistent interface (IC peaks, EC peaks).

    Args:
        data (pd.DataFrame): DataFrame containing foot sensor data with required columns
            based on method.
        method (str, optional): Segmentation method to use. Options:
            - "peak": Basic peak/trough detection (default)
            - "jasiewicz": Jasiewicz algorithm using roll and acceleration
            - "mod-jasiewicz": Modified Jasiewicz (Cionic) algorithm
            - "seel": Seel algorithm with foot-flat detection
        segmentation_range (tuple[float, float], optional): Time range to segment.
            Defaults to (0, np.inf).
        **segment_kwargs: Additional kwargs passed to segmentation function.

    Returns:
        tuple[list[int], list[int]]: (ic_peaks, ec_peaks) indices.

    Raises:
        ValueError: If method is unknown or required columns are missing.
    """
    common_kwargs = {
        'segmentation_range': segmentation_range,
        'plot_peaks': False,
        'verbose': False,
        **segment_kwargs,
    }

    if method == "peak":
        ic_peaks, ec_peaks, _ = segment_by_peaks(data, **common_kwargs)
    elif method == "jasiewicz":
        ec_peaks, ic_peaks, _, _, _ = segment_jasiewicz(data, **common_kwargs)
    elif method == "mod-jasiewicz":
        ec_peaks, ic_peaks, _ = segment_cionic(data, **common_kwargs)
    elif method == "seel":
        _, ec_peaks, ic_peaks, _, _ = segment_seel(
            data, segmentation_range=segmentation_range, verbose=False, **segment_kwargs
        )
    else:
        raise ValueError(
            f"Unknown segmentation method: {method}. "
            "Valid options: 'peak', 'jasiewicz', 'mod-jasiewicz', 'seel'"
        )

    return ic_peaks, ec_peaks


def get_required_columns(method: str) -> List[str]:
    """Get required columns for a segmentation method.

    Args:
        method (str): Segmentation method name.

    Returns:
        list[str]: List of required column names.
    """
    if method == "peak":
        return ['elapsed_s', 'roll']
    elif method == "seel":
        return [
            'elapsed_s',
            'accel_x',
            'accel_y',
            'accel_z',
            'gyro_x',
            'gyro_y',
            'gyro_z',
        ]
    return ['elapsed_s', 'roll', 'accel_x', 'accel_y', 'accel_z']
