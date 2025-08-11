"""
This module provides tools for calculating gait metrics from kinematic data streams.

It processes stride-based movement data, computes a variety of temporal and spatial
metrics for each stride, and outputs results in a structured CSV format. The module
supports flexible metric selection, stride segmentation, and optional toe-off event
detection. It is designed for use in gait analysis studies, biomechanics research,
and clinical movement assessments.

Typical usage involves providing a kinematic data stream, stride boundaries, and
optional metadata. The module extracts stride segments, computes metrics such as
stride time, cadence, peak/trough values, and stance/swing phase statistics, and
saves the results to disk for further analysis.

Usage Example (CLI):
    python3 tests/test_metrics_calculator.py

Usage Example (Code):

    metrics_calculator = gait_metrics.GaitMetricsCalculator(
        stream=kinematic_data_stream,
        stride_splits=stride_splits,
        shank_stream=kinematic_shank_stream,
        meta=gait_metrics.Metadata(
            org_shortname=org_shortname,
            study_shortname=study_shortname,
            collection_num=collection_num,
            position='r_shank',
            stream_name='euler',
            component='x',
        ),
    )
    metrics_df = metrics_calculator.calculate_metrics(output_path="recordings")

Note on stream data types:
    The input streams (stream, stride_splits, shank_stream) are expected to be NumPy
    record arrays with specific fields. For example, the stream should have an
    'elapsed_s' field and fields for the components being analyzed (e.g., 'x', 'y',
    'z', 'knee_flexion', etc.). The stride_splits should have 'start_s' and 'stop_s'
    fields indicating the time boundaries of each stride. Record arrays share many of
    the same properties as Pandas DataFrames, but are more memory efficient for large
    datasets.

Output:
    CSV file containing gait metrics for each stride. Each row represents a stride
    with columns for start time, stop time, elapsed time, and computed metrics.

Filename structure:
    {output_path}/{study_shortname}_{collection_num}_{position}_
    {stream}_{component}_gait_metrics.csv

Available metrics include:
    - stride_time: Duration of the stride.
    - cadence: Steps per minute.
    - peak_value: Maximum value of the component during the stride.
    - trough_value: Minimum value of the component during the stride.
    - start_heel_strike_value: Value at the start of the stride.
    - stop_heel_strike_value: Value at the end of the stride.
    - mean_value: Mean value during the stride.
    - median_value: Median value during the stride.
    - std_value: Standard deviation during the stride.
    - toe_off_value: Value at the toe-off event.
    - stance_time: Duration of the stance phase.
    - stance_mean_value: Mean value during stance phase.
    - stance_median_value: Median value during stance phase.
    - stance_std_value: Standard deviation during stance phase.
    - swing_time: Duration of the swing phase.
    - swing_mean_value: Mean value during swing phase.
    - swing_median_value: Median value during swing phase.
    - swing_std_value: Standard deviation during swing phase.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from cionic import kinematics


def compute_stride_time(stride_data: np.ndarray) -> float:
    """
    Compute the stride time for the given stride data.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.

    Returns:
        float: Duration of the stride, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data['elapsed_s'].max() - stride_data['elapsed_s'].min()


def compute_cadence(stride_data: np.ndarray) -> float:
    """
    Compute cadence (steps per minute) for the stride data.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.

    Returns:
        float: Cadence value, or None if data is empty or duration is zero.
    """
    if stride_data.shape[0] == 0:
        return None
    duration = compute_stride_time(stride_data)
    if duration is None or duration == 0:
        return None
    return 60 / duration


def compute_peak_value(stride_data: np.ndarray, component: str = 'x') -> float:
    """
    Compute the peak value of a component during the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Peak value, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[component].max()


def compute_trough_value(stride_data: np.ndarray, component: str = 'x') -> float:
    """
    Compute the trough value of a component during the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Trough value, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[component].min()


def compute_start_heel_strike_value(
    stride_data: np.ndarray, component: str = 'x'
) -> float:
    """
    Get the value of a component at the start of the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Value at start, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[0][component]


def compute_stop_heel_strike_value(
    stride_data: np.ndarray, component: str = 'x'
) -> float:
    """
    Get the value of a component at the end of the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Value at end, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[-1][component]


def compute_mean_value(stride_data: np.ndarray, component: str = 'x') -> float:
    """
    Compute the mean value of a component during the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Mean value, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[component].mean()


def compute_median_value(stride_data: np.ndarray, component: str = 'x') -> float:
    """
    Compute the median value of a component during the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Median value, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return np.median(stride_data[component])


def compute_std_value(stride_data: np.ndarray, component: str = 'x') -> float:
    """
    Compute the standard deviation of a component during the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        component (str): Component name to analyze.

    Returns:
        float: Standard deviation, or None if data is empty.
    """
    if stride_data.shape[0] == 0:
        return None
    return stride_data[component].std()


def compute_toe_off_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Get the value of a component at the toe off time.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Value at toe off, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    closest_index = np.abs(stride_data['elapsed_s'] - toe_off_time).argmin()
    return stride_data[closest_index][component]


def compute_stance_time(
    stride_data: np.ndarray, toe_off_time: Optional[float]
) -> float:
    """
    Compute the duration of the stance phase for the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.

    Returns:
        float: Stance duration, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    return toe_off_time - stride_data['elapsed_s'].min()


def compute_stance_mean_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute mean value of a component during the stance phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Mean value, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    stance_data = stride_data[stride_data['elapsed_s'] <= toe_off_time]
    if stance_data.shape[0] == 0:
        return None
    return compute_mean_value(stance_data, component)


def compute_stance_median_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute median value of a component during the stance phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Median value, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    stance_data = stride_data[stride_data['elapsed_s'] <= toe_off_time]
    if stance_data.shape[0] == 0:
        return None
    return compute_median_value(stance_data, component)


def compute_stance_std_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute standard deviation of a component during the stance phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Standard deviation, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    stance_data = stride_data[stride_data['elapsed_s'] <= toe_off_time]
    if stance_data.shape[0] == 0:
        return None
    return compute_std_value(stance_data, component)


def compute_swing_time(stride_data: np.ndarray, toe_off_time: Optional[float]) -> float:
    """
    Compute the duration of the swing phase for the stride.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.

    Returns:
        float: Swing duration, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    return stride_data['elapsed_s'].max() - toe_off_time


def compute_swing_mean_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute mean value of a component during the swing phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Mean value, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    swing_data = stride_data[stride_data['elapsed_s'] > toe_off_time]
    if swing_data.shape[0] == 0:
        return None
    return compute_mean_value(swing_data, component)


def compute_swing_median_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute median value of a component during the swing phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Median value, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    swing_data = stride_data[stride_data['elapsed_s'] > toe_off_time]
    if swing_data.shape[0] == 0:
        return None
    return compute_median_value(swing_data, component)


def compute_swing_std_value(
    stride_data: np.ndarray, toe_off_time: Optional[float], component: str = 'x'
) -> float:
    """
    Compute standard deviation of a component during the swing phase.

    Args:
        stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.
        toe_off_time (float): Toe off timestamp.
        component (str): Component name to analyze.

    Returns:
        float: Standard deviation, or None if data is empty or time is None.
    """
    if stride_data.shape[0] == 0 or toe_off_time is None:
        return None
    swing_data = stride_data[stride_data['elapsed_s'] > toe_off_time]
    if swing_data.shape[0] == 0:
        return None
    return compute_std_value(swing_data, component)


class Metric(Enum):
    STRIDE_TIME = 'stride_time'
    CADENCE = 'cadence'
    PEAK_VALUE = 'peak_value'
    TROUGH_VALUE = 'trough_value'
    START_HEEL_STRIKE_VALUE = 'start_heel_strike_value'
    STOP_HEEL_STRIKE_VALUE = 'stop_heel_strike_value'
    MEAN_VALUE = 'mean_value'
    MEDIAN_VALUE = 'median_value'
    STD_VALUE = 'std_value'
    STANCE_TIME = 'stance_time'
    SWING_TIME = 'swing_time'
    TOE_OFF_VALUE = 'toe_off_value'
    STANCE_MEAN_VALUE = 'stance_mean_value'
    STANCE_MEDIAN_VALUE = 'stance_median_value'
    STANCE_STD_VALUE = 'stance_std_value'
    SWING_MEAN_VALUE = 'swing_mean_value'
    SWING_MEDIAN_VALUE = 'swing_median_value'
    SWING_STD_VALUE = 'swing_std_value'


# Metrics that require only stride_data (no component or toe_off_time)
METRIC_FUNCTION_MAP_TIME = {
    Metric.STRIDE_TIME: compute_stride_time,
    Metric.CADENCE: compute_cadence,
}

# Metrics that require stride_data and component
METRIC_FUNCTION_MAP_COMPONENT = {
    Metric.PEAK_VALUE: compute_peak_value,
    Metric.TROUGH_VALUE: compute_trough_value,
    Metric.START_HEEL_STRIKE_VALUE: compute_start_heel_strike_value,
    Metric.STOP_HEEL_STRIKE_VALUE: compute_stop_heel_strike_value,
    Metric.MEAN_VALUE: compute_mean_value,
    Metric.MEDIAN_VALUE: compute_median_value,
    Metric.STD_VALUE: compute_std_value,
}

# Metrics that require stride_data and toe_off_time
METRIC_FUNCTION_MAP_PHASE_TIME = {
    Metric.STANCE_TIME: compute_stance_time,
    Metric.SWING_TIME: compute_swing_time,
}

# Metrics that require stride_data, toe_off_time, and component
METRIC_FUNCTION_MAP_PHASE_COMPONENT = {
    Metric.TOE_OFF_VALUE: compute_toe_off_value,
    Metric.STANCE_MEAN_VALUE: compute_stance_mean_value,
    Metric.STANCE_MEDIAN_VALUE: compute_stance_median_value,
    Metric.STANCE_STD_VALUE: compute_stance_std_value,
    Metric.SWING_MEAN_VALUE: compute_swing_mean_value,
    Metric.SWING_MEDIAN_VALUE: compute_swing_median_value,
    Metric.SWING_STD_VALUE: compute_swing_std_value,
}


@dataclass
class Metadata:
    position: str
    stream_name: str
    component: str
    org_shortname: Optional[str] = None
    study_shortname: Optional[str] = None
    collection_num: Optional[int] = None


class GaitMetricsCalculator:
    def __init__(
        self,
        stream: np.ndarray,
        stride_splits: np.ndarray,
        meta: Metadata,
        toe_off_times: Optional[np.ndarray] = None,
        shank_stream: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the GaitMetricsCalculator with input data and metadata.

        If toe_off_times are not provided, they will be computed from the shank_stream
        if available.

        Args:
            stream (np.ndarray): Kinematics input data. Numpy record array with fields
                including 'elapsed_s' and the component to analyze.
            stride_splits (np.ndarray): Array of stride splits. Numpy record array with
                fields 'start_s' and 'stop_s'.
            meta (Metadata): Metadata for analysis.
            toe_off_times (np.ndarray, optional): Toe off timestamps.
            shank_stream (np.ndarray, optional): Shank stream data. Numpy record array
                with 'elapsed_s' field, used to compute toe_off_times if not provided.
        """
        self.stream = stream
        self.stride_splits = stride_splits
        self.meta = meta
        if toe_off_times is not None:
            self.toe_off_times = toe_off_times
        elif shank_stream is not None:
            self.toe_off_times = self._compute_toe_offs(shank_stream)
        else:
            self.toe_off_times = None

    def _compute_toe_offs(self, shank_stream: np.ndarray) -> np.ndarray:
        """
        Compute toe off times from shank stream data.

        Args:
            shank_stream (np.ndarray): Shank stream data, NumPy record with 'elapsed_s'
                field.

        Returns:
            np.ndarray: Array of toe off timestamps.
        """
        grouped_toe_off_times = kinematics.get_grouped_walking_splits(
            shank_stream, factor=-1.0
        )
        toe_off_times = [item for sublist in grouped_toe_off_times for item in sublist]
        return np.array(toe_off_times)

    def _get_toe_off_time_in_stride(self, stride_data: np.ndarray) -> Optional[float]:
        """
        Get the toe off time within the stride data range.

        Args:
            stride_data (np.ndarray): Stride data, NumPy record with 'elapsed_s' field.

        Returns:
            float: Toe off timestamp, or None if not found.
        """
        if self.toe_off_times is None:
            return None
        start = stride_data['elapsed_s'].min()
        stop = stride_data['elapsed_s'].max()

        # Vectorized selection of toe_off_times within stride range
        valid_toe_offs = self.toe_off_times[
            (self.toe_off_times >= start) & (self.toe_off_times <= stop)
        ]
        if valid_toe_offs.shape[0] == 0:
            print(f"No toe_off_time found in stride range ({start}, {stop}).")
            return None
        if valid_toe_offs.shape[0] > 1:
            print(
                f"Multiple toe_off_times ({valid_toe_offs}) found in stride range "
                f"({start}, {stop}). Using the first."
            )
        return valid_toe_offs[0]

    def _output_metrics_to_csv(
        self, all_strides_metrics: pd.DataFrame, output_path: str
    ) -> None:
        """
        Output the computed metrics DataFrame to a CSV file.

        Args:
            all_strides_metrics (pd.DataFrame): DataFrame containing the metrics.
            output_path (str): Path to the output CSV file.
        """
        if (
            self.meta.org_shortname
            and self.meta.study_shortname
            and self.meta.collection_num
        ):
            output_path = os.path.join(
                output_path,
                self.meta.org_shortname,
                self.meta.study_shortname,
                str(self.meta.collection_num),
            )
            file_path = (
                f"{output_path}/{self.meta.org_shortname}_"
                f"{self.meta.study_shortname}_{self.meta.collection_num}_"
                f"{self.meta.position}_{self.meta.stream_name}_"
                f"{self.meta.component}_gait_metrics.csv"
            )
        else:
            file_path = (
                f"{output_path}/{self.meta.position}_{self.meta.stream_name}_"
                f"{self.meta.component}_gait_metrics.csv"
            )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        all_strides_metrics.to_csv(file_path, index=True)

    def calculate_metrics(
        self,
        metrics: Optional[list[Metric]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate selected gait metrics for all strides and save to CSV if needed.

        Args:
            metrics (list[Metric], optional): Metrics to compute.
            output_path (str, optional): Output directory path.

        Returns:
            pd.DataFrame: DataFrame of computed metrics for all strides.
        """
        if metrics is None:
            metrics = list(Metric)

        all_strides_metrics_list = []
        for stride in self.stride_splits:
            # Extract relevant data for the current stride
            stride_data = self.stream[
                (self.stream['elapsed_s'] >= stride['start_s'])
                & (self.stream['elapsed_s'] <= stride['stop_s'])
            ]
            toe_off_time = self._get_toe_off_time_in_stride(stride_data)

            stride_metrics = {
                **{'start_s': stride['start_s'], 'stop_s': stride['stop_s']},
                **{metric.value: None for metric in metrics},
            }
            for metric in metrics:
                if metric in METRIC_FUNCTION_MAP_TIME.keys():
                    func = METRIC_FUNCTION_MAP_TIME[metric]
                    stride_metrics[metric.value] = func(stride_data)
                elif metric in METRIC_FUNCTION_MAP_COMPONENT.keys():
                    func = METRIC_FUNCTION_MAP_COMPONENT[metric]
                    stride_metrics[metric.value] = func(
                        stride_data, self.meta.component
                    )
                elif metric in METRIC_FUNCTION_MAP_PHASE_TIME.keys():
                    func = METRIC_FUNCTION_MAP_PHASE_TIME[metric]
                    stride_metrics[metric.value] = func(stride_data, toe_off_time)
                elif metric in METRIC_FUNCTION_MAP_PHASE_COMPONENT.keys():
                    func = METRIC_FUNCTION_MAP_PHASE_COMPONENT[metric]
                    stride_metrics[metric.value] = func(
                        stride_data, toe_off_time, self.meta.component
                    )
                else:
                    print(f"Metric {metric} not recognized. Skipping.")

            all_strides_metrics_list.append(stride_metrics)

        all_strides_metrics = pd.DataFrame(all_strides_metrics_list)
        if output_path is not None:
            self._output_metrics_to_csv(all_strides_metrics, output_path)

        return all_strides_metrics


def compute_gait_metrics(
    stream: np.ndarray,
    stride_splits: np.ndarray,
    position: str,
    stream_name: str,
    component: str,
    org_shortname: Optional[str] = None,
    study_shortname: Optional[str] = None,
    collection_num: Optional[int] = None,
    toe_off_times: Optional[np.ndarray] = None,
    shank_stream: Optional[np.ndarray] = None,
    metrics: Optional[list[Metric]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute gait metrics for all strides and save to CSV if output_path is given.

    Args:
        stream (np.ndarray): Kinematics input data, NumPy record array with fields
            including 'elapsed_s' and the component to analyze.
        stride_splits (np.ndarray): Array of stride splits, NumPy record array with
            fields 'start_s' and 'stop_s'.
        position (str): Position identifier, e.g. 'r_shank'
        stream_name (str): Name of the stream.
        component (str): Component to analyze, e.g. 'x', 'knee_flexion'.
        org_shortname (str, optional): Organization ID.
        study_shortname (str, optional): Study name.
        collection_num (int, optional): Collection number.
        toe_off_times (np.ndarray, optional): Toe off timestamps.
        shank_stream (np.ndarray, optional): Shank stream data.
        metrics (list[Metric], optional): Metrics to compute.
        output_path (str, optional): Output directory path.

    Returns:
        pd.DataFrame: DataFrame of computed metrics for all strides.
    """
    metrics_calculator = GaitMetricsCalculator(
        stream=stream,
        stride_splits=stride_splits,
        meta=Metadata(
            position=position,
            stream_name=stream_name,
            component=component,
            org_shortname=org_shortname,
            study_shortname=study_shortname,
            collection_num=collection_num,
        ),
        toe_off_times=toe_off_times,
        shank_stream=shank_stream,
    )
    return metrics_calculator.calculate_metrics(
        metrics=metrics, output_path=output_path
    )
