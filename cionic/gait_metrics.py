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

Usage Example:

    metrics_calculator = gait_metrics.GaitMetricsCalculator(
        stream=kinematic_data_stream,
        stride_splits=stride_splits,
        shank_stream=kinematic_shank_stream,
        meta=gait_metrics.Metadata(
            orgid=orgid,
            study=study,
            collection_num=collection_num,
            position='r_shank',
            stream_name='euler',
            component='x',
        ),
    )
    metrics_df = metrics_calculator.calculate_metrics(output_path="recordings")

Output:
    CSV file containing gait metrics for each stride. Each row represents a stride
    with columns for start time, stop time, elapsed time, and computed metrics.

Filename structure:
    {output_path}/{study}_{collection_num}_{position}_
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
    TOE_OFF_VALUE = 'toe_off_value'
    STANCE_TIME = 'stance_time'
    STANCE_MEAN_VALUE = 'stance_mean_value'
    STANCE_MEDIAN_VALUE = 'stance_median_value'
    STANCE_STD_VALUE = 'stance_std_value'
    SWING_TIME = 'swing_time'
    SWING_MEAN_VALUE = 'swing_mean_value'
    SWING_MEDIAN_VALUE = 'swing_median_value'
    SWING_STD_VALUE = 'swing_std_value'


def compute_stride_time(stride_data: np.ndarray) -> float:
    """
    Compute the stride time for the given stride data.

    Args:
        stride_data (np.ndarray): Array of stride data.

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
        stride_data (np.ndarray): Array of stride data.

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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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
        stride_data (np.ndarray): Array of stride data.
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


@dataclass
class Metadata:
    position: int
    stream_name: str
    component: str
    orgid: Optional[str] = None
    study: Optional[str] = None
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

        Args:
            stream (np.ndarray): Kinematics input data.
            stride_splits (np.ndarray): Array of stride splits.
            meta (Metadata): Metadata for analysis.
            toe_off_times (np.ndarray, optional): Toe off timestamps.
            shank_stream (np.ndarray, optional): Shank stream data.
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
            shank_stream (np.ndarray): Shank stream data.

        Returns:
            np.ndarray: Array of toe off timestamps.
        """
        grouped_toe_off_times = kinematics.get_grouped_walking_splits(
            shank_stream, factor=-1.0
        )
        toe_off_times = [item for sublist in grouped_toe_off_times for item in sublist]
        return np.array(toe_off_times)

    def _get_toe_off_time(self, stride_data: np.ndarray) -> Optional[float]:
        """
        Get the toe off time within the stride data range.

        Args:
            stride_data (np.ndarray): Array of stride data.

        Returns:
            float: Toe off timestamp, or None if not found.
        """
        if self.toe_off_times is None:
            return None
        # Find the toe off time falling in the time range of the stride
        for toe_off_time in self.toe_off_times:
            if (
                stride_data['elapsed_s'].min()
                <= toe_off_time
                <= stride_data['elapsed_s'].max()
            ):
                return toe_off_time
        return None

    def _output_metrics_to_csv(
        self, all_strides_metrics: pd.DataFrame, output_path: str
    ) -> None:
        """
        Output the computed metrics DataFrame to a CSV file.

        Args:
            all_strides_metrics (pd.DataFrame): DataFrame containing the metrics.
            output_path (str): Path to the output CSV file.
        """
        if self.meta.orgid and self.meta.study and self.meta.collection_num:
            output_path = os.path.join(
                output_path,
                self.meta.orgid,
                self.meta.study,
                str(self.meta.collection_num),
            )
            file_path = (
                f"{output_path}/{self.meta.orgid}_"
                f"{self.meta.study}_{self.meta.collection_num}_"
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
            start_s = stride['start_s']
            stop_s = stride['stop_s']

            stride_data = self.stream[
                (self.stream['elapsed_s'] >= start_s)
                & (self.stream['elapsed_s'] <= stop_s)
            ]

            stride_metrics = {metric.value: None for metric in metrics}
            stride_metrics = {
                **{'start_s': start_s, 'stop_s': stop_s},
                **stride_metrics,
            }

            toe_off_time = self._get_toe_off_time(stride_data)
            for metric in metrics:
                if metric == Metric.STRIDE_TIME:
                    stride_metrics[metric.value] = compute_stride_time(stride_data)
                elif metric == Metric.CADENCE:
                    stride_metrics[metric.value] = compute_cadence(stride_data)
                elif metric == Metric.PEAK_VALUE:
                    stride_metrics[metric.value] = compute_peak_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.TROUGH_VALUE:
                    stride_metrics[metric.value] = compute_trough_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.START_HEEL_STRIKE_VALUE:
                    stride_metrics[metric.value] = compute_start_heel_strike_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.STOP_HEEL_STRIKE_VALUE:
                    stride_metrics[metric.value] = compute_stop_heel_strike_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.MEAN_VALUE:
                    stride_metrics[metric.value] = compute_mean_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.MEDIAN_VALUE:
                    stride_metrics[metric.value] = compute_median_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.STD_VALUE:
                    stride_metrics[metric.value] = compute_std_value(
                        stride_data, self.meta.component
                    )
                elif metric == Metric.TOE_OFF_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_toe_off_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.STANCE_TIME and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_stance_time(
                        stride_data, toe_off_time
                    )
                elif metric == Metric.STANCE_MEAN_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_stance_mean_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.STANCE_MEDIAN_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_stance_median_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.STANCE_STD_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_stance_std_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.SWING_TIME and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_swing_time(
                        stride_data, toe_off_time
                    )
                elif metric == Metric.SWING_MEAN_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_swing_mean_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.SWING_MEDIAN_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_swing_median_value(
                        stride_data, toe_off_time, self.meta.component
                    )
                elif metric == Metric.SWING_STD_VALUE and toe_off_time is not None:
                    stride_metrics[metric.value] = compute_swing_std_value(
                        stride_data, toe_off_time, self.meta.component
                    )

            all_strides_metrics_list.append(stride_metrics)
        all_strides_metrics = pd.DataFrame(all_strides_metrics_list)
        if output_path is not None:
            self._output_metrics_to_csv(all_strides_metrics, output_path)

        return all_strides_metrics


def compute_gait_metrics(
    stream: np.ndarray,
    stride_splits: np.ndarray,
    position: int,
    stream_name: str,
    component: str,
    orgid: Optional[str] = None,
    study: Optional[str] = None,
    collection_num: Optional[int] = None,
    toe_off_times: Optional[np.ndarray] = None,
    shank_stream: Optional[np.ndarray] = None,
    metrics: Optional[list[Metric]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute gait metrics for all strides and save to CSV if output_path is given.

    Args:
        stream (np.ndarray): Kinematics input data.
        stride_splits (np.ndarray): Array of stride splits.
        position (int): Position identifier, e.g. 'r_shank'
        stream_name (str): Name of the stream.
        component (str): Component to analyze, e.g. 'x', 'knee_flexion'.
        orgid (str, optional): Organization ID.
        study (str, optional): Study name.
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
            orgid=orgid,
            study=study,
            collection_num=collection_num,
        ),
        toe_off_times=toe_off_times,
        shank_stream=shank_stream,
    )
    return metrics_calculator.calculate_metrics(
        metrics=metrics, output_path=output_path
    )
