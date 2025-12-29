from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from cionic import api, npz_utils, stats, tools

FIGSIZE = (8, 5)
DPI = 100
LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
FONT_WEIGHT = "bold"

PLOTTING_METRIC_SPECIFICATION_LIST = [
    {
        "title": "Stride Time",
        "y_label": "Seconds",
        "metric_column": "stride_time",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Stance Time",
        "y_label": "Seconds",
        "metric_column": "stance_time",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Swing Time",
        "y_label": "Seconds",
        "metric_column": "swing_time",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Cadence",
        "y_label": "Steps per Minute",
        "metric_column": "cadence",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Shank Peak (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "peak_value",
        "position": "shank",
        "component": "x",
    },
    {
        "title": "Shank Trough (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "trough_value",
        "position": "shank",
        "component": "x",
    },
    {
        "title": "Shank Range (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "range_value",
        "position": "shank",
        "component": "x",
    },
    {
        "title": "Thigh Peak (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "peak_value",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Thigh Trough (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "trough_value",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Thigh Range (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "range_value",
        "position": "thigh",
        "component": "x",
    },
    {
        "title": "Foot Peak (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "peak_value",
        "position": "foot",
        "component": "x",
    },
    {
        "title": "Foot Trough (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "trough_value",
        "position": "foot",
        "component": "x",
    },
    {
        "title": "Foot Range (Sagittal)",
        "y_label": "Euler, (degrees)",
        "metric_column": "range_value",
        "position": "foot",
        "component": "x",
    },
    {
        "title": "Knee Flexion Peak",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "peak_value",
        "position": "knee_joint",
        "component": "knee_flexion",
    },
    {
        "title": "Knee Flexion Trough",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "trough_value",
        "position": "knee_joint",
        "component": "knee_flexion",
    },
    {
        "title": "Knee Valgus Stance Mean",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "stance_mean_value",
        "position": "knee_joint",
        "component": "knee_adduction",
    },
    {
        "title": "Knee Valgus Stance Std",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "stance_std_value",
        "position": "knee_joint",
        "component": "knee_adduction",
    },
    {
        "title": "Knee Valgus Swing Mean",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "swing_mean_value",
        "position": "knee_joint",
        "component": "knee_adduction",
    },
    {
        "title": "Knee Valgus Swing Std",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "swing_std_value",
        "position": "knee_joint",
        "component": "knee_adduction",
    },
    {
        "title": "Dorsiflexion Peak",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "peak_value",
        "position": "ankle_joint",
        "component": "dorsi_flexion",
    },
    {
        "title": "Dorsiflexion Trough",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "trough_value",
        "position": "ankle_joint",
        "component": "dorsi_flexion",
    },
    {
        "title": "Dorsiflexion Swing Mean",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "swing_mean_value",
        "position": "ankle_joint",
        "component": "dorsi_flexion",
    },
    {
        "title": "Inversion Swing Mean",
        "y_label": "Joint Angle, (degrees)",
        "metric_column": "swing_mean_value",
        "position": "ankle_joint",
        "component": "ankle_inversion",
    },
]

TOP_RELEVANT_PLOTTING_METRICS = [
    "Stride Time",
    "Stance Time",
    "Swing Time",
    "Cadence",
    "Knee Flexion Peak",
    "Knee Valgus Stance Mean",
    "Dorsiflexion Swing Mean",
    "Inversion Swing Mean",
]

# Color palette
COLORS = [
    "steelblue",
    "sandybrown",
    "firebrick",
    "olivedrab",
    "mediumpurple",
    "slategray",
    "peru",
    "cadetblue",
]


def lighten(color: str, amount: float = 0.3) -> Tuple[float, float, float]:
    """Lighten a color by blending it with white.

    Args:
        color: Color specification (name, hex, etc.)
        amount: Amount to lighten (0-1, where 1 is white)

    Returns:
        RGB tuple with lightened color values
    """
    c = to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)


def format_axis(ax: Axes) -> None:
    """Format plot axis with standard styling.

    Args:
        ax: Matplotlib axes object to format
    """
    ax.grid(axis="y", alpha=0.7, zorder=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = None,
    dpi: int = DPI,
    **kwargs,
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Create a subplot layout with standardized formatting for stream visualization.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches. If None, auto-calculated
            as (10, 3*nrows) for optimal stream visualization
        dpi: Figure resolution in dots per inch. Defaults to module DPI constant
        **kwargs: Additional arguments passed to plt.subplots()

    Returns:
        tuple: (fig, axs) - matplotlib Figure and array of Axes objects
    """
    if figsize is None:
        figsize = (10, 3 * nrows)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi, sharex=True, **kwargs
    )
    if isinstance(axs, np.ndarray):
        for ax in axs:
            format_axis(ax)
    else:
        format_axis(axs)
    return fig, axs


class StreamsPlotter:
    """
    Visualization class for plotting kinematic data streams with gait event annotations.

    This class provides functionality to visualize time-series kinematic data with
    overlaid gait events such as stride boundaries and walking periods. It's designed
    for exploratory data analysis and quality assessment of gait data.

    The plotter automatically downloads and processes NPZ data files, extracts relevant
    segments based on labels, and provides methods to create time-series plots with
    standardized formatting.

    Key Features:
    - Automatic data loading from Cionic API
    - Multi-stream plotting with synchronized time axes
    - Gait event overlays (stride splits, walking periods)
    - Standardized plot formatting and styling
    - Flexible subplot arrangements

    Typical Usage:
    # 1. Initialize with study/collection identifiers
    plotter = plotting.StreamsPlotter(
        org_shortname="cionic",
        study_shortname="reference_colls",
        collection_num=1,
        tokenpath="/home/jovyan/cionic-data/token.json",
    )

    # 2. Create subplot layout with subplots()
    fig, ax = plotter.subplots()

    # 3. Plot streams with plot_stream()
    plotter.plot_stream(
        ax=ax,
        label="unstimulated_walk",
        position=f"r_knee_joint",
        stream_name="euler",
        component="knee_flexion",
        label_name="Knee Flexion"
    )

    # 4. Add gait events with plot_stride_splits() and shade_walking_periods()
    kwargs = {"ax": ax, "label": "unstimulated_walk", "position": "r_shank"}
    plotter.plot_stride_splits(**kwargs)
    plotter.shade_walking_periods(**kwargs)

    # 5. Clip view to relevant periods with clip_non_gait_edges()
    plotter.clip_non_gait_edges(**kwargs)
    plt.show()
    """

    def __init__(
        self,
        org_shortname: str,
        study_shortname: str,
        collection_num: int,
        tokenpath: str,
        outdir: str = "recordings",
        segmented: bool = True,
        overwrite: bool = False,
        peak_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Initialize StreamsPlotter with data source parameters and download NPZ data.

        Args:
            org_shortname: Organization identifier (e.g., "cionic")
            study_shortname: Study protocol name (e.g., "Parkinsons")
            collection_num: Unique collection session identifier
            tokenpath: Path to authentication token file for API access
            outdir: Directory for caching downloaded NPZ files. Defaults to "recordings"
            segmented: Whether to download segmented data. Defaults to True
            overwrite: Whether to re-download existing NPZ files. Defaults to False
            peak_kwargs: Optional parameters for stride detection algorithms

        Raises:
            Exception: If API authentication fails or data cannot be downloaded
            FileNotFoundError: If token file is not found
        """
        self.org_shortname = org_shortname
        self.study_shortname = study_shortname
        self.collection_num = collection_num
        self.tokenpath = tokenpath
        self.outdir = outdir
        self.segmented = segmented
        self.npz = api.download_npz_from_metadata(
            org_shortname=self.org_shortname,
            study_shortname=self.study_shortname,
            collection_num=self.collection_num,
            tokenpath=self.tokenpath,
            outdir=self.outdir,
            segmented=segmented,
            overwrite=overwrite,
            peak_kwargs=peak_kwargs,
        )
        self.segs = self.npz['segments']

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[int, int] = None,
        dpi: int = DPI,
        **kwargs,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Create a subplot layout with standardized formatting for stream visualization.

        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Figure size (width, height) in inches. If None, auto-calculated
                as (10, 3*nrows) for optimal stream visualization
            dpi: Figure resolution in dots per inch. Defaults to module DPI constant
            **kwargs: Additional arguments passed to plt.subplots()

        Returns:
            tuple: (fig, axs) - matplotlib Figure and array of Axes objects
        """
        return create_subplots(nrows, ncols, figsize, dpi, **kwargs)

    def plot_stream(
        self,
        ax: Axes,
        label: str,
        position: str,
        stream_name: str,
        component: str,
        label_name: str = None,
        color: str = None,
        title: str = None,
        x_label: str = "Elapsed Time (s)",
        y_label: str = "Euler (deg)",
    ) -> None:
        """Plot kinematic stream data on the given axes."""
        segs_subset = self.segs[
            (self.segs["label"] == label)
            & (self.segs["position"] == position)
            & (self.segs["stream"] == stream_name)
        ]
        color = color if color else "steelblue"
        for i, seg in enumerate(segs_subset):
            stream = self.npz[seg["path"]]
            label_name = label_name if i == 0 else None
            ax.plot(
                stream["elapsed_s"], stream[component], label=label_name, color=color
            )
        ax.legend(loc="upper right")
        if title:
            ax.set_title(
                f"  {title}",
                loc="left",
                fontsize=TITLE_FONT_SIZE,
                fontweight=FONT_WEIGHT,
            )
        if x_label:
            ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
        if y_label:
            ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)

    def plot_stride_splits(
        self,
        ax: Axes,
        label: str,
        position: str,
        stream_name: str = "paired_stride_splits",
    ) -> None:
        """Plot stride boundary markers on the given axes."""
        segs = self.npz["segments"]
        segs = segs[segs["label"] == label]
        segment_nums = np.unique(segs["segment_num"])

        for segment_num in segment_nums:
            paired_stride_splits = npz_utils.retrieve_stream(
                npz=self.npz,
                position=position,
                stream=stream_name,
                segment_num=segment_num,
            )

            if paired_stride_splits is None or paired_stride_splits.shape[0] == 0:
                print(
                    f"No paired_stride_splits found for "
                    f"{position} {stream_name} {segment_num}."
                )
                continue

            for split in paired_stride_splits:
                ax.axvline(x=split['start_s'], color='green', linestyle='--', alpha=0.3)
                ax.axvline(x=split['stop_s'], color='red', linestyle='--', alpha=0.3)

    def shade_walking_periods(
        self,
        ax: Axes,
        label: str,
        position: str,
        stream_name: str = "walking_periods",
    ) -> None:
        """Shade walking periods on the given axes."""
        segs = self.npz["segments"]
        segs = segs[segs["label"] == label]
        segment_nums = np.unique(segs["segment_num"])

        for segment_num in segment_nums:
            walking_periods = npz_utils.retrieve_stream(
                npz=self.npz,
                position=position,
                stream=stream_name,
                segment_num=segment_num,
            )
            if walking_periods is None or walking_periods.shape[0] == 0:
                print(
                    f"No walking_periods found for "
                    f"{position} {stream_name} {segment_num}."
                )
                continue

            for period in walking_periods:
                ax.axvspan(
                    period["start_s"], period["stop_s"], color='blue', alpha=0.05
                )

    def clip_non_gait_edges(
        self,
        ax: Axes,
        label: Union[str, List[str]],
        position: str,
        stream_name: str = "walking_periods",
    ) -> None:
        """
        Clip plot view to exclude non-gait periods based on walking period boundaries.

        Args:
            ax: Matplotlib axes object to modify
            label: Label(s) to include in analysis. Can be a single string
                or list of strings for multiple labels
            position: Anatomical position (e.g., "l_shank")
            stream_name: Stream containing walking period data."
        """
        # Normalize label to always be a list
        if isinstance(label, str):
            label = [label]
        segs = self.npz["segments"]
        segs = segs[segs["stream"] == stream_name]
        segs = segs[np.isin(segs["label"], label)]

        min_time = np.inf
        max_time = -np.inf
        for seg in segs:
            walking_periods = npz_utils.retrieve_stream(
                npz=self.npz,
                position=position,
                stream=stream_name,
                segment_num=seg["segment_num"],
            )
            if walking_periods is None or walking_periods.shape[0] == 0:
                print(
                    f"No walking_periods found for "
                    f"{position} {stream_name} {seg['segment_num']}."
                )
                continue
            min_time = min(walking_periods["start_s"].min(), min_time)
            max_time = max(walking_periods["stop_s"].max(), max_time)
        ax.set_xlim([min_time - 3, max_time + 3])


class StreamsSplitsPlotter:
    """
    Provides visualization tools for plotting stride splits from NPZ data files.

    Key Features:
        - Downloads and caches NPZ data using API metadata and authentication.
        - Supports both segmented and unsegmented data.
        - Provides methods for creating subplots of stride splits for different labels
            and segments.
        - Allows customization of output directory, segmentation, and stride detection
            parameters.

    Typical Usage:
        # 1. Instantiate the class with organization, study, collection, and auth.
        plotter = StreamsSplitsPlotter(
            org_shortname="cionic",
            study_shortname="reference_colls",
            collection_num=1,
            tokenpath="/home/jovyan/cionic-data/token.json",
        )

        # 2. Use the `subplots` method to create custom figure layouts.
        fig, ax = plotter.subplots()

        # 3. Use the `plot_splits` method to visualize stride splits.
        plotter.plot_splits(
            ax=ax,
            label="unstimulated_walk",
            position=f"r_knee_joint",
            stream_name="euler",
            component="knee_flexion",
            splits_position="r_shank",
            label_name="Knee Flexion"
        )
        plt.show()
    """

    def __init__(
        self,
        org_shortname: str,
        study_shortname: str,
        collection_num: int,
        tokenpath: str,
        outdir: str = "recordings",
        segmented: bool = True,
        overwrite: bool = False,
        peak_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Initialize StreamsPlotter with data source parameters and download NPZ data.

        Args:
            org_shortname: Organization identifier (e.g., "cionic")
            study_shortname: Study protocol name (e.g., "Parkinsons")
            collection_num: Unique collection session identifier
            tokenpath: Path to authentication token file for API access
            outdir: Directory for caching downloaded NPZ files. Defaults to "recordings"
            segmented: Whether to download segmented data. Defaults to True
            overwrite: Whether to re-download existing NPZ files. Defaults to False
            peak_kwargs: Optional parameters for stride detection algorithms

        Raises:
            Exception: If API authentication fails or data cannot be downloaded
            FileNotFoundError: If token file is not found
        """
        self.org_shortname = org_shortname
        self.study_shortname = study_shortname
        self.collection_num = collection_num
        self.tokenpath = tokenpath
        self.outdir = outdir
        self.segmented = segmented
        self.npz = api.download_npz_from_metadata(
            org_shortname=self.org_shortname,
            study_shortname=self.study_shortname,
            collection_num=self.collection_num,
            tokenpath=self.tokenpath,
            outdir=self.outdir,
            segmented=segmented,
            overwrite=overwrite,
            peak_kwargs=peak_kwargs,
        )
        self.segs = self.npz['segments']

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[int, int] = None,
        dpi: int = DPI,
        **kwargs,
    ) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Create a subplot layout with standardized formatting for stream visualization.

        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Figure size (width, height) in inches. If None, auto-calculated
                as (10, 3*nrows) for optimal stream visualization
            dpi: Figure resolution in dots per inch. Defaults to module DPI constant
            **kwargs: Additional arguments passed to plt.subplots()

        Returns:
            tuple: (fig, axs) - matplotlib Figure and array of Axes objects
        """
        return create_subplots(nrows, ncols, figsize, dpi, **kwargs)

    def plot_splits(
        self,
        ax: Axes,
        label: str,
        position: str,
        stream_name: str,
        component: str,
        splits_position: str,
        individuals: bool = False,
        label_name: str = None,
        color: str = None,
        title: str = None,
        x_label: str = "Percent Gait Cycle (%)",
        y_label: str = "Euler (deg)",
    ) -> None:
        """Plot kinematic stream data on the given axes."""
        segs_subset = self.segs[
            (self.segs["label"] == label)
            & (self.segs["position"] == position)
            & (self.segs["stream"] == stream_name)
        ]
        color = color if color else "steelblue"

        long_matrix_list = []
        for seg in segs_subset:
            stream = self.npz[seg["path"]]
            stride_splits = npz_utils.retrieve_stream(
                npz=self.npz,
                position=splits_position,
                stream="paired_stride_splits",
                segment_num=seg["segment_num"],
            )
            matrix = tools.stream_splits_to_matrix(
                stream_data=stream,
                splits=stride_splits,
                ch_field=component,
                n_interp=tools.N_INTERP + 1,
                paired_splits=True,
            )
            long_matrix_list.append(matrix)

        if not long_matrix_list:
            return

        long_matrix = np.concatenate(long_matrix_list, axis=0)

        if individuals:
            for i in range(long_matrix.shape[0]):
                plot_label = label_name if i == 0 else None
                ax.plot(
                    long_matrix[i, :],
                    color=lighten(color, amount=0.05),
                    lw=0.8,
                    alpha=0.7,
                    label=plot_label,
                )
        else:
            stride_mean = np.mean(long_matrix, axis=0)
            ax.plot(
                stride_mean,
                color=color,
                lw=2,
                label=label_name,
            )
            stride_std = np.std(long_matrix, axis=0)
            ax.fill_between(
                np.arange(len(stride_mean)),
                stride_mean - stride_std,
                stride_mean + stride_std,
                color=color,
                alpha=0.3,
            )

        ax.legend(loc="upper right")
        if title:
            ax.set_title(
                f"  {title}",
                loc="left",
                fontsize=TITLE_FONT_SIZE,
                fontweight=FONT_WEIGHT,
            )
        if x_label:
            ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
        if y_label:
            ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)

        # Format x-axis as percentages
        def percent_formatter(x, pos):
            return f"{int(x / tools.N_INTERP * 100)}%"

        ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.set_xlim(0, tools.N_INTERP)


def violin_jitter(
    y: np.ndarray, center_x: float = 0, width: float = 0.25, n_points: int = 200
) -> np.ndarray:
    """
    Compute x jitter for scatter points to follow violin shape.

    Args:
        y: Array of data values
        center_x: X-position of this violin
        width: Max half-width of violin
        n_points: Number of points for KDE interpolation

    Returns:
        Array of jittered x-coordinates

    Note:
        For datasets with fewer than 3 points or with insufficient variance,
        falls back to minimal random jitter around the center position.
    """
    y = np.asarray(y)

    # Handle edge cases where KDE won't work properly
    if len(y) < 3:
        # Too few points for reliable KDE - use minimal jitter
        jitter_amount = width * 0.1  # Small fixed jitter
        return center_x + np.random.uniform(-jitter_amount, jitter_amount, len(y))

    # Check for insufficient variance (all values nearly identical)
    if np.std(y) < 1e-10:
        # All values are essentially the same - use minimal jitter
        jitter_amount = width * 0.05
        return center_x + np.random.uniform(-jitter_amount, jitter_amount, len(y))

    try:
        # Fit KDE on the data
        kde = gaussian_kde(y)
        ys = np.linspace(min(y), max(y), n_points)
        density = kde(ys)
        density /= density.max()  # normalize to [0,1]

        x_jittered = []
        for yi in y:
            # Interpolate normalized density at this yi
            d = np.interp(yi, ys, density)
            max_jitter = d * width * 0.95  # leave a little space
            jitter_val = np.random.uniform(-max_jitter, max_jitter)
            x_jittered.append(center_x + jitter_val)
        return np.array(x_jittered)

    except (np.linalg.LinAlgError, ValueError):
        # KDE failed (e.g., singular matrix) - fall back to uniform jitter
        jitter_amount = width * 0.3
        return center_x + np.random.uniform(-jitter_amount, jitter_amount, len(y))


class GroupedMetricsPlotter:
    """
    Visualization class for creating comparative plots of gait metrics across groups.

    It supports violin plots for showing full distributions and bar plots for
    statistical summaries with confidence intervals.

    Typical usage:
    1. Initialize with a metrics DataFrame containing multiple groups
    2. Define metric specifications (title, position, component, etc.)
    3. Generate violin plots to show distributions
    4. Generate bar plots to show statistical summaries (mean, std, CV)
    """

    def __init__(self, metrics: pd.DataFrame) -> None:
        """
        Initialize the plotter with metrics data.

        Args:
            metrics (pd.DataFrame): DataFrame containing gait metrics data.
                Must include columns:
                - group_name (str): Name of experimental group
                - group_color (str): Color specification for visualization
                - position (str): Anatomical position (e.g., 'thigh', 'shank')
                - component (str): Measurement component (e.g., 'x', 'knee_flexion')
                - Various metric columns (e.g., 'stride_time', 'peak_value', etc.)
        """
        self.metrics = metrics

    def violin_plot(
        self,
        metric_specification: Dict[str, str],
        figsize: Tuple[int, int] = FIGSIZE,
        dpi: int = DPI,
    ) -> Tuple[Optional[Figure], Optional[Axes]]:
        """
        Create a violin plot comparing metric distributions across experimental groups.

        Generates a violin plot showing the full probability distribution of a specific
        metric across different experimental groups. Each group is displayed as a violin
        shape showing the data distribution, with individual data points scattered over
        the violin using kernel density-based jittering.

        Args:
            metric_specification (dict): Dictionary specifying the metric to plot.
                Must contain keys:
                - title (str): Plot title
                - y_label (str): Y-axis label with units
                - metric_column (str): Column name in metrics DataFrame
                - position (str): Anatomical position to filter by
                - component (str): Component to filter by
            figsize (tuple, optional): Figure size (width, height) in inches.
                Defaults to FIGSIZE constant.
            dpi (int, optional): Figure resolution in dots per inch.
                Defaults to DPI constant.

        Returns:
            tuple: (fig, ax) - matplotlib Figure and Axes objects for further
                customization. Returns (None, None) if no matching data is found.
        """

        metrics = self.metrics[
            (self.metrics["position"] == metric_specification["position"])
            & (self.metrics["component"] == metric_specification["component"])
        ]
        if metrics.shape[0] == 0:
            print(
                f"No metrics found for {metric_specification['title']}. "
                f"No plot generated."
            )
            return None, None

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        group_names = metrics["group_name"].unique()
        for i, group_name in enumerate(group_names):
            group_metrics = metrics[metrics["group_name"] == group_name]
            group_color = group_metrics["group_color"].unique()
            assert group_color.shape[0] == 1, "Expected one unique color per group"
            violin = ax.violinplot(
                dataset=[group_metrics[metric_specification["metric_column"]]],
                positions=[i],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            assert len(violin["bodies"]) == 1
            violin["bodies"][0].set_facecolor("lightgray")
            violin["bodies"][0].set_edgecolor("darkgray")
            violin["bodies"][0].set_alpha(1.0)
            violin["bodies"][0].set_zorder(3)

            x_jittered = violin_jitter(
                group_metrics[metric_specification["metric_column"]], center_x=i
            )
            ax.scatter(
                x_jittered,
                group_metrics[metric_specification["metric_column"]],
                color=group_color[0],
                alpha=0.6,
                zorder=4,
            )
        format_axis(ax)
        ax.set_xticks(range(group_names.shape[0]))
        ax.set_xticklabels(
            group_names, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT
        )
        ax.set_ylabel(
            metric_specification["y_label"],
            fontsize=LABEL_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )
        ax.set_title(
            metric_specification["title"],
            loc="center",
            fontsize=TITLE_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )
        return fig, ax

    def statistical_summary_bar_plot(
        self,
        metric_specification: Dict[str, str],
        statistic: str = "mean",
        figsize: Tuple[int, int] = FIGSIZE,
        dpi: int = DPI,
    ) -> Tuple[Optional[Figure], Optional[Axes]]:
        """
        Create a bar plot showing statistical summaries with confidence intervals.

        Generates a bar chart comparing statistical summaries (mean, standard deviation,
        or coefficient of variation) across experimental groups. Each bar represents
        the computed statistic for one group, with error bars showing bootstrap
        confidence intervals.

        Args:
            metric_specification (dict): Dictionary specifying the metric to plot.
                Must contain keys:
                - title (str): Base plot title (statistic will be appended)
                - y_label (str): Y-axis label with units
                - metric_column (str): Column name in metrics DataFrame
                - position (str): Anatomical position to filter by
                - component (str): Component to filter by
            statistic (str, optional): Statistical summary to compute and display.
                Options: 'mean', 'std', 'cv' (coefficient of variation).
                Defaults to 'mean'.
            figsize (tuple, optional): Figure size (width, height) in inches.
                Defaults to FIGSIZE constant.
            dpi (int, optional): Figure resolution in dots per inch.
                Defaults to DPI constant.

        Returns:
            tuple: (fig, ax) - matplotlib Figure and Axes objects for further
                customization. Returns (None, None) if no matching data is found.

        Raises:
            ValueError: If statistic is not one of 'mean', 'std', or 'cv'.
        """
        if statistic not in ["mean", "std", "cv"]:
            raise ValueError(
                f"statistic must be one of 'mean', 'std', or 'cv'. Got {statistic}."
            )
        metrics = self.metrics[
            (self.metrics["position"] == metric_specification["position"])
            & (self.metrics["component"] == metric_specification["component"])
        ]
        if metrics.shape[0] == 0:
            print(
                f"No metrics found for {metric_specification['title']}. "
                f"No plot generated."
            )
            return None, None

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        group_names = metrics["group_name"].unique()

        for i, group_name in enumerate(group_names):
            group_metrics = metrics[metrics["group_name"] == group_name]

            group_color = group_metrics["group_color"].unique()
            assert group_color.shape[0] == 1, "Expected one unique color per group"
            stats_summary = stats.summarize_metric_distribution(
                values=group_metrics[metric_specification["metric_column"]],
            )
            ax.bar(
                x=i,
                height=stats_summary[statistic],
                color=lighten(group_color[0]),
                zorder=2,
            )
            ax.plot(
                [i, i],
                stats_summary[f"{statistic}_ci"],
                color="black",
                linewidth=2,
                zorder=4,
            )
        format_axis(ax)
        ax.set_xticks(range(group_names.shape[0]))
        ax.set_xticklabels(
            group_names, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT
        )
        y_label = metric_specification["y_label"] if statistic != "cv" else "CV (%)"
        ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
        ax.set_title(
            f"{metric_specification['title']}  |  {statistic.upper()}",
            loc="center",
            fontsize=TITLE_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )
        return fig, ax
