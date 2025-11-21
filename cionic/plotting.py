import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from cionic import api, npz_utils

FIGSIZE = (8, 6)
DPI = 100
LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
FONT_WEIGHT = "bold"


class StreamsPlotter:
    def __init__(
        self,
        org_shortname,
        study_shortname,
        collection_num,
        tokenpath,
        outdir,
        segmented,
        overwrite=False,
        peak_kwargs=None,
    ):
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

    def plot_stream(
        self, ax, label, label_name, position, stream_name, component, color, title
    ):
        segs_subset = self.segs[
            (self.segs["label"] == label)
            & (self.segs["position"] == position)
            & (self.segs["stream"] == stream_name)
        ]
        for i, seg in enumerate(segs_subset):
            stream = self.npz[seg["path"]]
            label_name = label_name if i == 0 else None
            ax.plot(
                stream["elapsed_s"], stream[component], label=label_name, color=color
            )
        ax.legend(loc="upper right")
        ax.set_title(f"    {title}", loc="left", fontweight="bold")
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_ylabel("Euler (deg)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    def plot_stride_splits(
        self, ax, label, position, stream_name="paired_stride_splits"
    ):
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
        ax,
        label,
        position,
        stream_name="walking_periods",
    ):
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
        ax,
        label_list,
        position,
        stream_name="walking_periods",
    ):
        segs = self.npz["segments"]
        segs = segs[segs["stream"] == stream_name]
        segs = segs[np.isin(segs["label"], label_list)]

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


def violin_jitter(y, center_x=0, width=0.2, n_points=200):
    """
    Compute x jitter for scatter points to follow violin shape.

    y: array of data values
    center_x: x-position of this violin
    width: max half-width of violin
    """
    # Fit KDE on the data
    kde = gaussian_kde(y)
    ys = np.linspace(min(y), max(y), n_points)
    density = kde(ys)
    density /= density.max()  # normalize to [0,1]

    x_jittered = []
    for yi in y:
        # Interpolate normalized density at this yi
        d = np.interp(yi, ys, density)
        max_jitter = d * width
        jitter_val = np.random.uniform(-max_jitter, max_jitter)
        x_jittered.append(center_x + jitter_val)
    return np.array(x_jittered)


class GroupedMetricsPlotter:
    def __init__(self, metrics):
        self.metrics = metrics

    def violin_plot(self, metric_specfication, figsize=FIGSIZE, dpi=DPI):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        metrics = self.metrics[
            (self.metrics["position"] == metric_specfication["position"])
            & (self.metrics["component"] == metric_specfication["component"])
        ]
        group_names = metrics["group_name"].unique()
        for i, group_name in enumerate(group_names):
            group_metrics = metrics[metrics["group_name"] == group_name]
            group_color = group_metrics["group_color"].unique()
            assert group_color.shape[0] == 1, "Expected one unique color per group"
            violin = ax.violinplot(
                dataset=[group_metrics[metric_specfication["metric_column"]]],
                positions=[i],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            assert len(violin["bodies"]) == 1
            violin["bodies"][0].set_facecolor("lightgray")
            violin["bodies"][0].set_edgecolor("black")

            x_jittered = violin_jitter(
                group_metrics[metric_specfication["metric_column"]], center_x=i
            )
            ax.scatter(
                x_jittered,
                group_metrics[metric_specfication["metric_column"]],
                color=group_color,
                alpha=0.6,
            )
        ax.grid(axis="y", alpha=0.7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks(range(group_names.shape[0]))
        ax.set_xticklabels(
            group_names, fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT
        )
        ax.set_ylabel(
            metric_specfication["y_label"],
            fontsize=LABEL_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )
        ax.set_title(
            metric_specfication["title"],
            loc="center",
            fontsize=TITLE_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )
        return fig, ax


# THIS CLASS IS DEPRECATED. CAN BE DELETED AFTER CONFIRMING NO USAGE.
class GroupedMetricsPlotterOld:
    def __init__(self, metrics, metric_name, column_prefix):
        self.metrics = metrics
        self.metric_name = metric_name
        self.column_prefix = column_prefix

    def plot(self, participant_id, side, ax):
        pid_metrics = self.metrics[self.metrics["pid"] == participant_id]
        x_tick_labels = []
        i = 0
        labels_in_legend = []
        for collection_num in pid_metrics["collection_num"].unique():
            coll_metrics = pid_metrics[pid_metrics["collection_num"] == collection_num]
            for label_name in coll_metrics["label_name"].unique():
                label_metrics = coll_metrics[
                    (coll_metrics["label_name"] == label_name)
                    & (coll_metrics["side"] == side)
                ]
                x_jittered = i + np.random.uniform(-0.2, 0.2, size=len(label_metrics))
                color = label_metrics["color"].unique()
                assert color.shape[0] == 1
                x_tick_labels.append(collection_num)
                if label_name not in labels_in_legend:
                    label_for_legend = label_name
                    labels_in_legend.append(label_name)
                else:
                    label_for_legend = None
                ax.scatter(
                    x_jittered,
                    label_metrics[self.metric_name],
                    color=color[0],
                    alpha=0.6,
                    label=label_for_legend,
                )
                violin = ax.violinplot(
                    dataset=[label_metrics[self.metric_name]],
                    positions=[i],
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                assert len(violin["bodies"]) == 1
                violin["bodies"][0].set_facecolor("lightgray")
                violin["bodies"][0].set_edgecolor("black")
                i += 1

        ax.set_xticks(range(i))
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel("Collection Number")
        metric_title = self.metric_name.upper().replace('_', ' ')
        if self.column_prefix is not None:
            metric_title = (
                f"{self.column_prefix.upper().replace('_', ' ')}  |  {metric_title}"
            )
        ax.set_title(
            f"Participant {participant_id}  |  {side.upper()}  |  {metric_title}",
            loc="left",
            fontweight="bold",
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc="upper right")
