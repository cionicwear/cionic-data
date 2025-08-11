"""
Demonstrates and tests usage of the GaitMetricsCalculator and related functions.

This module loads example kinematic data, extracts stride and segment information,
and computes gait metrics using various configurations. It saves results to CSV
files for inspection and visualizes stride and toe-off events using matplotlib.
The examples cover direct class usage, wrapper function calls, and metric subset
selection, providing practical reference for gait analysis workflows.

Example usage:
    python tests/test_metrics_calculator.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cionic import api, gait_metrics


def get_npz():
    download = (
        "cionic/collections/5hnxSDyWEZWDFEDw5pTcoMNhckJeD7N61K_fIpG1qFw/streams/npz"
    )
    npzpath = "/tmp/cionic_khe_13.npz"
    tokenpath = "token.json"

    if os.path.exists(npzpath):
        os.remove(npzpath)

    _ = api.auth(tokenpath=tokenpath)
    api.download_npz(npzpath, download, include_gait_splits=True)
    npz = np.load(npzpath)
    return npz


def main():
    npz = get_npz()

    segs = npz['segments']
    euler_shank_mask = (segs['stream'] == 'euler') & np.char.endswith(
        segs['position'], '_shank'
    )
    intervals_shank_mask = np.char.endswith(
        segs['position'], '_shank'
    ) & np.char.endswith(segs['path'], '_paired_stride_splits')

    stream_seg = segs[euler_shank_mask]
    stride_splits_seg = segs[intervals_shank_mask]

    assert stream_seg.shape[0] == 1, f"Expected 1 stream_seg, got {stream_seg.shape[0]}"
    assert (
        stride_splits_seg.shape[0] == 1
    ), f"Expected 1 stride_splits_seg, got {stride_splits_seg.shape[0]}"

    stream_seg = stream_seg[0]
    stride_splits_seg = stride_splits_seg[0]
    print(f"Strides segment: {stride_splits_seg['path']}")

    # Call the class directly.
    # Results CSV in tests/test_metrics_output/cionic/khe/3/
    metrics_calculator = gait_metrics.GaitMetricsCalculator(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        shank_stream=npz[stream_seg['path']],
        meta=gait_metrics.Metadata(
            org_shortname="cionic",
            study_shortname="khe",
            collection_num="3",
            position=stream_seg['position'],
            stream_name=stream_seg['stream'],
            component='x',
        ),
    )
    _ = metrics_calculator.calculate_metrics(output_path="tests/test_metrics_output")

    # Call the class directly, leaving out org_shortname, study_shortname,
    # and collection_num from Metadata.
    # Results CSV in tests/test_metrics_output/
    metrics_calculator = gait_metrics.GaitMetricsCalculator(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        shank_stream=npz[stream_seg['path']],
        meta=gait_metrics.Metadata(
            position=stream_seg['position'],
            stream_name=stream_seg['stream'],
            component='x',
        ),
    )
    _ = metrics_calculator.calculate_metrics(output_path="tests/test_metrics_output")

    # Call the class from compute_gait_metrics() wrapper.
    # Results CSV in tests/test_metrics_output/wrapper/cionic/khe/3/
    _ = gait_metrics.compute_gait_metrics(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        position=stream_seg['position'],
        stream_name=stream_seg['stream'],
        component='x',
        org_shortname="cionic",
        study_shortname="khe",
        collection_num="3",
        shank_stream=npz[stream_seg['path']],
        output_path="tests/test_metrics_output/wrapper",
    )

    # Call the class from compute_gait_metrics() wrapper,
    # leaving out org_shortname, study_shortname, collection_num from Metadata.
    # Results CSV in tests/test_metrics_output/wrapper/
    _ = gait_metrics.compute_gait_metrics(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        position=stream_seg['position'],
        stream_name=stream_seg['stream'],
        component='x',
        shank_stream=npz[stream_seg['path']],
        output_path="tests/test_metrics_output/wrapper",
    )

    # Call the class from compute_gait_metrics() wrapper,
    # using a subset of metrics.
    # Results CSV in tests/test_metrics_output/metrics_subset/
    metrics = [
        gait_metrics.Metric.STRIDE_TIME,
        gait_metrics.Metric.PEAK_VALUE,
    ]
    _ = gait_metrics.compute_gait_metrics(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        position=stream_seg['position'],
        stream_name=stream_seg['stream'],
        component='x',
        shank_stream=npz[stream_seg['path']],
        metrics=metrics,
        output_path="tests/test_metrics_output/metrics_subset",
    )

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot("elapsed_s", "x", "", data=npz[stream_seg['path']])

    vline_kwargs = dict(linestyle='--', alpha=0.5)
    for stride in npz[stride_splits_seg['path']]:
        ax.axvline(stride['start_s'], color='red', **vline_kwargs)
        ax.axvline(stride['stop_s'], color='green', zorder=-10)
    for toe_off_time in metrics_calculator.toe_off_times:
        ax.axvline(x=toe_off_time, color='gray', **vline_kwargs)
    plt.show()


if __name__ == "__main__":
    main()
