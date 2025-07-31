"""
Demonstrates and tests usage of the GaitMetricsCalculator and related functions.

This module loads example kinematic data, extracts stride and segment information,
and computes gait metrics using various configurations. It saves results to CSV
files for inspection and visualizes stride and toe-off events using matplotlib.
The examples cover direct class usage, wrapper function calls, and metric subset
selection, providing practical reference for gait analysis workflows.
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

    # download = (
    #     "cionic/collections/sjqW785pPLuWF-JxsD60r7WNmDS3Xy18p01Nzj7OVRI/streams/npz"
    # )
    # npzpath = "cionic_Apollo_Adult_897.npz"

    if os.path.exists(npzpath):
        os.remove(npzpath)

    _ = api.auth(tokenpath=tokenpath)
    api.download_npz(npzpath, download, include_gait_splits=True)
    npz = np.load(npzpath)
    return npz


def main():
    npz = get_npz()

    for seg in npz['segments']:
        is_euler = seg['stream'] == 'euler'
        is_shank = seg['position'].endswith('_shank')
        is_intervals = seg['stream'] == 'intervals'
        if is_euler and is_shank:
            print(f"Kinematics segment: {seg['path']}")
            stream_seg = seg
        if is_intervals and is_shank:
            print(f"Strides segment: {seg['path']}")
            stride_splits_seg = seg

    # Call the class directly.
    # Results CSV in tests/test_metrics_output/cionic/khe/3/
    metrics_calculator = gait_metrics.GaitMetricsCalculator(
        stream=npz[stream_seg['path']],
        stride_splits=npz[stride_splits_seg['path']],
        shank_stream=npz[stream_seg['path']],
        meta=gait_metrics.Metadata(
            orgid="cionic",
            study="khe",
            collection_num="3",
            position=stream_seg['position'],
            stream_name=stream_seg['stream'],
            component='x',
        ),
    )
    _ = metrics_calculator.calculate_metrics(output_path="tests/test_metrics_output")

    # Call the class directly, leaving out orgid, study, collection_num from Metadata.
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
        orgid="cionic",
        study="khe",
        collection_num="3",
        shank_stream=npz[stream_seg['path']],
        output_path="tests/test_metrics_output/wrapper",
    )

    # Call the class from compute_gait_metrics() wrapper,
    # leaving out orgid, study, collection_num from Metadata.
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
