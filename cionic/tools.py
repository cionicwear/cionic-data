import json
import logging
import os
import struct
import sys
from bisect import bisect_left
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp

from cionic import bno080frps, dsp, kinematics_setup, npz_utils

HP_PARAMS = {"filter_order": 5, "cutoff_freq": 50, "sampling_rate": 2000}
RMS_PARAMS = {"window_size": 301}

ADS119X_ID = 0xB6
ADS129X_ID = 0x92

LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
FONT_WEIGHT = "bold"


class AX:
    def __init__(self, axes, nrows, ncols):
        self.axes = axes
        self.nrows = nrows
        self.ncols = ncols

    def axs(self, row, col):
        if self.nrows > 1 and self.ncols > 1:
            return self.axes[row][col]
        elif self.nrows > 1:
            return self.axes[row]
        elif self.ncols > 1:
            return self.axes[col]
        else:
            return self.axes


def butter_highpass(filter_order, cutoff, sampling_freq):
    b, a = signal.butter(
        filter_order, cutoff, btype='highpass', analog=False, fs=sampling_freq
    )
    # scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    return b, a


def butter_highpass_filter(data, params):
    filter_order = params['filter_order']
    cutoff = params['cutoff_freq']
    sampling_freq = params['sampling_rate']

    b, a = butter_highpass(filter_order, cutoff, sampling_freq)

    # TODO : discuss filtfilt vs lfilter
    # return signal.lfilter(b, a, data)
    return signal.filtfilt(b, a, data)


def butter_lowpass(filter_order, cutoff, sampling_freq):
    b, a = signal.butter(
        filter_order, cutoff, btype='lowpass', analog=False, fs=sampling_freq
    )
    # scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    return b, a


def butter_lowpass_filter(data, params):
    filter_order = params['filter_order']
    cutoff = params['cutoff_freq']
    sampling_freq = params['sampling_rate']

    b, a = butter_lowpass(filter_order, cutoff, sampling_freq)

    # TODO : discuss filtfilt vs lfilter
    # return signal.lfilter(b, a, data)
    return signal.filtfilt(b, a, data)


def write_cmsis_coeff(coeff, taps_n, subsample, cutoff, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(f"{outdir}/fircoeff.h", "w") as f:
        f.write("/* AUTO GENERATED */\n\n")
        f.write("#ifndef __FIR_COEFF__\n")
        f.write("#define __FIR_COEFF__\n\n")
        f.write("#define FIR_COEFF_TAP_LEN (%d)\n" % taps_n)
        f.write("#define FIR_COEFF_SUBSAMPLE (%d)\n" % subsample)
        f.write("#define FIR_COEFF_CUTOFF (%d)\n" % cutoff)
        f.write("extern float armFirCoeffFloat[%d];\n" % taps_n)
        f.write("\n#endif\n")

    with open(f"{outdir}/fircoeff.c", "w") as f:
        f.write("/* AUTO GENERATED */\n\n")
        f.write("float armFirCoeffFloat[%d] = {\n" % taps_n)
        for c in reversed(coeff):
            f.write("    %.23f,\n" % c)
        f.write("};\n\n")


def fir_filter(data, params):
    taps_n = params['taps_n']
    cutoff = params['cutoff_freq']
    subsample = params.get('subsample', 1)
    sampling_freq = params['sampling_rate'] / subsample

    fir_coeff = signal.firwin(taps_n, cutoff=cutoff, pass_zero=False, fs=sampling_freq)
    if 'outdir' in params:
        print('writing fir')
        write_cmsis_coeff(fir_coeff, taps_n, subsample, cutoff, params['outdir'])

    # TODO : discuss filtfilt vs lfilter
    return signal.lfilter(fir_coeff, 1, data)
    # return signal.filtfilt(fir_coeff, 1, data)


def diff_filter(data, params):
    return np.diff(data, params['order'])


def no_filter(data, params):
    return data


def square(data):
    return data * data


def moving_avg_rms(data, window_size=301, mode='valid'):  # Calculate RMS
    # Moving avg RMS
    data2 = np.power(data, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(data2, window, mode))


def rms_data(data, window_size):
    out = []
    for i in range(0, len(data), window_size):
        d = data[i : i + window_size]
        out.append(np.sqrt(np.sum(np.power(d, 2)) / window_size))
    return np.array(out)


def rms_data_win(data, window_size, window_num):
    # windowed RMS roughly matching hub in how it downsamples as it RMSs
    sums = []
    for i in range(0, len(data), window_size):
        d = data[i : i + window_size]
        sos = np.sum(np.power(d, 2))
        sums.append(sos)
    out = []
    for i in range(0, len(sums)):
        d = sums[i : i + window_num]
        out.append(np.sqrt(np.sum(d) / (window_size * window_num)))
    return np.array(out)


def smooth_data(data, window_size=301):
    # Single variable moving average used in the hub for the processed
    # EMG stream as of 09.12.2019
    out = []
    smooth = 0
    for x in data:
        smooth -= smooth / window_size
        smooth += (x * x) / window_size
        out.append(smooth)
    return np.array(out)


def process_raw_emg(
    stream: np.ndarray,
    hp_filter=True,
    rectify=True,
    cutoff_spikes=True,
    max_norm=True,
    rms=True,
    n_cutoff_std=15,
    hp_params=HP_PARAMS,
    rms_params=RMS_PARAMS,
):
    if hp_filter:
        stream = butter_highpass_filter(stream, hp_params)
    if rectify:
        stream = np.abs(stream)
    if cutoff_spikes:
        cutoff = np.mean(stream) + n_cutoff_std * np.std(stream)
        cutoff_idx = np.argwhere(stream > cutoff)
        stream[cutoff_idx] = cutoff
    if max_norm:
        stream /= np.max(stream)
    if rms:
        stream = moving_avg_rms(stream, window_size=rms_params["window_size"])
    return stream


def process_raw_emg_stream(
    stream_timestamped,
    hp_filter=True,
    rectify=True,
    cutoff_spikes=False,
    max_norm=True,
    rms=True,
    n_cutoff_std=15,
    hp_params=HP_PARAMS,
    rms_params=RMS_PARAMS,
):
    fields = [name for name in stream_timestamped.dtype.names if name != "elapsed_s"]
    streams_list = []
    for field in fields:
        stream = process_raw_emg(
            stream_timestamped[field],
            hp_filter=hp_filter,
            rectify=rectify,
            cutoff_spikes=cutoff_spikes,
            max_norm=max_norm,
            rms=rms,
            n_cutoff_std=n_cutoff_std,
            hp_params=hp_params,
            rms_params=rms_params,
        )
        streams_list.append(stream)
    elapsed_s = stream_timestamped["elapsed_s"]
    if rms:
        start_idx = int((rms_params["window_size"] - 1) // 2)
        end_idx = -int(rms_params["window_size"] // 2)
        elapsed_s = elapsed_s[start_idx:end_idx]

    dtype = {
        "names": tuple(fields + ["elapsed_s"]),
        "formats": tuple(["f8"] * (len(fields) + 1)),
    }
    stream_timestamped = np.array(list(zip(*streams_list, elapsed_s)), dtype=dtype)
    return stream_timestamped


def convert_uV(raw_data, v_ref, channel_gain):
    """
    Convert data from bit counts to uV.
    Return type is pandas Series (RMS and FFT data are numpy arrays).
    Should probably pick one format and convert all others to it.
    """
    data_uV = (
        1e6 * raw_data * v_ref / (channel_gain * (2**23 - 1))
    )  # units: microvolts (uV)
    return data_uV


def find_closest_val(array, target_val):
    """
    Assumes 'array' is sorted.
    Returns (id, value) of 'array' element that is closest in value to 'target_num'.
    If two numbers are equally close, returns the smallest number.
    """
    pos = bisect_left(array, target_val)
    if pos == 0:
        return (0, array[0])
    if pos == len(array):
        return (len(array) - 1, array[len(array) - 1])
    before = array[pos - 1]
    after = array[pos]
    if (after - target_val) < (target_val - before):
        return (pos, after)
    else:
        return (pos - 1, before)


def compute_threshold_crossings(stream, threshold=0, crossing_type="both"):
    if crossing_type == "both":
        pos_crossings = compute_threshold_crossings(
            stream, threshold=threshold, crossing_type="positive"
        )
        neg_crossings = compute_threshold_crossings(
            stream, threshold=threshold, crossing_type="negative"
        )
        return np.concatenate([pos_crossings, neg_crossings], axis=0).reshape(-1)
    elif crossing_type == "positive":
        target = 2
    elif crossing_type == "negative":
        target = -2
    else:
        raise Exception("crossing_type options are 'positive' or 'negative'.")
    return np.argwhere(np.diff(np.sign(stream - threshold)) == target).reshape(-1)


def pair_two_arrays(array_1, array_2, threshold=0.5):
    paired_arrays = []
    for val_1 in array_1:
        val_2_candidate_idx = np.argmin(np.abs(array_2 - val_1))
        val_2_candidate = array_2[val_2_candidate_idx]
        if np.abs(val_1 - val_2_candidate) >= threshold:
            continue
        paired_arrays.append([val_1, val_2_candidate])
    return np.array(paired_arrays)


def _plot(ndarray, components, offset, off, axs, legend, y_column, color='', style='-'):
    """
    note: in the case of plotting multiple components only returns the last plot
    """
    ndkeys = ndarray.dtype.names
    comps = components if components else [k for k in ndkeys if k != y_column]
    for c in comps:
        offset += off
        x = ndarray[y_column]
        y = ndarray[c] + offset
        if color:
            (plot,) = axs.plot(x, y, style, color=color, label=legend)
        else:
            (plot,) = axs.plot(x, y, style, label=legend)
    axs.set_ylabel(f"{c}")
    return (offset, plot)


def simple_plot(
    streams,
    components=None,
    off=0,
    width=5,
    height=5,
    y_column='elapsed_s',
    title='',
    xlabel='',
    ylabel='',
    color='',
    ylim=None,
    leg_contents=None,
    style='-',
    legend_loc='best',
):
    if ylim is None:
        ylim = [0, 0]
    if leg_contents is None:
        leg_contents = []
    fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(width, height, forward=False)

    component_plot(
        axs,
        streams,
        components=components,
        off=off,
        y_column=y_column,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        ylim=ylim,
        leg_contents=leg_contents,
        style=style,
        legend_loc=legend_loc,
    )
    axs.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
    fig.show()


def plot_shades(axs, plots, legend, shades):
    ylims = axs.get_ylim()
    for shade in shades:
        name = shade['name']
        color = shade['color']
        first = True
        for pattern in shade['patterns']:
            if pattern[0] == pattern[1]:
                axs.vlines(
                    pattern[0], ylims[0], ylims[1], color=color, alpha=pattern[2]
                )
            else:
                plot = axs.fill_between(
                    pattern[0:2],
                    ylims[0],
                    ylims[1],
                    color=color,
                    alpha=pattern[2],
                    linewidth=0,
                )
                if first:
                    plots.append(plot)
                    legend.append(name)
                    first = False


def component_plot(
    axs,
    streams,
    components=None,
    off=0,
    y_column='elapsed_s',
    xlabel='',
    ylabel='',
    color='',
    ylim=None,
    leg_contents=None,
    style='-',
    legend_loc='best',
    shades=None,
):
    if ylim is None:
        ylim = [0, 0]
    if leg_contents is None:
        leg_contents = []
    if shades is None:
        shades = []
    if not isinstance(streams, list):
        streams = [streams]

    offset = 0
    legend = []
    plots = []
    for idx, stream in enumerate(streams):
        if color:
            (offset, plot) = _plot(
                stream,
                components,
                offset,
                off,
                axs,
                legend,
                y_column,
                color=color,
                style=style,
            )
        else:
            (offset, plot) = _plot(
                stream, components, offset, off, axs, legend, y_column, style=style
            )
        if idx < len(leg_contents):
            plots.append(plot)
            legend.append(leg_contents[idx])

    plot_shades(axs, plots, legend, shades)

    axs.legend(plots, legend, frameon=False, loc=legend_loc)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    if not xlabel:
        axs.set_xlabel(f"{y_column}", fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    else:
        axs.set_xlabel(f"{xlabel}", fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    axs.set_ylabel(f"{ylabel}", fontsize=LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
    if ylim != [0, 0]:
        axs.set_ylim(ylim)


def configurable_plot(
    streams,
    components=None,
    off=0,
    width=10,
    height=7,
    y_column='elapsed_s',
    title='',
    xlabel='',
    ylabel='',
    color='',
    ylim=None,
    ncols=1,
    same_plot=True,
    leg_contents=None,
    style='-',
    legend_loc='upper right',
    sharex=True,
    shades=None,
):
    if ylim is None:
        ylim = [0, 0]
    if leg_contents is None:
        leg_contents = []
    if shades is None:
        shades = []
    if same_plot:
        fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        fig.set_size_inches(width, height, forward=False)

        component_plot(
            axs,
            streams,
            components=components,
            off=off,
            y_column=y_column,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            ylim=ylim,
            leg_contents=leg_contents,
            style=style,
            legend_loc=legend_loc,
            shades=shades,
        )

        plt.tight_layout()
        plt.show()
    else:

        if not isinstance(streams, list):
            streams = [streams]

        num_plots = len(streams)
        nrows = int(np.ceil(num_plots / ncols))

        gs = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(figsize=(width, height))

        for idx, stream in enumerate(streams):
            legend = leg_contents[idx]
            if (idx > 0) and sharex:
                axs = fig.add_subplot(gs[idx], sharex=axs)
            else:
                axs = fig.add_subplot(gs[idx])

            component_plot(
                axs,
                [stream],
                components=components,
                off=0,
                y_column=y_column,
                xlabel=xlabel,
                ylabel=ylabel,
                color=color,
                ylim=ylim,
                leg_contents=[legend],
                style=style,
                legend_loc=legend_loc,
                shades=shades,
            )

        plt.suptitle(title, fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
        plt.tight_layout()
        plt.show()


def join_segments(npz, segments):
    data = {}
    for _, seg in segments.iterrows():
        position = seg.get('position')
        if not position:
            position = seg['device']
        key = f"{position}_{seg['stream']}"

        stream = npz[seg['path']].tolist()

        if key in data:
            # TODO : double check sorting
            data[key]['stream'].extend(stream)
        else:
            data[key] = {'stream': stream, 'dtype': npz[seg['path']].dtype}

            if seg.get('calibration'):
                data[key]['calibration'] = struct.unpack(
                    "<5f", eval(seg['calibration'])
                )
    return data


def regs_data(npz, regpath):
    regs = pd.DataFrame(npz[regpath])
    # convert the string columns from bytes to strings
    str_df = (
        regs[['regname', 'desc', 'parsedval']].stack().str.decode('utf-8').unstack()
    )
    for col in ['regname', 'desc', 'parsedval']:
        regs[col] = str_df[col]
    return regs


def segment_times(seg, times):
    start = seg.get('start_s', seg.get('first_elapsed_s'))
    end = seg.get('end_s', seg.get('last_elapsed_s'))
    if times:
        start = min(times[0], start)
        end = max(times[1], end)
    return [start, end]


def stream_regs(npz):
    device_regs = {}
    for _, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] == 'regs' and seg['device'] not in device_regs:
            device_regs[seg['device']] = regs_data(npz, seg['path'])
        elif seg['stream'] == 'frsp' and seg['device'] + '_frsp' not in device_regs:
            frps_data_frame = bno080frps.format_df(pd.DataFrame(npz[seg['path']]))
            device_regs[seg['device'] + '_frsp'] = frps_data_frame

    return device_regs


def stream_impedances(npz):
    """
    parse ads stored impedance values returning a dictionary keyed on device and field
    """
    impedances = {}
    for _, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] == 'emg':
            try:
                cal = seg['calibration']
                fields = seg['fields'].split()
                # TODO : eliminate eval
                impedance = struct.unpack(f"<{len(fields)}f", eval(cal))
                impedances[seg['device']] = {}
                for i, field in enumerate(fields):
                    impedances[seg['device']][field] = impedance[i]
            except Exception as e:
                print(e)
                print(f"no impedance for {seg['device']}")

    return impedances


def stream_calquat(stream, calibration):
    quats = []
    if calibration:
        # TODO : eliminate eval
        cal = struct.unpack("<5f", eval(calibration))
        upright = Rotation.from_quat(cal[0:4])
        forward = Rotation.from_quat([0, 0, np.sin(cal[4] / 2), np.cos(cal[4] / 2)])
        norm = Rotation.from_quat([0, 0, np.sin(-cal[4] / 2), np.cos(-cal[4] / 2)])

        for r in stream:
            orientation = Rotation.from_quat([r["i"], r["j"], r["k"], r["real"]])
            orientation = norm * orientation * upright * forward
            q = orientation.as_quat()
            quats.append((q[0], q[1], q[2], q[3], r['elapsed_s']))

    return np.array(
        quats,
        dtype={
            'names': ('x', 'y', 'z', 'w', 'elapsed_s'),
            'formats': ('f8', 'f8', 'f8', 'f8', 'f8'),
        },
    )


def process_joint_stream(
    npz: np.lib.npyio.NpzFile,
    group: str,
    position_name: str,
    values: dict,
    segment_num: int = None,
    label: str = None,
) -> Union[dict, None]:
    '''
    Processes two quaternion streams for a joint, converts them to Euler angles,
    and returns metadata and processed data. Called ONLY from get_joint_streams().

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        group (str): Group name (e.g., 'left', 'right').
        position_name (str): Name of the joint position.
        values (dict): Joint configuration dictionary.
        segment_num (int, optional): Segment number.
        label (str, optional): Segment label.

    Returns:
        dict or None: Metadata and processed data stream as a pandas DataFrame,
            or None if streams are missing.
    '''
    position_1_quats = npz_utils.retrieve_stream(
        npz=npz,
        position=values['segments'][0],
        stream='fquat',
        segment_num=segment_num,
    )
    position_2_quats = npz_utils.retrieve_stream(
        npz=npz,
        position=values['segments'][1],
        stream='fquat',
        segment_num=segment_num,
    )
    if position_1_quats is None or position_2_quats is None:
        return None

    device_name_1 = npz_utils.retrieve_segment_field(
        npz=npz,
        position=values['segments'][0],
        stream='fquat',
        field_name='device',
        segment_num=None,
    )
    device_name_2 = npz_utils.retrieve_segment_field(
        npz=npz,
        position=values['segments'][1],
        stream='fquat',
        field_name='device',
        segment_num=None,
    )

    df = pd.DataFrame(stream_quat2euler_joint(position_1_quats, position_2_quats))
    for angle_dict in values['angles']:
        df.rename(
            columns={angle_dict['rename'][0]: angle_dict['rename'][1]},
            inplace=True,
        )
        df[angle_dict['rename'][1]] = df[angle_dict['rename'][1]] * angle_dict['factor']

    col = df.pop('elapsed_s')
    df.insert(0, 'elapsed_s', col)

    meta = {
        'fields': ' '.join([c for c in df.columns.tolist() if c != 'elapsed_s']),
        'group': group,
        'position': f'{group[0]}_{position_name}',
        'device': f'{device_name_1}_{device_name_2}',
        'stream': 'euler',
        'path': f'{device_name_1}_{device_name_2}_fquat2euler',
        'start_s': df['elapsed_s'].min(),
        'end_s': df['elapsed_s'].max(),
        'duration_s': df['elapsed_s'].max() - df['elapsed_s'].min(),
        'nsamples': df.shape[0],
        'avg_rate_hz': df.shape[0] / (df['elapsed_s'].max() - df['elapsed_s'].min()),
        'data_stream': df,
    }
    if segment_num is not None:
        meta['segment_num'] = segment_num
    if label is not None:
        meta['label'] = label
    return meta


def get_joint_streams(
    npz: np.lib.npyio.NpzFile,
    included_groups: tuple[str] = ("left", "right"),
    allowable_segment_nums: Union[list[int], None] = None,
    segmented: bool = True,
) -> list[dict]:
    '''
    Generate joint angle data streams from an NPZ archive for specified body groups
    and segment numbers.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        included_groups (tuple[str], optional): Tuple of group names whose streams to
            return.
        allowable_segment_nums (list[int], optional): If provided, only segment numbers
            in this list will be processed.
        segmented (bool, optional): If True, loop over segments as in the default
            behavior. If False, ignore segments and just get stream data.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - relevant metadata fields
            - 'stream': A pandas DataFrame with elapsed time and joint angle data
    '''
    KINEMATICS_SETUP = kinematics_setup.kinematics_setup
    streams_data_packet = []

    groups = list(KINEMATICS_SETUP.keys())
    if segmented:
        segment_nums, labels = npz_utils.get_segment_nums_labels(npz)
        for segment_num, label in zip(segment_nums, labels):
            for group in groups:
                if group not in included_groups:
                    continue
                if allowable_segment_nums and segment_num not in allowable_segment_nums:
                    continue
                for position_name, values in KINEMATICS_SETUP[group]['angles'].items():
                    meta = process_joint_stream(
                        npz=npz,
                        group=group,
                        position_name=position_name,
                        values=values,
                        segment_num=segment_num,
                        label=label,
                    )
                    if meta is not None:
                        streams_data_packet.append(meta)
    else:
        for group in groups:
            if group not in included_groups:
                continue
            for position_name, values in KINEMATICS_SETUP[group]['angles'].items():
                meta = process_joint_stream(
                    npz=npz,
                    group=group,
                    position_name=position_name,
                    values=values,
                    segment_num=None,
                    label=None,
                )
                if meta is not None:
                    streams_data_packet.append(meta)

    return streams_data_packet


def get_limb_eulers(
    npz: np.lib.npyio.NpzFile, degrees: bool = True
) -> tuple[dict[str, np.recarray], np.recarray]:
    """
    Extract Euler angles from quaternion segments in a .npz file.

    Args:
        npz (np.lib.npyio.NpzFile): Opened .npz file containing segment data.
        degrees (bool, optional): If True, returns Euler angles in degrees.

    Returns:
        tuple:
            - dict[str, np.recarray]: Dict of euler_path names to Euler arrays.
            - np.recarray: Array of new segment metadata dicts for the Euler streams.
    """
    print("getting limb eulers from npz", file=sys.stderr)
    limb_eulers = {}
    new_limb_segments = []
    for seg in npz_utils.change_segments_column_dtype(npz['segments']):
        if seg['stream'] != 'fquat':
            continue
        stream = stream_quat2euler(
            stream=npz[seg['path']], calibration=seg['calibration'], degrees=degrees
        )
        euler_path = f'{seg["path"]}2euler'
        limb_eulers[euler_path] = stream

        new_segment = seg.copy()
        new_segment['path'] = euler_path
        new_segment['fields'] = 'x y z'
        new_segment['stream'] = 'euler'
        new_limb_segments.append(new_segment)

    return limb_eulers, np.array(new_limb_segments)


def get_joint_eulers(
    npz: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.recarray], list[np.recarray]]:
    '''
    Extracts joint Euler angle data and corresponding segment information from NPZ.

    Args:
        npz (np.lib.npyio.NpzFile): The NPZ file containing joint data.

    Returns:
        tuple:
            - joint_eulers (dict): A dictionary mapping each joint stream path to its
              Euler angle data as a NumPy ndarray.
            - new_joint_segments (list): A list of segment metadata arrays.
    '''
    print("getting joint eulers from npz", file=sys.stderr)
    segments = npz_utils.change_segments_column_dtype(npz['segments'])
    streams_data_packet = get_joint_streams(npz, segmented=False)
    joint_eulers = {}
    new_joint_segments = []

    for stream_data in streams_data_packet:
        data_stream = stream_data['data_stream']
        joint_eulers[stream_data['path']] = pandas_to_recarray(data_stream)

        seg_dtype = segments.dtype
        values = tuple(stream_data.get(name, '') for name in seg_dtype.names)

        new_segment = np.array([values], dtype=seg_dtype)[0]
        new_joint_segments.append(new_segment)

    return joint_eulers, new_joint_segments


def pandas_to_recarray(df: pd.DataFrame) -> np.recarray:
    """
    Convert a pandas DataFrame to a NumPy record array.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        np.recarray: The converted recarray.
    """
    array = np.array(
        list(df.itertuples(index=False)),
        dtype=np.dtype(
            {
                'names': df.columns.tolist(),
                'formats': [df[col].dtype for col in df.columns],
            }
        ),
    )
    return array


def stream_quat2euler(stream, calibration=None, degrees=True):
    if calibration:
        # TODO : eliminate eval
        cal = struct.unpack("<5f", eval(calibration))
        upright = Rotation.from_quat(cal[0:4])
        forward = Rotation.from_quat([0, 0, np.sin(cal[4] / 2), np.cos(cal[4] / 2)])
        norm = Rotation.from_quat([0, 0, np.sin(-cal[4] / 2), np.cos(-cal[4] / 2)])

    eulers = []
    for r in stream:
        orientation = Rotation.from_quat([r["i"], r["j"], r["k"], r["real"]])
        if calibration:
            orientation = norm * orientation * upright * forward

        e = orientation.as_euler('xyz', degrees=degrees)
        euler = (e[0], e[1], e[2], r['elapsed_s'])
        eulers.append(euler)

    return np.array(
        eulers,
        dtype={
            'names': ('x', 'y', 'z', 'elapsed_s'),
            'formats': ('f8', 'f8', 'f8', 'f8'),
        },
    )


def get_nonincreasing_indices(arr):
    """
    Returns all indices where an array is non increasing.
    This is used to remove nonincreasing timestamps for Slerp.
    """
    arr = np.maximum.accumulate(arr)
    ind = np.where(np.sign(np.diff(arr)) != 1)[0] + 1
    return ind


def remove_nonincreasing_entries(arr):
    """Removes all entries in an array with non increasing timestamps."""
    ind = get_nonincreasing_indices(arr["elapsed_s"])
    mask = np.ones(arr.size, dtype=bool)
    mask[ind] = False
    arr = arr[mask]
    return arr


def to_2d_array(arr):
    arr2d = np.zeros([arr.shape[0], len(arr.dtype)])
    for idx, name in enumerate(arr.dtype.names):
        arr2d[:, idx] = arr[name]
    return arr2d


def orientations_for_single_stream(arr, calibration=None):
    """Computes orientations from a single stream of quats in data."""
    ijkr_components = [name for name in arr.dtype.names if name != "elapsed_s"]
    quaternions = to_2d_array(arr[ijkr_components])
    sensor = Rotation.from_quat(quaternions)

    if calibration is not None:
        upright = Rotation.from_quat(calibration[0:4])
        forward = Rotation.from_quat(
            [0, 0, np.sin(calibration[4] / 2), np.cos(calibration[4] / 2)]
        )
        norm = Rotation.from_quat(
            [0, 0, np.sin(-calibration[4] / 2), np.cos(-calibration[4] / 2)]
        )
        orientation = norm * sensor * upright * forward
    else:
        orientation = sensor

    orientation_dict = {'orientation': orientation, 'elapsed_s': arr["elapsed_s"]}
    return orientation_dict


def get_interpolated_array_from_two_arrays(arr_1, arr_2):
    """Returns an interpolated sythetic array given two arrays.
    Used specifically in the context of time arrays.
    """
    sampling_rate = np.mean([np.median(np.diff(arr_1)), np.median(np.diff(arr_2))])
    min_timestamp = max(arr_1.min(), arr_2.min())
    max_timestamp = min(arr_1.max(), arr_2.max())
    elapsed_s_interp = np.arange(min_timestamp, max_timestamp, sampling_rate)
    # Remove edge case arrays with values exceeding min_timestamp, max_timestamp.
    # Not sure why this happens, but this line seems to fix the issue.
    elapsed_s_interp = elapsed_s_interp[
        (elapsed_s_interp <= max_timestamp) & (elapsed_s_interp >= min_timestamp)
    ]
    return elapsed_s_interp


def calculate_angle(q1, q2, seq="xyz", degrees=True):
    """Calculates the relative angle between two rotations and returns Euler."""
    relative = q1.inv() * q2
    return relative.as_euler(seq, degrees=degrees)


def stream_quat2euler_joint(quat_stream_1, quat_stream_2):
    # remove samples with nonincreasing timestamps.
    n_1 = quat_stream_1.shape[0]
    n_2 = quat_stream_2.shape[0]
    quat_stream_1 = remove_nonincreasing_entries(quat_stream_1)
    quat_stream_2 = remove_nonincreasing_entries(quat_stream_2)

    if n_1 != quat_stream_1.shape[0]:
        print(
            f"quat_stream_1 {n_1 - quat_stream_1.shape[0]} nonincreasing samples "
            f"removed (of {quat_stream_1.shape[0]} total samples)."
        )
    if n_2 != quat_stream_2.shape[0]:
        print(
            f"quat_stream_2 {n_2 - quat_stream_2.shape[0]} nonincreasing samples "
            f"removed (of {quat_stream_2.shape[0]} total samples)."
        )

    # TODO incorporate calibration features.
    calibration_1 = None
    calibration_2 = None

    # Get rotations for each stream.
    orientations_1 = orientations_for_single_stream(
        quat_stream_1, calibration=calibration_1
    )  # dict with many Rotations and time array
    orientations_2 = orientations_for_single_stream(
        quat_stream_2, calibration=calibration_2
    )  # dict with many Rotations and time array

    elapsed_s_interp = get_interpolated_array_from_two_arrays(
        orientations_1["elapsed_s"], orientations_2["elapsed_s"]
    )

    # Get interpolated rotations for each stream.
    slerp_1 = Slerp(orientations_1["elapsed_s"], orientations_1["orientation"])
    slerp_2 = Slerp(orientations_2["elapsed_s"], orientations_2["orientation"])
    orientations_1_interp = slerp_1(elapsed_s_interp)
    orientations_2_interp = slerp_2(elapsed_s_interp)

    relative = orientations_1_interp.inv() * orientations_2_interp
    eulers = relative.as_euler(seq="xyz", degrees=True)
    eulers = np.concatenate((eulers, elapsed_s_interp.reshape(-1, 1)), axis=1)
    eulers_tupled = []
    for idx in range(eulers.shape[0]):
        eulers_tupled.append(tuple(eulers[idx, :]))

    eulers_vectorized = np.array(
        eulers_tupled,
        dtype={
            'names': ('x', 'y', 'z', 'elapsed_s'),
            'formats': ('f8', 'f8', 'f8', 'f8'),
        },
    )
    return eulers_vectorized


def stream_data(npz, streams, degrees=True):
    data = {}
    components = []
    times = None
    for _, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] not in streams:
            continue

        device_name = seg['device']
        stream_name = seg['stream']
        try:
            stream = npz[seg['path']]
        except Exception:
            continue
        times = segment_times(seg, times)

        # create the device stream
        # and add fields to the components array
        device_stream = f"{device_name}_{stream_name}"
        if device_stream not in data:
            data[device_stream] = {'stream': stream.tolist(), 'dtype': stream.dtype}
            p = seg.get('position')
            if not p:
                p = "."
            components += [
                f"{device_name} {stream_name} {f} {p}"
                for f in stream.dtype.names
                if f != "elapsed_s"
            ]
        else:
            data[device_stream]['stream'].extend(stream.tolist())

    return (
        {k: np.array(v['stream'], dtype=v['dtype']) for k, v in data.items()},
        times,
        components,
    )


CHSETMAP = {
    "c1": "CH1SET",
    "c2": "CH2SET",
    "c3": "CH3SET",
    "c4": "CH4SET",
    "c5": "CH5SET",
    "c6": "CH6SET",
    "c7": "CH7SET",
    "c8": "CH8SET",
}


def regs_get(regs, regname, position='first'):
    """
    If there are register changes during a collection, multiple register values are
    appended together in the registers dataframe. 'position' allows the user to specify
    which register value they want to get ("first", "last", or specify an index)
    """
    if position == 'first':
        return regs[regs.regname == regname].regval.iloc[0]
    elif position == 'last':
        return regs[regs.regname == regname].regval.iloc[-1]
    else:
        return regs[regs.regname == regname].regval.iloc[position]


def regs_convert_uV(data, regs, channel, gain=None, position='first'):
    """
    convert from bits to uV based on register values
    """
    ads_id = {0: "UNKNOWN", ADS119X_ID: "ADS119X", ADS129X_ID: "ADS129X"}[
        regs_get(regs, "ID")
    ]
    try:
        regval = regs_get(regs, CHSETMAP[channel], position)
        gain = (
            {0: 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8, 6: 12, 8: 0}[(regval >> 4) & 0b1111]
            if gain is None
            else gain
        )  # Needed 0b1111 to distinguish between gain of 6 and OFF
        regval = regs_get(regs, "CONFIG3", position)
        vref = 4.0 if ((regval >> 5) & 0b1) else 2.4
        print(
            f"\nConvert {ads_id} ch [{channel}] to uV with register values vref "
            f"[{vref}V] and gain [{gain}]"
        )
    except Exception as e:
        logging.error(
            f"\nCould not convert {ads_id} ch [{channel}] from registers assuming "
            f"vref [4.0] and gain [12]"
        )
        logging.error(e)
        gain = 12
        vref = 4.0

    return convert_uV(data, vref, gain)


def get_current(loff_regval):
    """
    Get current value from register settings (LOFF is reg # 4 )
    Do automatic checks for bad/sub-optimal LOFF settings.
    Data sheet (pg 70 LOFF): https://www.ti.com/lit/ds/symlink/ads1296.pdf
    """
    # --- Checks ---
    # Fifth Bit: Check we are not using pullup/pulldown resistors mode
    # (use current source mode only, more accurate)
    if (loff_regval >> 4) & 0b1:
        print(
            f'WARNING: Impedance measurement was made with pullup/pulldown resistor '
            f'mode. This is less accurate. (regval: {loff_regval})'
        )

    # First/Second Bits: Check that they are valid
    if (loff_regval & 0b11) == 2:
        print(
            f'WARNING: Not valid values for the lead-off frequency (first two bits). '
            f'(regval: {loff_regval})'
        )

    # 6th/7th/8th Bits: Check that max LOFF current accuracy levels are set
    if ((loff_regval >> 5) & 0b111) > 0:
        print(
            f'WARNING: The LOFF Comparator Threshold value (6th-8th bits) is not set '
            f'to max possible accuracy. (regval: {loff_regval}) '
        )

    # --- Get Current ---
    # Second bit: DC or AC
    if loff_regval == 0:
        print(
            f'WARNING: current is off. Cannot calculate impedance. '
            f'(regval: {loff_regval})'
        )
        raise ValueError
    elif [(loff_regval >> 1) & 0b1]:  # DC LOFF Current
        # Third and Forth Bits: Current Level
        current = {0: 6, 1: 12, 2: 18, 3: 24}[  # nA DC  # nA DC  # nA DC  # nA DC
            (loff_regval >> 2) & 0b11
        ]
        print(
            f'Current from registers is: {current}nA DC, will be using 2x that in imp '
            f'calculations: {current*2}nA AC. (regval: {loff_regval})'
        )
        current = (
            2.0 * current
        )  # Assume manual FLIP method (eg. 24nA DC --> 48nA AC peak-to-peak)
        return current

    else:  # AC LOFF Current
        # Third and Forth Bits: Current Level
        current = {0: 6, 1: 12, 2: 18, 3: 24}[  # nA AC  # nA AC  # nA AC  # nA AC
            (loff_regval >> 2) & 0b11
        ]
        print(f'Current from registers is: {current}nA AC. (regval: {loff_regval})')
        return current


def regs_convert_uV_MOhms(data, regs, current=None, position='first'):
    """
    convert from uV to MOhms based on register values
    Data sheet (pg 70): https://www.ti.com/lit/ds/symlink/ads1296.pdf
    """

    # If not manually-defined, read current from register settings
    if not current:
        regval = regs_get(regs, "LOFF", position)
        current = get_current(regval)
    else:
        print(f'Using user-defined current: {current}nA AC')
    data = data / current / 1000.0  # uV / nA / 1000 = MOhms
    return data


def regs_gain_all(regs):
    """
    Loop through all 8 channels and return dictionary of gain for each channel
        Input
            regs: dataframe of register values
        Output
            all_gains: dict of gain for each of the 8 channels (defined by CHSETMAP)
                e.g. {'c1': 12, 'c2':12, 'c3':0, ..., 'c8':0}
    """
    all_gains = {}
    for c in CHSETMAP:
        all_gains[c] = regs_gain(regs, c)
    return all_gains


def regs_gain(regs, channel, position='first'):
    """
    Returns gain for a particular channel
        Inputs
            regs - dataframe of register values
            channel - options: 'c1' through 'c8'
        Outputs
            gain - gain value for the specified channel (float)
    """
    try:
        regval = regs_get(regs, CHSETMAP[channel], position='first')
        bitval = (regval >> 4) & 0b1111
        if bitval > 7:
            gain = 0  # channel off
        else:
            gain = {0: 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8, 6: 12}[bitval]
    except Exception:
        logging.error("could not convert from registers assuming gain [12]")
        gain = 12.0
    gain = float(gain)
    return gain


def regs_vref(regs, position='first'):
    """
    Returns v_ref for a particular channel
        Inputs
            regs - dataframe of register values
            channel - options: 'c1' through 'c8'
        Outputs
            vref - reference voltage value for the specified channel (float)
    """
    try:
        regval = regs_get(regs, "CONFIG3", position)
        vref = 4.0 if ((regval >> 5) & 0b1) else 2.4
    except Exception:
        logging.error("could not convert from registers assuming vref [4.0]")
        vref = 4.0
    vref = float(vref)
    return vref


def regs_sampling_rate(regs):
    """
    pull sampling rate from register value

    for ads129x High Rate fMOD = fclk/4
    for ads129x Low Rate fMOD = flck/8
    for ads119x fMOD = flck/16
    """
    try:
        idval = regs_get(regs, "ID")
        regval = regs_get(regs, "CONFIG1")
        fclk = 2048000
        hr = (regval >> 7) & 0b1

        if idval == ADS119X_ID:
            fmod = int(fclk / 16)
        elif hr:
            fmod = int(fclk / 4)
        else:
            fmod = int(fclk / 8)
        rate = {
            0: fmod >> 4,
            1: fmod >> 5,
            2: fmod >> 6,
            3: fmod >> 7,
            4: fmod >> 8,
            5: fmod >> 9,
            6: fmod >> 10,
        }[regval & 0b111]
    except Exception as e:
        logging.error("Could not read sampling rate assuming 2000")
        logging.error(e)
        return 2000

    return rate


def load_streams(npz, df=None, convert=True, degrees=False, **kwargs):
    """
    Produces a list of streams from a given npz file

    Parameters:
      npz:      npz file (required)
      df:       pandas dataframe of filtered segments (optional)
      convert:  convert 'emg' data to uV and 'fquat' data to euler
      degrees:  True to presnet eulers in degrees, false for radians
      **kwargs: filter npz segments by key value (string or list)

    Returns:
      list of dictionaries matching filter criteria
                      [{ 'stream'   : (string),
                         'position' : (string),
                         'label'    : (string),
                         'segment'  : (string),
                         'values'   : (ndarray)
                      }]
    """
    if df is None:
        df = pd.DataFrame(npz['segments'])

    # filter dataframe usings kwargs
    for k, v in kwargs.items():
        print(f"filter {k} == {v}")
        if type(v) is list:
            df = df[df[k].isin(v)]
        else:
            df = df[df[k] == v]

    # add label and segment_num if not available
    if 'label' not in df:
        df['label'] = 'none'
    if 'segment_num' not in df:
        df['segment_num'] = '0'

    results = []
    for stm, pos, lbl, pth, dev, sgm in zip(
        df['stream'],
        df['position'],
        df['label'],
        df['path'],
        df['device'],
        df['segment_num'],
    ):
        values = npz[pth]
        # convert adc uV values into properly scaled emg data
        if convert and stm == 'emg':
            print("converting EMG to uV")
            regs = regs_data(npz, f"{dev}_regs")
            for name in values.dtype.names:
                if name != "elapsed_s":
                    values[name] = regs_convert_uV(values[name], regs, name)
        # convert fquat in eulers
        elif convert and stm == 'fquat':
            print("converting FQUAT to euler")
            values = stream_quat2euler(values, calibration=None, degrees=degrees)

        results.append(
            {
                'stream': stm,
                'position': pos,
                'label': lbl,
                'segment': sgm,
                'values': values,
            }
        )
    return results


def legend_name(device, position, stream, field):
    parts = [stream, field]
    parts.insert(0, position if position and len(position) > 1 else device)
    return " ".join(parts)


def compute_signals(
    data_dict, regs_dict, streams, times, fil, rms, fft, scales, chunk_length=None
):

    signals = {
        'sig': [],
        'rms': [],
        'fft': [],
        'rms_chunks': [],
        'med_freq_chunks': [],
        'time_chunks': [],
    }
    legends = {
        'sig': [],
        'rms': [],
        'fft': [],
        'rms_chunks': [],
        'med_freq_chunks': [],
        'time_chunks': [],
    }

    stream_scales = {}

    for stream_name in streams:
        # get data and regs based on key
        (device, stream, field, *pos) = stream_name.split()
        position = '_'.join(pos)
        ln = legend_name(device, position, stream, field)
        data = data_dict[f"{device}_{stream}"]
        regs = regs_dict.get(device)

        # filter data by time
        first_sample = np.argwhere(data['elapsed_s'] >= times[0])[0][0]
        last_sample = np.argwhere(data['elapsed_s'] <= times[1])[-1][0]
        elapsed = data['elapsed_s'][first_sample:last_sample]
        output = data[field][first_sample:last_sample]

        if stream_scales.get(stream, None) is None:
            if len(stream_scales) < len(scales):
                stream_scales[stream] = scales[len(stream_scales)]
            else:
                stream_scales[stream] = scales[-1]

        scale = stream_scales[stream]

        if stream == "emg":
            fil['sampling_rate'] = regs_sampling_rate(regs)
            output = regs_convert_uV(output, regs, field)
            if fil['filter'] is not None:
                output = fil['filter'](output, fil)

            # filtered signal array
            output = output * scale
            sig_arr = np.array(
                list(tuple(zip(output, elapsed))),
                dtype={'names': ('stream', 'elapsed_s'), 'formats': ('f8', 'f8')},
            )

            # calculate fft on filtered signal
            f, data_fft = dsp.calculate_welch_density(output, fil['sampling_rate'])
            mean_freq = dsp.mean_freq(data_fft, f)
            med_freq = dsp.median_freq(data_fft, f)
            if fft == 'cdf':
                y = dsp.calculate_cdf(data_fft, f)
                fft_arr = np.array(
                    list(tuple(zip(y, f))),
                    dtype={'names': ('probability', 'Hz'), 'formats': ('f8', 'f8')},
                )
            else:
                y = dsp.dB_scale(data_fft)
                fft_arr = np.array(
                    list(tuple(zip(y, f))),
                    dtype={'names': ('db', 'Hz'), 'formats': ('f8', 'f8')},
                )

            # calculate RMS on filtered signal
            rms_out = moving_avg_rms(output, rms)
            rms_arr = np.array(
                list(tuple(zip(rms_out, elapsed))),
                dtype={'names': ('uv', 'elapsed_s'), 'formats': ('f8', 'f8')},
            )

            # calculate FFT chunks on filtered signal
            if chunk_length:
                nPts = chunk_length.value * fil['sampling_rate']
                mean_freq_chunks, med_freq_chunks, time_chunks = dsp.calc_freq_chunks(
                    output, elapsed, nPts, fil['sampling_rate']
                )
                freq_chunks_arr = np.array(
                    list(tuple(zip(mean_freq_chunks, med_freq_chunks, time_chunks))),
                    dtype={
                        'names': ('Mean_Hz', 'Median_Hz', 'elapsed_s'),
                        'formats': ('f8', 'f8', 'f8'),
                    },
                )
                leg = np.array([': Mean Freq', ': Median Freq'])
                chunks_device_legend = []
                for d in [device]:
                    chunks_device_legend.extend((d + s for s in leg))
                chunks_position_legend = []
                for p in [position]:
                    chunks_position_legend.extend((p + s for s in leg))

            signals['sig'].append(sig_arr)
            legends['sig'].append(ln)
            signals['rms'].append(rms_arr)
            legends['rms'].append(f"{ln} RMS")
            signals['fft'].append(fft_arr)
            legends['fft'].append(
                f"{ln}: Mean {mean_freq:.1f}Hz, Median {med_freq:.1f}Hz"
            )
            if chunk_length:
                signals['med_freq_chunks'].append(freq_chunks_arr)
                legends['med_freq_chunks'].append(
                    chunks_position_legend or chunks_device_legend
                )
        else:
            output = output * scale
            sig_arr = np.array(
                list(tuple(zip(output, elapsed))),
                dtype={'names': ('stream', 'elapsed_s'), 'formats': ('f8', 'f8')},
            )
            signals['sig'].append(sig_arr)
            legends['sig'].append(ln)

    return signals, legends


def csv_imu_convert(frame):
    eulers = []
    for _, r in frame.iterrows():
        orientation = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])
        e = orientation.as_euler('xyz', degrees=True)
        euler = (r['elapsed'], r['limb'], e[0], e[1], e[2])
        eulers.append(euler)
    return pd.DataFrame(eulers, columns=['elapsed', 'limb', 'x', 'y', 'z'])


def csv_limb_streams(df, skew):
    excluded = ['limb', 'elapsed']
    streams = {}
    if 'limb' in df:
        for limb in set(df['limb']):
            frame = df[
                df['limb'] == limb
            ].copy()  # explictly ask for copy to avoid warning
            if len(frame) == 0:
                continue
            # subset of streams that are not all zeroes
            comp = [
                x
                for x in frame.keys()
                if x not in excluded and not (frame[x] == 0).all()
            ]
            # add a column elapsed_s that is
            offset = frame['elapsed'].iloc[0] + skew
            frame['elapsed_s'] = (frame['elapsed'] - offset) / 10000
            streams[limb] = frame[[*comp, "elapsed_s"]].to_records(index=False)
    return streams


def csv_streams(files, directory, skew=0):
    streams = {}
    for file in files:
        (name, ext) = file.split(".")
        if ext == "csv":
            if directory:
                path = f"{directory}/{file}"
            else:
                path = file
            streams.update(csv_limb_streams(pd.read_csv(path), skew=skew))
    return streams


def collection_metadata(directory):
    path = "metadata.json"
    if directory:
        path = f"{directory}/{path}"
    with open(path, 'r') as f:
        data = json.load(f)
        return data


def csv_signals(data_dict, streams, times, scales=None, align=True):
    signals = []
    legends = []
    for stream_name in streams:
        parts = stream_name.split()
        print(parts)
        if len(parts) == 3:
            # has limb
            (stream, limb, comp) = parts
            frame = data_dict[stream]
            frame = frame[
                frame['limb'] == limb
            ].copy()  # explictly ask for copy to avoid warning
        else:
            (stream, comp) = parts
            frame = data_dict[stream].copy()

        frame = frame[(frame['elapsed'] >= times[0]) & (frame['elapsed'] < times[1])]
        print(np.min(frame['elapsed']), np.max(frame['elapsed']))
        signals.append(frame[['elapsed', comp]].to_records(index=False))
        legends.append(stream_name)
    return (signals, legends)


def get_device_group(npz, device_str):
    """Extracts the group ("left" or "right") for a device name.

    Args:
        npz: npz files
        device_str: name of the device, e.g. "DC_3801113"

    Returns:
        Group for the device
    """
    segments = pd.DataFrame(npz["segments"])
    device_segments = segments[segments["device"] == device_str]
    if any(device_segments["position"].str.startswith("r_")):
        return "right"
    elif any(device_segments["position"].str.startswith("l_")):
        return "left"
    else:
        raise Exception(f"Unable to determine group for device {device_str}")


def check_if_group_exists(segments, group):
    """Returns True if group, i.e. left or right, exists in collection data."""
    check = np.any(np.char.startswith(segments["position"], f"{group[0]}_"))
    return check


def gauss_standardize_array(array):
    """Gaussian standardization of an array.

    Args:
        array: numpy array of shape (n, )

    Returns:
        array: Gauss standardized numpy array of shape (n, )
    """
    return (array - np.mean(array)) / np.std(array)


def unit_normalize_array(array):
    """Unit normalize an array so all elements are the set [0, 1].

    Args:
        array: numpy array of shape (n, )

    Returns:
        array: Unit normalized numpy array of shape (n, )
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def calc_impedance(
    raw_data: np.array,
    window_size: int,
    num_windows: int,
    regs_dict,
    ch,
    current,
    gain=None,
    position='last',
):
    """
    Calculate impedance (MOhms) from raw bitcount values (EMG stream of NPZ)

    Impedance is calculated by (1) taking raw ADC bitcount values, (2) converting them
    to uV,(3) windowing the data so that 1 waveform period (squarewave) is in the
    window, (4) taking the peak-to-peak value of each window (uV), and (5) converting
    those values to MOhms by dividing by the current (R = V/I).

    Outputs
        + data_MOhms - final result (array of impedance values in MOhms)
        + data_uV, data_33Hz, data_ptp - intermediate values in calculating impedance.
            Can be useful for auditing pipeline.

    """
    # Convert to uV
    data = regs_convert_uV(raw_data, regs_dict, ch, gain=gain, position=position)
    data_uV = data

    # Windowing (Non-Sliding): 2000Hz to 33Hz
    data = np.reshape(
        data[: int(window_size * num_windows)], (num_windows, window_size)
    )
    data_33Hz = data

    # Peak-to-peak
    data = np.ptp(data, axis=1)
    data_ptp = data

    # uV -> MOhm
    data = regs_convert_uV_MOhms(data, regs_dict, current=current, position=position)
    data_MOhms = data

    return data_MOhms, data_ptp, data_33Hz, data_uV


def stream_splits_to_matrix(
    stream_data, splits, ch_field, n_interp, paired_splits=False
):
    """
    Converts a stream with timestamp splits to a matrix,
    where each row is an interpolated split.

    Args:
        stream_data: <array> data with columns "elapsed_s" and ch_field (possibly more)
        splits: <list> timestamp splits
        ch_field: <str> column name of channel field
        n_interp: <int> number of interpolation points, dictates n_columns in matrix

    Returns:
        2D matrix array of interpolated splits
    """
    n_splits = len(splits) - 1
    if paired_splits:
        n_splits += 1
    matrix = []
    for idx in range(n_splits):
        if paired_splits:
            start_split, end_split = splits[idx][0], splits[idx][1]
        else:
            start_split, end_split = splits[idx], splits[idx + 1]

        split_data = get_sub_stream(stream_data, start_split, end_split)
        if split_data.shape[0] == 0:
            continue
        interp_time = np.linspace(start_split, end_split, n_interp)
        split_data_interp = np.interp(
            interp_time, split_data["elapsed_s"], split_data[ch_field]
        )
        matrix.append(split_data_interp)

    return np.array(matrix)


def get_sub_stream(stream, start_timestamp, stop_timestamp):
    stream = stream[
        (stream["elapsed_s"] >= start_timestamp)
        & (stream["elapsed_s"] <= stop_timestamp)
    ]
    return stream
