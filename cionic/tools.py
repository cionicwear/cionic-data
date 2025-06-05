import json
import logging
import os
import struct
import sys
from bisect import bisect_left

import cionic.bno080frps as b_frps
import cionic.kinematics_setup as kc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy import signal
from scipy.spatial.transform import Rotation, Slerp

sys.path.append(os.path.dirname(__file__))
import cionic.dsp as dsp

HP_PARAMS = {"filter_order": 5, "cutoff_freq": 50, "sampling_rate": 2000}
RMS_PARAMS = {"window_size": 301}

ADS119X_ID = 0xB6
ADS129X_ID = 0x92


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

    nyquist_f = sampling_freq / 2.0
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
    # Moving avg RMS: https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
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
    # Single variable moving average used in the hub for the processed EMG stream as of 09.12.2019
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
    Convert data from bit counts to uV
    """
    data_uV = (
        1e6 * raw_data * v_ref / (channel_gain * (2**23 - 1))
    )  # units: microvolts (uV)
    return data_uV  # these are pandas Series (RMS and FFT data are numpy arrays). Should probably pick one format and convert all others to it.


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
    ylim=[0, 0],
    leg_contents=[],
    style='-',
    legend_loc='best',
):
    fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(width, height, forward=False)

    component_plot(
        axs,
        streams,
        components=components,
        off=off,
        y_column=y_column,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        ylim=ylim,
        leg_contents=leg_contents,
        style=style,
        legend_loc=legend_loc,
    )

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
    title='',
    xlabel='',
    ylabel='',
    color='',
    ylim=[0, 0],
    leg_contents=[],
    style='-',
    legend_loc='best',
    shades=[],
):

    if type(streams) != type([]):
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
        axs.set_xlabel(f"{y_column}")
    else:
        axs.set_xlabel(f"{xlabel}")
    axs.set_ylabel(f"{ylabel}")
    axs.set_title(title)
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
    ylim=[0, 0],
    ncols=1,
    same_plot=True,
    leg_contents=[],
    style='-',
    legend_loc='upper right',
    sharex=True,
    shades=[],
):
    if same_plot:
        fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
        fig.set_size_inches(width, height, forward=False)

        component_plot(
            axs,
            streams,
            components=components,
            off=off,
            y_column=y_column,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            ylim=ylim,
            leg_contents=leg_contents,
            style=style,
            legend_loc=legend_loc,
            shades=shades,
        )

        fig.show()
    else:

        if type(streams) != type([]):
            streams = [streams]

        num_plots = len(streams)
        nrows = int(np.ceil(num_plots / ncols))

        gs = gridspec.GridSpec(nrows, ncols)
        scale = max(ncols, nrows)
        fig = plt.figure(figsize=(width, height))

        for idx, stream in enumerate(streams):
            legend = leg_contents[idx]
            if (idx > 0) and sharex == True:
                axs = fig.add_subplot(gs[idx], sharex=axs)
            else:
                axs = fig.add_subplot(gs[idx])

            component_plot(
                axs,
                [stream],
                components=components,
                off=0,
                y_column=y_column,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                color=color,
                ylim=ylim,
                leg_contents=[legend],
                style=style,
                legend_loc=legend_loc,
                shades=shades,
            )

        plt.suptitle(title)
        plt.tight_layout()
        fig.show()


def join_segments(npz, segments):
    data = {}
    for index, seg in segments.iterrows():
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
    for index, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] == 'regs' and seg['device'] not in device_regs:
            device_regs[seg['device']] = regs_data(npz, seg['path'])
        elif seg['stream'] == 'frsp' and seg['device'] + '_frsp' not in device_regs:
            frps_data_frame = b_frps.format_df(pd.DataFrame(npz[seg['path']]))
            device_regs[seg['device'] + '_frsp'] = frps_data_frame

    return device_regs


def stream_impedances(npz):
    """
    parse ads stored impedance values returning a dictionary keyed on device and field
    """
    impedances = {}
    for index, seg in pd.DataFrame(npz['segments']).iterrows():
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
    """Returns an interpolated sythetic array given two arrays. Used specifically in the context of time arrays."""
    sampling_rate = np.mean([np.median(np.diff(arr_1)), np.median(np.diff(arr_2))])
    min_timestamp = max(arr_1.min(), arr_2.min())
    max_timestamp = min(arr_1.max(), arr_2.max())
    elapsed_s_interp = np.arange(min_timestamp, max_timestamp, sampling_rate)
    # Remove edge case arrays created with values exceeding min_timestamp, max_timestamp. Not sure why this happens, but this line seems to fix the issue.
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
            f"quat_stream_1 {n_1 - quat_stream_1.shape[0]} nonincreasing samples removed (of {quat_stream_1.shape[0]} total samples)."
        )
    if n_2 != quat_stream_2.shape[0]:
        print(
            f"quat_stream_2 {n_2 - quat_stream_2.shape[0]} nonincreasing samples removed (of {quat_stream_2.shape[0]} total samples)."
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
    for index, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] not in streams:
            continue

        device_name = seg['device']
        stream_name = seg['stream']
        try:
            stream = npz[seg['path']]
        except:
            continue
        times = segment_times(seg, times)

        # turn fquat into cailbrated quat
        if stream_name in ['fquat', 'r_knee', 'l_knee']:
            print(f"converting {device_name} fquat to calibrated euler")
            stream_name = "".join(e for e in stream_name if e.isalpha()) + '2euler'
            stream = stream_quat2euler(stream, seg['calibration'], degrees=degrees)

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


def get_stream_data_joints(npz, streams):
    KINEMATICS_CONFIG = kc.get_kinematics_config()
    device_dict = {}
    for index, seg in pd.DataFrame(npz['segments']).iterrows():
        if seg['stream'] == 'fquat':
            device_dict[seg.get('position')] = npz[seg['path']]

    joint_dict = {}
    joint_component_list = []
    sides = ['left', 'right']
    joints = ['knee', 'ankle']
    for joint in joints:
        for side in sides:
            body_part_config = getattr(getattr(KINEMATICS_CONFIG, side), joint)
            factors = {
                'x': body_part_config.x.factor,
                'y': body_part_config.y.factor,
                'z': body_part_config.z.factor,
            }
            limb_comp_1, limb_comp_2 = body_part_config.limb_components

            if (
                f'{side[0]}_{limb_comp_1}' in device_dict.keys()
                and f'{side[0]}_{limb_comp_2}' in device_dict.keys()
            ):
                joint_stream = stream_quat2euler_joint(
                    device_dict[f'{side[0]}_{limb_comp_1}'],
                    device_dict[f'{side[0]}_{limb_comp_2}'],
                )
                joint_stream_name = f'{side[0]}_{joint}_joint_fquat2euler'
                for dtype in joint_stream.dtype.fields:
                    if dtype != 'elapsed_s':
                        joint_stream[dtype] = factors[dtype] * joint_stream[dtype]
                        joint_component_list.append(
                            f'{side[0]}_{joint}_joint fquat2euler {dtype} {side[0]}_{joint}'
                        )
                joint_dict[joint_stream_name] = joint_stream
    return joint_dict, joint_component_list


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
            f"\nConvert {ads_id} ch [{channel}] to uV with register values vref [{vref}V] and gain [{gain}]"
        )
    except Exception as e:
        logging.error(
            f"\nCould not convert {ads_id} ch [{channel}] from registers assuming vref [4.0] and gain [12]"
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
    # Fifth Bit: Check we are not using pullup/pulldown resistors mode (use current source mode only, more accurate)
    if (loff_regval >> 4) & 0b1:
        print(
            f'WARNING: Impedance measurement was made with pullup/pulldown resistor mode. This is less accurate. (regval: {loff_regval})'
        )

    # First/Second Bits: Check that they are valid
    if (loff_regval & 0b11) == 2:
        print(
            f'WARNING: Not valid values for the lead-off frequency (first two bits). (regval: {loff_regval})'
        )

    # 6th/7th/8th Bits: Check that max LOFF current accuracy levels are set
    if ((loff_regval >> 5) & 0b111) > 0:
        print(
            f'WARNING: The LOFF Comparator Threshold value (6th-8th bits) is not set to max possible accuracy. (regval: {loff_regval}) '
        )

    # --- Get Current ---
    # Second bit: DC or AC
    if loff_regval == 0:
        print(
            f'WARNING: current is off. Cannot calculate impedance. (regval: {loff_regval})'
        )
        raise ValueError
    elif [(loff_regval >> 1) & 0b1]:  # DC LOFF Current
        # Third and Forth Bits: Current Level
        current = {0: 6, 1: 12, 2: 18, 3: 24}[  # nA DC  # nA DC  # nA DC  # nA DC
            (loff_regval >> 2) & 0b11
        ]
        print(
            f'Current from registers is: {current}nA DC, will be using 2x that in imp calculations: {current*2}nA AC. (regval: {loff_regval})'
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
            all_gains: dictionary of gain for each of the 8 channels (defined by CHSETMAP)
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
    except:
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
    except:
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
    for i, r in frame.iterrows():
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


def find_nearest_idx(array, value):
    """Finds the index of an array whose element is closest to the value.

    Args:
        array: numpy array
        value: value to find nearest element in array

    Returns:
        index in array with closest element to value
    """
    idx = (np.abs(array - value)).argmin()
    return idx

    """
    To suppress function/library prints
        with tools.HidePrints():
            INSERT FUNCTION CALLS
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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

    Impedance is calculated by (1) taking raw ADC bitcount values, (2) converting them to uV,
    (3) windowing the data so that 1 waveform period (squarewave) is in the window, (4) taking
    the peak-to-peak value of each window (uV), and (5) converting those values to MOhms by dividing
    by the current (R = V/I).

    Outputs
        + data_MOhms - final result (array of impedance values in MOhms)
        + data_uV, data_33Hz, data_ptp - intermediate values in calculating impedance. Can be useful for auditing pipeline.

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
    Converts a stream with timestamp splits to a matrix, where each row is an interpolated split.

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
