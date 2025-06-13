"""
Helper functions to interpret trigger data
"""

import enum
import json
import warnings

import matplotlib.pyplot as plt
import numpy
import pandas

from cionic import tools


def get_enum_member_by_value(enum_type, value):
    for member in enum_type:
        if member.value == value:
            return member
    return None


#
# TODO: these constant values should come from the shared constants
# https://github.com/cionicwear/cde/issues/419
#
class TriggerKind(enum.IntEnum):
    TRIGGER_NONE = 0
    TRIGGER_FES = 1
    TRIGGER_KEY = 2
    TRIGGER_MTR = 3
    TRIGGER_VAL = 4
    TRIGGER_SSP = 5
    TRIGGER_ZSP = 6
    TRIGGER_NUM_TYPES = 7


FES_CHANNEL = numpy.uint64(0x0000000000FFFFFF)  # 24 bits for mux configuration
FES_INTENSITY = numpy.uint64(0x0000000FFF000000)  # 12 bits for intensity
FES_FORM = numpy.uint64(0x000000F000000000)  # 4  bits for waveform
FES_PAYLOAD = numpy.uint64(0xFFFFFF0000000000)  # 24 bits for payload


class FESForm(enum.IntEnum):
    FES_OFF = 0
    FES_TRAPEZOID = 1
    FES_RAMP = 2
    FES_TRAPEZOID_PERC = 3
    FES_RAMP_PERC = 4
    FES_TRAPEZOID_EXT = 5
    FES_RAMP_EXT = 6


FES_TRAPEZOID_UP = (numpy.uint64(0x000000FF),)  # ramp up time in centiseconds
FES_TRAPEZOID_TOP = (numpy.uint64(0x0000FF00),)  # top time in centiseconds
FES_TRAPEZOID_DN = (numpy.uint64(0x00FF0000),)  # ramp dn time in centiseconds

FES_RAMP_RAMP = (numpy.uint64(0x000000FF),)  # ramp up time in centiseconds

CS_TO_S = 0.01
DS_TO_S = 0.1

MAX_INTENSITY_MA = 100
MAX_INTENSITY_FW = 4000
INTERP_FREQ = 100


class FESMuscleName(enum.IntEnum):
    ZRH = 1
    ZRQ = 2
    ZLQ = 3
    ZLH = 4
    ZRT = 5
    ZLT = 6
    ZRC = 7
    ZLC = 8
    ZRP = 9
    ZLP = 10


class FESSetting(enum.IntEnum):
    FESMETA_MUX = 0
    FESMETA_FREQUENCY = 1
    FESMETA_DISCHARGE = 2
    FESMETA_PULSEWIDTH = 3
    FESMETA_TIMEOUT = 4
    FESMETA_PERCENT = 5
    FESMETA_PAUSED = 6


COLORS = ['red', 'blue', 'green', 'orange', 'brown']


def mask_count(mask):
    """
    Determine how much to bit shift (right) a value to match first set bit of the mask
    :param mask: bitmask value
    :return: number of bits to shift
    """
    m = mask
    bc = numpy.uint64(0)
    one = numpy.uint64(1)
    while bc < 64:
        if numpy.bitwise_and(m, one) == one:
            break
        m = m >> one
        bc = bc + one
    return bc


def mask_down(value, mask):
    """
    Get the unpacked value of the field specified by the mask.
    :param value: original value (or array of values)
    :param mask: bitmask
    :return: unpacked value (or array of values)
    """
    return numpy.bitwise_and(value, mask) >> mask_count(mask)


def get_algo(npz, aid):
    """
    Return the algo stream specified by the aid (stream num)
    :param npz: numpy object containing the streams
    :param aid: stream num
    :return: dataframe of the stream
    """
    columns_to_rename = ['c1091c-1212', 'stream_num']
    df = pandas.DataFrame(npz['segments'])
    for column in columns_to_rename:
        df.columns = df.columns.str.replace(column, 'streamnum')
    return df.query(f"streamnum=='{aid}'")


def get_algo_stream(npz, algo_stream_num):
    """
    Return the algo stream data specified by the aid (stream num)
    :param npz: numpy object containing the streams
    :param algo_stream_num: stream num
    :return: numpy array of stream data
    """
    algo = get_algo(npz, algo_stream_num).squeeze()
    algo_stream = npz[algo["path"]]
    return algo_stream


def get_algo_stream_by_name(npz, algo_stream_name, side):
    """
    Return the algo stream data specified by the algo stream name and side
    :param npz: numpy object containing the streams
    :param algo_stream_name: algo stream name
    :param side: str side of body
    :return: numpy array of stream data
    """
    segs = pandas.DataFrame(npz["segments"])
    segs = segs[segs["position"].str.startswith(f"{side[0].lower()}_")]
    segs = segs[segs["stream"] == algo_stream_name].squeeze()
    algo_stream = npz[segs["path"]]
    return algo_stream


def get_action(npz, stream):
    """
    Return a dataframe of interpreted action stream data
    :param npz: nump object containing the collection streams
    :param stream: stream name, e.g., 'DC_2752556_action'
    :return: interpreted dataframe with the following columns:
        kind
        algo_stream_num
        muscle
        intensity
        form
        tval
        elapsed_s
    """
    action_stream = []
    for action in npz[stream]:
        if action['kind'] == TriggerKind.TRIGGER_FES.value:
            tcmd = action['tcmd']
            muscle = mask_down(tcmd, FES_CHANNEL)
            intensity = mask_down(tcmd, FES_INTENSITY)
            form = mask_down(tcmd, FES_FORM)
            form_val = FESForm(form).name
            if form in (
                FESForm.FES_TRAPEZOID.value,
                FESForm.FES_TRAPEZOID_PERC.value,
                FESForm.FES_TRAPEZOID_EXT.value,
            ):
                payload = mask_down(tcmd, FES_PAYLOAD)
                up = mask_down(payload, numpy.uint64(FES_TRAPEZOID_UP))[0]
                top = mask_down(payload, numpy.uint64(FES_TRAPEZOID_TOP))[0]
                dn = mask_down(payload, numpy.uint64(FES_TRAPEZOID_DN))[0]
                form_val += f" {up} {top} {dn}"
            elif form in (
                FESForm.FES_RAMP.value,
                FESForm.FES_RAMP_PERC.value,
                FESForm.FES_RAMP_EXT.value,
            ):
                payload = mask_down(tcmd, FES_PAYLOAD)
                ramp = mask_down(payload, numpy.uint64(FES_RAMP_RAMP))[0]
                form_val += f" {ramp}"

            action_stream.append(
                [
                    TriggerKind(action['kind']).name,
                    action['aid'],
                    muscle,
                    intensity,
                    form_val,
                    action['tval'],
                    action['elapsed_s'],
                ]
            )
    return pandas.DataFrame(
        data=action_stream,
        columns=[
            'kind',
            'algo_stream_num',
            'muscle',
            'intensity',
            'form',
            'tval',
            'elapsed_s',
        ],
    )


def get_action_csv(action_stream):
    """
    Return a dataframe of interpreted action stream data
    :param npz: nump object containing the collection streams
    :param stream: stream name, e.g., 'DC_2752556_action'
    :return: interpreted dataframe with the following columns:
        kind
        algo_stream_num
        muscle
        intensity
        form
        tval
        elapsed_s
    """
    actions = action_stream["key"].to_numpy(dtype=numpy.uint64)
    triggers = action_stream["trigger"].to_numpy(dtype=numpy.uint64)
    elapses = action_stream["elapsed"].to_numpy(dtype=numpy.uint64)
    vals = action_stream["val"].to_numpy(dtype=numpy.uint64)
    action_stream = []
    for idx, tcmd in enumerate(actions):
        if triggers[idx] == TriggerKind.TRIGGER_FES.value:
            muscle = mask_down(tcmd, FES_CHANNEL)
            name = FESMuscleName(muscle).name if muscle != 0 else None
            intensity = mask_down(tcmd, FES_INTENSITY)
            form = mask_down(tcmd, FES_FORM)
            form_val = FESForm(form).name
            if form in (
                FESForm.FES_TRAPEZOID.value,
                FESForm.FES_TRAPEZOID_PERC.value,
                FESForm.FES_TRAPEZOID_EXT.value,
            ):
                payload = mask_down(tcmd, FES_PAYLOAD)
                up = mask_down(payload, numpy.uint64(FES_TRAPEZOID_UP))[0]
                top = mask_down(payload, numpy.uint64(FES_TRAPEZOID_TOP))[0]
                dn = mask_down(payload, numpy.uint64(FES_TRAPEZOID_DN))[0]
                form_val += f" {up} {top} {dn}"
            elif form in (
                FESForm.FES_RAMP.value,
                FESForm.FES_RAMP_PERC.value,
                FESForm.FES_RAMP_EXT.value,
            ):
                payload = mask_down(tcmd, FES_PAYLOAD)
                ramp = mask_down(payload, numpy.uint64(FES_RAMP_RAMP))[0]
                form_val += f" {ramp}"

            action_stream.append(
                [
                    TriggerKind(triggers[idx]).name,
                    muscle,
                    name,
                    intensity,
                    form_val,
                    vals[idx],
                    elapses[idx],
                ]
            )
    return pandas.DataFrame(
        data=action_stream,
        columns=['kind', 'muscle', 'name', 'intensity', 'form', 'tval', 'elapsed'],
    )


def compute_ramp(last, start, form):
    form_params = form.split(' ')

    # cycle forms
    if form_params[0] in [FESForm.FES_TRAPEZOID_PERC.name, FESForm.FES_RAMP_PERC.name]:
        if last and start:
            cycle = start - last
            percent = float(form_params[1])
            ramp = cycle * percent / 100
            return ramp
    # EXT forms
    if form_params[0] in [FESForm.FES_TRAPEZOID_EXT.name, FESForm.FES_RAMP_EXT.name]:
        return numpy.float(form_params[1]) * DS_TO_S
    # regular forms
    if form_params[0] in [FESForm.FES_TRAPEZOID.name, FESForm.FES_RAMP.name]:
        return numpy.float(form_params[1]) * CS_TO_S

    # time forms - tbd
    return 0


def compute_stims(
    npz,
    actions,
    times=None,
    colors=COLORS,
    ts='elapsed_s',
):
    """
    Return a dictionary of stim patterns
    :param npz: nump object containing the collection streams
    :param actions: list of strings coressponding to action streams
    :return:
    """
    stims = []
    for stream, muscles in actions.items():
        tdf = get_action(npz, stream)
        stims += compute_stim_muscles(tdf, muscles, times=times, colors=colors, ts=ts)
    return stims


def compute_stim_muscles(
    tdf,
    muscles,
    times=None,
    colors=COLORS,
    ts='elapsed',
):
    """
    Return a dictionary of stim patterns
    :param stim_frame: pandas dataframe containing muscle stims
    :param muscles: list of muscle names
    :return:
    """
    stims = []
    for idx, muscle in enumerate(muscles):
        muscle_num = int(muscle)
        name = FESMuscleName(muscle_num).name if muscle_num != 0 else None
        muscle_actions = tdf[tdf["muscle"] == muscle_num]
        color = colors[idx]
        start = None
        start_ramp = None
        last = None
        patterns = []
        for _, row in muscle_actions.iterrows():
            elapsed = row[ts]
            if times and (elapsed < times[0] or elapsed > times[1]):
                if row['intensity'] == 0:
                    last = elapsed
                continue
            if row['intensity'] == 0 and start:
                end_ramp = compute_ramp(last, start, row['form'])
                # relax trigger can fire before ramp finishes
                if elapsed < start + start_ramp:
                    start_ramp = elapsed - start
                # ramp up
                patterns.append([start, start + start_ramp, 0.1])
                # top
                patterns.append([start + start_ramp, elapsed, 0.2])
                # ramp down
                patterns.append([elapsed, elapsed + end_ramp, 0.1])

                # if lines desired
                # patterns.append([elapsed, elapsed, 0.5])
                # patterns.append([start+start_ramp, start+start_ramp, 0.5])

                last = start
                start = None
                start_ramp = None
            elif not start and row['intensity'] > 0:
                start = elapsed
                start_ramp = compute_ramp(last, start, row['form'])

        stims.append(
            {'name': f"stim {muscle} {name}", 'color': color, 'patterns': patterns}
        )
    return stims


def get_action_streams(npz, segment_num=None):
    """
    Return a list of action streams
    :param npz: nump object containing the collection streams
    :return:
    """
    segments = npz["segments"]
    if segment_num is not None and "segment_num" in segments.dtype.names:
        segments = segments[segments["segment_num"] == segment_num]
    return segments[segments["stream"] == "action"]["path"].tolist()


def check_for_action(streams):
    actions = {}
    new_streams = []
    for stream_name in streams:
        # get data and regs based on key
        (device, stream, field, pos, *extra) = stream_name.split()
        if stream == "action" and field == "stim":
            stream_name = f"{device}_{stream}"
            actions[stream_name] = actions.get(stream_name, []) + [pos]
        else:
            new_streams.append(stream_name)
    return (actions, new_streams)


def check_for_action_csv(streams):
    muscles = []
    new_streams = []
    for stream_name in streams:
        # get data and regs based on key
        (field, pos, *extra) = stream_name.split()
        if field == "stim":
            muscles.append(pos)
        else:
            new_streams.append(stream_name)
    return (muscles, new_streams)


def get_action_muscles(npz, segment_num=None):
    muscles = []
    for stream in get_action_streams(npz, segment_num=segment_num):
        (dev, num, action) = stream.split('_')
        tdf = get_action(npz, stream)
        for muscle in set(tdf["muscle"]):
            muscle_name = FESMuscleName(muscle).name
            muscles.append(f"{dev}_{num} {action} stim {muscle} {muscle_name}")
    return muscles


def get_fesmeta(npz, side=None):
    """
    Return a dataframe of interpreted fesmeta data
    :param npz: nump object containing the collection streams
    :param side: str side of body
    :return: interpreted dataframe with the following columns:
        setting
        muscle
        val
        elapsed_s
    """
    if side is None:
        # assumes sleeve on one leg only
        (fesmeta_file,) = [f for f in npz.files if 'fesmeta' in f]
    else:
        # allows for sleeve on both legs
        assert (
            side.lower() == "left" or side.lower() == "right"
        ), f"side must be 'left' or 'right', given: {side}"
        segs = pandas.DataFrame(npz["segments"])
        segs = segs[segs["position"].str.startswith(f"{side[0].lower()}_")]
        segs = segs[segs["device"].str.startswith("DC")]
        device_list = segs["device"].unique()
        assert device_list.shape[0] == 1
        device = device_list[0]
        fesmeta_file = f"{device}_fesmeta"

    fesmeta_steam = []
    for fesmeta_row in npz[fesmeta_file]:
        setting = FESSetting(fesmeta_row['setting']).name
        key = fesmeta_row['key']
        muscle = FESMuscleName(key).name if key != 0 else None
        val = fesmeta_row['val']
        if setting == "FES_META_MUX":
            # convert bit field to readable circuit switches
            pass
        elapsed_s = fesmeta_row['elapsed_s']
        fesmeta_steam.append([setting, muscle, val, elapsed_s])

    return pandas.DataFrame(
        data=fesmeta_steam, columns=['setting', 'muscle', 'val', 'elapsed_s']
    )


def get_fes_trigger_metadata(npz, side):
    """
    Return a list of dicts with trigger metadata from protocol
    :param npz: nump object containing the collection streams
    :param side: str side of body
    :return: list of dicts
    """

    protocol = json.loads(npz["protocol.json"])
    protocol_fes = protocol["setup"].get("fes")
    if protocol_fes is None:
        return None

    for protocol_fes_zone in protocol_fes:
        if side == protocol_fes_zone["zone"]:
            triggers = protocol_fes_zone["triggers"]
            return triggers
    return None


def get_fes_trigger_algo_names(npz, side):
    """
    Return a list of dicts with trigger algo names
    :param npz: nump object containing the collection streams
    :param side: str side of body
    :return: list of dicts with keys:
        contract
        relax
    """
    triggers = get_fes_trigger_metadata(npz, side)
    if triggers is None:
        return None
    algo_names = []
    for trigger in triggers:
        algo_names.append(
            {"contract": trigger["contract"]["algo"], "relax": trigger["relax"]["algo"]}
        )
    return algo_names


def extract_muscle_actions_from_action_stream(muscle, action_stream):
    muscle_action_stream = (
        action_stream[action_stream["muscle"] == muscle]
        .sort_values(by="elapsed_s")
        .reset_index(drop=True)
    )
    return muscle_action_stream


def compute_stim_profile_from_action_uninterpolated(muscle_action_stream):
    elapsed_list = [muscle_action_stream.loc[0, "elapsed_s"]]
    intensity_list = [0.0]
    prev_intensity = 0.0
    prev_trigger_type = "relax"
    prev_timeout = 200 * CS_TO_S
    for _, action in muscle_action_stream.iterrows():
        form_split = action["form"].split(" ")
        intensity = action["tval"] / MAX_INTENSITY_FW * MAX_INTENSITY_MA
        timestamp = action["elapsed_s"]

        ramp = numpy.float(form_split[1]) * CS_TO_S

        if (
            prev_trigger_type == "contract"
            and timestamp > elapsed_list[-2] + prev_timeout
        ):
            # TODO: logic to handle timeouts
            # elapsed_list += 2 * [elapsed_list[-2] + prev_timeout]
            # intensity_list += 2 * [prev_intensity, 0]
            # print(timestamp)
            # print(form_split)
            pass

        elapsed_list += [timestamp, timestamp + ramp]
        intensity_list += [prev_intensity, intensity]

        prev_intensity = intensity

        if form_split[0] == "FES_TRAPEZOID":
            prev_trigger_type = "contract"
            prev_timeout = numpy.float(form_split[2]) * CS_TO_S
        else:
            prev_trigger_type = "relax"

    return numpy.array(
        list(tuple(zip(intensity_list, elapsed_list))),
        dtype={"names": ("intensity", "elapsed_s"), "formats": ("f8", "f8")},
    )


def compute_stim_profile_from_action(muscle_action_stream, interp_freq=INTERP_FREQ):
    assert interp_freq >= 1, f"interp_freq must be > 1. {interp_freq} was given."
    muscle_stim_profile = compute_stim_profile_from_action_uninterpolated(
        muscle_action_stream
    )
    if interp_freq == 1:
        return muscle_stim_profile

    min_timestamp = numpy.min(muscle_stim_profile["elapsed_s"])
    max_timestamp = numpy.max(muscle_stim_profile["elapsed_s"])
    time_range = max_timestamp - min_timestamp
    n_interp = numpy.int(time_range * interp_freq)
    elapsed_interp = numpy.linspace(min_timestamp, max_timestamp, n_interp)
    intensity_interp = numpy.interp(
        elapsed_interp,
        muscle_stim_profile["elapsed_s"],
        muscle_stim_profile["intensity"],
    )
    return numpy.array(
        list(zip(intensity_interp, elapsed_interp)),
        dtype={"names": ("intensity", "elapsed_s"), "formats": ("f8", "f8")},
    )


def extract_all_stim_profiles(npz, segment_num=None, interp_freq=INTERP_FREQ):
    action_stream_names = get_action_streams(npz, segment_num=segment_num)
    if len(action_stream_names) == 0:
        return None
    stim_profiles_dict = {}
    for action_stream_name in action_stream_names:
        action_stream = get_action(npz, action_stream_name)
        muscles = action_stream["muscle"].unique()
        for muscle in muscles:
            muscle_action_stream = extract_muscle_actions_from_action_stream(
                muscle, action_stream
            )
            muscle_stim_profile = compute_stim_profile_from_action(
                muscle_action_stream, interp_freq=interp_freq
            )
            stim_profiles_dict[FESMuscleName[muscle]] = muscle_stim_profile
    return stim_profiles_dict


def merge_trigger_timestamps_to_plot_stream(
    contract_timestamps: numpy.ndarray,
    relax_timestamps: numpy.ndarray,
    on_ramp: float,
    off_ramp: float,
    interp_hz: int = 100,
):
    """Depricated. This used the protocol to look up ramps.
    The new version uses fes_form in the action_stream.
    """
    synthetic_plot_stream_x, synthetic_plot_stream_y = [], []

    for contract_timestamp in contract_timestamps:
        synthetic_plot_stream_x += [contract_timestamp, contract_timestamp + on_ramp]
        synthetic_plot_stream_y += [0, 1]

        next_relax_timestamp = numpy.min(
            numpy.where(
                relax_timestamps - contract_timestamp > 0, relax_timestamps, numpy.inf
            )
        )

        synthetic_plot_stream_x += [
            next_relax_timestamp,
            next_relax_timestamp + off_ramp,
        ]
        synthetic_plot_stream_y += [1, 0]

    min_timestamp = min(numpy.min(contract_timestamps), numpy.min(relax_timestamps))
    max_timestamp = max(numpy.max(contract_timestamps), numpy.max(relax_timestamps))
    n_interp = int((max_timestamp - min_timestamp) * interp_hz)
    interp_stream = numpy.linspace(min_timestamp, max_timestamp, n_interp)

    synthetic_plot_stream_y = numpy.interp(
        interp_stream, synthetic_plot_stream_x, synthetic_plot_stream_y
    )
    synthetic_plot_stream_x = interp_stream
    return numpy.array(
        list(zip(synthetic_plot_stream_y, synthetic_plot_stream_x)),
        dtype={"formats": ("f8", "f8"), "names": ("trigger", "elapsed_s")},
    )


def get_synthetic_trigger_streams(npz, side):
    """Depricated. This used the protocol to look up ramps.
    The new version uses fes_form in the action_stream.
    """
    warnings.warn(
        "get_synthetic_trigger_streams_depr() is depricated. "
        "Use get_synthetic_stim_profiles() instead."
    )
    trigger_metadata_list = get_fes_trigger_metadata(npz, side)
    synthetic_trigger_stream_dict = {}
    for trigger_metadata in trigger_metadata_list:
        muscle = trigger_metadata["muscle"]
        contract_stream_name = trigger_metadata["contract"]["algo"]
        relax_stream_name = trigger_metadata["relax"]["algo"]
        contract_ramp = trigger_metadata["contract"].get("ramp", 0) / 1000.0
        relax_ramp = trigger_metadata["relax"].get("ramp", 0) / 1000.0
        contract_timestamps = get_algo_stream_by_name(
            npz=npz, algo_stream_name=contract_stream_name, side=side
        )["elapsed_s"]
        relax_timestamps = get_algo_stream_by_name(
            npz=npz, algo_stream_name=relax_stream_name, side=side
        )["elapsed_s"]
        synthetic_trigger_stream = merge_trigger_timestamps_to_plot_stream(
            contract_timestamps=contract_timestamps,
            relax_timestamps=relax_timestamps,
            on_ramp=contract_ramp,
            off_ramp=relax_ramp,
        )
        synthetic_trigger_stream_dict[muscle] = synthetic_trigger_stream
    return synthetic_trigger_stream_dict


def plot_stims_with_streams(npz, stim_config, streams, times=None, width=10, height=7):
    """ """
    fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    fig.set_size_inches(width, height, forward=False)

    legend = []
    plots = []
    for stream in streams:
        sv = stream['values']
        if times:
            sv = sv[sv['elapsed_s'] >= times[0]]
            sv = sv[sv['elapsed_s'] <= times[1]]
        (plot,) = axs.plot(sv['elapsed_s'], sv['x'])
        plots.append(plot)
        legend.append(stream['position'])

    stims = compute_stims(npz, stim_config, times=times)
    tools.plot_shades(axs, plots, legend, stims)
    axs.legend(plots, legend)

    fig.show()
