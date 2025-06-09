from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal


@dataclass
class AxisOperation:
    rename: str
    factor: np.float64


@dataclass
class BodyPartConfig:
    limb_components: Optional[List[str]]
    x: AxisOperation
    y: AxisOperation
    z: AxisOperation


@dataclass
class BodyPartListConfig:
    pelvis: BodyPartConfig
    hip: BodyPartConfig
    knee: BodyPartConfig
    ankle: BodyPartConfig
    foot: BodyPartConfig

    @property
    def valid_attrs(self):
        return sorted(self.__dict__.keys())


@dataclass
class KinematicsJointConfig:
    left: BodyPartListConfig
    right: BodyPartListConfig


left_pelvis = BodyPartConfig(
    limb_components=["hip", "thigh"],
    x=AxisOperation(rename="pelvic_tilt", factor=-1.0),
    y=AxisOperation(rename="pelvic_obliquity", factor=1.0),
    z=AxisOperation(rename="pelvic_int_rotation", factor=-1.0),
)

left_hip = BodyPartConfig(
    limb_components=None,
    x=AxisOperation(rename="hip_flexion", factor=1.0),
    y=AxisOperation(rename="hip_adduction", factor=-1.0),
    z=AxisOperation(rename="hip_int_rotation", factor=-1.0),
)

left_knee = BodyPartConfig(
    limb_components=["thigh", "shank"],
    x=AxisOperation(rename="knee_flexion", factor=-1.0),
    y=AxisOperation(rename="knee_adduction", factor=-1.0),
    z=AxisOperation(rename="knee_int_rotation", factor=-1.0),
)

left_ankle = BodyPartConfig(
    limb_components=["shank", "foot"],
    x=AxisOperation(rename="dorsi_flexion", factor=1.0),
    y=AxisOperation(rename="ankle_inversion", factor=-1.0),
    z=AxisOperation(rename="ankle_int_rotation", factor=-1.0),
)

left_foot = BodyPartConfig(
    limb_components=None,
    x=AxisOperation(rename="foot_floor_angle", factor=1.0),
    y=AxisOperation(rename="foot_supination", factor=-1.0),
    z=AxisOperation(rename="foot_int_rotation", factor=1.0),
)

right_pelvis = BodyPartConfig(
    limb_components=["hip", "thigh"],
    x=AxisOperation(rename="pelvic_tilt", factor=-1.0),
    y=AxisOperation(rename="pelvic_obliquity", factor=-1.0),
    z=AxisOperation(rename="pelvic_int_rotation", factor=1.0),
)

right_hip = BodyPartConfig(
    limb_components=None,
    x=AxisOperation(rename="hip_flexion", factor=1.0),
    y=AxisOperation(rename="hip_adduction", factor=1.0),
    z=AxisOperation(rename="hip_int_rotation", factor=1.0),
)

right_knee = BodyPartConfig(
    limb_components=["thigh", "shank"],
    x=AxisOperation(rename="knee_flexion", factor=-1.0),
    y=AxisOperation(rename="knee_adduction", factor=1.0),
    z=AxisOperation(rename="knee_int_rotation", factor=1.0),
)

right_ankle = BodyPartConfig(
    limb_components=["shank", "foot"],
    x=AxisOperation(rename="dorsi_flexion", factor=1.0),
    y=AxisOperation(rename="ankle_inversion", factor=1.0),
    z=AxisOperation(rename="ankle_int_rotation", factor=1.0),
)

right_foot = BodyPartConfig(
    limb_components=None,
    x=AxisOperation(rename="foot_floor_angle", factor=1.0),
    y=AxisOperation(rename="foot_supination", factor=-1.0),
    z=AxisOperation(rename="foot_int_rotation", factor=1.0),
)

kinematics_config = KinematicsJointConfig(
    left=BodyPartListConfig(
        pelvis=left_pelvis,
        hip=left_hip,
        knee=left_knee,
        ankle=left_ankle,
        foot=left_foot,
    ),
    right=BodyPartListConfig(
        pelvis=right_pelvis,
        hip=right_hip,
        knee=right_knee,
        ankle=right_ankle,
        foot=right_foot,
    ),
)


def get_kinematics_config():
    return kinematics_config


kinematics_setup = {
    'left': {
        'angles': {
            ('upright', 'hips'): {
                '-x': 'pelvic_tilt',
                'y': 'pelvic_obliquity',
                '-z': 'pelvic_int_rotation',
            },
            ('l_thigh', 'l_shank'): {
                '-x': 'knee_flexion',
                '-y': 'knee_adduction',
                '-z': 'knee_int_rotation',
            },
            ('hips', 'l_thigh'): {
                'x': 'hip_flexion',
                '-y': 'hip_adduction',
                '-z': 'hip_int_rotation',
            },
            ('l_shank', 'l_foot'): {
                'x': 'dorsi_flexion',
                '-y': 'ankle_inversion',
                '-z': 'ankle_int_rotation',
            },
            ('upright', 'l_foot'): {
                'x': 'foot_floor_angle',
                '-y': 'foot_supination',
                '-z': 'foot_int_rotation',
            },
            ('upright', 'l_shank'): {
                'x': 'shank_floor_angle',
                '-y': 'shank_adduction',
                '-z': 'shank_int_rotation',
            },
            ('upright', 'l_thigh'): {
                'x': 'thigh_floor_angle',
                '-y': 'thigh_adduction',
                '-z': 'thigh_int_rotation',
            },
        },
        'emgs': ['l_shank_emg', 'l_thigh_emg', 'l_emg'],
        'pressures': ['l_footbed'],
    },
    'right': {
        'angles': {
            ('upright', 'hips'): {
                '-x': 'pelvic_tilt',
                '-y': 'pelvic_obliquity',
                'z': 'pelvic_int_rotation',
            },
            ('r_thigh', 'r_shank'): {
                '-x': 'knee_flexion',
                'y': 'knee_adduction',
                'z': 'knee_int_rotation',
            },
            ('hips', 'r_thigh'): {
                'x': 'hip_flexion',
                'y': 'hip_adduction',
                'z': 'hip_int_rotation',
            },
            ('r_shank', 'r_foot'): {
                'x': 'dorsi_flexion',
                'y': 'ankle_inversion',
                'z': 'ankle_int_rotation',
            },
            ('upright', 'r_foot'): {
                'x': 'foot_floor_angle',
                'y': 'foot_supination',
                'z': 'foot_int_rotation',
            },
            ('upright', 'r_shank'): {
                'x': 'shank_floor_angle',
                '-y': 'shank_adduction',
                '-z': 'shank_int_rotation',
            },
            ('upright', 'r_thigh'): {
                'x': 'thigh_floor_angle',
                '-y': 'thigh_adduction',
                '-z': 'thigh_int_rotation',
            },
        },
        'emgs': ['r_shank_emg', 'r_thigh_emg', 'r_emg'],
        'pressures': ['r_footbed'],
    },
}

csv_position_streams = {
    'hips': 'fquat',
    'l_thigh': 'fquat',
    'r_thigh': 'fquat',
    'l_shank': 'fquat',
    'r_shank': 'fquat',
    'l_foot': 'fquat',
    'r_foot': 'fquat',
    'l_shank_emg': 'emg',
    'r_thigh_emg': 'emg',
    'l_emg': 'emg',
    'r_emg': 'emg',
    'l_shank_emg': 'emg',
    'r_shank_emg': 'emg',
    'l_thigh_emg': 'emg',
    'r_thigh_emg': 'emg',
    'l_emg': 'emg',
    'r_emg': 'emg',
    'l_footbed': 'emg',
    'r_footbed': 'emg',
}


neutral_offsets = {
    'left': {
        'pelvic_tilt': 7,
    },
    'right': {
        'pelvic_tilt': 7,
    },
}


def signal_peaks(data, config, elapsed):
    splits = []
    (peaks, meta) = signal.find_peaks(data, **config)
    for i, idx in enumerate(peaks):
        cur_height = meta['peak_heights'][i]
        splits.append((elapsed[idx], cur_height))
    return splits


def signal_troughs(data, config, elapsed):
    data = -data
    return signal_peaks(data, config, elapsed)


def zero_crossings(data, config, elapsed):
    splits = []
    crosses = np.diff(np.sign(data))
    for i, cross in enumerate(crosses):
        if cross != 0 and config['sign'] == 'a':
            splits.append((elapsed[i], 10))
        elif cross > 0 and config['sign'] == '+':
            splits.append((elapsed[i], 10))
        elif cross < 0 and config['sign'] == '-':
            splits.append((elapsed[i], 10))
    return splits


def crossings(data, config, elapsed):
    splits = []
    sign = config.get(
        'sign', 0
    )  # -1 negative crossing 0 any crossing 1 positive crossing
    cross = config.get('cross', 0)
    distance = config.get('distance', 1)
    previous = data[0] > cross
    i = 0
    while i < len(data):
        current = data[i] > cross
        if current != previous:
            if previous and sign < 1:
                splits.append((elapsed[i], 10))
                i += distance
            if current and sign > -1:
                splits.append((elapsed[i], 10))
                i += distance
            previous = current
        else:
            i += 1
    return splits


# configure to split on x euler
heel_strike = {
    'left': {
        'split': ('l_shank', 'euler', 'x'),
        'func': signal_peaks,
        'config': {'height': 0, 'distance': 60},
    },
    'right': {
        'split': ('r_shank', 'euler', 'x'),
        'func': signal_peaks,
        'config': {'height': 0, 'distance': 60},
    },
}

toe_off = {
    'left': {
        'split': ('l_shank', 'euler', 'x'),
        'func': signal_troughs,
        'config': {'height': 0, 'distance': 60},
    },
    'right': {
        'split': ('r_shank', 'euler', 'x'),
        'func': signal_troughs,
        'config': {'height': 0, 'distance': 60},
    },
}

max_knee = {
    'left': {
        'split': ('knee_flexion', 'angle', 'degrees'),
        'func': signal_peaks,
        'config': {'height': 0, 'distance': 60},
    },
    'right': {
        'split': ('knee_flexion', 'angle', 'degrees'),
        'func': signal_peaks,
        'config': {'height': 0, 'distance': 60},
    },
}

mid_swing = {
    'left': {
        'split': ('l_shank', 'euler', 'x'),
        'func': zero_crossings,
        'config': {'sign': '+'},
    },
    'right': {
        'split': ('r_shank', 'euler', 'x'),
        'func': zero_crossings,
        'config': {'sign': '+'},
    },
}

mid_stance = {
    'left': {
        'split': ('l_shank', 'euler', 'x'),
        'func': zero_crossings,
        'config': {'sign': '-'},
    },
    'right': {
        'split': ('r_shank', 'euler', 'x'),
        'func': zero_crossings,
        'config': {'sign': '-'},
    },
}

pconf = {
    'heel_strike': heel_strike,
    'toe_off': toe_off,
    'max_knee': max_knee,
    'mid_swing': mid_swing,
    'mid_stance': mid_stance,
}
