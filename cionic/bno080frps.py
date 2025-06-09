import pprint

import pandas as pd

reports = {
    '0X1AC9': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Environmental sensor - Humidity calibration',
        'word_parser': {},
    },
    '0X1B2A': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Pickup detector configuration',
        'word_parser': {},
    },
    '0X1F1F': {
        'comments': 'proprietary',
        'no_of_words': 0,
        'type_of_FRS': 'Dynamic calibration',
        'word_parser': {},
    },
    '0X2D3E': {
        'comments': '4.3.4',
        'no_of_words': 4,
        'type_of_FRS': 'System orientation',
        'word_parser': {
            'rotation quat w': (3, 0, 32, 's.30'),
            'rotation quat x': (0, 0, 32, 's.30'),
            'rotation quat y': (1, 0, 32, 's.30'),
            'rotation quat z': (2, 0, 32, 's.30'),
        },
    },
    '0X2D41': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Primary accelerometer orientation',
        'word_parser': {},
    },
    '0X2D43': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Screen rotation accelerometer orientation',
        'word_parser': {},
    },
    '0X2D46': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Gyroscope orientation',
        'word_parser': {},
    },
    '0X2D4C': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Magnetometer orientation',
        'word_parser': {},
    },
    '0X39AF': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Environmental sensor - Pressure calibration',
        'word_parser': {},
    },
    '0X39B1': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Environmental sensor - Ambient light calibration',
        'word_parser': {},
    },
    '0X3E2D': {
        'comments': '4.3.5 rotation vector',
        'no_of_words': 4,
        'type_of_FRS': 'AR/VR stabilization – rotation vector',
        'word_parser': {
            'max error 0 -pi': (2, 0, 32, 'u.29'),
            'max rotation 0 - pi': (1, 0, 32, 'u.29'),
            'scaling 0 - 1.0': (0, 0, 32, 'u.30'),
            'stability magnitude': (3, 0, 32, 'u.29'),
        },
    },
    '0X3E2E': {
        'comments': '4.3.5 game rotation vector',
        'no_of_words': 4,
        'type_of_FRS': 'AR/VR stabilization – game rotation vector',
        'word_parser': {
            'max error 0 -pi': (2, 0, 32, 'u.29'),
            'max rotation 0 - pi': (1, 0, 32, 'u.29'),
            'scaling 0 - 1.0': (0, 0, 32, 'u.30'),
            'stability magnitude': (3, 0, 32, 'u.29'),
        },
    },
    '0X4B4B': {
        'comments': 'serial number',
        'no_of_words': 1,
        'type_of_FRS': 'Serial number',
        'word_parser': {'serial number': (0, 0, 32, 'u.0')},
    },
    '0X4D20': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Environmental sensor - Temperature calibration',
        'word_parser': {},
    },
    '0X4D4D': {
        'comments': 'proprietary',
        'no_of_words': 0,
        'type_of_FRS': 'Nominal calibration – AGM',
        'word_parser': {},
    },
    '0X4DA2': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Environmental sensor - Proximity calibration',
        'word_parser': {},
    },
    '0X4E4E': {
        'comments': 'proprietary',
        'no_of_words': 0,
        'type_of_FRS': 'Nominal calibration - SRA',
        'word_parser': {},
    },
    '0X74B4': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'User record',
        'word_parser': {},
    },
    '0X7979': {
        'comments': 'proprietary',
        'no_of_words': 0,
        'type_of_FRS': 'Static calibration – AGM',
        'word_parser': {},
    },
    '0X7D7D': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Shake detector configuration',
        'word_parser': {},
    },
    '0X8A8A': {
        'comments': 'proprietary',
        'no_of_words': 0,
        'type_of_FRS': 'Static calibration – SRA',
        'word_parser': {},
    },
    '0XA1A1': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'UART Output Format Selection',
        'word_parser': {},
    },
    '0XA1A2': {
        'comments': '',
        'no_of_words': 7,
        'type_of_FRS': 'Gyro-Integrated Rotation Vector configuration',
        'word_parser': {
            'alpha default 0.3': (4, 0, 32, 'u.20'),
            'beta default 0.11': (5, 0, 32, 'u.20'),
            'gamma default 0.002': (6, 0, 32, 'u.20'),
            'max error': (2, 0, 32, 'u.29'),
            'prediction amount in sec': (3, 0, 32, 'u.10'),
            'reference type': (0, 0, 16, 'u.0'),
            'reserved': (1, 0, 32, 'u.0'),
        },
    },
    '0XA1A3': {
        'comments': '0 - false, 1 - true',
        'no_of_words': 1,
        'type_of_FRS': 'Fusion Control Flags',
        'word_parser': {'enable mag stab. GRV': (0, 0, 32, 'u.0')},
    },
    '0XC274': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Significant Motion detector configuration',
        'word_parser': {},
    },
    '0XD3E2': {
        'comments': '4.3.3',
        'no_of_words': 3,
        'type_of_FRS': 'MotionEngine power management',
        'word_parser': {
            'delta accel in mg': (2, 16, 'u.0'),
            'delta orientation': (0, 0, 32, 's.28'),
            'stable duration in sec': (2, 0, 16, 'u.0'),
            'stable threshold': (1, 0, 32, 's.25'),
        },
    },
    '0XD401': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'ALS Calibration',
        'word_parser': {},
    },
    '0XD402': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Proximity Sensor Calibration',
        'word_parser': {},
    },
    '0XD403': {
        'comments': '0 - gyro period, 1 - use timestamps',
        'no_of_words': 1,
        'type_of_FRS': 'MotionEngine Time Source Selection',
        'word_parser': {'time source': (0, 0, 32, 'u.0')},
    },
    '0XD7D7': {
        'comments': '',
        'no_of_words': 1,
        'type_of_FRS': 'Maximum fusion period',
        'word_parser': {'max fusion period in u-sec': (0, 0, 32, 'u.0')},
    },
    '0XED85': {
        'comments': '',
        'no_of_words': 2,
        'type_of_FRS': 'Stability detector configuration',
        'word_parser': {
            'accel threshold in m/s^2': (0, 0, 32, 's.24'),
            'duration in micro seconds': (1, 0, 32, 'u.0'),
        },
    },
    '0XED87': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Sleep detector configuration',
        'word_parser': {},
    },
    '0XED88': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Activity Tracker configuration',
        'word_parser': {},
    },
    '0XED89': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Tilt detector configuration',
        'word_parser': {},
    },
    '0XEE51': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Circle detector configuration',
        'word_parser': {},
    },
    '0XEF27': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Pocket detector configuration',
        'word_parser': {},
    },
    '0XFC94': {
        'comments': '?',
        'no_of_words': 0,
        'type_of_FRS': 'Flip detector configuration',
        'word_parser': {},
    },
}


class FRSRecords:
    def __init__(self):
        self.record_dict = {}

    def create_rec(self, FRS_index, no_of_words, word_parser, comments=""):
        self.record_dict[FRS_index] = {}
        self.add(FRS_index, "no_of_words", no_of_words)
        self.add(FRS_index, "word_parser", word_parser)
        self.add(FRS_index, "comments", comments)

    def add(self, FRS_index, characteristic, value):
        self.record_dict[FRS_index][characteristic] = value

    def retrieve(self, FRS_index, characteristic):
        return self.record_dict[FRS_index][characteristic]

    def print_dict(self):
        temp = {}
        for key, val in self.record_dict.items():
            temp[hex(key).upper()] = val
        pprint.pprint(temp)


frs_records = FRSRecords()
for report_id, report in reports.items():
    frs_records.create_rec(
        report_id, report["no_of_words"], report["word_parser"], report["comments"]
    )
    frs_records.add(report_id, "type_of_FRS", report["type_of_FRS"])


def parse_fixed_point_number(parser, word_list):
    word = word_list[parser[0]]
    offset = parser[1]
    no_of_bits = parser[2]
    format_ = parser[3]
    mask = 0
    for i in range(no_of_bits):
        mask |= 1 << offset + i
    word = word & mask
    word = word >> offset
    s, q = format_.split('.')
    q = int(q)
    if s == 's':
        if word > 2 ** (no_of_bits - 1) - 1:
            word = word - 2 ** (no_of_bits)
    elif s != 'u':
        raise Exception('unrecognized number format in the FRS parsing')
    return word * 2 ** (-q)


def format_df(df):
    new_col_names = ['frs_index', 'frs', 'frs_param', 'value', 'comments']
    f_df = pd.DataFrame(columns=new_col_names)

    # blocks of related FRS records
    blocks = []
    prev_frs = None
    for index, row in df.iterrows():
        frs = int(row['frs'])
        if frs == prev_frs:
            blocks[-1].append(row)
        else:
            blocks.append([])
            blocks[-1].append(row)
            prev_frs = frs

    # * for each block of rows, generate rows that correspond to the no of words
    # that are associated with that FRS index
    # * skip parsing if the no of words was set to zero
    # * flag error if not enough packets have been recieved
    for block in blocks:
        row = block[0]
        frs = hex(int(row['frs'])).upper()
        elapsed_s = row['elapsed_s']
        offsets = [int(row['offset'])]
        d = [int(row['d0']), int(row['d1'])]
        for row in block[1:]:
            d.extend([int(row['d0']), int(row['d1'])])
            offsets.append(int(row['offset']))
        description = frs_records.retrieve(frs, 'type_of_FRS')
        no_of_words = frs_records.retrieve(frs, 'no_of_words')
        word_parser = frs_records.retrieve(frs, 'word_parser')
        comments = frs_records.retrieve(frs, 'comments')
        if no_of_words == 0:
            new_row = {
                'frs_index': frs,
                'frs_param': '',
                'value': None,
                'frs': description,
                'comments': "skipping parsing this FRS record",
                'offsets': offsets,
                'elapsed_s': elapsed_s,
            }
            f_df = pd.concat([f_df, pd.DataFrame([new_row])], ignore_index=True)
        elif no_of_words <= len(d):
            for frs_param, parser in word_parser.items():
                value = parse_fixed_point_number(parser, d)
                new_row = {
                    'frs_index': frs,
                    'frs_param': frs_param,
                    'value': value,
                    'frs': description,
                    'comments': comments,
                    'offsets': offsets,
                    'elapsed_s': elapsed_s,
                }
                f_df = pd.concat([f_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            new_row = {
                'frs_index': frs,
                'frs_param': '',
                'value': None,
                'frs': description,
                'comments': "not enough frames to parse this FRS record",
                'offsets': offsets,
                'elapsed_s': elapsed_s,
            }
            f_df = pd.concat([f_df, pd.DataFrame([new_row])], ignore_index=True)

    return f_df
