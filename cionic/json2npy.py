import json

import numpy as np


def from_jsonl(fp):
    'Return list of dicts from jsonl in open file.'
    return [json.loads(line) for line in fp if line]


class JSONL2NPY:
    def __init__(self):
        self.formats = {}

    def convert_array(self, f):
        """
        read jsonl data from iterator
        and return npy ndarray
        """
        self.measure(f)
        return self.to_nparray(from_jsonl(f))

    def measure(self, f):
        """
        measures the formats and sets the default type
        for each column in the ndarray
        """
        for _, line in enumerate(f):
            data = json.loads(line)
            [self.npformat(k, v) for k, v in data.items()]

    def to_nparray(self, objs):
        """
        fills data for each row of the ndarray
        from the jsonl row or default data type

        TODO: explore memmap numpy file for supporting long files
        """
        keys = self.formats.keys()
        arrs = []
        for data in objs:
            ts = tuple([self.npdata(data, k) for k in keys])
            arrs.append(ts)

        return np.array(arrs, dtype=[self.npdtype(k) for k in keys])

    def npdata(self, data, k):
        """
        get data for column **k** from json object **data**

        apply type conversion falling back to the default value
        """
        if data.get(k) is not None:
            try:
                if data[k] != '':
                    return (self.formats[k]['t'])(data[k])
            except Exception as e:
                print(f"Exception for column {k} - {e}")
        return self.formats[k]['d']

    def npdtype(self, k):
        """
        return numpy dtype object for column **k**
        """
        f = self.formats[k]
        return (k, f['t'], f['l']) if f['t'] == str else (k, f['t'])

    def npformat(self, k, obj):
        """
        sets the (type, length, and default value) for the object obj for column k

        currently only supports column types of string, float, int

        default object type is set to string and can be *upgraded* to a hard type

        length property is relevant to string types only
        """
        prev = self.formats.get(k, {'t': str, 'l': 1, 'd': ''})
        if prev['t'] == str and obj is not None:
            if isinstance(obj, str):
                prev['l'] = max(prev['l'], len(obj))
                prev['d'] = ''
            else:
                prev['t'] = type(obj)
                prev['l'] = 0
                prev['d'] = 0
        self.formats[k] = prev


def to_nparray(objs):
    j = JSONL2NPY()

    for row in objs:
        for k, v in row.items():
            j.npformat(k, v)

    return j.to_nparray(objs)
