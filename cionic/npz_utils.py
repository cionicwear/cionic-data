import json


def get_segment_nums_labels(npz):
    '''
    Extract unique segment numbers and their corresponding labels from NPZ archive.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.

    Returns:
        tuple[list[int], list[str]]: tuple containing two lists —
            the first with unique segment numbers, and the second with their
            corresponding labels.
    '''
    segment_nums = []
    labels = []
    for line in npz['segments.jsonl'].split(b'\n'):
        if line:
            segment = json.loads(line)
            segment_num = segment.get('segment_num')
            if segment_num is not None and segment_num not in segment_nums:
                segment_nums.append(segment_num)
                labels.append(segment.get('label'))
    return segment_nums, labels


def get_relevant_npz_segments(npz, csvs):
    '''
    Filter segments from an NPZ archive based on specified stream types.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        csvs (list[str]): List of stream types to include (e.g., ['fquat']).

    Returns:
        list[dict]: A list of segment dictionaries matching the specified stream types.
    '''
    npz_segments = []
    for line in npz['segments.jsonl'].split(b'\n'):
        if not line:
            continue
        segment = json.loads(line)
        if csvs and segment['stream'] in csvs:
            npz_segments.append(segment)
    return npz_segments


def retrieve_stream(npz, position, stream, segment_num=False):
    '''
    Retrieve a specific data stream segment from an NPZ archive.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        position (str): Position on body, e.g. "r_shank".
        stream (str): Stream name, e.g. "fquat".
        segment_num (int or bool): Segment number to match in the segment metadata,
            or False to ignore.

    Returns:
        np.ndarray or None: The matched data segment if found, otherwise None.
    '''
    for line in npz['segments.jsonl'].split(b'\n'):
        if line:
            segment = json.loads(line)
            if (
                position == segment.get('position')
                and stream == segment.get('stream')
                and (segment_num is False or segment_num == segment.get('segment_num'))
            ):
                return npz[segment['path']]
    return None


def retrieve_segment_field(npz, position, stream, field_name, segment_num=False):
    '''
    Retrieve a specific field from a line in the segments file in an NPZ archive.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        position (str): Position on body, e.g. "r_shank".
        stream (str): Stream name, e.g. "fquat".
        field_name (str): Name of the field to retrieve from the segment metadata.
        segment_num (int or bool): Segment number to match in the segment metadata,
            or False to ignore.

    Returns:
        np.ndarray or None: The matched data segment if found, otherwise None.
    '''
    for line in npz['segments.jsonl'].split(b'\n'):
        if line:
            segment = json.loads(line)
            if (
                position == segment.get('position')
                and stream == segment.get('stream')
                and (segment_num is False or segment_num == segment.get('segment_num'))
            ):
                return segment[field_name]
    return None
