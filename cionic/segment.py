"""
Module for segmenting npzs
"""

import collections
import io
import os
import pathlib
import zipfile

import numpy as np

from cionic import from_jsonl, to_jsonl, to_nparray

META_STEMS = ['PATCHES', 'devices', 'segments']


def parse_metatables(inputs):
    '''Parse metadata tables from inputs pathnames. Return combined metatables mapping.
    Each row includes source_stem for which input it came from.
    '''
    metatables = collections.defaultdict(list)  # [fnstem] -> metadata table
    for infn in inputs:
        instem = pathlib.Path(infn).stem
        with zipfile.ZipFile(infn) as inzf:
            for zi in inzf.infolist():
                zipath = pathlib.Path(zi.filename)
                if zipath.suffix == '.jsonl':
                    for row in from_jsonl(inzf.open(zi)):
                        row['source_stem'] = instem
                        metatables[zipath.stem].append(row)
    return metatables


def find_segment(sourcestem, innerstem, segs):
    for segrow in segs:
        if segrow['path'] == innerstem and segrow['source_stem'] == sourcestem:
            return segrow


def keys_match(keys, row):
    for k, v in keys.items():
        if row[k] != v:
            return False
    return True


def split_segment(data, t0, t1):
    i0, i1 = None, None
    times = data['elapsed_s']

    if t1 is None:
        i1 = len(data)
    elif t1 < times[0]:  # data completely after time range
        return None

    if t0 is None:
        i0 = 0
    elif t0 > times[-1]:  # data completely before time range
        return None

    if i0 is None:
        i0 = np.searchsorted(times, [t0])[0]

    if i1 is None:
        i1 = np.searchsorted(times, [t1])[0]

    return data[i0:i1]


def load_npy(fp):
    return np.load(io.BytesIO(fp.read()))


def progress_none(s):
    pass


def segmentize(inputs, boundaries, output, progress=progress_none):
    metatables = parse_metatables(inputs)

    segments = metatables['segments']
    metatables['segments'] = []  # new segments

    # write new output .npz
    with zipfile.ZipFile(output, mode='w', compression=zipfile.ZIP_DEFLATED) as outzf:
        # copy all data
        for infn in inputs:
            instem = pathlib.Path(infn).stem
            with zipfile.ZipFile(infn) as inzf:
                for zi in inzf.infolist():
                    zipath = pathlib.Path(zi.filename)

                    if zipath.suffix == '.jsonl' or zipath.stem in metatables:
                        continue

                    if len(inputs) > 1:
                        outstem = f'{instem}_{zipath.stem}'
                    else:
                        outstem = zipath.stem

                    progress(zi.filename)
                    segmeta = find_segment(instem, zipath.stem, segments)

                    if not segmeta:
                        progress(f'"{zipath.stem}" not found in segments table')
                        continue

                    if segmeta['stream'] == 'regs':
                        # HACK: copy 'regs' table
                        metatables['segments'].append(dict(segmeta))
                        with outzf.open(f'{outstem}.npy', mode='w') as outregsfp:
                            outregsfp.write(inzf.open(zi).read())
                        continue

                    segdata = load_npy(inzf.open(zi))
                    for i, boundary in enumerate(boundaries):

                        # proposal all boundaries are relevant
                        # segment number used to remove duplicates
                        # if not keys_match(boundary['keys'], segmeta):
                        #    continue

                        newsegdata = split_segment(
                            segdata, boundary.get('start_s'), boundary.get('end_s')
                        )
                        if newsegdata is None or len(newsegdata) == 0:
                            continue

                        suffix = boundary.get('add', {}).get('segment_num', i)
                        stem = f'{outstem}_{suffix:03}'

                        newsegmeta = dict(segmeta)
                        newsegmeta['start_s'] = float(newsegdata[0]['elapsed_s'])
                        newsegmeta['end_s'] = float(newsegdata[-1]['elapsed_s'])
                        newsegmeta['duration_s'] = (
                            newsegmeta['end_s'] - newsegmeta['start_s']
                        )
                        newsegmeta['path'] = stem

                        for k, v in boundary.get('add', {}).items():
                            if k in newsegmeta and newsegmeta[k] != v:
                                progress(
                                    f'cannot replace {k} ("{newsegmeta[k]}") with "{v}"'
                                )
                            else:
                                newsegmeta[k] = v

                        metatables['segments'].append(dict(newsegmeta))

                        with outzf.open(stem + '.npy', mode='w') as outsegfp:
                            np.save(outsegfp, newsegdata)

        # add npy and jsonl metadata tables
        for tblname, tbl in metatables.items():
            if tblname in META_STEMS:
                with outzf.open(tblname + '.npy', mode='w') as fp:
                    np.save(fp, to_nparray(tbl), allow_pickle=False)
                outzf.writestr(tblname + '.jsonl', to_jsonl(tbl))

        outzf.writestr('gwlabels.jsonl', to_jsonl(boundaries))


def load_boundary_times(npz, segfile='gwlabels.jsonl'):
    times = []
    max_time = np.max(npz['segments']['end_s'])
    boundaries = from_jsonl(npz[segfile].split(b"\n"))
    for i, boundary in enumerate(boundaries):
        label = boundary.get('add', {}).get('label', i).replace(" ", "_")
        segment = boundary.get('add', {}).get('segment_num', i)

        # update boundary entry with cleansed data
        boundary.update(
            {
                'label': label,
                'segment': segment,
                'start_s': boundary.get('start_s') or 0,
                'end_s': boundary.get('end_s') or max_time,
            }
        )
        times.append(boundary)
    return times


def load_segmented(npzpath, segfile='gwlabels.jsonl', segsuffix='_seg'):
    npz = np.load(npzpath)

    if isinstance(segfile, str) and segfile in npz:
        boundaries = load_boundary_times(npz, segfile)
    else:
        return npz

    output = os.path.splitext(npzpath)[0] + segsuffix + '.npz'

    # if no boundaries in the npz return as is
    if len(boundaries) == 0:
        return npz

    # dedupe boundaries by segment num
    dedupe = {b['segment']: b for b in boundaries}
    boundaries = list(dedupe.values())

    segmentize([npzpath], boundaries, output, progress=progress_none)
    return np.load(output)
