#!/usr/bin/env python3

import argparse
import pathlib
import sys

import pandas as pd

import cionic
from cionic import kinematics, kinematics_setup, npz_utils, tools

__usage__ = '''
./scripts/download.py
    [orgid]
    [studyid]
    [-p <protocol shortname>]
    [-n <collection numbers to download>]
    [-c <streams to csv>]
    [-o <output directory>]
    [-t <filepath to tokenfile>]
    [-l <limit>]
    [-f <additional collection files>]

Common usage examples:

Print help
./scripts/download.py -h

Interactive org and study - default limit = 20
./scripts/download.py

Interactive org and study - download last 5 collections including collection files
./scripts/download.py -f -l 5

Interactive org and study - create csv files for emg and fquat streams
./scripts/download.py -c emg fquat

Download from cionic org, sample study and quad-assist protocol
./scripts/download.py cionic sample -p quad-assist

Download collections 253 & 254 from cionic org, sample study
./scripts/download.py cionic sample -n 253 254
'''


KINEMATICS_SETUP = kinematics_setup.kinematics_setup


def download_npz(collection, urlroot, fileroot, nameroot):
    '''
    Download and return NPZ file.

    Args:
        collection (dict): Dictionary containing collection metadata. Must include
            keys 'num' and 'xid'.
        urlroot (str): Base URL for constructing the full download path.
        fileroot (str): Root directory where the files should be saved locally.
        nameroot (str): Typically orgid and studyid.

    Returns:
        np.lib.npyio.NpzFile: NPZ file
    '''
    colnum = collection['num']
    coldir = f"{fileroot}/{colnum}"
    download = f"{urlroot}/{collection['xid']}/streams/npz"
    npzpath = f"{coldir}/{nameroot}_{colnum}.npz"
    pathlib.Path(coldir).mkdir(parents=True, exist_ok=True)

    # exit if npz already exists
    if pathlib.Path(npzpath).exists():
        print(f"already exists {npzpath}", file=sys.stderr)
        return cionic.load_segmented(npzpath)

    cionic.download_npz(npzpath, download)

    # exit if npz was not downloaded
    if not pathlib.Path(npzpath).exists():
        print(f"missing streams {npzpath}", file=sys.stderr)
        return None

    # segment npz
    return cionic.load_segmented(npzpath)


def download_files(collection, urlroot, fileroot):
    '''
    Download files associated with a collection to a local directory.

    Args:
        collection (dict): Dictionary containing collection metadata. Must include
            keys 'num' and 'xid'.
        urlroot (str): Base URL for constructing the full download path.
        fileroot (str): Root directory where the files should be saved locally.

    Returns:
        None
    '''
    colnum = collection['num']
    files_dir = f"{fileroot}/{colnum}/files/"
    files_url = f"{urlroot}/{collection['xid']}/files"
    cionic.download_files(files_url, files_dir, exclude=[".CDE", ".npz"])


def output_joint_streams(collection, fileroot, npz):
    '''
    Save joint angle data streams from an NPZ archive to CSV files.

    Args:
        collection (dict): Dictionary containing collection metadata, including 'num'.
        fileroot (str): Root path where output CSV files should be saved.
        npz ( np.lib.npyio.NpzFile): Loaded NPZ archive.

    Returns:
        None
    '''
    colnum = collection['num']
    joint_streams_packet = tools.return_joint_streams(npz)
    for stream_data in joint_streams_packet:
        outpath = (
            f'{fileroot}/{colnum}/{stream_data["group"][0]}_'
            f'{stream_data["position_name"]}_euler_{stream_data["segment_num"]:>03}_'
            f'{stream_data["label"]}.csv'
        )
        print(f"Saving {outpath}")
        stream_data["stream"].to_csv(outpath, index=False)


def make_csv_stream(collection, fileroot, npz, segment):
    '''
    Save a raw stream from an NPZ archive as CSV file.

    Args:
        collection (dict): Dictionary containing collection metadata, including a 'num'.
        fileroot (str): Root path where output CSV files should be saved.
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive containing the data.
        segment (dict): Dictionary describing the stream segment (from NPZ).

    Returns:
        None
    '''
    # construct file path
    colnum = collection['num']
    outpath = (
        f"{fileroot}/{colnum}/{segment['position']}_"
        f"{segment['path']}_{segment['label']}.csv"
    )
    print(f"Saving {outpath}")
    # load array into pandas
    arr = npz[segment['path']]
    df = pd.DataFrame(arr)
    # rename columns
    if segment.get('chanpos'):
        remap = dict(zip(segment['fields'].split(), segment['chanpos'].split()))
        df.rename(columns=remap, inplace=True)
    # pop elapsed_s to front for convenience
    col = df.pop('elapsed_s')
    df.insert(0, 'elapsed_s', col)
    # save to csv
    df.to_csv(outpath, index=False)

    if segment['stream'] == 'fquat':
        outpath = (
            f"{fileroot}/{colnum}/{segment['position']}_"
            f"{segment['path'].replace('fquat', 'euler')}_{segment['label']}.csv"
        )
        df_euler = pd.DataFrame(tools.stream_quat2euler(arr))
        # pop elapsed_s to front for convenience
        col = df_euler.pop('elapsed_s')
        df_euler.insert(0, 'elapsed_s', col)
        df_euler.to_csv(outpath, index=False)


def make_csv_splits(collection, fileroot, component, segment, splits_matrix):
    '''
    Save a splits stream as CSV file.

    Args:
        collection (dict): Dictionary containing collection metadata, including a 'num'.
        fileroot (str): Root path where output CSV files should be saved.
        component (str): channel name for split stream, e.g. 'x', 'y', 'z'.
        segment (dict): Dictionary describing the stream segment (from NPZ).
        splits_matrix (numpy.NDArray): 2D matrix of split stream data

    Returns:
        None
    '''
    # construct file path
    colnum = collection['num']
    outpath = (
        f"{fileroot}/{colnum}/splits_{segment['position']}_{component}_"
        f"{segment['path']}_{segment['label']}.csv"
    )
    print(f"Saving {outpath}")
    pd.DataFrame(splits_matrix).to_csv(outpath, index=False)


def output_streams(c, fileroot, npz, segments, csvs):
    '''
    Save specified data streams from an NPZ archive to CSV files.

    Args:
        c (dict): Collection metadata, including a 'num' key.
        fileroot (str): Root path for where the CSV files should be saved.
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive containing the data streams.
        segments (list[dict]): List of segment definitions.
        csvs (list[str]): List of stream types (e.g., ['fquat']) to export.

    Returns:
        list[dict]: The input list of segment definitions.
    '''
    for segment in segments:
        make_csv_stream(c, fileroot, npz, segment)
    if 'fquat' in csvs:
        output_joint_streams(c, fileroot, npz)
    return segments


def output_split_streams(c, fileroot, npz, segments, csvs):
    '''
    Save split-phase data streams from an NPZ archive to CSV files,
    including joint angle segments.

    Args:
        c (dict): Collection metadata, including a 'num' key.
        fileroot (str): Root path where CSV files should be saved.
        npz (np.lib.npyio.NpzFile): Loaded NPZ archive.
        segments (list[dict]): List of segment metadata dicts.
        csvs (list[str]): List of stream types (e.g., ['fquat']),
           indicating which types to process and save.

    Returns:
        None
    '''
    segment_nums, _ = npz_utils.get_segment_nums_labels(npz)

    # get walking intervals
    for candidate_segment in segments:
        segment_num = candidate_segment['segment_num']
        if (
            segment_num in segment_nums
            and 'shank' in candidate_segment['position']
            and candidate_segment['stream'] == 'fquat'
        ):
            # get paired splits from shank eulers
            paired_splits = kinematics.get_paired_walking_splits(
                kinematic_time_series=tools.stream_quat2euler(
                    npz[candidate_segment['path']]
                )
            )
            if len(paired_splits) == 0:
                continue

            path = (
                f'{fileroot}/{c["num"]}/paired_splits_{segment_num:>03}_'
                f'{candidate_segment["label"]}.csv'
            )
            pd.DataFrame([{'start': x[0], 'stop': x[1]} for x in paired_splits]).to_csv(
                path, index=False
            )

            group = {'l': 'left', 'r': 'right'}.get(candidate_segment['position'][0])
            for segment in segments:
                if (
                    segment['segment_num'] != segment_num
                    or candidate_segment['position'][0] != segment['position'][0]
                ):
                    continue
                # segment num and side must match
                stream = npz[segment['path']]
                components = [
                    name for name in stream.dtype.names if name != "elapsed_s"
                ]
                for component in components:
                    splits_matrix = tools.stream_splits_to_matrix(
                        stream_data=stream,
                        splits=paired_splits,
                        ch_field=component,
                        n_interp=100,
                        paired_splits=True,
                    )
                    make_csv_splits(c, fileroot, component, segment, splits_matrix)
                if 'fquat' in csvs and segment['stream'] == 'fquat':
                    euler_stream = tools.stream_quat2euler(stream)
                    components = [
                        name for name in euler_stream.dtype.names if name != "elapsed_s"
                    ]
                    for component in components:
                        splits_matrix = tools.stream_splits_to_matrix(
                            stream_data=euler_stream,
                            splits=paired_splits,
                            ch_field=component,
                            n_interp=100,
                            paired_splits=True,
                        )
                        segment['path'] = segment['path'].replace('fquat', 'euler')
                        make_csv_splits(c, fileroot, component, segment, splits_matrix)
            # joint euler streams
            joint_euler_streams_packet = tools.return_joint_streams(
                npz, included_groups=(group), allowable_segment_nums=[segment_num]
            )
            for stream_data in joint_euler_streams_packet:
                components = [
                    name
                    for name in stream_data['stream'].columns
                    if name != "elapsed_s"
                ]
                for component in components:
                    splits_matrix = tools.stream_splits_to_matrix(
                        stream_data=stream_data['stream'],
                        splits=paired_splits,
                        ch_field=component,
                        n_interp=100,
                        paired_splits=True,
                    )
                    joint_segment = {
                        'position': (
                            f'{stream_data["group"][0]}_'
                            f'{stream_data["position_name"]}_euler'
                        ),
                        'path': f'{stream_data["segment_num"]:>03}',
                        'label': stream_data['label'],
                    }
                    make_csv_splits(
                        c, fileroot, component, joint_segment, splits_matrix
                    )


def load_collections(collections, urlroot, fileroot, nameroot, files, csvs):
    '''
    Load and process multiple data collections from a remote source,
    saving streams to CSV.

    Args:
        collections (list[dict]): List of collection metadata dictionaries.
        urlroot (str): Base URL for downloading files and NPZ archives.
        fileroot (str): Local root directory for saving CSV outputs.
        nameroot (str): Typically orgid and studyid.
        files (bool): Whether to download raw files for each collection.
        csvs (list[str]): List of stream types (e.g., ['fquat']) to extract and save.

    Returns:
        None
    '''
    for c in collections:
        if files:
            download_files(c, urlroot, fileroot)
        npz = download_npz(c, urlroot, fileroot, nameroot)

        if not npz:
            continue
        segments = npz_utils.get_relevant_npz_segments(npz, csvs)
        output_streams(c, fileroot, npz, segments, csvs)
        output_split_streams(c, fileroot, npz, segments, csvs)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=__usage__)
    parser.add_argument('orgid', nargs='?', help='organization shortname')
    parser.add_argument('studyid', nargs='?', help='study shortname')
    parser.add_argument(
        '-c',
        dest='csvs',
        nargs="+",
        required=False,
        help='generate CSV for listed streams',
    )
    parser.add_argument(
        '-f',
        dest='files',
        action='store_true',
        help='download additional collection files',
    )
    parser.add_argument(
        '-l',
        dest='limit',
        default=20,
        type=int,
        help='number of most recent collections to fetch',
    )
    parser.add_argument(
        '-n',
        dest='nums',
        nargs='+',
        type=int,
        required=False,
        help='collection numbers to download',
    )
    parser.add_argument(
        '-o', dest='outdir', default="./recordings", help='directory to store output'
    )
    parser.add_argument(
        '-p',
        dest='protoid',
        type=str,
        required=False,
        help='protocol shortname to download',
    )
    parser.add_argument(
        '-t',
        dest='token',
        default='token.json',
        help='path to auth credentials json file',
    )
    args = parser.parse_args(sys.argv[1:])

    # select orgid
    tokenpath = args.token
    orgs = cionic.auth(tokenpath=tokenpath)
    if args.orgid is None:
        for i, o in enumerate(orgs):
            print(f"{i} : {o['shortname']}")
        choice = int(input("Choose an org\n"))
        args.orgid = orgs[choice]['shortname']

    # fetch studies
    studies = cionic.get_cionic(f"{args.orgid}/studies")
    if studies is None:
        print(f"Studies not found not for org [{args.orgid}]")
        return

    # select or match study_id
    sxid = None
    if args.studyid is None:
        for i, s in enumerate(studies):
            print(f"{i} : {s['shortname']}")
        choice = int(input("Choose a study\n"))
        sxid = studies[choice]['xid']
        args.studyid = studies[choice]['shortname']
    else:
        for _, s in enumerate(studies):
            if args.studyid == s['shortname']:
                sxid = s['xid']

    # exit if study cannot be matched
    if sxid is None:
        print(f"Study [{args.studyid}] not found for org [{args.orgid}]")
        return

    # fetch study protocols
    protocols = cionic.get_cionic(f"{args.orgid}/protocols?sxid={sxid}")
    named_protos = {p['shortname']: p['xid'] for p in protocols}

    # match study or print the protocols in the selected study
    if args.protoid is None:
        print(
            f"Fetching [{args.limit}] collections for org [{args.orgid}] "
            f"study [{args.studyid}] all protocols"
        )
        for p in protocols:
            print(f"  {p['shortname']}")
        collections = cionic.get_cionic(f"{args.orgid}/collections?sxid={sxid}")
    elif pxid := named_protos.get(args.protoid):
        print(
            f"Fetching [{args.limit}] collections for org [{args.orgid}] "
            f"study [{args.studyid}] proto [{args.protoid}]"
        )
        collections = cionic.get_cionic(f"{args.orgid}/collections?protoxid={pxid}")
    else:
        print(
            f"Protocol [{args.protoid}] not found for org [{args.orgid}] "
            f"study [{args.studyid}]"
        )
        return

    # filter down by collection numbers
    if args.nums:
        collections = [coll for coll in collections if coll['num'] in args.nums]

    # sort by created time and limit fetch
    collections = sorted(collections, key=lambda collection: -collection['created_ts'])
    if args.limit:
        collections = collections[0 : args.limit]

    # download and parse
    fileroot = f"{args.outdir}/{args.orgid}/{args.studyid}"
    urlroot = f"{args.orgid}/collections"
    nameroot = f"{args.orgid}_{args.studyid}"
    load_collections(collections, urlroot, fileroot, nameroot, args.files, args.csvs)


if __name__ == '__main__':
    main()
