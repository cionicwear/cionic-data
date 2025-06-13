#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys

import pandas as pd

import cionic
from cionic import kinematics_setup, tools

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
    colnum = collection['num']
    files_dir = f"{fileroot}/{colnum}/files/"
    files_url = f"{urlroot}/{collection['xid']}/files"
    cionic.download_files(files_url, files_dir, exclude=[".CDE", ".npz"])


def retrieve_stream(npz, position, stream, segment_num):
    for line in npz['segments.jsonl'].split(b'\n'):
        if line:
            segment = json.loads(line)
            if (
                position == segment.get('position')
                and stream == segment.get('stream')
                and segment_num == segment.get('segment_num')
            ):
                return npz[segment['path']]
    return None


def get_segment_nums_labels(npz):
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


def output_joint_streams(collection, fileroot, npz):
    colnum = collection['num']
    segment_nums, labels = get_segment_nums_labels(npz)
    for segment_num, label in zip(segment_nums, labels):
        groups = list(KINEMATICS_SETUP.keys())
        for group in groups:
            for positions, orientations in KINEMATICS_SETUP[group]['angles'].items():
                position_1_quats = retrieve_stream(
                    npz=npz,
                    position=positions[0],
                    stream='fquat',
                    segment_num=segment_num,
                )
                position_2_quats = retrieve_stream(
                    npz=npz,
                    position=positions[1],
                    stream='fquat',
                    segment_num=segment_num,
                )
                if position_1_quats is None or position_2_quats is None:
                    continue

                df = pd.DataFrame(
                    tools.stream_quat2euler_joint(position_1_quats, position_2_quats)
                )
                for axis, kinematic_name in orientations.items():
                    if axis[0] == '-':
                        df[axis[-1]] = -df[axis[-1]]
                        df.rename(columns={axis[-1]: kinematic_name}, inplace=True)
                    else:
                        df.rename(columns={axis: kinematic_name}, inplace=True)
                col = df.pop('elapsed_s')
                df.insert(0, 'elapsed_s', col)

                if 'upright' in positions[0] and 'hips' in positions[1]:
                    position_name = 'pelvis_joint'
                elif 'hips' in positions[0] and 'thigh' in positions[1]:
                    position_name = 'hip_joint'
                elif 'thigh' in positions[0] and 'shank' in positions[1]:
                    position_name = 'knee_joint'
                elif 'shank' in positions[0] and 'foot' in positions[1]:
                    position_name = 'ankle_joint'
                else:
                    position_name = f'{positions[0]}_{positions[1]}'

                outpath = (
                    f'{fileroot}/{colnum}/{group[0]}_'
                    f'{position_name}_euler_{segment_num:>03}_{label}.csv'
                )
                print(f"Saving {outpath}")
                df.to_csv(outpath, index=False)


def make_csv(collection, fileroot, npz, segment):
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


def output_streams(c, fileroot, npz, segments, csvs):
    for line in npz['segments.jsonl'].split(b'\n'):
        if line:
            segment = json.loads(line)
            segments.append(segment)
            if csvs and segment['stream'] in csvs:
                make_csv(c, fileroot, npz, segment)
    if 'fquat' in csvs:
        output_joint_streams(c, fileroot, npz)
    return segments


def output_split_streams():
    # TODO
    pass


def load_collections(collections, urlroot, fileroot, nameroot, files, csvs):
    segments = []
    for c in collections:
        try:
            if files:
                download_files(c, urlroot, fileroot)
            npz = download_npz(c, urlroot, fileroot, nameroot)

            if npz:
                segments = output_streams(c, fileroot, npz, segments, csvs)
                output_split_streams()

        except Exception as e:
            print(e)

    return segments


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
