#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.append('.')
import cionic

__usage__ = '''
./scripts/download.py [orgid] [studyid]
    [-n <collection numbers to download>]
    [-c <streams to csv>]
    [-o <output directory>]
    [-t <filepath to tokenfile>]
    [-l <limit>]
    [-f]

Common usage examples:
./scripts/download.py -h                            (print help)
./scripts/download.py                               (interactive org and study - default limit = 20)
./scripts/download.py -f -l 5                       (interactive org and study - download last 5 collections including collection files)
./scripts/download.py -c emg fquat                  (interactive org and study - create csv files for emg and fquat streams)
./scripts/download.py cionic sample -p quad-assist  (download from cionic org, sample study and quad-assist protocol)
./scripts/download.py cionic sample -n 253 254      (download collections 253 & 254 from cionic org, sample study)

'''


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


def make_csv(collection, fileroot, npz, segment):
    # construct file path
    colnum = collection['num']
    outpath = f"{fileroot}/{colnum}/{segment['path']}_{segment['label']}.csv"
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


def load_collections(collections, urlroot, fileroot, nameroot, files, csvs):
    segments = []
    for c in collections:
        try:
            if files:
                download_files(c, urlroot, fileroot)
            npz = download_npz(c, urlroot, fileroot, nameroot)

            if npz:
                for line in npz['segments.jsonl'].split(b'\n'):
                    if line:
                        segment = json.loads(line)
                        segments.append(segment)
                        if csvs and segment['stream'] in csvs:
                            make_csv(c, fileroot, npz, segment)
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
        for i, s in enumerate(studies):
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
