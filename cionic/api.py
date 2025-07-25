"""
Module for making calls to CIONIC REST APIs.
"""

import http.client
import io
import json
import os
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

from cionic import json2npy, segmenter, tools

apiver = '0.22'
server = None
authtoken = None


def flat_name(prefix, suffix):
    if prefix is None:
        return suffix
    else:
        return f'{prefix}_{suffix}'


def flatten(json, prefix=None):
    if isinstance(json, dict):
        result = {}
        for k, v in json.items():
            fn = flat_name(prefix, k)
            flat = flatten(v, fn)
            if isinstance(flat, dict):
                result.update(flat)
            else:
                result[fn] = flat
        return result
    elif isinstance(json, list):
        return [flatten(v, prefix) for i, v in enumerate(json)]
    else:
        return json


def ensure_parent(path):
    path = pathlib.Path(path)
    try:
        path.parent.mkdir(parents=True)
    except FileExistsError:
        pass

    assert path.parent.exists()
    assert path.parent.is_dir()

    return path


def web_url(urlpath):
    return f"https://{server}/{urlpath}"


def get_cionic(urlpath, microservice='c', cachepath=None, ver=apiver, **kwargs):
    'Get and cache cionic API JSON result.  Return dict'
    if cachepath:
        cachepath = ensure_parent(cachepath)
        if cachepath.exists():
            print(f'using cached {cachepath}', file=sys.stderr)
            return json.loads(cachepath.open().read())
    else:
        cachepath = None

    url = f'https://{server}/{microservice}/v{ver}/{urlpath}'
    print(f'fetching {url} {kwargs if kwargs else ""}', file=sys.stderr)
    r = requests.get(url, headers={'x-cionic-user': authtoken}, params=kwargs)

    if r.status_code != 200:
        print(r, file=sys.stderr)
        return None

    if cachepath:
        with cachepath.open(mode='w') as fp:
            fp.write(json.dumps(r.json()))

    return r.json()


def post_cionic(
    urlpath,
    microservice='c',
    cachepath=None,
    json=None,
    ver=apiver,
    ret_status=False,
    **kwargs,
):
    url = f'https://{server}/{microservice}/v{ver}/{urlpath}'
    print(f'posting {url} {kwargs} {json}', file=sys.stderr)

    if json:
        r = requests.post(url, headers={'x-cionic-user': authtoken}, json=json)
    else:
        r = requests.post(url, headers={'x-cionic-user': authtoken}, params=kwargs)

    if ret_status:
        return r.status_code

    if r.status_code not in [200]:
        print(r, file=sys.stderr)
        return None

    return r.json()


def put_cionic(
    urlpath,
    microservice='c',
    cachepath=None,
    json=None,
    ver=apiver,
    ret_status=False,
    **kwargs,
):
    url = f'https://{server}/{microservice}/v{ver}/{urlpath}'
    print('putting ' + url, file=sys.stderr)

    if json:
        r = requests.put(url, headers={'x-cionic-user': authtoken}, json=json)
    else:
        r = requests.put(url, headers={'x-cionic-user': authtoken}, params=kwargs)

    if ret_status:
        return r.status_code

    if r.status_code not in [200]:
        print(r, file=sys.stderr)
        return None

    return r.json()


def delete_cionic(urlpath, microservice='c', ver=apiver, ret_status=False):
    url = f'https://{server}/{microservice}/v{ver}/{urlpath}'
    print(f'deleting {url}', file=sys.stderr)

    r = requests.delete(url, headers={'x-cionic-user': authtoken})

    if ret_status:
        return r.status_code

    if r.status_code not in [200]:
        print(r, file=sys.stderr)
        return None

    return r.json()


def get_user(email):
    'Return user dictionary for email'
    return get_cionic('accounts', microservice='a', email=email)


def create_user(orgid, email):
    'Create a new user with passed email address'
    json = {'user_email': email, 'user_name': email}
    return post_cionic(f'{orgid}/accounts', microservice='a', json=json)


ORG_ROLES = {'analyst': 1, 'collector': 2, 'admin': 3}

ROLE_ADD_RESPONSE = {
    202: 'Success',
    409: 'Already Granted',
}

ROLE_REM_RESPONSE = {
    202: 'Success',
    404: 'Not Granted',
}


def add_roles(orgid, xid, roles):
    for role in roles:
        if rid := ORG_ROLES.get(role):
            status = post_cionic(
                f'{orgid}/accounts/{xid}/roles',
                microservice='a',
                json={'role': rid},
                ret_status=True,
            )
            print(f'Role {role} added <{ROLE_ADD_RESPONSE.get(status, status)}>')
        else:
            print(f'Role {role} unknown')


def remove_roles(orgid, xid, roles):
    for role in roles:
        if rid := ORG_ROLES.get(role):
            status = delete_cionic(
                f'{orgid}/accounts/{xid}/roles/{rid}', microservice='a', ret_status=True
            )
            print(f'Role {role} removed <{ROLE_REM_RESPONSE.get(status, status)}>')
        else:
            print(f'Role {role} unknown')


def local_npz(num, npzdir, suffix=''):
    npzname = f"{num}{suffix}.npz"
    return pathlib.Path(npzdir) / npzname


def get_study_metadata(sxid, orgid, tokenpath=None, **kwargs):
    """
    Fetch metadata for a specific study by sxid, caching into npzdir.
    Return list of collection metadata as flattened dicts.

    sxid: study xid
    tokenpath: pathname with auth token
    """
    if tokenpath is not None:
        auth(tokenpath)
    study_json = get_cionic(
        f'{orgid}/collections', microservice='w', sxid=sxid, **kwargs
    )
    if study_json is None:
        #
        # If study has too much data, metadata request times out and responds with 502:
        # https://github.com/cionicwear/cionic-collection/issues/251
        # As a workaround, we can stitch it together by getting the metadata
        #  for each protocol in the study
        #
        print('Fetching study timed out. Fetching by protocol instead')
        protos = get_cionic(f'{orgid}/protocols', microservice='c', sxid=sxid)
        colls = []
        for proto in protos:
            pcolls = get_cionic(
                f'{orgid}/collections',
                microservice='w',
                sxid=sxid,
                pxid=proto['xid'],
                **kwargs,
            )
            for pcoll in pcolls:
                colls.append(pcoll)
        study_json = colls

    return flatten(study_json)


def download_collections(
    colls, npzdir, tokenpath=None, segfile=None, segsuffix=None, segroot=None
):
    '''Download .npz files to npzdir for the given list of collections.
    Return list of segment metadata dicts.

    colls: entries returned by meta(), possibly filtered
    npzdir: output directory to store downloaded npz data files
    tokenpath: pathname with auth token
    segfile: file of labels to segment on
    '''
    if tokenpath is not None:
        auth(tokenpath)

    keys = [
        'config',
        'collector_name',
        'participant_name',
        'participant_age',
        'participant_safe_meta_height',
        'participant_safe_meta_height_ft',
        'participant_safe_meta_height_in',
        'participant_safe_meta_weight',
        'protocol_title',
        'removed',
        'status',
        'study_title',
        'time_created',
        'xid',
        'num',
    ]

    servermeta = {}
    for c in colls:
        servermeta[c['num']] = {k: v for k, v in c.items() if k in keys}
        download_file(local_npz(c['num'], npzdir), c['npz'])

    segs = []

    for num, collmeta in servermeta.items():
        print('processing collection %s' % num)
        if segroot is not None:
            segpath = f"{npzdir}/{num}_{segroot}.jsonl"
            segfile = open(segpath)
        npz = segmenter.load_segmented(
            local_npz(num, npzdir), segfile=segfile, segsuffix=segsuffix
        )
        for line in npz['segments.jsonl'].split(b'\n'):
            if line:
                segmeta = json.loads(line)
                segmeta.update(collmeta)
                segs.append(segmeta)

    print('study fetch complete', file=sys.stderr)
    return segs


def package_npz(segments, npzdir, npzpath, segsuffix=''):
    """
    Package list of <segments> with data in <npzdir> into <npzpath> as .npz file.
    Manipulate path in npz/segments table to point to internal .npy file.

    segments: list of segments from segments.jsonl
    npzdir: directory of cached data .npz collection files
    npzpath: pathname for output .npz
    """
    segment_jsonl = []

    gwlabels_written = set()
    with zipfile.ZipFile(npzpath, mode='w', compression=zipfile.ZIP_DEFLATED) as outnpz:
        for seg in segments:
            try:
                collpath = local_npz(seg['num'], npzdir, suffix=segsuffix)
                with zipfile.ZipFile(collpath) as zf:
                    with zf.open(seg['path'] + '.npy') as fp:
                        seg['origpath'] = seg['path']
                        seg['path'] = ('{num}' + segsuffix + '_{path}').format(**seg)
                        outnpz.writestr(seg['path'] + '.npy', fp.read())
                    if seg['num'] not in gwlabels_written:
                        with zf.open('gwlabels.jsonl') as fp:
                            outnpz.writestr(
                                ('{num}' + segsuffix + '_gwlabels.jsonl').format(**seg),
                                fp.read(),
                            )
                        gwlabels_written.add(seg['num'])
                segment_jsonl.append(json.dumps(seg))
            except Exception as e:
                print(collpath, seg.get('path', '<None>'), str(e), file=sys.stderr)

        segnpy = json2npy.JSONL2NPY().convert_array(segment_jsonl)
        output = io.BytesIO()
        np.save(output, segnpy)
        outnpz.writestr('segments.npy', output.getvalue())
        outnpz.writestr('segments.jsonl', '\n'.join(segment_jsonl))
        print("study package complete", file=sys.stderr)


def download_file(destpath, url, headers=None):
    'Download response from url to destpath'
    if headers is None:
        headers = {}
    destpath = ensure_parent(destpath)

    if destpath.exists():
        print(f"already exists {destpath}", file=sys.stderr)
        return 0

    print(f"getting {destpath}", file=sys.stderr)

    r = requests.get(url, stream=True, headers=headers)
    with destpath.open(mode='wb') as fp:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            fp.write(chunk)
    return 1


def download_npz(destpath, urlpath):
    npz = get_cionic(urlpath)
    status = download_file(destpath, npz['streams.npz'])
    if status == 0:
        return 0
    include_eulers_to_npz(destpath)


def download_files(urlpath, directory, include=None, exclude=None, ver=apiver):
    results = []
    files = get_cionic(urlpath)
    for filename, data in files.items():
        name, extension = os.path.splitext(filename)
        if exclude is not None and extension in exclude:
            continue
        if include is not None and extension not in include:
            continue
        absolute = data['url']
        destpath = f"{directory}{filename}"
        results.append(filename)
        download_file(destpath, absolute, headers={'x-cionic-user': authtoken})
    return results


# TODO: fix signal.ipynb
# TODO: fix gait.ipynb


def add_arrays_to_npz_and_store(
    npz: np.lib.npyio.NpzFile,
    array_dict: dict[str, np.ndarray],
    destpath: str,
) -> None:
    """
    Adds arrays from `array_dict` to an existing NumPy `.npz` file object `npz`,
    and saves the combined arrays to a new `.npz` file at `destpath`.
    Handles both .npy arrays and .jsonl files (as bytes).

    Parameters:
        npz (numpy.lib.npyio.NpzFile): An opened `.npz` file object containing arrays.
        array_dict (dict): Dictionary of arrays to add.
        destpath (str): Destination file path to save the updated `.npz` file.
        keep_existing_segments_file (bool): If True, keeps the 'segments' in the npz.

    Returns:
        None
    """
    arrays_to_write = {}
    for file in npz.files:
        arrays_to_write[file] = npz[file]
    arrays_to_write.update(array_dict)

    with zipfile.ZipFile(destpath, mode='w', compression=zipfile.ZIP_DEFLATED) as outzf:
        for file, arr in arrays_to_write.items():
            if file == 'segments.jsonl':
                continue
            if file == 'segments':
                # Convert JSONL bytes since segments file is updated with Eulers.
                outzf.writestr(
                    'segments.jsonl',
                    json2npy.structured_array_to_jsonl_bytes(arr),
                )
            if (
                file.endswith('.jsonl')
                or file.endswith('.json')
                or file.endswith('.csv')
                or file == "ERRORS"
            ):
                # Write as bytes (if not already bytes, encode as utf-8)
                if isinstance(arr, bytes):
                    outzf.writestr(file, arr)
                else:
                    outzf.writestr(file, arr.encode('utf-8'))
            else:
                with outzf.open(f"{file}.npy", mode='w') as fp:
                    np.save(fp, arr, allow_pickle=False)


def get_limb_eulers(
    npz: np.lib.npyio.NpzFile, degrees: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Extract Euler angles from quaternion segments in a .npz file.

    Args:
        npz (np.lib.npyio.NpzFile): Opened .npz file containing segment data.
        degrees (bool, optional): If True, returns Euler angles in degrees.

    Returns:
        tuple:
            - dict[str, np.ndarray]: Dict of euler_path names to Euler arrays.
            - np.ndarray: Array of new segment metadata dicts for the Euler streams.
    """
    print("Getting limb eulers from npz", file=sys.stderr)
    limb_eulers = {}
    new_limb_segments = []
    for seg in change_segments_column_dtype(npz['segments']):
        if seg['stream'] != 'fquat':
            continue
        stream = tools.stream_quat2euler(
            stream=npz[seg['path']], calibration=seg['calibration'], degrees=degrees
        )
        euler_path = f'{seg["path"]}2euler'
        limb_eulers[euler_path] = stream

        new_segment = seg.copy()
        new_segment['path'] = euler_path
        new_segment['fields'] = 'x y z'
        new_segment['stream'] = 'euler'
        new_limb_segments.append(new_segment)

    return limb_eulers, np.array(new_limb_segments)


def pandas_to_ndarray(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a pandas DataFrame to a NumPy ndarray.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        np.ndarray: The converted ndarray.
    """
    array = np.array(
        list(df.itertuples(index=False)),
        dtype=np.dtype(
            {
                'names': df.columns.tolist(),
                'formats': [df[col].dtype for col in df.columns],
            }
        ),
    )
    return array


def get_joint_eulers(
    npz: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    '''
    Extracts joint Euler angle data and corresponding segment information from NPZ.

    Args:
        npz (np.lib.npyio.NpzFile): The NPZ file containing joint data.

    Returns:
        tuple:
            - joint_eulers (dict): A dictionary mapping each joint stream path to its
              Euler angle data as a NumPy ndarray.
            - new_joint_segments (list): A list of segment metadata arrays.
    '''
    print("Getting joint eulers from npz", file=sys.stderr)
    segments = change_segments_column_dtype(npz['segments'])
    streams_data_packet = tools.get_joint_streams(npz, segmented=False)
    joint_eulers = {}
    new_joint_segments = []

    for stream_data in streams_data_packet:
        data_stream = stream_data['data_stream']
        joint_eulers[stream_data['path']] = pandas_to_ndarray(data_stream)

        seg_dtype = segments.dtype
        values = tuple(stream_data.get(name, '') for name in seg_dtype.names)

        new_segment = np.array([values], dtype=seg_dtype)[0]
        new_joint_segments.append(new_segment)

    return joint_eulers, new_joint_segments


def change_segments_column_dtype(segments: np.ndarray, dtype_dict=None) -> np.ndarray:
    '''
    Change dtype of specified columns in a structured numpy array.

    Args:
        segments (np.ndarray): Structured numpy array (like pandas DataFrame).
        new_dtypes (list of tuples): List of (field, dtype) to update.

    Returns:
        np.ndarray: New array with updated dtypes.
    '''
    if dtype_dict is None:
        dtype_dict = {
            'position': 'U20',
            'device': 'U40',
            'path': 'U100',
        }

    # Build new dtype: update only specified fields, keep others the same
    new_dtype = []
    for name, oldtype in segments.dtype.descr:
        if name in dtype_dict.keys():
            new_dtype.append((name, dtype_dict[name]))
        else:
            new_dtype.append((name, oldtype))

    # Create new array and copy data
    new_segments = np.empty(segments.shape, dtype=new_dtype)
    for name in segments.dtype.names:
        new_segments[name] = segments[name]

    return new_segments


def include_eulers_to_npz(destpath: str) -> None:
    """
    Load a .npz file, compute limb and joint Euler angles, update segments,
    and save the updated arrays back to the .npz file.

    Args:
        destpath (str): Path to the .npz file to update.
    """
    try:
        npz = np.load(destpath)
    except FileNotFoundError:
        print(f"File {destpath} not found.", file=sys.stderr)
        return

    limb_eulers, new_limb_segments = get_limb_eulers(npz)
    joint_eulers, new_joint_segments = get_joint_eulers(npz)

    updated_segments = np.concatenate(
        [
            change_segments_column_dtype(npz['segments']),
            new_limb_segments,
            new_joint_segments,
        ]
    )

    array_dict = {**limb_eulers, **joint_eulers, 'segments': updated_segments}
    add_arrays_to_npz_and_store(npz, array_dict, destpath)


def list_files(directory, include=None, exclude=None):
    results = []
    for filename in os.listdir(directory):
        name, extension = os.path.splitext(filename)
        if exclude is not None and extension in exclude:
            continue
        if include is not None and extension not in include:
            continue
        results.append(filename)
    return results


def auth(tokenpath=None, domain=None):
    """
    Parse a token.json file to get the Cionic credentials for future API requests.
    If tokenpath is not specified, use the CIONIC_ACCESS_TOKEN from the environment
    to retrieve the Cionic credentials.

    TODO: Get rid of the server and authtoken globals

    :param tokenpath: path to token.json (include filename)
    :param domain: if not using tokenpath, specify cionic domain
        (defaults to CIONIC_OAUTH_SERVER from the environment)
    :return: list of the user's org shortnames
    """
    global server, authtoken
    if tokenpath is None:
        access_token = os.environ.get('CIONIC_ACCESS_TOKEN')
        domain = os.environ.get('CIONIC_OAUTH_SERVER')
        if (access_token is None) or (domain is None):
            print(
                'CIONIC AUTH ERROR: No tokenpath specified and no CIONIC_ACCESS_TOKEN'
                'or CIONIC_OAUTH_SERVER in the environment.\n'
                'Please logout/login.'
            )

        #
        # GET the user's Cionic credentials from the OAuth API
        #
        ouser_resp = requests.get(
            f'https://{domain}/oauth/user',
            headers={'Authorization': f'Bearer {access_token}'},
        )
        if ouser_resp.status_code != http.client.OK:
            print(
                '''
CIONIC AUTH ERROR: OAuth token failed. Please logout/login.
            '''
            )

        ouser = ouser_resp.json()
        authtoken = ouser['atok']
        server = domain
        return ouser['orgs']

    #
    # tokenpath overrides the env var
    #
    with open(tokenpath) as tokfp:
        d = json.loads(tokfp.read())
        server = d['domain']
        authtoken = d['token']
        return d['orgs']
