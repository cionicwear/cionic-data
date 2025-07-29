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
import requests

from cionic import json2npy, npz_utils, segmenter, tools

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


def download_file(destpath: str, url: str, headers: dict = None) -> bool:
    """
    Download the content from the given URL and save it to the destination path.

    Args:
        destpath (str): The local file path where the downloaded content will be saved.
        url (str): The URL to download the content from.
        headers (dict, optional): Optional HTTP headers to include in the request.

    Returns:
        bool: True if the file was downloaded, False if it already exists.
    """
    if headers is None:
        headers = {}
    destpath = ensure_parent(destpath)

    if destpath.exists():
        print(f"already exists {destpath}", file=sys.stderr)
        return False

    print(f"getting {destpath}", file=sys.stderr)

    r = requests.get(url, stream=True, headers=headers)
    with destpath.open(mode='wb') as fp:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            fp.write(chunk)
    return True


def download_npz(
    destpath: str, urlpath: str, include_eulers=True, include_gait_splits=True
) -> None:
    """
    Downloads a .npz file from a specified URL path and saves it to the destpath.

    Args:
        destpath (str): The local file path where the .npz file will be saved.
        urlpath (str): The URL or identifier used to locate the .npz file.

    Returns:
        None
    """
    npz = get_cionic(urlpath)
    status = download_file(destpath, npz['streams.npz'])
    if status is False:
        return
    if include_eulers:
        include_eulers_to_npz(destpath)
    if include_gait_splits:
        # TODO: Implement gait splits processing
        raise NotImplementedError(
            "The 'include_gait_splits' functionality is not yet implemented."
        )


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

    limb_eulers, new_limb_segments = tools.get_limb_eulers(npz)
    joint_eulers, new_joint_segments = tools.get_joint_eulers(npz)

    updated_segments = np.concatenate(
        [
            npz_utils.change_segments_column_dtype(npz['segments']),
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
                "CIONIC AUTH ERROR: OAuth token authentication failed. "
                "Please logout and login again.",
                file=sys.stderr,
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
