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

from cionic import json2npy, kinematics, npz_utils, segmenter, tools

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

    if r.status_code not in [
        200,
        201,  # 201 is standard response for resource creation (for getting gcs urls)
        202,  # 202 for accepted request (update collection files after upload)
    ]:
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


def _get_collection_xid(
    org_shortname: str, study_shortname: str, collection_num: str
) -> str:
    """
    Get the collection XID from the collection metadata.

    Args:
        org_shortname (str): The organization shortname.
        study_shortname (str): The study shortname.
        collection_num (str): The collection number.

    Returns:
        str: The collection XID.

    Raises:
        RuntimeError: If the collection cannot be found.
    """
    kwargs = {"study": study_shortname, "num": collection_num}
    (collection,) = get_cionic(f'{org_shortname}/collections', **kwargs)
    return collection['xid']


def _check_file_exists(org_shortname: str, collection_xid: str, filename: str) -> bool:
    """
    Check if a file already exists in a collection.

    Args:
        org_shortname (str): The organization shortname.
        collection_xid (str): The collection XID.
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    urlpath = f"{org_shortname}/collections/{collection_xid}/files"
    existing_files = get_cionic(urlpath)
    return existing_files is not None and filename in existing_files


def _get_upload_url(org_shortname: str, collection_xid: str, filename: str) -> str:
    """
    Get the upload URL for a file.

    Args:
        org_shortname (str): The organization shortname.
        collection_xid (str): The collection XID.
        filename (str): The name of the file to upload.

    Returns:
        str: The upload URL for the file.

    Raises:
        RuntimeError: If unable to get an upload URL.
    """
    urlpath = f"{org_shortname}/collections/{collection_xid}/files"
    response = post_cionic(urlpath, json={'files': [filename]})

    if not response:
        raise RuntimeError(f"Failed to get upload URL for file: {filename}")

    gcs_url = response.get(filename)
    if not gcs_url:
        raise RuntimeError(f"No upload URL found for file: {filename}")

    return gcs_url


def _upload_file(filepath, gcs_url):
    """
    Upload a file to a specified GCS URL.

    Args:
        filepath (str): The local file path to upload.
        gcs_url (str): The GCS URL to upload the file to.

    Returns:
        bool: True if the file was uploaded successfully, False otherwise.
    """
    try:
        with open(filepath, 'rb') as file:
            # file can be any type, octet stream is generic
            response = requests.put(
                gcs_url, data=file, headers={'Content-Type': 'application/octet-stream'}
            )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error uploading file: {e}", file=sys.stderr)
        return False


def _update_collection_files(
    orgname, collection_xid, uploaded_files, participant_xid=None
):
    """
    Updates a collection by marking files as uploaded after verifying they exist
    in the storage bucket. Required after uploading files to get the collection
    out of "in progress" state.

    Args:
        orgname (str): Organization name
        collection_xid (str): Collection ID to update
        uploaded_files (list[str]): List of filenames that have been uploaded
        participant_xid (str, optional): Participant ID if updating participant

    Returns:
        dict: Updated collection data if successful, None otherwise
    """
    data = {"uploaded_files": uploaded_files}
    if participant_xid:
        data["participant_xid"] = participant_xid

    return post_cionic(f"{orgname}/collections/{collection_xid}", json=data)


def upload_file_from_metadata(
    tokenpath: str,
    org_shortname: str,
    study_shortname: str,
    collection_num: str,
    filepath: str,
    overwrite: bool = False,
) -> bool:
    """
    Upload a file to a specific collection in the Cionic platform using metadata.

    Args:
        tokenpath (str): Path to the authentication token file.
        org_shortname (str): The organization shortname.
        study_shortname (str): The study shortname.
        collection_num (str): The collection number.
        filepath (str): The local file path to upload.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
            Defaults to False.

    Returns:
        bool: True if the file was uploaded successfully, False otherwise.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}", file=sys.stderr)
        return False

    # Initialize authentication
    auth(tokenpath=tokenpath)
    filename = os.path.basename(filepath)

    try:
        # Get collection ID and check file existence
        collection_xid = _get_collection_xid(
            org_shortname, study_shortname, collection_num
        )

        if _check_file_exists(org_shortname, collection_xid, filename):
            if not overwrite:
                print(
                    f"File {filename} already exists in collection. \
                     Use overwrite=True to replace.",
                    file=sys.stderr,
                )
                return False
            print(f"File {filename} exists, overwriting...", file=sys.stderr)

        # Get upload URL and perform upload
        gcs_url = _get_upload_url(org_shortname, collection_xid, filename)
        success = _upload_file(filepath, gcs_url)
        if success:
            _update_collection_files(
                orgname=org_shortname,
                collection_xid=collection_xid,
                uploaded_files=[filename],
            )
            return True

    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return False


def download_file(
    destpath: str, url: str, overwrite: bool = False, headers: dict = None
) -> bool:
    """
    Download the content from the given URL and save it to the destination path.

    Args:
        destpath (str): The local file path where the downloaded content will be saved.
        overwrite (bool): If True, overwrite the file if it already exists.
        url (str): The URL to download the content from.
        headers (dict, optional): Optional HTTP headers to include in the request.

    Returns:
        bool: True if the file was downloaded, False if it already exists.
    """
    if headers is None:
        headers = {}
    destpath = ensure_parent(destpath)

    if destpath.exists() and not overwrite:
        print(f"already exists {destpath}", file=sys.stderr)
        return False

    print(f"getting {destpath}", file=sys.stderr)

    r = requests.get(url, stream=True, headers=headers)
    with destpath.open(mode='wb') as fp:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            fp.write(chunk)
    return True


def download_npz_from_metadata(
    org_shortname: str,
    study_shortname: str,
    collection_num: int,
    tokenpath: str,
    outdir: str = '.',
    overwrite: bool = False,
    segmented: bool = False,
    include_eulers: bool = True,
    include_gait_splits: bool = True,
) -> np.lib.npyio.NpzFile:
    """
    Downloads a .npz file associated with a specific org, study, and collection and
    loads it. User friendly version.

    Args:
        org_shortname (str): Organization ID.
        study_shortname (str): Short name of the study.
        collection_num (int): Collection number within the study.
        tokenpath (str): Path to the authentication token file. Must be specified.
        outdir (str, optional): Output directory for the downloaded file.
        overwrite (bool, optional): Whether to overwrite the file if it exists.
        segmented (bool, optional): If True, loads and returns segmented data.
        include_eulers (bool, optional): Whether to include Eulers in the download.
        include_gait_splits (bool, optional): Whether to include gait splits in
            the download.

    Returns:
        np.lib.npyio.NpzFile: Loaded .npz file.
    """
    destpath = (
        f"{outdir}/{org_shortname}/{study_shortname}/{collection_num}/"
        f"{org_shortname}_{study_shortname}_{collection_num}.npz"
    )
    if os.path.exists(destpath) and not overwrite:
        print(f"already exists {destpath}", file=sys.stderr)
        if segmented:
            return segmenter.load_segmented(destpath)
        else:
            return np.load(destpath)
    if tokenpath is None:
        raise ValueError("tokenpath must be specified to download NPZ from metadata.")

    auth(tokenpath=tokenpath)
    kwargs = {"study": study_shortname, "num": collection_num}
    (collection,) = get_cionic(f'{org_shortname}/collections', **kwargs)

    urlpath = f"{org_shortname}/collections/{collection['xid']}/streams/npz"

    download_npz(
        destpath=destpath,
        urlpath=urlpath,
        overwrite=overwrite,
        include_eulers=include_eulers,
        include_gait_splits=include_gait_splits,
    )

    if segmented:
        return segmenter.load_segmented(destpath)
    else:
        return np.load(destpath)


def download_npz(
    destpath: str,
    urlpath: str,
    overwrite: bool = False,
    include_eulers: bool = True,
    include_gait_splits: bool = True,
) -> None:
    """
    Downloads a .npz file from a specified URL path and saves it to the destpath.

    Args:
        destpath (str): The local file path where the .npz file will be saved.
        urlpath (str): The URL or identifier used to locate the .npz file.
        overwrite (bool): If True, overwrite the file if it already exists.
        include_eulers (bool): Whether to include Eulers in the download.
        include_gait_splits (bool): Whether to include gait splits in the download.

    Returns:
        None
    """
    npz_dict = get_cionic(urlpath)

    status = download_file(destpath, npz_dict['streams.npz'], overwrite=overwrite)
    if status is False:
        return
    if include_eulers:
        include_eulers_to_npz(destpath)
    if include_gait_splits:
        include_gait_splits_to_npz(destpath)


def download_files_from_metadata(
    org_shortname: str,
    study_shortname: str,
    collection_num: int,
    tokenpath: str,
    outdir: str = '.',
    include: list = None,
    exclude: list = None,
) -> tuple[str, str]:
    """
    Downloads files from a collection using metadata, with optional file filtering.

    Args:
        org_shortname (str): Organization ID.
        study_shortname (str): Short name of the study.
        collection_num (int): Collection number within the study.
        tokenpath (str): Path to the authentication token file.
        outdir (str, optional): Base output directory for downloaded files.
            Defaults to current directory.
        include (list, optional): List of file extensions to include
            (e.g., ['.npz', '.json']).
        exclude (list, optional): List of file extensions to exclude.

    Returns:
        tuple[str, str]: Tuple of (API urlpath, local destination path) used
            for the download.

    """
    if tokenpath is None:
        raise ValueError("tokenpath must be specified to download NPZ from metadata.")

    auth(tokenpath=tokenpath)
    kwargs = {"study": study_shortname, "num": collection_num}
    (collection,) = get_cionic(f'{org_shortname}/collections', **kwargs)

    urlpath = f'{org_shortname}/collections/{collection["xid"]}/files'
    destpath = f'{outdir}/{org_shortname}/{study_shortname}/{collection_num}/'

    download_files(urlpath, destpath, include=include, exclude=exclude)
    return urlpath, destpath


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
    array_dict: dict[str, np.recarray],
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


def include_gait_splits_to_npz(destpath: str) -> None:
    '''
    Loads a NumPy .npz file from the specified path, computes gait split arrays and
    updated segments, and adds them to the .npz file.

    Args:
        destpath (str): The file path to the .npz file to be updated.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified file does not exist.
    '''
    try:
        npz = np.load(destpath)
    except FileNotFoundError:
        print(f"File {destpath} not found.", file=sys.stderr)
        return

    new_files, updated_segments = get_splits_arrays_and_segments(npz=npz)

    array_dict = {**new_files, 'segments': updated_segments}
    add_arrays_to_npz_and_store(npz, array_dict, destpath)


def create_new_segment_helper(
    segment: np.void, path: str, stream: str = 'intervals'
) -> np.void:
    """
    Create a new segment structured array entry for walking intervals.

    Args:
        segment (np.void): Existing segment structured array entry.
        path (str): Path for the new segment.
        stream (str): Stream name for the new segment (default 'intervals').

    Returns:
        np.void: New segment structured array entry with updated fields.
    """
    new_segment = np.zeros((), dtype=segment.dtype)
    new_segment['path'] = path
    new_segment['position'] = segment['position']
    new_segment['device'] = segment['device']
    new_segment['stream'] = stream
    return new_segment


def get_splits_arrays_and_segments(
    npz: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.recarray], np.recarray]:
    '''
    Processes segments in the given npz file, generating new arrays and segments
    for walking intervals and paired walking splits based on segment properties.

    Args:
        npz (np.lib.npyio.NpzFile): NPZ file containing 'segments' and time series.

    Returns:
        tuple[dict[str, np.recarray], np.recarray]: A dictionary of new arrays keyed
        by path, and an updated array of segments.
    '''
    new_files = {}
    new_segments = []
    for segment in npz['segments']:
        is_shank = '_shank' in segment['position']
        is_thigh = '_thigh' in segment['position']
        is_euler = segment['stream'] == 'euler'

        if is_shank and is_euler:
            grouped_walking_periods = get_grouped_walking_periods_as_array(
                kinematic_time_series=npz[segment['path']],
            )
            path = f'{segment["device"]}_walking_periods'
            new_files[path] = grouped_walking_periods

            new_segment = create_new_segment_helper(segment=segment, path=path)
            new_segments.append(new_segment)

        if (is_shank or is_thigh) and is_euler:
            paired_stride_splits = get_paired_stride_splits_as_array(
                kinematic_time_series=npz[segment['path']],
                component="x",
                n_start_remove=0,
                n_stop_remove=0,
            )
            path = f'{segment["device"]}_paired_stride_splits'
            new_files[path] = paired_stride_splits

            new_segment = create_new_segment_helper(segment=segment, path=path)
            new_segments.append(new_segment)

    if len(new_files) == 0:
        raise RuntimeError(
            "No gait splits computed from the provided NPZ file. This likely means "
            "include_eulers is set to False. Try setting include_eulers to True when "
            "calling download_npz()."
        )
    updated_segments = np.concatenate(
        [
            npz_utils.change_segments_column_dtype(npz['segments']),
            new_segments,
        ]
    )
    return new_files, updated_segments


def get_grouped_walking_periods_as_array(
    kinematic_time_series: np.recarray,
    component: str = "x",
    peak_kwargs: dict = None,
) -> np.recarray:
    '''
    Convert grouped walking splits from time series into a structured NumPy array.

    Args:
        kinematic_time_series (np.recarray): Input time series data.
        component (str): Component to analyze (default "x").
        peak_kwargs (dict, optional): Arguments for peak detection.

    Returns:
        np.recarray: Structured array with start, stop, and elapsed times.
    '''
    grouped_walking_periods = kinematics.get_grouped_walking_splits(
        kinematic_time_series=kinematic_time_series,
        component=component,
        peak_kwargs=peak_kwargs,
    )
    grouped_walking_periods_array = np.array(
        [
            (group[0], group[-1], group[-1] - group[0])
            for group in grouped_walking_periods
        ],
        dtype=np.dtype([('start_s', 'f8'), ('stop_s', 'f8'), ('elapsed_s', 'f8')]),
    )
    return grouped_walking_periods_array


def get_paired_stride_splits_as_array(
    kinematic_time_series: np.recarray,
    component: str = "x",
    n_start_remove: int = 0,
    n_stop_remove: int = 0,
    peak_kwargs: dict = None,
) -> np.recarray:
    '''
    Convert paired walking splits from time series into a structured NumPy array.

    Args:
        kinematic_time_series (np.recarray): Input time series data.
        component (str): Component to analyze (default "x").
        n_start_remove (int): Number of splits to remove from start.
        n_stop_remove (int): Number of splits to remove from end.
        peak_kwargs (dict, optional): Arguments for peak detection.

    Returns:
        np.recarray: Structured array with start, stop, and elapsed times.
    '''
    paired_stride_splits = kinematics.get_paired_walking_splits(
        kinematic_time_series=kinematic_time_series,
        component=component,
        n_start_remove=n_start_remove,
        n_stop_remove=n_stop_remove,
        peak_kwargs=peak_kwargs,
    )
    paired_stride_splits_array = np.array(
        [(start, stop, stop - start) for start, stop in paired_stride_splits],
        dtype=np.dtype([('start_s', 'f8'), ('stop_s', 'f8'), ('elapsed_s', 'f8')]),
    )
    return paired_stride_splits_array


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
