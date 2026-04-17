"""
Usage: pytest tests/test_npz_utils.py

Unit tests for the `retrieve_stream_generalized`, `retrieve_stream`, and
`retrieve_segment_field` functions in the `npz_utils` module.

These tests verify the correct behavior of stream retrieval from an NPZ data source
based on specified field filters. The tests cover the following scenarios:

- Ensuring that a `MultipleStreamsFoundError` is raised when multiple streams match
  the provided filters.
- Confirming that the correct stream is returned when a single match is found.
- Ensuring that a `ValueError` is raised when an invalid field is specified.
- Verifying that `segment_num=None` (the default) skips segment filtering.
- Verifying that an explicit `segment_num` integer filters to the correct segment.
- Verifying that `segment_num=0` is treated as a valid filter.

Fixtures:
    npz: Loads the example NPZ file for use in the tests.
    segmented_npz: Minimal in-memory NPZ with two segments for testing segment_num
        filtering in retrieve_stream and retrieve_segment_field.

Test Cases:
    - test_multiple_streams_found: Checks error handling for multiple matching streams.
    - test_stream_found: Validates successful retrieval of a single matching stream.
    - test_bad_field_identified: Ensures that a `ValueError` is raised when an invalid
        field is specified.
    - test_retrieve_stream_with_segment_num: Filters by segment_num and
        returns correct data.
    - test_retrieve_stream_none_raises_when_ambiguous: segment_num=None skips filter.
    - test_retrieve_stream_default_is_none: Default kwarg behaves like segment_num=None.
    - test_retrieve_segment_field_with_segment_num: Returns field from matching segment.
    - test_retrieve_segment_field_none_returns_first_match: None skips segment filter.
    - test_retrieve_segment_field_default_is_none: Default kwarg behaves like None.
    - test_retrieve_stream_segment_num_zero: segment_num=0 filters correctly.
    - test_retrieve_segment_field_segment_num_zero: segment_num=0 returns correct field.
"""

import io
import json
import zipfile

import numpy as np
import pytest

from cionic import npz_utils


@pytest.fixture(scope="module")
def npz():
    return np.load("npz_examples/example.npz")


def _npy_bytes(arr):
    """Serialize a numpy array to .npy-format bytes."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


@pytest.fixture(scope="module")
def segmented_npz(tmp_path_factory):
    """
    Minimal NPZ with three fquat segments for l_shank (segment_num 0, 1, and 2).

    Segment 0 is included to ensure that segment_num=0 is treated as a valid filter.
    Built via zipfile so that segments.jsonl is stored as raw bytes (matching the real
    NPZ format) and stream data arrays are stored as .npy entries.
    """
    tmp = tmp_path_factory.mktemp("npz")
    path = str(tmp / "segmented.npz")

    data_0 = np.zeros(10, dtype=np.float32)
    data_1 = np.ones(10, dtype=np.float32)
    data_2 = np.ones(10, dtype=np.float32) * 2

    segments = np.array(
        [
            ("l_shank", "fquat", 0, "l_shank_fquat_0"),
            ("l_shank", "fquat", 1, "l_shank_fquat_1"),
            ("l_shank", "fquat", 2, "l_shank_fquat_2"),
        ],
        dtype=[
            ("position", "<U20"),
            ("stream", "<U16"),
            ("segment_num", "<i4"),
            ("path", "<U32"),
        ],
    )
    jsonl = (
        json.dumps(
            {
                "position": "l_shank",
                "stream": "fquat",
                "segment_num": 0,
                "device": "dev_zero",
            }
        )
        + "\n"
        + json.dumps(
            {
                "position": "l_shank",
                "stream": "fquat",
                "segment_num": 1,
                "device": "dev_a",
            }
        )
        + "\n"
        + json.dumps(
            {
                "position": "l_shank",
                "stream": "fquat",
                "segment_num": 2,
                "device": "dev_b",
            }
        )
        + "\n"
    ).encode()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("segments.npy", _npy_bytes(segments))
        zf.writestr("l_shank_fquat_0.npy", _npy_bytes(data_0))
        zf.writestr("l_shank_fquat_1.npy", _npy_bytes(data_1))
        zf.writestr("l_shank_fquat_2.npy", _npy_bytes(data_2))
        zf.writestr("segments.jsonl", jsonl)  # raw bytes — no .npy magic prefix

    return np.load(path)


def test_multiple_streams_found(npz):
    """
    Test that retrieve_stream_generalized raises a MultipleStreamsFoundError
    when multiple streams match the provided field_filters.

    Args:
        npz: A fixture or mock representing the NPZ data source.

    Raises:
        npz_utils.MultipleStreamsFoundError: If more than one stream matches
        field_filters.
    """
    field_filters = {"position": "l_shank"}
    with pytest.raises(npz_utils.MultipleStreamsFoundError):
        npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)


def test_stream_found(npz):
    """
    Test that retrieve_stream_generalized returns a stream when a single
    matching stream is found.

    Args:
        npz: A fixture or mock representing the NPZ data source.

    Asserts:
        That the function returns the expected stream when a single match is found.
    """
    field_filters = {"position": "l_shank", "stream": "euler"}
    stream = npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)
    assert stream is not None


def test_bad_field_identified(npz):
    """
    Test that retrieve_stream_generalized raises a ValueError when an invalid
    field is specified in the filters.

    Args:
        npz: An npz data structure to be queried.

    Raises:
        ValueError: If an invalid field is specified in the filters.
    """
    field_filters = {"position": "l_shank", "stream": "euler", "invalid_field": "value"}
    with pytest.raises(ValueError):
        npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)


# ---------------------------------------------------------------------------
# retrieve_stream — segment_num sentinel tests
# ---------------------------------------------------------------------------


def test_retrieve_stream_with_segment_num(segmented_npz):
    """Passing a specific segment_num returns only that segment's data."""
    stream = npz_utils.retrieve_stream(
        npz=segmented_npz, position="l_shank", stream="fquat", segment_num=1
    )
    assert stream is not None
    assert np.all(stream == 1.0)


def test_retrieve_stream_none_raises_when_ambiguous(segmented_npz):
    """segment_num=None skips the filter, so two matching segments → error."""
    with pytest.raises(npz_utils.MultipleStreamsFoundError):
        npz_utils.retrieve_stream(
            npz=segmented_npz, position="l_shank", stream="fquat", segment_num=None
        )


def test_retrieve_stream_default_is_none(segmented_npz):
    """Omitting segment_num behaves identically to segment_num=None."""
    with pytest.raises(npz_utils.MultipleStreamsFoundError):
        npz_utils.retrieve_stream(npz=segmented_npz, position="l_shank", stream="fquat")


# ---------------------------------------------------------------------------
# retrieve_segment_field — segment_num sentinel tests
# ---------------------------------------------------------------------------


def test_retrieve_segment_field_with_segment_num(segmented_npz):
    """Passing segment_num returns the field value from the matching segment only."""
    device = npz_utils.retrieve_segment_field(
        npz=segmented_npz,
        position="l_shank",
        stream="fquat",
        field_name="device",
        segment_num=1,
    )
    assert device == "dev_a"


def test_retrieve_segment_field_none_returns_first_match(segmented_npz):
    """segment_num=None ignores segment filtering and returns the first match."""
    device = npz_utils.retrieve_segment_field(
        npz=segmented_npz,
        position="l_shank",
        stream="fquat",
        field_name="device",
        segment_num=None,
    )
    assert device == "dev_zero"


def test_retrieve_segment_field_default_is_none(segmented_npz):
    """Omitting segment_num behaves identically to segment_num=None."""
    device = npz_utils.retrieve_segment_field(
        npz=segmented_npz,
        position="l_shank",
        stream="fquat",
        field_name="device",
    )
    assert device == "dev_zero"


# ---------------------------------------------------------------------------
# Regression: segment_num=0 must not be dropped by a falsy check
# ---------------------------------------------------------------------------


def test_retrieve_stream_segment_num_zero(segmented_npz):
    """segment_num=0 is a valid filter value and must not be treated as falsy."""
    stream = npz_utils.retrieve_stream(
        npz=segmented_npz, position="l_shank", stream="fquat", segment_num=0
    )
    assert stream is not None
    assert np.all(stream == 0.0)


def test_retrieve_segment_field_segment_num_zero(segmented_npz):
    """segment_num=0 returns the correct field and must not be treated as falsy."""
    device = npz_utils.retrieve_segment_field(
        npz=segmented_npz,
        position="l_shank",
        stream="fquat",
        field_name="device",
        segment_num=0,
    )
    assert device == "dev_zero"
