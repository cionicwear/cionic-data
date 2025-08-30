"""
Unit tests for the `retrieve_stream_generalized` function in the `npz_utils` module.

These tests verify the correct behavior of stream retrieval from an NPZ data source
based on specified field filters. The tests cover the following scenarios:

- Ensuring that a `MultipleStreamsFoundError` is raised when multiple streams match
  the provided filters.
- Confirming that the correct stream is returned when a single match is found.
- Ensuring that a `ValueError` is raised when an invalid field is specified.

Fixtures:
    npz: Loads the example NPZ file for use in the tests.

Test Cases:
    - test_multiple_streams_found: Checks error handling for multiple matching streams.
    - test_stream_found: Validates successful retrieval of a single matching stream.
    - test_bad_field_identified: Ensures that a `ValueError` is raised when an invalid
        field is specified.
"""

import numpy as np
import pytest

from cionic import npz_utils


@pytest.fixture(scope="module")
def npz():
    return np.load("example.npz")


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
