import numpy as np
import pytest

from cionic import npz_utils


@pytest.fixture(scope="module")
def npz():
    return np.load("example.npz")


def test_multiple_streams_found(npz):
    field_filters = {"position": "l_shank"}
    with pytest.raises(npz_utils.MultipleStreamsFoundError):
        npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)


def test_stream_found(npz):
    field_filters = {"position": "l_shank", "stream": "euler"}
    stream = npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)
    assert stream is not None


def test_stream_not_found(npz):
    field_filters = {"position": "l_shank", "stream": "fake_stream"}
    with pytest.raises(npz_utils.StreamNotFoundError):
        npz_utils.retrieve_stream_generalized(npz=npz, field_filters=field_filters)
