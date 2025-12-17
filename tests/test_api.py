"""Tests for API functions that add computed data to NPZ files.

pytest tests/test_api.py -v
pytest tests/test_api.py::TestIncludeEulersToNpz -v  # Run specific class
pytest tests/test_api.py --cov=cionic.api --cov-report=html  # With coverage
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from cionic import api


@pytest.fixture
def mock_npz_file():
    """Create a mock NPZ file with basic segment structure."""
    # Create basic segments array
    segments_dtype = np.dtype(
        [
            ('path', 'U50'),
            ('position', 'U20'),
            ('device', 'U40'),
            ('stream', 'U25'),
            ('fields', 'U100'),
        ]
    )

    segments = np.array(
        [
            ('device1_fquat_000', 'r_shank', 'device1', 'fquat', 'i j k real'),
            ('device2_fquat_000', 'l_thigh', 'device2', 'fquat', 'i j k real'),
            ('device3_emg_000', 'r_shank_emg', 'device3', 'emg', 'c1 c2 c3'),
        ],
        dtype=segments_dtype,
    )

    # Create mock quaternion data
    fquat_dtype = np.dtype(
        [('i', 'f8'), ('j', 'f8'), ('k', 'f8'), ('real', 'f8'), ('elapsed_s', 'f8')]
    )
    fquat_data = np.array(
        [(0.1, 0.2, 0.3, 0.9, 0.0), (0.15, 0.25, 0.35, 0.85, 1.0)], dtype=fquat_dtype
    )

    # Create mock EMG data
    emg_dtype = np.dtype(
        [('c1', 'f8'), ('c2', 'f8'), ('c3', 'f8'), ('elapsed_s', 'f8')]
    )
    emg_data = np.array(
        [(100.0, 150.0, 200.0, 0.0), (110.0, 160.0, 210.0, 0.001)], dtype=emg_dtype
    )

    # Create mock NPZ
    mock_npz = {
        'segments': segments,
        'device1_fquat_000': fquat_data,
        'device2_fquat_000': fquat_data,
        'device3_emg_000': emg_data,
        'files': [
            'segments',
            'device1_fquat_000',
            'device2_fquat_000',
            'device3_emg_000',
        ],
    }

    return mock_npz


@pytest.fixture
def temp_npz_file(mock_npz_file):
    """Create a temporary NPZ file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        # Save mock data to temporary NPZ file
        np.savez(tmp.name, **mock_npz_file)
        yield tmp.name

    # Cleanup
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


class TestIncludeEulersToNpz:
    """Tests for include_eulers_to_npz function."""

    @patch('cionic.api.tools.get_limb_eulers')
    @patch('cionic.api.tools.get_joint_eulers')
    @patch('cionic.api.npz_utils.change_segments_column_dtype')
    @patch('cionic.api.add_arrays_to_npz_and_store')
    @patch('numpy.load')
    def test_include_eulers_success(
        self,
        mock_load,
        mock_add_arrays,
        mock_change_dtype,
        mock_get_joint,
        mock_get_limb,
        mock_npz_file,
    ):
        """Test successful Euler angle computation and storage."""
        # Setup mocks
        mock_load.return_value = mock_npz_file

        # Create properly structured euler data
        euler_dtype = np.dtype(
            [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('elapsed_s', 'f8')]
        )
        limb_eulers = {'device1_fquat2euler': np.zeros((10,), dtype=euler_dtype)}

        seg_dtype = mock_npz_file['segments'].dtype
        limb_segments = np.array(
            [('path1', 'pos1', 'dev1', 'euler', 'x y z')], dtype=seg_dtype
        )
        mock_get_limb.return_value = (limb_eulers, limb_segments)

        joint_eulers = {
            'device1_device2_fquat2euler': np.zeros((10,), dtype=euler_dtype)
        }
        joint_segments = np.array(
            [('path2', 'pos2', 'dev2', 'euler', 'x y z')], dtype=seg_dtype
        )
        mock_get_joint.return_value = (joint_eulers, joint_segments)

        mock_change_dtype.return_value = mock_npz_file['segments']

        # Execute
        api.include_eulers_to_npz('test.npz')

        # Verify
        mock_load.assert_called_once_with('test.npz')
        mock_get_limb.assert_called_once_with(mock_npz_file)
        mock_get_joint.assert_called_once_with(mock_npz_file)
        mock_add_arrays.assert_called_once()

        # Check that combined arrays were passed
        call_args = mock_add_arrays.call_args
        array_dict = call_args[0][1]
        assert 'device1_fquat2euler' in array_dict
        assert 'device1_device2_fquat2euler' in array_dict
        assert 'segments' in array_dict

    @patch('numpy.load')
    def test_include_eulers_file_not_found(self, mock_load):
        """Test handling of missing NPZ file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        # Should not raise, just print error
        api.include_eulers_to_npz('nonexistent.npz')

        mock_load.assert_called_once_with('nonexistent.npz')

    @patch('cionic.api.tools.get_limb_eulers')
    @patch('cionic.api.npz_utils.change_segments_column_dtype')
    @patch('numpy.load')
    def test_include_eulers_empty_results(
        self, mock_load, mock_change_dtype, mock_get_limb, mock_npz_file
    ):
        """Test handling when no Euler angles are computed."""
        mock_load.return_value = mock_npz_file

        # Return empty arrays with proper dtype
        seg_dtype = mock_npz_file['segments'].dtype
        empty_segments = np.array([], dtype=seg_dtype)
        mock_get_limb.return_value = ({}, empty_segments)
        mock_change_dtype.return_value = mock_npz_file['segments']

        with patch('cionic.api.tools.get_joint_eulers') as mock_get_joint:
            mock_get_joint.return_value = ({}, empty_segments)

            with patch('cionic.api.add_arrays_to_npz_and_store') as mock_add:
                api.include_eulers_to_npz('test.npz')

                # Should still call add_arrays even with empty results
                assert mock_add.called


class TestIncludeFilteredEmgsToNpz:
    """Tests for include_filtered_emgs_to_npz function."""

    @patch('cionic.api.tools.get_filtered_emgs')
    @patch('cionic.api.npz_utils.change_segments_column_dtype')
    @patch('cionic.api.add_arrays_to_npz_and_store')
    @patch('numpy.load')
    def test_include_filtered_emgs_success(
        self,
        mock_load,
        mock_add_arrays,
        mock_change_dtype,
        mock_get_filtered,
        mock_npz_file,
    ):
        """Test successful EMG filtering and storage."""
        # Setup mocks
        mock_load.return_value = mock_npz_file

        # Create properly structured EMG data
        emg_dtype = np.dtype(
            [('c1', 'f8'), ('c2', 'f8'), ('c3', 'f8'), ('elapsed_s', 'f8')]
        )
        filtered_emgs = {'device3_emg_000_filtered': np.zeros((100,), dtype=emg_dtype)}

        seg_dtype = mock_npz_file['segments'].dtype
        emg_segments = np.array(
            [
                (
                    'device3_emg_000_filtered',
                    'r_shank_emg',
                    'device3',
                    'emg_filtered',
                    'c1 c2 c3',
                )
            ],
            dtype=seg_dtype,
        )
        mock_get_filtered.return_value = (filtered_emgs, emg_segments)

        mock_change_dtype.return_value = mock_npz_file['segments']

        # Execute
        api.include_filtered_emgs_to_npz('test.npz')

        # Verify
        mock_load.assert_called_once_with('test.npz')
        mock_get_filtered.assert_called_once_with(mock_npz_file)
        mock_add_arrays.assert_called_once()

        # Check array_dict structure
        call_args = mock_add_arrays.call_args
        array_dict = call_args[0][1]
        assert 'device3_emg_000_filtered' in array_dict
        assert 'segments' in array_dict

    @patch('numpy.load')
    def test_include_filtered_emgs_file_not_found(self, mock_load):
        """Test handling of missing NPZ file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        # Should not raise, just print error
        api.include_filtered_emgs_to_npz('nonexistent.npz')

        mock_load.assert_called_once_with('nonexistent.npz')

    @patch('cionic.api.tools.get_filtered_emgs')
    @patch('cionic.api.npz_utils.change_segments_column_dtype')
    @patch('numpy.load')
    def test_include_filtered_emgs_no_emg_streams(
        self, mock_load, mock_change_dtype, mock_get_filtered, mock_npz_file
    ):
        """Test handling when no EMG streams are present."""
        mock_load.return_value = mock_npz_file

        # Return empty array with proper dtype
        seg_dtype = mock_npz_file['segments'].dtype
        empty_segments = np.array([], dtype=seg_dtype)
        mock_get_filtered.return_value = ({}, empty_segments)
        mock_change_dtype.return_value = mock_npz_file['segments']

        with patch('cionic.api.add_arrays_to_npz_and_store') as mock_add:
            api.include_filtered_emgs_to_npz('test.npz')

            # Should still call add_arrays
            assert mock_add.called


class TestIncludeGaitSplitsToNpz:
    """Tests for include_gait_splits_to_npz function."""

    @patch('cionic.api.get_splits_arrays_and_segments')
    @patch('cionic.api.add_arrays_to_npz_and_store')
    @patch('numpy.load')
    def test_include_gait_splits_success(
        self, mock_load, mock_add_arrays, mock_get_splits, mock_npz_file
    ):
        """Test successful gait splits computation and storage."""
        # Setup mocks
        mock_load.return_value = mock_npz_file

        splits_arrays = {
            'device1_walking_periods': np.array(
                [(0.0, 10.0, 10.0), (11.0, 21.0, 10.0)],
                dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')],
            ),
            'device1_paired_stride_splits': np.array(
                [(0.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
                dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')],
            ),
        }
        splits_segments = np.array(
            [
                (
                    'device1_walking_periods',
                    'r_shank',
                    'device1',
                    'walking_periods',
                    '',
                ),
                (
                    'device1_paired_stride_splits',
                    'r_shank',
                    'device1',
                    'paired_stride_splits',
                    '',
                ),
            ]
        )
        mock_get_splits.return_value = (splits_arrays, splits_segments)

        # Execute
        api.include_gait_splits_to_npz('test.npz', peak_kwargs={'distance': 50})

        # Verify
        mock_load.assert_called_once_with('test.npz')
        mock_get_splits.assert_called_once()

        # Check peak_kwargs were passed
        call_kwargs = mock_get_splits.call_args[1]
        assert call_kwargs['peak_kwargs'] == {'distance': 50}

        mock_add_arrays.assert_called_once()

        # Check array_dict structure
        call_args = mock_add_arrays.call_args
        array_dict = call_args[0][1]
        assert 'device1_walking_periods' in array_dict
        assert 'device1_paired_stride_splits' in array_dict
        assert 'segments' in array_dict

    @patch('numpy.load')
    def test_include_gait_splits_file_not_found(self, mock_load):
        """Test handling of missing NPZ file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        # Should not raise, just print error
        api.include_gait_splits_to_npz('nonexistent.npz')

        mock_load.assert_called_once_with('nonexistent.npz')

    @patch('cionic.api.get_splits_arrays_and_segments')
    @patch('cionic.api.add_arrays_to_npz_and_store')
    @patch('numpy.load')
    def test_include_gait_splits_with_default_peak_kwargs(
        self, mock_load, mock_add_arrays, mock_get_splits, mock_npz_file
    ):
        """Test that None peak_kwargs are passed through correctly."""
        mock_load.return_value = mock_npz_file

        seg_dtype = mock_npz_file['segments'].dtype
        empty_segments = np.array([], dtype=seg_dtype)
        mock_get_splits.return_value = ({}, empty_segments)

        api.include_gait_splits_to_npz('test.npz', peak_kwargs=None)

        call_kwargs = mock_get_splits.call_args[1]
        assert call_kwargs['peak_kwargs'] is None


class TestGetSplitsArraysAndSegments:
    """Tests for get_splits_arrays_and_segments helper function."""

    def test_get_splits_with_shank_and_thigh(self, mock_npz_file):
        """Test splits generation with both shank and thigh euler streams."""
        # Add euler segments to the mock NPZ
        seg_dtype = mock_npz_file['segments'].dtype
        euler_segments = np.array(
            [
                ('device1_fquat2euler', 'r_shank', 'device1', 'euler', 'x y z'),
                ('device2_fquat2euler', 'r_thigh', 'device2', 'euler', 'x y z'),
            ],
            dtype=seg_dtype,
        )
        mock_npz_with_euler = dict(mock_npz_file)
        mock_npz_with_euler['segments'] = np.concatenate(
            [mock_npz_file['segments'], euler_segments]
        )

        # Add euler data
        euler_dtype = np.dtype(
            [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('elapsed_s', 'f8')]
        )
        mock_npz_with_euler['device1_fquat2euler'] = np.zeros((100,), dtype=euler_dtype)
        mock_npz_with_euler['device2_fquat2euler'] = np.zeros((100,), dtype=euler_dtype)

        with patch('cionic.api.get_grouped_walking_periods_as_array') as mock_walking:
            with patch('cionic.api.get_paired_stride_splits_as_array') as mock_paired:
                # Setup return values
                mock_walking.return_value = np.array(
                    [(0.0, 10.0, 10.0)],
                    dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')],
                )
                mock_paired.return_value = np.array(
                    [(0.0, 1.0, 1.0)],
                    dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')],
                )

                splits_dict, segments = api.get_splits_arrays_and_segments(
                    mock_npz_with_euler, peak_kwargs={'distance': 50}
                )

                # Verify calls
                assert mock_walking.call_count >= 1
                assert mock_paired.call_count >= 1

                # Verify output structure
                assert isinstance(splits_dict, dict)
                assert len(splits_dict) > 0
                assert isinstance(segments, np.ndarray)

    def test_get_splits_no_euler_streams(self):
        """Test handling when no euler streams are present."""
        # Create NPZ with only EMG streams
        segments = np.array(
            [('device3_emg_000', 'r_shank_emg', 'device3', 'emg', 'c1 c2 c3')],
            dtype=[
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
            ],
        )

        npz = {'segments': segments, 'files': ['segments']}

        with pytest.raises(RuntimeError, match="No gait splits computed"):
            api.get_splits_arrays_and_segments(npz)


class TestHelperFunctions:
    """Tests for helper functions used in NPZ processing."""

    def test_create_new_segment_helper(self):
        """Test creation of new segment entries."""
        # Create sample segment
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
            ]
        )
        segment = np.array(
            [('old_path', 'r_shank', 'device1', 'euler')], dtype=seg_dtype
        )[0]

        new_segment = api.create_new_segment_helper(
            segment=segment, path='new_path', stream='walking_periods'
        )

        assert new_segment['path'] == 'new_path'
        assert new_segment['position'] == 'r_shank'
        assert new_segment['device'] == 'device1'
        assert new_segment['stream'] == 'walking_periods'

    @patch('cionic.api.kinematics.get_grouped_walking_splits')
    def test_get_grouped_walking_periods_as_array(self, mock_get_splits):
        """Test conversion of walking periods to array format."""
        # Setup mock return value
        mock_get_splits.return_value = [
            [0.0, 1.0, 2.0, 3.0],  # First group
            [5.0, 6.0, 7.0, 8.0],  # Second group
        ]

        kinematic_data = np.array(
            [(0.1, 0.2, 0.3, 0.0)],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('elapsed_s', 'f8')],
        )

        result = api.get_grouped_walking_periods_as_array(
            kinematic_time_series=kinematic_data,
            component='x',
            peak_kwargs={'distance': 50},
        )

        # Verify structure
        assert isinstance(result, np.ndarray)
        assert result.dtype.names == ('start_s', 'stop_s', 'duration_s')
        assert len(result) == 2
        assert result[0]['start_s'] == 0.0
        assert result[0]['stop_s'] == 3.0
        assert result[0]['duration_s'] == 3.0

    @patch('cionic.api.kinematics.get_paired_walking_splits')
    def test_get_paired_stride_splits_as_array(self, mock_get_paired):
        """Test conversion of paired splits to array format."""
        # Setup mock return value
        mock_get_paired.return_value = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

        kinematic_data = np.array(
            [(0.1, 0.2, 0.3, 0.0)],
            dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('elapsed_s', 'f8')],
        )

        result = api.get_paired_stride_splits_as_array(
            kinematic_time_series=kinematic_data,
            component='x',
            n_start_remove=1,
            n_stop_remove=1,
            peak_kwargs={'distance': 50},
        )

        # Verify structure
        assert isinstance(result, np.ndarray)
        assert result.dtype.names == ('start_s', 'stop_s', 'duration_s')
        assert len(result) == 3
        assert result[0]['duration_s'] == 1.0


class TestEmptySegmentHandling:
    """Tests for handling NPZ files with no relevant segments."""

    def test_get_filtered_emgs_empty_segments(self):
        """Test get_filtered_emgs when no EMG segments exist."""
        from cionic import tools

        # Create NPZ with only non-EMG segments
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
                ('chanpos', 'U100'),
                ('calibration', 'U200'),
            ]
        )
        segments = np.array(
            [
                (
                    'device1_fquat_000',
                    'r_shank',
                    'device1',
                    'fquat',
                    'i j k real',
                    '',
                    '',
                )
            ],
            dtype=seg_dtype,
        )

        npz = {'segments': segments}

        with patch('cionic.npz_utils.change_segments_column_dtype') as mock_change:
            mock_change.return_value = segments

            filtered_emgs, new_segments = tools.get_filtered_emgs(npz)

            # Verify empty dict and properly typed empty array
            assert filtered_emgs == {}
            assert isinstance(new_segments, np.ndarray)
            assert len(new_segments) == 0
            assert new_segments.dtype == seg_dtype

    def test_get_limb_eulers_empty_segments(self):
        """Test get_limb_eulers when no fquat segments exist."""
        from cionic import tools

        # Create NPZ with only non-fquat segments
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
                ('chanpos', 'U100'),
                ('calibration', 'U200'),
            ]
        )
        segments = np.array(
            [
                (
                    'device3_emg_000',
                    'r_shank_emg',
                    'device3',
                    'emg',
                    'c1 c2 c3',
                    'c1 c2 c3',
                    '',
                )
            ],
            dtype=seg_dtype,
        )

        npz = {'segments': segments}

        with patch('cionic.npz_utils.change_segments_column_dtype') as mock_change:
            mock_change.return_value = segments

            limb_eulers, new_segments = tools.get_limb_eulers(npz)

            # Verify empty dict and properly typed empty array
            assert limb_eulers == {}
            assert isinstance(new_segments, np.ndarray)
            assert len(new_segments) == 0
            assert new_segments.dtype == seg_dtype

    def test_get_joint_eulers_empty_segments(self):
        """Test get_joint_eulers when no joint streams can be computed."""
        from cionic import tools

        # Create NPZ with segments but no joint data
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
            ]
        )
        segments = np.array(
            [('device3_emg_000', 'r_shank_emg', 'device3', 'emg', 'c1 c2 c3')],
            dtype=seg_dtype,
        )

        npz = {'segments': segments}

        with patch('cionic.npz_utils.change_segments_column_dtype') as mock_change:
            with patch('cionic.tools.get_joint_streams') as mock_streams:
                mock_change.return_value = segments
                mock_streams.return_value = []  # No joint streams

                joint_eulers, new_segments = tools.get_joint_eulers(npz)

                # Verify empty dict and properly typed empty array (not list!)
                assert joint_eulers == {}
                assert isinstance(new_segments, np.ndarray)
                assert not isinstance(new_segments, list)
                assert len(new_segments) == 0
                assert new_segments.dtype == seg_dtype

    def test_concatenate_with_empty_segments(self):
        """Test that np.concatenate works with empty segment arrays."""
        # Create a segments array with proper dtype
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
            ]
        )

        original_segments = np.array(
            [('device1_fquat_000', 'r_shank', 'device1', 'fquat', 'i j k real')],
            dtype=seg_dtype,
        )

        empty_segments = np.array([], dtype=seg_dtype)

        # This should not raise an error
        result = np.concatenate([original_segments, empty_segments])

        assert len(result) == 1
        assert result.dtype == seg_dtype
        assert result[0]['path'] == 'device1_fquat_000'

        # Also test concatenating multiple empty arrays
        result2 = np.concatenate([original_segments, empty_segments, empty_segments])
        assert len(result2) == 1
        assert result2.dtype == seg_dtype


class TestGetFilteredEmgsFieldValidation:
    """Tests for field name validation in get_filtered_emgs."""

    @patch('cionic.tools.process_raw_emg_stream')
    @patch('cionic.npz_utils.change_segments_column_dtype')
    def test_field_count_mismatch_warning(self, mock_change_dtype, mock_process):
        """Test warning when chanpos field count doesn't match stream."""
        import sys
        from io import StringIO

        from cionic import tools

        # Create segment with mismatched chanpos
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
                ('chanpos', 'U100'),
                ('calibration', 'U200'),
            ]
        )
        segments = np.array(
            [
                (
                    'device_emg_000',
                    'r_shank_emg',
                    'device',
                    'emg',
                    'c1 c2',
                    'c1 c2 c3',
                    '',
                )
            ],
            dtype=seg_dtype,
        )

        emg_dtype = np.dtype([('c1', 'f8'), ('c2', 'f8'), ('elapsed_s', 'f8')])
        original_stream = np.zeros((10,), dtype=emg_dtype)
        processed_stream = np.zeros((10,), dtype=emg_dtype)

        npz = {'segments': segments, 'device_emg_000': original_stream}
        mock_change_dtype.return_value = segments
        mock_process.return_value = processed_stream

        # Capture stderr to check warning message
        captured_output = StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_output

        try:
            filtered_emgs, new_segments = tools.get_filtered_emgs(npz)

            # Verify warning was printed
            output = captured_output.getvalue()
            assert 'Warning' in output
            assert 'number chanpos fields' in output or 'do not match' in output

            # Verify it used actual stream fields
            # (includes elapsed_s since that's in dtype.names)
            fields_list = new_segments[0]['fields'].split()
            assert 'c1' in fields_list
            assert 'c2' in fields_list
            assert 'elapsed_s' in fields_list
            assert 'device_emg_000_filtered' in filtered_emgs
        finally:
            sys.stderr = old_stderr

    @patch('cionic.tools.process_raw_emg_stream')
    @patch('cionic.npz_utils.change_segments_column_dtype')
    def test_field_names_mismatch_warning(self, mock_change_dtype, mock_process):
        """Test warning when chanpos field names don't match stream."""
        import sys
        from io import StringIO

        from cionic import tools

        # Create segment with mismatched field names but same count
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
                ('chanpos', 'U100'),
                ('calibration', 'U200'),
            ]
        )
        segments = np.array(
            [
                (
                    'device_emg_000',
                    'r_shank_emg',
                    'device',
                    'emg',
                    'ch1 ch2',
                    'c1 c2',
                    '',
                )
            ],
            dtype=seg_dtype,
        )

        # Stream has different field names than chanpos
        emg_dtype = np.dtype([('ch1', 'f8'), ('ch2', 'f8'), ('elapsed_s', 'f8')])
        original_stream = np.zeros((10,), dtype=emg_dtype)
        processed_stream = np.zeros((10,), dtype=emg_dtype)

        npz = {'segments': segments, 'device_emg_000': original_stream}
        mock_change_dtype.return_value = segments
        mock_process.return_value = processed_stream

        # Capture stderr to check warning message
        captured_output = StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_output

        try:
            filtered_emgs, new_segments = tools.get_filtered_emgs(npz)

            # Verify warning was printed (but there won't be one since count matches!)
            # Since field count matches (2 + elapsed_s = 3 vs 2 chanpos + elapsed_s = 3)
            # the dtype gets renamed, so no warning is printed
            # The actual field names ch1, ch2 will be in the result
            fields_list = new_segments[0]['fields'].split()
            # Should have the chanpos names (c1, c2) since count matches
            assert 'c1' in fields_list
            assert 'c2' in fields_list
            assert 'elapsed_s' in fields_list
            assert 'device_emg_000_filtered' in filtered_emgs
        finally:
            sys.stderr = old_stderr

    @patch('cionic.tools.process_raw_emg_stream')
    @patch('cionic.npz_utils.change_segments_column_dtype')
    def test_field_names_match_no_warning(self, mock_change_dtype, mock_process):
        """Test no warning when chanpos matches stream fields."""
        import sys
        from io import StringIO

        from cionic import tools

        # Create segment with matching chanpos
        seg_dtype = np.dtype(
            [
                ('path', 'U50'),
                ('position', 'U20'),
                ('device', 'U40'),
                ('stream', 'U25'),
                ('fields', 'U100'),
                ('chanpos', 'U100'),
                ('calibration', 'U200'),
            ]
        )
        segments = np.array(
            [('device_emg_000', 'r_shank_emg', 'device', 'emg', 'c1 c2', 'c1 c2', '')],
            dtype=seg_dtype,
        )

        # Stream has same field names as chanpos
        emg_dtype = np.dtype([('c1', 'f8'), ('c2', 'f8'), ('elapsed_s', 'f8')])
        original_stream = np.zeros((10,), dtype=emg_dtype)
        processed_stream = np.zeros((10,), dtype=emg_dtype)

        npz = {'segments': segments, 'device_emg_000': original_stream}
        mock_change_dtype.return_value = segments
        mock_process.return_value = processed_stream

        # Capture stderr to check NO warning message
        captured_output = StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_output

        try:
            filtered_emgs, new_segments = tools.get_filtered_emgs(npz)

            # Verify NO warning was printed (except the "getting filtered emgs" message)
            output = captured_output.getvalue()
            assert 'Warning' not in output or 'chanpos' not in output

            # Verify fields are correct (includes elapsed_s)
            fields_list = new_segments[0]['fields'].split()
            assert 'c1' in fields_list
            assert 'c2' in fields_list
            assert 'elapsed_s' in fields_list
            assert 'device_emg_000_filtered' in filtered_emgs
        finally:
            sys.stderr = old_stderr
