"""
Test suite for segmentizer.py module.

This module contains comprehensive pytest tests for the segmentize_walking_periods
function from cionic.segmenter. The tests cover various scenarios including normal
operation, edge cases, and error conditions using in-memory ZIP file operations.

Usage:
    Run all tests:
        $ pytest test_segmenter.py

    Run with verbose output:
        $ pytest -v test_segmenter.py

    Run specific test class:
        $ pytest test_segmenter.py::TestSegmentizeWalkingPeriods

    Run specific test method:
        $ pytest test_segmenter.py::TestSegmentizeWalkingPeriods::test_example_function

Test Coverage:
    - Single boundary segmentation covering full time range
    - Multiple boundary segmentation with separate time windows
    - Boundaries with no overlapping walking periods (empty segments)
    - Partial overlap scenarios where boundaries intersect with some periods
    - Empty walking periods data handling
    - Metadata preservation and updating across segmentation operations

Fixtures:
    - sample_walking_periods: Creates 5 realistic walking periods spanning 0-8.2 seconds
    - empty_walking_periods: Creates empty structured array for edge case testing
    - sample_segmeta: Standard segment metadata dictionary for testing
    - mock_zip_files: Complete in-memory ZIP file setup for file I/O testing

Test Data Structure:
    Walking periods use numpy structured arrays with float64 fields:
    - 'start_s': Period start time in seconds
    - 'stop_s': Period end time in seconds
    - 'duration_s': Period duration in seconds

Expected Behavior:
    The function now returns a list of metadata dictionaries, one for each
    processed boundary, rather than just the last boundary's metadata.
    Each returned dictionary contains:
    - Timing information (start_s, end_s, duration_s)
    - File information (path, nsamples)
    - Segment identification (segment_num, label)
    - Original metadata preservation

Testing Strategy:
    Uses in-memory ZIP files to avoid filesystem dependencies and ensure
    test isolation. All file operations are mocked using BytesIO objects,
    allowing for fast and reliable testing without temporary file creation.
"""

import io
import zipfile

import numpy as np
import pytest

from cionic.segmenter import segmentize_walking_periods


class TestSegmentizeWalkingPeriods:
    """Test suite for segmentize_walking_periods function."""

    @pytest.fixture
    def sample_walking_periods(self):
        """Create sample walking periods data."""
        return np.array(
            [
                (0.0, 1.0, 1.0),  # period 0: 0-1s
                (1.5, 3.0, 1.5),  # period 1: 1.5-3s
                (3.2, 4.8, 1.6),  # period 2: 3.2-4.8s
                (5.0, 6.5, 1.5),  # period 3: 5-6.5s
                (7.0, 8.2, 1.2),  # period 4: 7-8.2s
            ],
            dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')],
        )

    @pytest.fixture
    def empty_walking_periods(self):
        """Create empty walking periods data."""
        return np.array(
            [], dtype=[('start_s', 'f8'), ('stop_s', 'f8'), ('duration_s', 'f8')]
        )

    @pytest.fixture
    def sample_segmeta(self):
        """Create sample segment metadata."""
        return {
            'device': 'test_device',
            'position': 'r_shank',
            'stream': 'walking_periods',
            'original_start_s': 0.0,
            'original_end_s': 10.0,
        }

    @pytest.fixture
    def mock_zip_files(self, sample_walking_periods):
        """Create mock ZIP files for testing."""
        # Create input ZIP file
        input_buffer = io.BytesIO()
        with zipfile.ZipFile(input_buffer, 'w') as inzf:
            data_buffer = io.BytesIO()
            np.save(data_buffer, sample_walking_periods)
            inzf.writestr('walking_periods.npy', data_buffer.getvalue())
        input_buffer.seek(0)

        # Create output ZIP file
        output_buffer = io.BytesIO()

        inzf = zipfile.ZipFile(input_buffer, 'r')
        outzf = zipfile.ZipFile(output_buffer, 'w')
        zi = inzf.getinfo('walking_periods.npy')

        return inzf, zi, outzf, output_buffer

    def test_single_boundary_full_range(self, mock_zip_files, sample_segmeta):
        """Test segmentation with single boundary covering full time range."""
        inzf, zi, outzf, output_buffer = mock_zip_files

        boundaries = [
            {
                'start_s': 0.0,
                'end_s': 10.0,
                'add': {'segment_num': 0, 'label': 'full_walk'},
            }
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=sample_segmeta,
        )

        # Should return a list with one metadata dict
        assert isinstance(result_list, list)
        assert len(result_list) == 1

        result = result_list[0]
        assert result['start_s'] == 0.0
        assert result['end_s'] == 8.2
        assert result['duration_s'] == 8.2
        assert result['path'] == 'test_000'
        assert result['nsamples'] == 5
        assert result['segment_num'] == 0
        assert result['label'] == 'full_walk'

        # Verify file was written
        outzf.close()
        output_buffer.seek(0)
        with zipfile.ZipFile(output_buffer, 'r') as verify_zf:
            assert 'test_000.npy' in verify_zf.namelist()

        inzf.close()

    def test_multiple_boundaries(self, mock_zip_files, sample_segmeta):
        """Test segmentation with multiple boundaries."""
        inzf, zi, outzf, output_buffer = mock_zip_files

        boundaries = [
            {
                'start_s': 0.0,
                'end_s': 3.0,
                'add': {'segment_num': 0, 'label': 'early_walk'},
            },
            {
                'start_s': 3.0,
                'end_s': 6.0,
                'add': {'segment_num': 1, 'label': 'mid_walk'},
            },
            {
                'start_s': 6.0,
                'end_s': 10.0,
                'add': {'segment_num': 2, 'label': 'late_walk'},
            },
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=sample_segmeta,
        )

        # Should return list with metadata for all boundaries
        assert isinstance(result_list, list)
        assert len(result_list) == 3

        # Check each boundary result
        assert result_list[0]['segment_num'] == 0
        assert result_list[0]['label'] == 'early_walk'
        assert result_list[1]['segment_num'] == 1
        assert result_list[1]['label'] == 'mid_walk'
        assert result_list[2]['segment_num'] == 2
        assert result_list[2]['label'] == 'late_walk'

        # Verify all files were written
        outzf.close()
        output_buffer.seek(0)
        with zipfile.ZipFile(output_buffer, 'r') as verify_zf:
            files = verify_zf.namelist()
            assert 'test_000.npy' in files
            assert 'test_001.npy' in files
            assert 'test_002.npy' in files

        inzf.close()

    def test_boundary_with_no_overlapping_periods(self, mock_zip_files, sample_segmeta):
        """Test boundary that doesn't overlap with any walking periods."""
        inzf, zi, outzf, output_buffer = mock_zip_files

        boundaries = [
            {
                'start_s': 10.0,  # After all walking periods
                'end_s': 15.0,
                'add': {'segment_num': 0, 'label': 'no_data'},
            }
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=sample_segmeta,
        )

        assert len(result_list) == 1
        result = result_list[0]

        # Should have empty segment - uses boundary times when no data
        assert result['nsamples'] == 0
        assert result['start_s'] == 10.0  # Uses boundary start when no data
        assert result['end_s'] == 15.0  # Uses boundary end when no data

        inzf.close()
        outzf.close()

    def test_partial_overlap_boundary(self, mock_zip_files, sample_segmeta):
        """Test boundary that partially overlaps with walking periods."""
        inzf, zi, outzf, output_buffer = mock_zip_files

        boundaries = [
            {
                'start_s': 2.0,  # Overlaps with periods 1 and 2
                'end_s': 4.0,
                'add': {'segment_num': 0, 'label': 'partial'},
            }
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=sample_segmeta,
        )

        assert len(result_list) == 1
        result = result_list[0]

        # Should have 2 periods that overlap with 2.0-4.0 range
        assert result['nsamples'] == 2
        assert result['start_s'] == 2.0  # Clipped to boundary
        assert result['end_s'] == 4.0  # Clipped to boundary

        inzf.close()
        outzf.close()

    def test_empty_walking_periods(self, empty_walking_periods, sample_segmeta):
        """Test with empty walking periods array."""
        # Create mock files with empty data
        input_buffer = io.BytesIO()
        with zipfile.ZipFile(input_buffer, 'w') as inzf_create:
            data_buffer = io.BytesIO()
            np.save(data_buffer, empty_walking_periods)
            inzf_create.writestr('walking_periods.npy', data_buffer.getvalue())
        input_buffer.seek(0)

        output_buffer = io.BytesIO()
        inzf = zipfile.ZipFile(input_buffer, 'r')
        outzf = zipfile.ZipFile(output_buffer, 'w')
        zi = inzf.getinfo('walking_periods.npy')

        boundaries = [
            {'start_s': 0.0, 'end_s': 5.0, 'add': {'segment_num': 0, 'label': 'empty'}}
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=sample_segmeta,
        )

        assert len(result_list) == 1
        result = result_list[0]

        assert result['nsamples'] == 0
        assert result['start_s'] == 0.0
        assert result['end_s'] == 5.0

        inzf.close()
        outzf.close()

    def test_segment_metadata_preservation(self, mock_zip_files):
        """Test that original segment metadata is preserved and updated correctly."""
        inzf, zi, outzf, output_buffer = mock_zip_files

        original_segmeta = {
            'device': 'test_device',
            'position': 'r_shank',
            'stream': 'walking_periods',
            'sampling_rate': 100.0,
            'custom_field': 'preserved',
        }

        boundaries = [
            {
                'start_s': 0.0,
                'end_s': 5.0,
                'add': {'segment_num': 42, 'label': 'test_segment'},
            }
        ]

        result_list = segmentize_walking_periods(
            inzf=inzf,
            zi=zi,
            outzf=outzf,
            boundaries=boundaries,
            outstem='test',
            segmeta=original_segmeta,
        )

        assert len(result_list) == 1
        result = result_list[0]

        # Original metadata should be preserved
        assert result['device'] == 'test_device'
        assert result['position'] == 'r_shank'
        assert result['stream'] == 'walking_periods'
        assert result['sampling_rate'] == 100.0
        assert result['custom_field'] == 'preserved'

        # New metadata should be added
        assert result['segment_num'] == 42
        assert result['label'] == 'test_segment'
        assert result['path'] == 'test_042'

        inzf.close()
        outzf.close()


if __name__ == "__main__":
    pytest.main([__file__])
