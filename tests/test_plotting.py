"""Tests for plotting module configuration handling.

Run with:
    pytest tests/test_plotting.py -v
    pytest tests/test_plotting.py::TestStreamsPlotterConfig -v
"""

from cionic import plotting


class TestStreamsPlotterConfig:
    """Test StreamsPlotter stream_config parameter handling."""

    def test_default_stream_config(self, mocker):
        """Test that default stream configuration is applied when none provided."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        _ = plotting.StreamsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
        )

        # Verify download was called with default config
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is True
        assert call_kwargs['include_filtered_emgs'] is True
        assert call_kwargs['include_gait_splits'] is True

    def test_custom_stream_config(self, mocker):
        """Test that custom stream configuration overrides defaults."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        custom_config = {
            'include_eulers': False,
            'include_filtered_emgs': True,
            'include_gait_splits': False,
        }

        _ = plotting.StreamsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
            stream_config=custom_config,
        )

        # Verify download was called with custom config
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is False
        assert call_kwargs['include_filtered_emgs'] is True
        assert call_kwargs['include_gait_splits'] is False

    def test_partial_stream_config_override(self, mocker):
        """Test that partial config override merges with defaults."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        # Only override one setting
        partial_config = {'include_filtered_emgs': False}

        _ = plotting.StreamsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
            stream_config=partial_config,
        )

        # Verify defaults are kept except for overridden value
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is True  # default
        assert call_kwargs['include_filtered_emgs'] is False  # overridden
        assert call_kwargs['include_gait_splits'] is True  # default

    def test_other_parameters_passed_through(self, mocker):
        """Test that other parameters are still passed correctly."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        _ = plotting.StreamsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=42,
            tokenpath="/custom/path.json",
            outdir="/custom/outdir",
            segmented=False,
            overwrite=True,
            stream_config={'include_eulers': False},
            peak_kwargs={'min_height': 10},
        )

        # Verify all parameters passed correctly
        mock_download.assert_called_once()
        call_args = mock_download.call_args

        assert call_args.kwargs['org_shortname'] == "test_org"
        assert call_args.kwargs['study_shortname'] == "test_study"
        assert call_args.kwargs['collection_num'] == 42
        assert call_args.kwargs['tokenpath'] == "/custom/path.json"
        assert call_args.kwargs['outdir'] == "/custom/outdir"
        assert call_args.kwargs['segmented'] is False
        assert call_args.kwargs['overwrite'] is True
        assert call_args.kwargs['include_eulers'] is False
        assert call_args.kwargs['peak_kwargs'] == {'min_height': 10}

    def test_empty_stream_config_uses_defaults(self, mocker):
        """Test that empty dict still applies defaults."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        _ = plotting.StreamsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
            stream_config={},
        )

        # Verify defaults are applied even with empty dict
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is True
        assert call_kwargs['include_filtered_emgs'] is True
        assert call_kwargs['include_gait_splits'] is True


class TestStreamsSplitsPlotterConfig:
    """Test StreamsSplitsPlotter stream_config parameter handling."""

    def test_default_stream_config(self, mocker):
        """Test that default stream configuration is applied when none provided."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        _ = plotting.StreamsSplitsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
        )

        # Verify download was called with default config
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is True
        assert call_kwargs['include_filtered_emgs'] is True
        assert call_kwargs['include_gait_splits'] is True

    def test_custom_stream_config(self, mocker):
        """Test that custom stream configuration overrides defaults."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        custom_config = {
            'include_eulers': False,
            'include_filtered_emgs': True,
            'include_gait_splits': False,
        }

        _ = plotting.StreamsSplitsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
            stream_config=custom_config,
        )

        # Verify download was called with custom config
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is False
        assert call_kwargs['include_filtered_emgs'] is True
        assert call_kwargs['include_gait_splits'] is False

    def test_partial_stream_config_override(self, mocker):
        """Test that partial config override merges with defaults."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        # Only override one setting
        partial_config = {'include_gait_splits': False}

        _ = plotting.StreamsSplitsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=1,
            tokenpath="/path/to/token.json",
            stream_config=partial_config,
        )

        # Verify defaults are kept except for overridden value
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs

        assert call_kwargs['include_eulers'] is True  # default
        assert call_kwargs['include_filtered_emgs'] is True  # default
        assert call_kwargs['include_gait_splits'] is False  # overridden

    def test_other_parameters_passed_through(self, mocker):
        """Test that other parameters are still passed correctly."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        _ = plotting.StreamsSplitsPlotter(
            org_shortname="test_org",
            study_shortname="test_study",
            collection_num=99,
            tokenpath="/another/path.json",
            outdir="/another/outdir",
            segmented=True,
            overwrite=False,
            stream_config={'include_filtered_emgs': False},
            peak_kwargs={'min_prominence': 5},
        )

        # Verify all parameters passed correctly
        mock_download.assert_called_once()
        call_args = mock_download.call_args

        assert call_args.kwargs['org_shortname'] == "test_org"
        assert call_args.kwargs['study_shortname'] == "test_study"
        assert call_args.kwargs['collection_num'] == 99
        assert call_args.kwargs['tokenpath'] == "/another/path.json"
        assert call_args.kwargs['outdir'] == "/another/outdir"
        assert call_args.kwargs['segmented'] is True
        assert call_args.kwargs['overwrite'] is False
        assert call_args.kwargs['include_filtered_emgs'] is False
        assert call_args.kwargs['peak_kwargs'] == {'min_prominence': 5}


class TestStreamConfigBackwardCompatibility:
    """Test that the new approach doesn't break existing usage patterns."""

    def test_streams_plotter_basic_usage(self, mocker):
        """Test basic usage without stream_config still works."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        plotter = plotting.StreamsPlotter(
            org_shortname="cionic",
            study_shortname="reference_colls",
            collection_num=1,
            tokenpath="/path/to/token.json",
        )

        # Should succeed without errors
        assert plotter.npz is not None
        assert mock_download.called

    def test_splits_plotter_basic_usage(self, mocker):
        """Test basic usage without stream_config still works."""
        mock_npz = {'segments': mocker.MagicMock()}
        mock_download = mocker.patch(
            'cionic.plotting.api.download_npz_from_metadata', return_value=mock_npz
        )

        plotter = plotting.StreamsSplitsPlotter(
            org_shortname="cionic",
            study_shortname="reference_colls",
            collection_num=1,
            tokenpath="/path/to/token.json",
        )

        # Should succeed without errors
        assert plotter.npz is not None
        assert mock_download.called
