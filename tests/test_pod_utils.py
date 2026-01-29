"""
Usage: pytest tests/test_pod_utils.py

Comprehensive unit tests for the `load_imu_from_csv` function in the `pod_utils` module.

These tests verify the correct behavior of loading and preprocessing IMU data from CSV
files based on specified limb positions. The tests provide full code coverage including:

- Normal case with valid files and data
- Handling of missing or invalid input files
- Different time source fallbacks (quat, accel, gyro)
- Missing columns and fallback column names
- Euler angle unwrapping (on/off)
- Partial data availability (quat only, accel only, gyro only)
- Edge cases and error conditions

Test Cases:
    - test_load_imu_from_csv_normal: Validates loading and processing of IMU data
    - test_load_imu_from_csv_missing_file: Ensures handling of missing input files
    - test_load_imu_from_csv_empty: Ensures handling of empty input files
    - test_load_imu_from_csv_only_imu: Tests with only IMU file (no EMG)
    - test_load_imu_from_csv_only_emg: Tests with only EMG file (no IMU)
    - test_load_imu_from_csv_missing_both_files: Tests when both files are missing
    - test_load_imu_from_csv_time_from_accel: Tests time extraction from accel
    - test_load_imu_from_csv_time_from_gyro: Tests time extraction from gyro
    - test_load_imu_from_csv_no_time_data: Tests when no time data is available
    - test_load_imu_from_csv_no_data_for_position: Tests when position not found
    - test_load_imu_from_csv_fallback_column_names: Tests fallback colnames
    - test_load_imu_from_csv_missing_quat_columns: Tests missing quaternion columns
    - test_load_imu_from_csv_invalid_quaternion: Tests invalid quaternion data
    - test_load_imu_from_csv_no_euler_conversion: Tests when euler conversion fails
    - test_load_imu_from_csv_unwrap_euler_false: Tests with unwrap_euler=False
    - test_load_imu_from_csv_unwrap_euler_true: Tests with unwrap_euler=True
    - test_load_imu_from_csv_different_lengths: Tests trimming to shortest length
    - test_load_imu_from_csv_empty_after_filtering: Tests empty data after filtering
"""

import numpy as np
import pandas as pd

from cionic.pod_utils import load_imu_from_csv


def make_test_csvs(tmp_path, position="r_shank"):
    """Helper to create minimal valid imu.csv and emg.csv for a given limb position.

    Args:
        tmp_path: Temporary directory path for test files.
        position: Limb position string for IMU data.
    """
    imu = pd.DataFrame(
        {
            "limb": [position] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    # Making emg longer to ensure trimming works
    emg = pd.DataFrame(
        {
            "limb": [f"{position}_accel", f"{position}_gyro"] * 4,
            "elapsed": [0, 0, 10000, 10000, 20000, 20000, 30000, 30000],
            "x": [1, 11, 2, 12, 3, 13, 0, 0],
            "y": [4, 14, 5, 15, 6, 16, 0, 0],
            "z": [7, 17, 8, 18, 9, 19, 0, 0],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)
    emg.to_csv(tmp_path / "emg.csv", index=False)


def test_load_imu_from_csv_normal(tmp_path):
    """Test normal case: valid files and data for a limb position.

    Args:
        tmp_path: Temporary directory path for test files.

    Asserts:
        Existence of loaded and processed DataFrame columns.
        Correct values in loaded DataFrame.
        Correct trimming to shortest input length.
    """
    make_test_csvs(tmp_path)
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    # Should be 0, 10000, 20000 hundos -> 0, 1, 2 s
    assert "elapsed_s" in df.columns
    assert np.allclose(df["elapsed_s"].values, [0, 1, 2])

    # Should be all 0s and 1s for quaternions
    assert "fquat_i" in df.columns
    assert "fquat_j" in df.columns
    assert "fquat_k" in df.columns
    assert "fquat_real" in df.columns
    assert np.allclose(df["fquat_i"].values, [0, 0, 0])
    assert np.allclose(df["fquat_j"].values, [0, 0, 0])
    assert np.allclose(df["fquat_k"].values, [0, 0, 0])
    assert np.allclose(df["fquat_real"].values, [1, 1, 1])

    # Euler angles from (0,0,0,1) quaternion should be all zero
    assert "roll" in df.columns
    assert "pitch" in df.columns
    assert "yaw" in df.columns
    assert np.allclose(df["roll"].values, [0, 0, 0])
    assert np.allclose(df["pitch"].values, [0, 0, 0])
    assert np.allclose(df["yaw"].values, [0, 0, 0])

    # Should be exactly what was in emg.csv for accel
    assert "accel_x" in df.columns
    assert "accel_y" in df.columns
    assert "accel_z" in df.columns
    assert np.allclose(df["accel_x"].values, [1, 2, 3])
    assert np.allclose(df["accel_y"].values, [4, 5, 6])
    assert np.allclose(df["accel_z"].values, [7, 8, 9])

    # Should be exactly what was in emg.csv for gyro
    assert "gyro_x" in df.columns
    assert "gyro_y" in df.columns
    assert "gyro_z" in df.columns
    assert np.allclose(df["gyro_x"].values, [11, 12, 13])
    assert np.allclose(df["gyro_y"].values, [14, 15, 16])
    assert np.allclose(df["gyro_z"].values, [17, 18, 19])

    assert len(df) == 3  # should match shortest input length


def test_load_imu_from_csv_missing_file(tmp_path):
    """Test missing emg.csv file returns None and prints error.

    Args:
        tmp_path: Temporary directory path for test files.

    Asserts:
        That the function returns None when a required file is missing.
    """
    (tmp_path / "imu.csv").write_text("")
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_empty(tmp_path):
    """Test empty files return None and print error.

    Args:
        tmp_path: Temporary directory path for test files.

    Asserts:
        That the function returns None when input files are empty.
    """
    pd.DataFrame().to_csv(tmp_path / "imu.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "emg.csv", index=False)
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_only_imu(tmp_path):
    """Test loading with only IMU file (no EMG file).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "elapsed_s" in df.columns
    assert "fquat_i" in df.columns
    assert "roll" in df.columns
    # Should not have accel or gyro
    assert "accel_x" not in df.columns
    assert "gyro_x" not in df.columns


def test_load_imu_from_csv_only_emg(tmp_path):
    """Test loading with only EMG file (no IMU file).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel", "r_shank_gyro"] * 2,
            "elapsed": [0, 0, 10000, 10000],
            "x": [1, 11, 2, 12],
            "y": [4, 14, 5, 15],
            "z": [7, 17, 8, 18],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "elapsed_s" in df.columns
    assert "accel_x" in df.columns
    assert "gyro_x" in df.columns
    # Should not have quaternion data
    assert "fquat_i" not in df.columns


def test_load_imu_from_csv_missing_both_files(tmp_path):
    """Test when both files are missing.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_time_from_accel(tmp_path):
    """Test time extraction from accelerometer when quat has no time.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    # IMU without elapsed column
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    # EMG with elapsed column
    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "elapsed_s" in df.columns
    assert np.allclose(df["elapsed_s"].values, [0, 1, 2])


def test_load_imu_from_csv_time_from_gyro(tmp_path):
    """Test time extraction from gyroscope when quat and accel have no time.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    # IMU without elapsed column
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    # EMG with elapsed only in gyro (no accel rows)
    emg = pd.DataFrame(
        {
            "limb": ["r_shank_gyro"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [11, 12, 13],
            "y": [14, 15, 16],
            "z": [17, 18, 19],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "elapsed_s" in df.columns


def test_load_imu_from_csv_no_time_data(tmp_path):
    """Test when no time data is available.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_no_data_for_position(tmp_path):
    """Test when requested position is not found in data.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["l_shank"] * 3,  # Different position
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["l_shank_accel", "l_shank_gyro"] * 2,
            "elapsed": [0, 0, 10000, 10000],
            "x": [1, 11, 2, 12],
            "y": [4, 14, 5, 15],
            "z": [7, 17, 8, 18],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")  # Requesting different position
    assert df is None


def test_load_imu_from_csv_fallback_column_names(tmp_path):
    """Test fallback to alternative column names (ax/ay/az, gx/gy/gz).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    # EMG with fallback column names - accel uses ax/ay/az, gyro uses gx/gy/gz
    emg_accel = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "ax": [1, 2, 3],  # Using 'ax' instead of 'x'
            "ay": [4, 5, 6],  # Using 'ay' instead of 'y'
            "az": [7, 8, 9],  # Using 'az' instead of 'z'
        }
    )
    emg_gyro = pd.DataFrame(
        {
            "limb": ["r_shank_gyro"] * 3,
            "elapsed": [0, 10000, 20000],
            "gx": [21, 22, 23],  # Using 'gx' for gyro
            "gy": [24, 25, 26],
            "gz": [27, 28, 29],
        }
    )
    emg = pd.concat([emg_accel, emg_gyro], ignore_index=True)
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "accel_x" in df.columns
    assert "accel_y" in df.columns
    assert "accel_z" in df.columns
    assert "gyro_x" in df.columns
    assert "gyro_y" in df.columns
    assert "gyro_z" in df.columns
    assert np.allclose(df["accel_x"].values, [1, 2, 3])
    assert np.allclose(df["gyro_x"].values, [21, 22, 23])


def test_load_imu_from_csv_missing_quat_columns(tmp_path):
    """Test when quaternion columns are missing (KeyError should be caught)

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            # Missing x, y, z, w columns
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "fquat_i" not in df.columns
    assert "accel_x" in df.columns


def test_load_imu_from_csv_invalid_quaternion(tmp_path):
    """Test handling of invalid quaternion data.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [np.nan, 0, 0],  # Invalid quaternion
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    # Should handle gracefully - may skip euler conversion but still return data
    assert df is not None
    assert "accel_x" in df.columns


def test_load_imu_from_csv_euler_conversion(tmp_path):
    """Test normal euler conversion with valid quaternion data.

    Note: The quaternion [0, 0, 0, 1] is the identity quaternion and converts
    successfully to euler angles [0, 0, 0]. This test verifies that the function
    handles normal quaternion-to-euler conversion correctly.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    # Valid quaternion data (identity quaternion)
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    # Should have successfully converted quaternion to euler angles
    assert "roll" in df.columns
    assert "pitch" in df.columns
    assert "yaw" in df.columns


def test_load_imu_from_csv_unwrap_euler_false(tmp_path):
    """Test with unwrap_euler=False.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    make_test_csvs(tmp_path)
    df = load_imu_from_csv(str(tmp_path), "r_shank", unwrap_euler=False)

    assert df is not None
    assert "roll" in df.columns
    assert "pitch" in df.columns
    assert "yaw" in df.columns
    # Should not have raw_roll, raw_pitch, raw_yaw when unwrap_euler=False
    assert "raw_roll" not in df.columns
    assert "raw_pitch" not in df.columns
    assert "raw_yaw" not in df.columns


def test_load_imu_from_csv_unwrap_euler_true(tmp_path):
    """Test with unwrap_euler=True (default).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    make_test_csvs(tmp_path)
    df = load_imu_from_csv(str(tmp_path), "r_shank", unwrap_euler=True)

    assert df is not None
    assert "roll" in df.columns
    assert "pitch" in df.columns
    assert "yaw" in df.columns
    # Should have raw_roll, raw_pitch, raw_yaw when unwrap_euler=True
    assert "raw_roll" in df.columns
    assert "raw_pitch" in df.columns
    assert "raw_yaw" in df.columns


def test_load_imu_from_csv_different_lengths(tmp_path):
    """Test trimming to shortest data length.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    # IMU with 3 samples
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    # EMG with 5 samples (longer)
    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel", "r_shank_gyro"] * 5,
            "elapsed": [0, 0, 10000, 10000, 20000, 20000, 30000, 30000, 40000, 40000],
            "x": [1, 11, 2, 12, 3, 13, 4, 14, 5, 15],
            "y": [4, 14, 5, 15, 6, 16, 7, 17, 8, 18],
            "z": [7, 17, 8, 18, 9, 19, 10, 20, 11, 21],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    # Should be trimmed to shortest length (3 from IMU)
    assert len(df) == 3


def test_load_imu_from_csv_empty_after_filtering(tmp_path):
    """Test when data is empty after filtering by position.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    # Files exist but no matching position
    imu = pd.DataFrame(
        {
            "limb": ["other_position"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["other_position_accel"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_accel_only_no_gyro(tmp_path):
    """Test with only accelerometer data (no gyro).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel"] * 3,  # Only accel, no gyro
            "elapsed": [0, 10000, 20000],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "accel_x" in df.columns
    assert "gyro_x" not in df.columns


def test_load_imu_from_csv_gyro_only_no_accel(tmp_path):
    """Test with only gyroscope data (no accel).

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    emg = pd.DataFrame(
        {
            "limb": ["r_shank_gyro"] * 3,  # Only gyro, no accel
            "elapsed": [0, 10000, 20000],
            "x": [11, 12, 13],
            "y": [14, 15, 16],
            "z": [17, 18, 19],
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    assert "gyro_x" in df.columns
    assert "accel_x" not in df.columns


def test_load_imu_from_csv_missing_accel_gyro_columns(tmp_path):
    """Test when accel/gyro columns are missing.

    Args:
        tmp_path: Temporary directory path for test files.
    """
    imu = pd.DataFrame(
        {
            "limb": ["r_shank"] * 3,
            "elapsed": [0, 10000, 20000],
            "x": [0, 0, 0],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "w": [1, 1, 1],
        }
    )
    imu.to_csv(tmp_path / "imu.csv", index=False)

    # EMG with missing x/y/z columns
    emg = pd.DataFrame(
        {
            "limb": ["r_shank_accel", "r_shank_gyro"] * 3,
            "elapsed": [0, 0, 10000, 10000, 20000, 20000],
            # Missing x, y, z columns
        }
    )
    emg.to_csv(tmp_path / "emg.csv", index=False)

    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is not None
    # Should still have quaternion data
    assert "fquat_i" in df.columns
    assert "accel_x" not in df.columns
    assert "gyro_x" not in df.columns
