import numpy as np
import pandas as pd

from cionic.pod_utils import load_imu_from_csv


def make_test_csvs(tmp_path, position="r_shank"):
    """Helper to create minimal valid imu.csv and emg.csv for a given limb position."""
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
    """Test normal case: valid files and data for a limb position."""
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
    """Test missing emg.csv file returns None and prints error."""
    (tmp_path / "imu.csv").write_text("")
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None


def test_load_imu_from_csv_empty(tmp_path):
    """Test empty files return None and print error."""
    pd.DataFrame().to_csv(tmp_path / "imu.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "emg.csv", index=False)
    df = load_imu_from_csv(str(tmp_path), "r_shank")
    assert df is None
