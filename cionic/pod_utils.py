"""Preprocessing Utilities for IMU and Sensor Data.

This module provides functions for loading and preprocessing IMU and sensor data
from various file formats (CSV, NPZ). It includes utilities for:
- Loading foot IMU data from CSV files using the Cionic API
"""

# Third-party imports
import pandas as pd

# Local imports
from .orientation import orientation_quaternion_to_euler
from .unwrap import Unwrap


def load_imu_from_csv(
    path: str, position: str, unwrap_euler: bool = True
) -> pd.DataFrame:
    """Load and preprocess limb-specific IMU data from CSV files.

    Loads IMU and EMG data from CSV files, extracts limb-specific data,
    and processes quaternions into Euler angles.

    Args:
        path (Union[str, Path]): Path to directory containing imu.csv and emg.csv
        position (str): Position on body, e.g. "r_shank".
        unwrap_euler (bool, optional): Whether to unwrap euler angles. Default True.
    Returns:
        pd.DataFrame: Processed data with columns:
            - elapsed_s: Time in seconds
            - euler: List of [roll, pitch, yaw] angles
            - fquat_i/j/k/real: Quaternion components
            - roll/pitch/yaw: Euler angles (optionally unwrapped)
            - raw_roll/pitch/yaw: Raw Euler angles before unwrapping (if unwrap_euler)
            - accel_x/y/z: Acceleration components
            - gyro_x/y/z: Gyroscope components
    """
    # Load CSV files
    imu_df = pd.read_csv(f"{path}/imu.csv")
    other_df = pd.read_csv(f"{path}/emg.csv")

    # Extract limb-specific data
    quat = imu_df[imu_df["limb"] == position]
    accel = other_df[other_df["limb"] == f"{position}_accel"]
    gyro = other_df[other_df["limb"] == f"{position}_gyro"]

    # Convert time to seconds
    time = (quat["elapsed"] - quat["elapsed"].min()) / 10000.0

    # Convert quaternions to euler angles
    eulers = [
        orientation_quaternion_to_euler(
            [
                quat["x"].iloc[i],
                quat["y"].iloc[i],
                quat["z"].iloc[i],
                quat["w"].iloc[i],
            ]
        )
        for i in range(len(quat))
    ]

    # Return early if all have no data
    if all(len(x) == 0 for x in [time, eulers, accel, gyro]):
        print(f"No data found for {position}")
        return None

    # Find minimum length across all data
    min_length = min(len(x) for x in [time, eulers, accel, gyro] if len(x) > 0)

    # Create unified DataFrame, try 'ax, ay, az' for accel and 'gx, gy, gz' for gyro
    # if this doesn't work. This will depend on how the fields were named in the
    # collection device. This is compatible with current cionic-circuitpython FW
    # as of 12/15/2025.
    try:
        df = pd.DataFrame(
            {
                "elapsed_s": time[:min_length],
                "fquat_i": quat["x"].values[:min_length],
                "fquat_j": quat["y"].values[:min_length],
                "fquat_k": quat["z"].values[:min_length],
                "fquat_real": quat["w"].values[:min_length],
                "euler": eulers[:min_length],
                "accel_x": accel["x"].values[:min_length],
                "accel_y": accel["y"].values[:min_length],
                "accel_z": accel["z"].values[:min_length],
                "gyro_x": gyro["x"].values[:min_length],
                "gyro_y": gyro["y"].values[:min_length],
                "gyro_z": gyro["z"].values[:min_length],
            }
        ).reset_index(drop=True)
    except KeyError:
        df = pd.DataFrame(
            {
                "elapsed_s": time[:min_length],
                "fquat_i": quat["x"].values[:min_length],
                "fquat_j": quat["y"].values[:min_length],
                "fquat_k": quat["z"].values[:min_length],
                "fquat_real": quat["w"].values[:min_length],
                "euler": eulers[:min_length],
                "accel_x": accel["ax"].values[:min_length],
                "accel_y": accel["ay"].values[:min_length],
                "accel_z": accel["az"].values[:min_length],
                "gyro_x": gyro["gx"].values[:min_length],
                "gyro_y": gyro["gy"].values[:min_length],
                "gyro_z": gyro["gz"].values[:min_length],
            }
        ).reset_index(drop=True)

    # Extract euler angles
    df[["roll", "pitch", "yaw"]] = pd.DataFrame(df["euler"].tolist(), index=df.index)
    if unwrap_euler:
        # Save raw angles
        df[["raw_roll", "raw_pitch", "raw_yaw"]] = df[["roll", "pitch", "yaw"]]

        # Process each angle
        for angle in ["roll", "pitch", "yaw"]:
            unwrapper = Unwrap()
            df[angle] = [unwrapper.process(x) for x in df[angle]]
            df[angle] -= df[angle].mean()  # center around 0

    return df
