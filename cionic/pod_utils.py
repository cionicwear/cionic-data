"""Preprocessing Utilities for IMU and Sensor Data.

This module provides functions for loading and preprocessing IMU and sensor data
from various file formats (CSV, NPZ). It includes utilities for:
- Loading foot IMU data from CSV files using the Cionic API
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_imu_from_csv(
    path: str, position: str, unwrap_euler: bool = True
) -> pd.DataFrame:
    """Load and preprocess limb-specific IMU data from CSV files.

    Loads IMU and EMG data from CSV files, extracts limb-specific data,
    and processes quaternions into Euler angles. Handles cases where only
    quaternion, accelerometer, or gyroscope data may be available.

    Args:
        path (Union[str, Path]): Path to directory containing imu.csv and emg.csv
        position (str): Position on body, e.g. "r_shank".
        unwrap_euler (bool, optional): Whether to unwrap euler angles. Default True.
    Returns:
        pd.DataFrame: Processed data with available columns:
            - elapsed_s: Time in seconds
            - fquat_i/j/k/real: Quaternion components (if quat available)
            - roll/pitch/yaw: Euler angles (optionally unwrapped, if quat available)
            - raw_roll/pitch/yaw: Raw Euler angles pre-unwrapping (if unwrap_euler)
            - accel_x/y/z: Acceleration components (if accel available)
            - gyro_x/y/z: Gyroscope components (if gyro available)
    """
    # Load CSV files
    try:
        imu_df = pd.read_csv(f"{path}/imu.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        imu_df = None

    try:
        other_df = pd.read_csv(f"{path}/emg.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        other_df = None

    if imu_df is None and other_df is None:
        print(f"Missing or invalid files: {path}/imu.csv and {path}/emg.csv")
        return None

    # Extract limb-specific data
    quat = pd.DataFrame()
    accel = pd.DataFrame()
    gyro = pd.DataFrame()

    if imu_df is not None:
        quat = imu_df[imu_df["limb"] == position]

    if other_df is not None:
        accel = other_df[other_df["limb"] == f"{position}_accel"]
        gyro = other_df[other_df["limb"] == f"{position}_gyro"]

    # Determine time source - prefer quat, fallback to accel or gyro
    time = None
    if len(quat) > 0 and "elapsed" in quat.columns:
        time = (quat["elapsed"] - quat["elapsed"].min()) / 10000.0
    elif len(accel) > 0 and "elapsed" in accel.columns:
        time = (accel["elapsed"] - accel["elapsed"].min()) / 10000.0
    elif len(gyro) > 0 and "elapsed" in gyro.columns:
        time = (gyro["elapsed"] - gyro["elapsed"].min()) / 10000.0
    if time is None or len(time) == 0:
        print(f"No time data found for {position}")
        return None

    # Convert quaternions to euler angles (radians) using scipy
    eulers = None
    if len(quat) > 0:
        try:
            quats = quat[["x", "y", "z", "w"]].values
            eulers = R.from_quat(quats).as_euler('xyz', degrees=False)
        except (KeyError, ValueError):
            eulers = None

    # Return early if no data found
    if all(len(x) == 0 for x in [quat, accel, gyro]):
        print(f"No data found for {position}")
        return None

    # Find minimum length across all available data
    available_data = [time]
    if len(quat) > 0:
        available_data.append(quat)
    if len(accel) > 0:
        available_data.append(accel)
    if len(gyro) > 0:
        available_data.append(gyro)

    min_length = min(len(x) for x in available_data if len(x) > 0)

    # Build DataFrame with only available data
    df_dict = {"elapsed_s": time[:min_length]}

    # Add quaternion data if available
    if len(quat) > 0:
        try:
            df_dict["fquat_i"] = quat["x"].values[:min_length]
            df_dict["fquat_j"] = quat["y"].values[:min_length]
            df_dict["fquat_k"] = quat["z"].values[:min_length]
            df_dict["fquat_real"] = quat["w"].values[:min_length]
            if eulers is not None:
                df_dict["roll"] = eulers[:min_length, 0]
                df_dict["pitch"] = eulers[:min_length, 1]
                df_dict["yaw"] = eulers[:min_length, 2]
        except KeyError:
            pass  # Skip quat if columns don't exist

    # Add accelerometer data if available
    if len(accel) > 0:
        try:
            df_dict["accel_x"] = accel["x"].values[:min_length]
            df_dict["accel_y"] = accel["y"].values[:min_length]
            df_dict["accel_z"] = accel["z"].values[:min_length]
        except KeyError:
            try:
                # Fallback to 'ax', 'ay', 'az'
                df_dict["accel_x"] = accel["ax"].values[:min_length]
                df_dict["accel_y"] = accel["ay"].values[:min_length]
                df_dict["accel_z"] = accel["az"].values[:min_length]
            except KeyError:
                pass  # Skip accel if columns don't exist

    # Add gyroscope data if available
    if len(gyro) > 0:
        try:
            df_dict["gyro_x"] = gyro["x"].values[:min_length]
            df_dict["gyro_y"] = gyro["y"].values[:min_length]
            df_dict["gyro_z"] = gyro["z"].values[:min_length]
        except KeyError:
            try:
                # Fallback to 'gx', 'gy', 'gz'
                df_dict["gyro_x"] = gyro["gx"].values[:min_length]
                df_dict["gyro_y"] = gyro["gy"].values[:min_length]
                df_dict["gyro_z"] = gyro["gz"].values[:min_length]
            except KeyError:
                pass  # Skip gyro if columns don't exist

    df = pd.DataFrame(df_dict).reset_index(drop=True)

    # Unwrap euler angles if available and requested
    if unwrap_euler and "roll" in df.columns and len(df) > 0:
        try:
            # Save raw angles
            df[["raw_roll", "raw_pitch", "raw_yaw"]] = df[["roll", "pitch", "yaw"]]

            # Use numpy.unwrap for each angle (expects radians)
            for angle in ["roll", "pitch", "yaw"]:
                arr = df[angle].values
                unwrapped = np.unwrap(arr)
                df[angle] = unwrapped - np.mean(unwrapped)  # center around 0
        except (ValueError, KeyError):
            pass  # Skip unwrapping if it fails

    return df
