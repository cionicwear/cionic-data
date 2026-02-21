"""Preprocessing Utilities for IMU and Sensor Data.

This module provides functions for loading and preprocessing IMU and sensor data
from various file formats (CSV, NPZ). It includes utilities for:
- Loading foot IMU data from CSV files using the Cionic API
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_imu_from_csv(
    path: str,
    position: str,
    unwrap_euler: bool = True,
    time_align: bool = True,
    side: str = None,
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
        time = quat["elapsed"] / 10000.0
    elif len(accel) > 0 and "elapsed" in accel.columns:
        time = accel["elapsed"] / 10000.0
    elif len(gyro) > 0 and "elapsed" in gyro.columns:
        time = gyro["elapsed"] / 10000.0
    if time is None or len(time) == 0:
        print(f"No time data found for {position}")
        return None

    # Time align and zero start
    if time_align and side is not None:
        # Step 1: Sync clocks (e.g., foot clock -> thigh clock)
        time = time_align_from_shank(time, path, position, side)
        # Step 2: Global offset (shank t_0 -> 0.0)
        time = zero_start_from_shank(time, path, side)
    else:
        # Fallback: just make the individual sensor start at 0
        time = time - time.iloc[0]

    # Convert quaternions to euler angles (radians) using scipy
    eulers = None
    if len(quat) > 0:
        try:
            quats = quat[["x", "y", "z", "w"]].values
            eulers = R.from_quat(quats).as_euler('xyz', degrees=True)
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

            # Use numpy.unwrap for each angle
            for angle in ["roll", "pitch", "yaw"]:
                arr = np.radians(df[angle].values)
                unwrapped = np.unwrap(arr)
                df[angle] = np.degrees(unwrapped - np.mean(unwrapped))  # center on 0
        except (ValueError, KeyError):
            print("Euler unwrapping failed, returning raw angles.")
            pass  # Skip unwrapping if it fails

    return df


def time_align_from_shank(
    time: pd.Series, path: Union[str, Path], position: str, side: str
) -> pd.DataFrame:
    """Align limb-specific IMU data to a common thigh reference frame.

    Calculates a temporal offset by finding the nearest row-index neighbor
    between the target position and the thigh anchor in the master imu.csv.
    This offset is then applied to the input DataFrame's elapsed time.

    DISCLAIMER: This function assumes that the imu.csv is populated in
    real-time. It relies on the row-proximity of sensor entries as the
    ground truth for synchronization, rather than the internal timestamps
    of the individual sensors.

    Args:
    df (pd.DataFrame): The limb-specific data to be aligned. 'elapsed_s' in seconds.
    path (Union[str, Path]): Path to directory containing imu.csv. 'elapsed' in hundos.
    position (str): Position on body, e.g. "r_shank".
    side (str): The body side to align to ("left" or "right").

    Returns:
    pd.DataFrame: Data with the 'elapsed' column shifted to the
    thigh's time frame.
    """

    if side.lower() == "left":
        target_limb = "l_shank"
    elif side.lower() == "right":
        target_limb = "r_shank"
    else:
        print(
            f"Invalid side: {side}. Must be 'left' or 'right'. Returning original data."
        )
        return time
    try:
        # Loading without predefined names to handle potential headers;
        # ensure 'limb' and 'elapsed' are present.
        imu_df = pd.read_csv(Path(path) / "imu.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: imu.csv not found at {path}. Returning original data.")
        return time

    # Check if 'limb' and 'elapsed' columns exist
    if 'limb' not in imu_df.columns or 'elapsed' not in imu_df.columns:
        print("Warning: imu.csv missing 'limb' or 'elapsed' columns.")
        return time

    pos_indices = imu_df[imu_df['limb'] == position].index
    target_indices = imu_df[imu_df['limb'] == target_limb].index

    if pos_indices.empty or target_indices.empty:
        print(f"Warning: {position} or {target_limb} not found in imu.csv.")
        return time

    # 1. Find Start Pair (using whichever starts later to ensure overlap)
    start_idx_ref = max(pos_indices[0], target_indices[0])
    first_pos_idx = pos_indices[np.abs(pos_indices - start_idx_ref).argmin()]
    closest_target_start_idx = target_indices[
        np.abs(target_indices - first_pos_idx).argmin()
    ]

    t_pos_start = imu_df.loc[first_pos_idx, 'elapsed']
    t_target_start = imu_df.loc[closest_target_start_idx, 'elapsed']

    # 2. Find End Pair (using whichever ends earlier to ensure overlap)
    end_idx_ref = min(pos_indices[-1], target_indices[-1])
    last_pos_idx = pos_indices[np.abs(pos_indices - end_idx_ref).argmin()]
    closest_target_end_idx = target_indices[
        np.abs(target_indices - last_pos_idx).argmin()
    ]

    t_pos_end = imu_df.loc[last_pos_idx, 'elapsed']
    t_target_end = imu_df.loc[closest_target_end_idx, 'elapsed']

    # 3. Apply the temporal shift (based on the start pair)
    offset = (t_target_start - t_pos_start) / 10000.0
    time_aligned = time.copy() + offset

    # 4. Compute and apply dynamic drift factor
    pos_duration = t_pos_end - t_pos_start
    target_duration = t_target_end - t_target_start

    # Calculate drift: ratio of thigh-time to pos-time minus 1
    computed_drift = (target_duration / pos_duration) - 1 if pos_duration != 0 else 0
    zero_start_for_warp = time_aligned - time_aligned.iloc[0]
    time_aligned = time_aligned + zero_start_for_warp * computed_drift

    return time_aligned


def zero_start_from_shank(
    time: pd.Series, path: Union[str, Path], side: str
) -> pd.Series:
    """Offsets the aligned time series so the shank's first recording is t=0.

    Args:
        time (pd.Series): The synchronized elapsed time (in seconds).
        path (Union[str, Path]): Path to directory containing imu.csv.
        side (str): The body side to align to ("left" or "right").

    Returns:
        pd.Series: Time series starting relative to the shank's first entry.
    """
    if side.lower() == "left":
        target_limb = "l_shank"
    elif side.lower() == "right":
        target_limb = "r_shank"
    else:
        print(
            f"Invalid side: {side}. Must be 'left' or 'right'. Returning original data."
        )
        return time

    try:
        imu_df = pd.read_csv(Path(path) / "imu.csv")
        thigh_entries = imu_df[imu_df["limb"] == target_limb]

        if not thigh_entries.empty:
            # Get the absolute first timestamp of the shank in seconds
            shank_start_s = thigh_entries["elapsed"].iloc[0] / 10000.0
            return time - shank_start_s

    except Exception as e:
        print(f"Zero start failed: {e}")

    # Fallback: if shank not found, just zero the series to its own start
    print(f"Warning: {target_limb} not found in imu.csv. Zeroing to own start.")
    return time - time.iloc[0]
