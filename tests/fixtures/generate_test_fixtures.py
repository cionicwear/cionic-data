#!/usr/bin/env python3
"""Generate test fixtures for foot segmentation integration tests.

This script pre-processes real data from collection 1274 to create minimal
test fixtures for integration testing. It extracts only the necessary columns
and time range to minimize repository size.

Usage:
    python tests/fixtures/generate_test_fixtures.py

The script will:
1. Load foot IMU data from collection 1274
2. Filter to time range 9-70 seconds (as used in validation notebook)
3. Extract only required columns for segmentation methods
4. Save as compressed numpy file to tests/fixtures/
"""

import sys
from pathlib import Path

import numpy as np

from cionic import pod_utils

# Add cionic-data root to path to import cionic modules
script_dir = Path(__file__).parent
fixtures_dir = script_dir  # Script is in fixtures directory
tests_dir = fixtures_dir.parent
cionic_data_dir = tests_dir.parent

sys.path.insert(0, str(cionic_data_dir))


def generate_foot_imu_fixture():
    """Generate foot IMU fixture from collection 1274."""
    # Paths relative to cionic-data root
    recordings_dir = cionic_data_dir / "recordings"
    collection_dir = recordings_dir / "cionic" / "demo-develop" / "1274"
    output_fixtures_dir = fixtures_dir  # Save to same directory as script

    # Create fixtures directory if it doesn't exist
    output_fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Check if collection data exists
    if not collection_dir.exists():
        print(f"Collection directory not found: {collection_dir}")
        print("Please ensure collection 1274 data is downloaded.")
        print("You can download it using the API or manually.")
        return False

    # Load foot IMU data
    print(f"Loading foot IMU data from {collection_dir}...")
    foot_data = pod_utils.load_imu_from_csv(
        str(collection_dir), "l_foot", unwrap_euler=True
    )

    if foot_data is None or foot_data.empty:
        print("Failed to load foot IMU data")
        return False

    print(f"Loaded {len(foot_data)} rows")
    print(f"Columns: {list(foot_data.columns)}")

    # Filter to time range 9-70 seconds (as used in validation notebook)
    segmentation_range = (9, 70)
    print(f"Filtering to time range {segmentation_range}...")

    mask = (foot_data['elapsed_s'] >= segmentation_range[0]) & (
        foot_data['elapsed_s'] <= segmentation_range[1]
    )
    filtered_data = foot_data[mask].copy().reset_index(drop=True)

    print(f"Filtered to {len(filtered_data)} rows")

    # Extract only required columns for all segmentation methods
    # Peak method: elapsed_s, roll
    # Jasiewicz/Cionic: elapsed_s, roll, accel_x, accel_y, accel_z
    # Seel: elapsed_s, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    required_columns = [
        'elapsed_s',
        'roll',
        'pitch',
        'yaw',  # Euler angles
        'accel_x',
        'accel_y',
        'accel_z',  # Acceleration
        'gyro_x',
        'gyro_y',
        'gyro_z',  # Gyroscope
    ]

    # Only include columns that exist in the data
    available_columns = [
        col for col in required_columns if col in filtered_data.columns
    ]
    fixture_data = filtered_data[available_columns].copy()

    print(f"Extracted {len(available_columns)} columns: {available_columns}")

    # Save as compressed numpy file
    output_path = output_fixtures_dir / "foot_imu_1274_9_70s.npz"
    print(f"Saving to {output_path}...")

    # Convert DataFrame to dict for npz format
    # npz can't directly save DataFrames, so we save arrays and metadata
    save_dict = {}
    for col in fixture_data.columns:
        if fixture_data[col].dtype == 'object':
            # Handle object columns (like 'euler' which is a list)
            # Try to convert to numpy array, or skip if not possible
            try:
                # Try to convert list column to array
                if isinstance(fixture_data[col].iloc[0], (list, np.ndarray)):
                    # Save as list of arrays
                    save_dict[col] = np.array(
                        [np.array(x) for x in fixture_data[col]], dtype=object
                    )
                else:
                    # Skip non-numeric object columns
                    continue
            except (TypeError, ValueError):
                # Skip columns that can't be converted
                continue
        else:
            save_dict[col] = fixture_data[col].values

    # Save column names and dtypes for reconstruction
    save_dict['_columns'] = np.array(fixture_data.columns.tolist(), dtype=object)
    save_dict['_dtypes'] = np.array(
        [str(dtype) for dtype in fixture_data.dtypes.values], dtype=object
    )

    np.savez_compressed(output_path, **save_dict)

    # Calculate file size
    file_size = output_path.stat().st_size
    print(f"Saved fixture: {file_size / 1024:.2f} KB")

    # Verify we can load it back
    print("Verifying fixture...")
    loaded = np.load(output_path, allow_pickle=True)
    print(f"Loaded {len(loaded['_columns'])} columns")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Foot Segmentation Test Fixtures")
    print("=" * 60)

    success = generate_foot_imu_fixture()

    if success:
        print("\n✓ Fixture generation completed successfully!")
    else:
        print("\n✗ Fixture generation failed!")
        sys.exit(1)
