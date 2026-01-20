# Test Fixtures

This directory contains pre-processed test data for testing.

## Architecture

To minimize data size in the repository while testing on real data:

1. **Pre-processed Data**: Extract only the necessary columns and time range (9-70 seconds) from collection 1274
2. **Compressed Format**: Save as `.npz` (numpy compressed) format for minimal file size
3. **Fixture Loading**: Use pytest fixtures to load data automatically
4. **Integration Tests**: Test all 4 segmentation algorithms on this real-world data

## Files

- `foot_imu_1274_9_70s.npz`: Pre-processed foot IMU data from collection 1274, time range 9-70 seconds
  - Contains only columns needed for all segmentation methods
  - Compressed numpy format for minimal size

## Regenerating Fixtures

If you need to regenerate the fixture data (e.g., after updating data processing):

```bash
# From cionic-data directory
python tests/fixtures/generate_test_fixtures.py
```

This script:
1. Loads data from collection 1274
2. Extracts foot IMU data using `pod_utils.load_imu_from_csv`
3. Filters to time range 9-70 seconds
4. Saves only required columns as compressed numpy file to `tests/fixtures/`

## Data Source

- Collection: 1274 (demo-develop)
- Position: l_foot (left foot)
- Time Range: 9-70 seconds (as used in validation notebook)
- Columns: All columns required by segmentation methods:
  - `elapsed_s`: Time in seconds
  - `roll`, `pitch`, `yaw`: Euler angles
  - `accel_x`, `accel_y`, `accel_z`: Acceleration
  - `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope
