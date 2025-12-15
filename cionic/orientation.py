"""Quaternion Orientation Utilities.

This module provides functions for working with quaternion orientations,
including conversions to Euler angles and quaternion operations.

The quaternion format used is [x, y, z, w] where:
- x, y, z are the vector components
- w is the scalar component
"""

import math


def orientation_quaternion_to_euler(q: list[float]) -> list[float]:
    """Convert a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (list[float]): Quaternion in [x, y, z, w] format

    Returns:
        list[float]: Euler angles [roll, pitch, yaw] in radians where:
            - roll: rotation around x-axis
            - pitch: rotation around y-axis
            - yaw: rotation around z-axis
    """
    # initialize euler
    e = [0.0, 0.0, 0.0]
    # roll (x-axis rotation)
    sinr = 2.0 * (q[3] * q[0] + q[1] * q[2])
    cosr = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])
    e[0] = math.atan2(sinr, cosr)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[3] * q[1] - q[2] * q[0])
    if abs(sinp) < 1:
        e[1] = math.asin(sinp)
    else:
        e[1] = math.copysign(math.pi / 2, sinp)
    # yaw (z-axis rotation)
    siny = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    e[2] = math.atan2(siny, cosy)
    return e


def orientation_multiply(a: list[float], b: list[float]) -> list[float]:
    """Multiply two quaternions.

    Implements the Hamilton product of two quaternions.

    Args:
        a (list[float]): First quaternion in [x, y, z, w] format
        b (list[float]): Second quaternion in [x, y, z, w] format

    Returns:
        list[float]: Resulting quaternion in [x, y, z, w] format
    """
    out = [0.0, 0.0, 0.0, 0.0]
    # val rw = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    # val rx = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    out[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1]
    # val ry = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    out[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0]
    # val rz = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    out[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3]
    return out


def orientation_inverse(a: list[float]) -> list[float]:
    """Calculate the inverse of a quaternion.

    The inv of a quaternion q is the conjugate of q divided by the norm of q squared.

    Args:
        a (list[float]): Input quaternion in [x, y, z, w] format

    Returns:
        list[float]: Inverse quaternion in [x, y, z, w] format
    """
    out = [0.0, 0.0, 0.0, 0.0]
    d = a[3] * a[3] + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
    out[0] = -a[0] / d
    out[1] = -a[1] / d
    out[2] = -a[2] / d
    out[3] = a[3] / d
    return out


def orientation_difference(a: list[float], b: list[float]) -> list[float]:
    """Calculate the difference between two quaternions.

    Returns the quaternion that rotates from orientation a to orientation b.
    Computed as: b * a^(-1)

    Args:
        a (list[float]): First quaternion in [x, y, z, w] format
        b (list[float]): Second quaternion in [x, y, z, w] format

    Returns:
        list[float]: Difference quaternion in [x, y, z, w] format
    """
    inverse = orientation_inverse(a)
    out = orientation_multiply(inverse, b)
    return out
