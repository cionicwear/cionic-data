"""Angle Unwrapping Utility.

This module provides a class for unwrapping angular values that wrap around at
2π intervals. It includes a modified unwrapping algorithm that:
1. Restricts wrap-around to single intervals (-1, 0, 1)
2. Enforces consistent wrap direction
3. Uses a configurable threshold for wrap detection
"""

import numpy as np


class Unwrap:
    """Unwraps angular values that wrap around at 2π intervals.

    This class maintains state between calls to handle continuous unwrapping
    of a sequence of angles. It uses a modified algorithm that:
    - Detects wrapping using a threshold of 10π/8 radians
    - Restricts wrap-around to single intervals (-2π, 0, +2π)
    - Enforces consistent wrap direction once established

    Attributes:
        prev_x (float): Previous input angle
        curr_x (float): Current unwrapped angle
        wrap_around (int): Current wrap count (-1, 0, or 1)
        wrapdir_set (bool): Whether wrap direction has been established
        wrapdir (int): Established wrap direction (-1 or 1)
    """

    def __init__(self):
        """Initialize unwrapper with default state."""
        self.prev_x = 0
        self.curr_x = 0
        self.wrap_around = 0

        # uncomment for applications where rotation is limited to 360 degrees
        # # Variables for wrap direction consistency
        # self.wrapdir_set = False
        # self.wrapdir = 0

    def process(self, x: float) -> float:
        """Process a new angle value, unwrapping if necessary.

        Uses a threshold of 10π/8 radians to detect wrapping. Once a wrap
        direction is established (positive or negative), only allows wrapping
        in that direction and restricts wrap_around to [-1, 1].

        Args:
            x (float): New angle value in radians

        Returns:
            float: Unwrapped angle value in radians
        """
        threshold = 10 * np.pi / 8  # 10/8 of a full circle in radians
        if x - self.prev_x < -threshold:
            self.wrap_around += 1
            # uncomment for applications where rotation is limited to 360 degrees
            # if not self.wrapdir_set:
            #     self.wrapdir = 1
            #     self.wrapdir_set = True
        elif x - self.prev_x > threshold:
            self.wrap_around -= 1
            # uncomment for applications where rotation is limited to 360 degrees
            # if not self.wrapdir_set:
            #     self.wrapdir = -1
            #     self.wrapdir_set = True

        # uncomment for applications where rotation is limited to 360 degrees
        # # Only allow wrap in one direction once established
        # if self.wrapdir == -1 and self.wrap_around > 0:
        #     self.wrap_around = 0
        # if self.wrapdir == 1 and self.wrap_around < 0:
        #     self.wrap_around = 0

        # # Restrict wrap_around to -1, 0, 1
        # self.wrap_around = min(self.wrap_around, 1)
        # self.wrap_around = max(self.wrap_around, -1)

        # Update prev_x and curr_x values
        self.prev_x = x
        self.curr_x = x + self.wrap_around * 2 * np.pi
        return self.curr_x
