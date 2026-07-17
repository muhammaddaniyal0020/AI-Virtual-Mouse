"""
util.py
-------
Utility / helper functions for the AI Virtual Mouse project.

This module contains pure, side-effect-free helper functions used by
main.py for:
    * Geometry (angles, distances between hand landmarks)
    * Interpolation / mapping of coordinate ranges
    * Exponential Moving Average (EMA) cursor smoothing
    * Value clamping

Keeping these functions free of any OpenCV / MediaPipe / PyAutoGUI
dependencies makes them easy to unit test in isolation.
"""

import numpy as np


def get_angle(a, b, c):
    """
    Calculate the angle (in degrees) formed at vertex `b` by the rays
    b->a and b->c. Used to determine whether a finger is bent (small
    angle) or straight/extended (large angle).

    Args:
        a, b, c: (x, y) tuples/lists representing landmark coordinates.
                 `b` is the vertex (e.g. a knuckle joint).

    Returns:
        float: Angle in degrees, in the range [0, 180].
    """
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(np.degrees(radians))

    # Normalize to the [0, 180] range so results are consistent
    # regardless of rotation direction.
    if angle > 180:
        angle = 360 - angle

    return angle


def distance_between_points(p1, p2):
    """
    Compute the raw Euclidean distance between two (x, y) points.

    Args:
        p1, p2: (x, y) tuples/lists.

    Returns:
        float: Euclidean distance between p1 and p2.
    """
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def interpolate(value, from_range, to_range):
    """
    Linearly map `value` from `from_range` to `to_range`.

    Args:
        value: The value to remap.
        from_range: [min, max] of the source range.
        to_range: [min, max] of the destination range.

    Returns:
        float: The remapped, clamped value.
    """
    return float(np.interp(value, from_range, to_range))


def get_distance(landmark_list):
    """
    Compute the distance between the first two points in
    `landmark_list`, interpolated from normalized (0-1) camera space
    into a 0-1000 scale. Kept for backward compatibility with the
    original project API (used throughout main.py's gesture checks).

    Args:
        landmark_list: list of at least two (x, y) points.

    Returns:
        float or None: Interpolated distance on a 0-1000 scale, or
        None if fewer than two points are supplied.
    """
    if landmark_list is None or len(landmark_list) < 2:
        return None

    p1, p2 = landmark_list[0], landmark_list[1]
    raw_distance = distance_between_points(p1, p2)
    return interpolate(raw_distance, [0, 1], [0, 1000])


def clamp(value, min_value, max_value):
    """
    Clamp `value` so that it lies within [min_value, max_value].

    Args:
        value: The value to clamp.
        min_value: Lower bound.
        max_value: Upper bound.

    Returns:
        The clamped value.
    """
    return max(min_value, min(value, max_value))


def smooth_cursor(previous_point, target_point, smoothing):
    """
    Apply Exponential Moving Average (EMA) smoothing between the
    previously smoothed cursor position and the newly measured target
    position. This removes high-frequency jitter from hand-tracking
    noise while keeping the cursor responsive.

    Args:
        previous_point: (x, y) previous smoothed cursor location, or
            None on the very first call (no smoothing is applied).
        target_point: (x, y) raw target location for this frame.
        smoothing: float in [0, 1). Higher values produce more
            smoothing (and more lag); lower values are more
            responsive but noisier. 0 disables smoothing entirely.

    Returns:
        (x, y): The new smoothed cursor location.
    """
    if previous_point is None:
        return target_point

    smoothing = clamp(smoothing, 0.0, 0.99)
    alpha = 1.0 - smoothing  # weight given to the new sample

    new_x = previous_point[0] + (target_point[0] - previous_point[0]) * alpha
    new_y = previous_point[1] + (target_point[1] - previous_point[1]) * alpha

    return (new_x, new_y)
