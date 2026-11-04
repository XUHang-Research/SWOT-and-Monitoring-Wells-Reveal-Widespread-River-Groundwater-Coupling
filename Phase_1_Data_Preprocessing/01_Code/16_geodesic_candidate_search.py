"""Conservative unit-sphere search bounds for WGS84 geodesic screening."""
import numpy as np
from pyproj import Geod

WGS84 = Geod(ellps='WGS84')
# The meridional radius at the equator is the smallest WGS84 metric scale.
# Ellipsoid path length >= this scale * unit-sphere path length. Therefore
# using it (rather than the mean Earth radius) cannot omit an in-range pair.
MIN_WGS84_CURVATURE_RADIUS_M = WGS84.b ** 2 / WGS84.a


def conservative_chord_radius(max_center_distance_m):
    distance = float(max_center_distance_m)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('search distance must be finite and nonnegative')
    angle = min(distance / MIN_WGS84_CURVATURE_RADIUS_M, np.pi)
    # A small outward numerical margin also protects exact-boundary pairs.
    return float(2 * np.sin(angle / 2) + 1e-12)
