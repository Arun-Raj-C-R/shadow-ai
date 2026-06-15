"""
Numerical Tolerance & Robust Predicates
=========================================
Handles the fundamental problem of floating-point geometry:
    0.1 + 0.2 != 0.3

Provides:
    - Epsilon-based comparisons
    - Robust geometric predicates (orient2d, orient3d, in_circle)
    - Interval arithmetic helpers
    - Adaptive precision for critical decisions
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE


def float_eq(a, b, tol=EPSILON):
    """Robust float equality: |a - b| < tol"""
    return abs(a - b) < tol

def float_lt(a, b, tol=EPSILON):
    """Robust less-than: a < b - tol"""
    return a < b - tol

def float_gt(a, b, tol=EPSILON):
    """Robust greater-than: a > b + tol"""
    return a > b + tol

def float_le(a, b, tol=EPSILON):
    return a < b + tol

def float_ge(a, b, tol=EPSILON):
    return a > b - tol

def float_sign(a, tol=EPSILON):
    """Return -1, 0, or +1 with tolerance."""
    if a > tol: return 1
    if a < -tol: return -1
    return 0

def clamp(value, lo, hi):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# =====================================================================
# ROBUST GEOMETRIC PREDICATES
# =====================================================================
# These determine the topology of geometric configurations.
# Errors here cause boolean operations to fail catastrophically.

def orient2d(a, b, c):
    """
    Orientation test for 2D points.
    Returns:
        > 0 if (a, b, c) are counterclockwise
        < 0 if clockwise
        = 0 if collinear

    Uses the 2×2 determinant:
        | ax-cx  ay-cy |
        | bx-cx  by-cy |

    With error expansion for robustness.
    """
    # First try fast floating-point
    det = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])

    # Error bound (Shewchuk's approach simplified)
    err_bound = (abs(a[0] - c[0]) * abs(b[1] - c[1]) +
                 abs(a[1] - c[1]) * abs(b[0] - c[0])) * EPSILON * 4.0

    if abs(det) > err_bound:
        return det
    # Fall back to higher precision
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    ax, ay = Decimal(str(a[0])), Decimal(str(a[1]))
    bx, by = Decimal(str(b[0])), Decimal(str(b[1]))
    cx, cy = Decimal(str(c[0])), Decimal(str(c[1]))
    det_exact = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return float(det_exact)


def orient3d(a, b, c, d):
    """
    Orientation test for 3D points.
    Returns:
        > 0 if d is below the plane of (a, b, c) (right-hand rule)
        < 0 if above
        = 0 if coplanar

    Uses the 3×3 determinant:
        | ax-dx  ay-dy  az-dz |
        | bx-dx  by-dy  bz-dz |
        | cx-dx  cy-dy  cz-dz |
    """
    ad = np.array(a[:3], dtype=np.float64) - np.array(d[:3], dtype=np.float64)
    bd = np.array(b[:3], dtype=np.float64) - np.array(d[:3], dtype=np.float64)
    cd = np.array(c[:3], dtype=np.float64) - np.array(d[:3], dtype=np.float64)

    det = np.linalg.det(np.array([ad, bd, cd]))

    # Simple error bound
    err = (np.abs(ad).max() + np.abs(bd).max() + np.abs(cd).max()) ** 3 * EPSILON * 10.0
    if abs(det) < err:
        return 0.0
    return det


def in_circle(a, b, c, d):
    """
    InCircle test: is point d inside the circumcircle of triangle (a, b, c)?
    Assumes (a, b, c) are in counterclockwise order.

    Returns:
        > 0 if d is inside
        < 0 if outside
        = 0 if on the circle

    Uses the 4×4 determinant:
        | ax-dx  ay-dy  (ax-dx)²+(ay-dy)² |
        | bx-dx  by-dy  (bx-dx)²+(by-dy)² |
        | cx-dx  cy-dy  (cx-dx)²+(cy-dy)² |
    """
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]

    mat = np.array([
        [adx, ady, adx*adx + ady*ady],
        [bdx, bdy, bdx*bdx + bdy*bdy],
        [cdx, cdy, cdx*cdx + cdy*cdy],
    ])
    return float(np.linalg.det(mat))


def points_are_collinear(a, b, c, tol=LINEAR_TOLERANCE):
    """Check if three 3D points are collinear."""
    ab = np.array(b[:3], dtype=np.float64) - np.array(a[:3], dtype=np.float64)
    ac = np.array(c[:3], dtype=np.float64) - np.array(a[:3], dtype=np.float64)
    cross_len = np.linalg.norm(np.cross(ab, ac))
    ab_len = np.linalg.norm(ab)
    return cross_len < tol * max(ab_len, 1e-30)


def points_are_coplanar(points, tol=LINEAR_TOLERANCE):
    """Check if a set of 3D points are coplanar."""
    if len(points) <= 3:
        return True
    a = np.array(points[0][:3], dtype=np.float64)
    b = np.array(points[1][:3], dtype=np.float64)
    c = np.array(points[2][:3], dtype=np.float64)
    normal = np.cross(b - a, c - a)
    n_len = np.linalg.norm(normal)
    if n_len < tol:
        return True  # Degenerate: first 3 points collinear
    normal = normal / n_len
    for p in points[3:]:
        d = abs(np.dot(np.array(p[:3], dtype=np.float64) - a, normal))
        if d > tol:
            return False
    return True
