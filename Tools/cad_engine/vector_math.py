"""
Vector Mathematics for CAD
============================
Core 3D vector operations — the foundation of all CAD geometry.

Implements:
    - Vec3 class with operator overloading
    - Dot/cross product, normalization, projection, reflection
    - Rotation (axis-angle via Rodrigues' formula)
    - Angle between vectors, distance computations
    - Plane operations (signed distance, projection)
    - Line-line closest points
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE, ANGULAR_TOLERANCE


class Vec3:
    """3D Vector with operator overloading. Stores numpy array internally."""

    __slots__ = ('_data',)

    def __init__(self, x=0.0, y=0.0, z=0.0):
        if isinstance(x, np.ndarray):
            self._data = x.astype(np.float64).ravel()[:3].copy()
        elif isinstance(x, (list, tuple)):
            self._data = np.array(x[:3], dtype=np.float64)
        elif isinstance(x, Vec3):
            self._data = x._data.copy()
        else:
            self._data = np.array([float(x), float(y), float(z)], dtype=np.float64)

    @property
    def x(self): return self._data[0]
    @x.setter
    def x(self, val): self._data[0] = float(val)
    @property
    def y(self): return self._data[1]
    @y.setter
    def y(self, val): self._data[1] = float(val)
    @property
    def z(self): return self._data[2]
    @z.setter
    def z(self, val): self._data[2] = float(val)
    @property
    def array(self): return self._data.copy()

    def dot(self, other):
        """a·b = ax*bx + ay*by + az*bz"""
        return float(np.dot(self._data, _to_array(other)))

    def cross(self, other):
        """a×b = (ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx)"""
        return Vec3(np.cross(self._data, _to_array(other)))

    def length(self):
        return float(np.linalg.norm(self._data))

    def length_sq(self):
        return float(np.dot(self._data, self._data))

    def normalize(self):
        L = self.length()
        return Vec3(0, 0, 0) if L < EPSILON else Vec3(self._data / L)

    def normalized(self):
        return self.normalize()

    def distance_to(self, other):
        return float(np.linalg.norm(self._data - _to_array(other)))

    def distance_sq_to(self, other):
        d = self._data - _to_array(other)
        return float(np.dot(d, d))

    def project_onto(self, other):
        b = _to_array(other)
        bb = np.dot(b, b)
        return Vec3(0, 0, 0) if bb < EPSILON else Vec3(b * (np.dot(self._data, b) / bb))

    def reject_from(self, other):
        return self - self.project_onto(other)

    def reflect(self, normal):
        n = _to_array(normal)
        n = n / np.linalg.norm(n)
        return Vec3(self._data - 2.0 * np.dot(self._data, n) * n)

    def rotate_axis_angle(self, axis, angle_rad):
        """Rodrigues: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))"""
        k = _to_array(axis)
        k_norm = np.linalg.norm(k)
        if k_norm < EPSILON:
            return Vec3(self._data.copy())
        k = k / k_norm
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return Vec3(self._data * c + np.cross(k, self._data) * s + k * np.dot(k, self._data) * (1 - c))

    def rotate_x(self, a):
        c, s = np.cos(a), np.sin(a)
        return Vec3(self.x, self.y*c - self.z*s, self.y*s + self.z*c)

    def rotate_y(self, a):
        c, s = np.cos(a), np.sin(a)
        return Vec3(self.x*c + self.z*s, self.y, -self.x*s + self.z*c)

    def rotate_z(self, a):
        c, s = np.cos(a), np.sin(a)
        return Vec3(self.x*c - self.y*s, self.x*s + self.y*c, self.z)

    def angle_to(self, other):
        b = _to_array(other)
        return float(np.arctan2(np.linalg.norm(np.cross(self._data, b)), np.dot(self._data, b)))

    def signed_angle_to(self, other, ref_normal):
        angle = self.angle_to(other)
        if self.cross(other).dot(ref_normal) < 0:
            angle = -angle
        return angle

    def is_zero(self, tol=LINEAR_TOLERANCE): return self.length_sq() < tol * tol
    def is_parallel_to(self, o, tol=ANGULAR_TOLERANCE):
        return self.cross(o).length() < tol * max(self.length(), 1e-30) * max(Vec3(o).length(), 1e-30)
    def is_perpendicular_to(self, o, tol=ANGULAR_TOLERANCE):
        return abs(self.dot(o)) < tol * max(self.length(), 1e-30) * max(Vec3(o).length(), 1e-30)
    def is_equal_to(self, other, tol=LINEAR_TOLERANCE): return self.distance_to(other) < tol

    def __add__(self, o): return Vec3(self._data + _to_array(o))
    def __radd__(self, o): return Vec3(_to_array(o) + self._data)
    def __sub__(self, o): return Vec3(self._data - _to_array(o))
    def __rsub__(self, o): return Vec3(_to_array(o) - self._data)
    def __mul__(self, s): return Vec3(self._data * float(s))
    def __rmul__(self, s): return Vec3(self._data * float(s))
    def __truediv__(self, s): return Vec3(self._data / float(s))
    def __neg__(self): return Vec3(-self._data)
    def __eq__(self, o): return self.is_equal_to(o)
    def __repr__(self): return f"Vec3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"
    def __iter__(self): return iter(self._data)
    def __getitem__(self, i): return self._data[i]
    def __setitem__(self, i, v): self._data[i] = float(v)
    def __len__(self): return 3
    def copy(self): return Vec3(self._data.copy())
    def to_list(self): return [float(self.x), float(self.y), float(self.z)]
    def to_tuple(self): return (float(self.x), float(self.y), float(self.z))


def _to_array(v):
    if isinstance(v, Vec3): return v._data
    if isinstance(v, np.ndarray): return v.astype(np.float64)
    return np.array(v, dtype=np.float64)

def dot(a, b): return float(np.dot(_to_array(a), _to_array(b)))
def cross(a, b): return Vec3(np.cross(_to_array(a), _to_array(b)))
def normalize(v):
    arr = _to_array(v); L = np.linalg.norm(arr)
    return Vec3(0, 0, 0) if L < EPSILON else Vec3(arr / L)
def distance(a, b): return float(np.linalg.norm(_to_array(a) - _to_array(b)))
def lerp(a, b, t): return Vec3(_to_array(a) + t * (_to_array(b) - _to_array(a)))
def midpoint(a, b): return lerp(a, b, 0.5)
def triple_product(a, b, c):
    return float(np.dot(_to_array(a), np.cross(_to_array(b), _to_array(c))))

def signed_distance_to_plane(point, plane_point, plane_normal):
    n = _to_array(plane_normal); n_len = np.linalg.norm(n)
    if n_len < EPSILON: return 0.0
    return float(np.dot(_to_array(point) - _to_array(plane_point), n / n_len))

def project_point_to_plane(point, plane_point, plane_normal):
    d = signed_distance_to_plane(point, plane_point, plane_normal)
    n = normalize(plane_normal)
    return Vec3(_to_array(point) - d * n._data)

def closest_point_on_line(point, line_origin, line_dir):
    o, d = _to_array(line_origin), _to_array(line_dir)
    d_len = np.linalg.norm(d)
    if d_len < EPSILON: return Vec3(o)
    d = d / d_len
    return Vec3(o + np.dot(_to_array(point) - o, d) * d)

def closest_point_on_segment(point, seg_start, seg_end):
    a, b = _to_array(seg_start), _to_array(seg_end)
    ab = b - a; ab_sq = np.dot(ab, ab)
    if ab_sq < EPSILON: return Vec3(a)
    t = max(0.0, min(1.0, np.dot(_to_array(point) - a, ab) / ab_sq))
    return Vec3(a + t * ab)

def line_line_closest_points(p1, d1, p2, d2):
    p1a, d1a = _to_array(p1), _to_array(d1)
    p2a, d2a = _to_array(p2), _to_array(d2)
    w0 = p1a - p2a
    a, b, c = np.dot(d1a, d1a), np.dot(d1a, d2a), np.dot(d2a, d2a)
    d, e = np.dot(d1a, w0), np.dot(d2a, w0)
    denom = a * c - b * b
    if abs(denom) < EPSILON:
        t, s = 0.0, (d / b if abs(b) > EPSILON else 0.0)
    else:
        t, s = (b * e - c * d) / denom, (a * e - b * d) / denom
    cp1, cp2 = Vec3(p1a + t * d1a), Vec3(p2a + s * d2a)
    return cp1, cp2, cp1.distance_to(cp2)
