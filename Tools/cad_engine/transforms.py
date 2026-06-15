"""
4×4 Transformation Matrices for CAD
======================================
Homogeneous coordinate transforms — translation, rotation, scaling, affine.

All transforms use 4×4 matrices operating on homogeneous coordinates [x,y,z,1]:

    [x']   [m00 m01 m02 tx] [x]
    [y'] = [m10 m11 m12 ty] [y]
    [z']   [m20 m21 m22 tz] [z]
    [1 ]   [ 0   0   0   1] [1]

Supports hierarchical transforms (local/world coordinate systems).
"""

import numpy as np
from .constants import EPSILON, DEG_TO_RAD
from .vector_math import Vec3, _to_array


class Transform:
    """4×4 homogeneous transformation matrix."""

    __slots__ = ('_matrix',)

    def __init__(self, matrix=None):
        if matrix is not None:
            self._matrix = np.array(matrix, dtype=np.float64).reshape(4, 4)
        else:
            self._matrix = np.eye(4, dtype=np.float64)

    @property
    def matrix(self): return self._matrix.copy()

    @property
    def inverse(self):
        return Transform(np.linalg.inv(self._matrix))

    # ── Factory Methods ───────────────────────────────────────────

    @staticmethod
    def identity():
        return Transform()

    @staticmethod
    def translation(dx, dy=None, dz=None):
        """T = [[I, t], [0, 1]]"""
        if isinstance(dx, (Vec3, list, tuple, np.ndarray)):
            v = _to_array(dx)
            dx, dy, dz = v[0], v[1], v[2]
        elif dy is None: dy = 0.0
        if dz is None: dz = 0.0
        m = np.eye(4, dtype=np.float64)
        m[0, 3], m[1, 3], m[2, 3] = float(dx), float(dy), float(dz)
        return Transform(m)

    @staticmethod
    def scaling(sx, sy=None, sz=None):
        if sy is None: sy = sx
        if sz is None: sz = sx
        m = np.eye(4, dtype=np.float64)
        m[0, 0], m[1, 1], m[2, 2] = float(sx), float(sy), float(sz)
        return Transform(m)

    @staticmethod
    def rotation_x(angle_rad):
        """Rotation about X axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        m = np.eye(4, dtype=np.float64)
        m[1, 1], m[1, 2] = c, -s
        m[2, 1], m[2, 2] = s, c
        return Transform(m)

    @staticmethod
    def rotation_y(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        m = np.eye(4, dtype=np.float64)
        m[0, 0], m[0, 2] = c, s
        m[2, 0], m[2, 2] = -s, c
        return Transform(m)

    @staticmethod
    def rotation_z(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        m = np.eye(4, dtype=np.float64)
        m[0, 0], m[0, 1] = c, -s
        m[1, 0], m[1, 1] = s, c
        return Transform(m)

    @staticmethod
    def rotation_axis(axis, angle_rad):
        """Rotation about arbitrary axis (Rodrigues → matrix form)."""
        k = _to_array(axis)
        k_len = np.linalg.norm(k)
        if k_len < EPSILON:
            return Transform.identity()
        k = k / k_len
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) * c + (1 - c) * np.outer(k, k) + s * K
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = R
        return Transform(m)

    @staticmethod
    def rotation_euler(rx, ry, rz, order='xyz'):
        """Euler angle rotation (radians)."""
        rotations = {'x': Transform.rotation_x, 'y': Transform.rotation_y, 'z': Transform.rotation_z}
        result = Transform.identity()
        for axis_char in order:
            angle = {'x': rx, 'y': ry, 'z': rz}[axis_char]
            result = result.compose(rotations[axis_char](angle))
        return result

    @staticmethod
    def look_at(eye, target, up=None):
        """View matrix looking from eye to target."""
        if up is None: up = Vec3(0, 1, 0)
        e, t, u = _to_array(eye), _to_array(target), _to_array(up)
        fwd = t - e; fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, u); right = right / np.linalg.norm(right)
        up_new = np.cross(right, fwd)
        m = np.eye(4, dtype=np.float64)
        m[0, :3], m[1, :3], m[2, :3] = right, up_new, -fwd
        m[0, 3], m[1, 3], m[2, 3] = -np.dot(right, e), -np.dot(up_new, e), np.dot(fwd, e)
        return Transform(m)

    @staticmethod
    def mirror_plane(plane_point, plane_normal):
        """Mirror/reflection across a plane."""
        n = _to_array(plane_normal)
        n = n / np.linalg.norm(n)
        p = _to_array(plane_point)
        d = -np.dot(n, p)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] -= 2.0 * np.outer(n, n)
        m[:3, 3] = -2.0 * d * n
        return Transform(m)

    # ── Operations ────────────────────────────────────────────────

    def apply_point(self, point):
        """Transform a point (w=1): P' = M · [x, y, z, 1]^T"""
        p = np.append(_to_array(point), 1.0)
        result = self._matrix @ p
        return Vec3(result[:3] / result[3] if abs(result[3]) > EPSILON else result[:3])

    def apply_vector(self, vector):
        """Transform a direction vector (w=0): V' = M · [x, y, z, 0]^T"""
        v = np.append(_to_array(vector), 0.0)
        result = self._matrix @ v
        return Vec3(result[:3])

    def apply_normal(self, normal):
        """Transform a normal vector: N' = (M^{-T}) · N"""
        inv_T = np.linalg.inv(self._matrix).T
        n = np.append(_to_array(normal), 0.0)
        result = inv_T @ n
        return Vec3(result[:3]).normalize()

    def apply_points(self, points):
        """Transform multiple points efficiently."""
        pts = np.array([_to_array(p) for p in points], dtype=np.float64)
        ones = np.ones((len(pts), 1))
        homo = np.hstack([pts, ones])
        result = (self._matrix @ homo.T).T
        return [Vec3(r[:3] / r[3] if abs(r[3]) > EPSILON else r[:3]) for r in result]

    def compose(self, other):
        """Compose transforms: self * other (self applied AFTER other)."""
        return Transform(self._matrix @ other._matrix)

    def then(self, other):
        """Chain: apply self first, then other."""
        return Transform(other._matrix @ self._matrix)

    def decompose(self):
        """Decompose into translation, rotation (3x3), and scale."""
        t = Vec3(self._matrix[0, 3], self._matrix[1, 3], self._matrix[2, 3])
        R = self._matrix[:3, :3].copy()
        sx = np.linalg.norm(R[:, 0])
        sy = np.linalg.norm(R[:, 1])
        sz = np.linalg.norm(R[:, 2])
        if sx > EPSILON: R[:, 0] /= sx
        if sy > EPSILON: R[:, 1] /= sy
        if sz > EPSILON: R[:, 2] /= sz
        return {'translation': t, 'rotation': R, 'scale': Vec3(sx, sy, sz)}

    def is_identity(self, tol=EPSILON):
        return np.allclose(self._matrix, np.eye(4), atol=tol)

    def determinant(self):
        return float(np.linalg.det(self._matrix))

    def __mul__(self, other):
        if isinstance(other, Transform):
            return self.compose(other)
        if isinstance(other, Vec3):
            return self.apply_point(other)
        raise TypeError(f"Cannot multiply Transform by {type(other)}")

    def __repr__(self):
        return f"Transform(\n{np.array2string(self._matrix, precision=6)}\n)"


class CoordinateSystem:
    """Local coordinate system defined by origin + 3 orthonormal axes."""

    def __init__(self, origin=None, x_axis=None, y_axis=None, z_axis=None):
        self.origin = Vec3(origin) if origin else Vec3(0, 0, 0)
        self.x_axis = Vec3(x_axis).normalize() if x_axis else Vec3(1, 0, 0)
        self.z_axis = Vec3(z_axis).normalize() if z_axis else Vec3(0, 0, 1)
        # Ensure orthogonality
        self.y_axis = self.z_axis.cross(self.x_axis).normalize()
        self.z_axis = self.x_axis.cross(self.y_axis).normalize()

    def to_world_transform(self):
        """Transform from local coordinates to world coordinates."""
        m = np.eye(4, dtype=np.float64)
        m[:3, 0] = self.x_axis._data
        m[:3, 1] = self.y_axis._data
        m[:3, 2] = self.z_axis._data
        m[:3, 3] = self.origin._data
        return Transform(m)

    def to_local_transform(self):
        """Transform from world coordinates to local coordinates."""
        return self.to_world_transform().inverse

    def local_to_world(self, point):
        return self.to_world_transform().apply_point(point)

    def world_to_local(self, point):
        return self.to_local_transform().apply_point(point)
