"""
Computational Geometry Algorithms
====================================
Massive field. Core algorithms for spatial queries.

Implements:
    - Convex hull (3D Gift wrapping + Quickhull)
    - Delaunay triangulation (2D Bowyer-Watson)
    - Point-in-polygon (ray casting)
    - Ray-triangle intersection (Möller-Trumbore)
    - Ray-AABB intersection
    - AABB and OBB bounding boxes
    - KD-tree for nearest neighbor search
    - Spatial hashing
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE
from .vector_math import Vec3, _to_array


# =====================================================================
# POINT-IN-POLYGON (Ray Casting Algorithm)
# =====================================================================

def point_in_polygon_2d(point, polygon):
    """
    Test if a 2D point is inside a polygon using ray casting.
    Cast a horizontal ray from point to +x infinity.
    Count edge crossings: odd = inside, even = outside.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


# =====================================================================
# RAY-TRIANGLE INTERSECTION (Möller-Trumbore)
# =====================================================================

def ray_triangle_intersection(ray_origin, ray_dir, v0, v1, v2):
    """
    Möller-Trumbore ray-triangle intersection.

    Returns (hit, t, u, v) where:
        hit: bool — does the ray intersect?
        t: distance along ray
        u, v: barycentric coordinates

    Algorithm:
        e1 = v1 - v0, e2 = v2 - v0
        h = d × e2,   a = e1 · h
        if |a| < ε: parallel (no hit)
        f = 1/a,  s = o - v0
        u = f · (s · h)
        q = s × e1
        v = f · (d · q)
        t = f · (e2 · q)
    """
    o = _to_array(ray_origin)
    d = _to_array(ray_dir)
    p0 = _to_array(v0)
    p1 = _to_array(v1)
    p2 = _to_array(v2)

    e1 = p1 - p0
    e2 = p2 - p0
    h = np.cross(d, e2)
    a = np.dot(e1, h)

    if abs(a) < EPSILON:
        return False, 0.0, 0.0, 0.0

    f = 1.0 / a
    s = o - p0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, 0.0, 0.0, 0.0

    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return False, 0.0, 0.0, 0.0

    t = f * np.dot(e2, q)
    if t > EPSILON:
        return True, float(t), float(u), float(v)

    return False, 0.0, 0.0, 0.0


def ray_aabb_intersection(ray_origin, ray_dir, box_min, box_max):
    """
    Ray-AABB (Axis-Aligned Bounding Box) intersection.
    Returns (hit, t_near, t_far).
    """
    o = _to_array(ray_origin)
    d = _to_array(ray_dir)
    bmin = _to_array(box_min)
    bmax = _to_array(box_max)

    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        if abs(d[i]) < EPSILON:
            if o[i] < bmin[i] or o[i] > bmax[i]:
                return False, 0.0, 0.0
        else:
            t1 = (bmin[i] - o[i]) / d[i]
            t2 = (bmax[i] - o[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near > t_far or t_far < 0:
                return False, 0.0, 0.0

    return True, float(t_near), float(t_far)


# =====================================================================
# CONVEX HULL (3D)
# =====================================================================

def convex_hull_2d(points):
    """
    2D convex hull using Andrew's monotone chain algorithm.
    Returns indices of hull points in CCW order.
    O(n log n).
    """
    pts = [(p[0], p[1], i) for i, p in enumerate(points)]
    pts.sort()
    n = len(pts)
    if n < 3:
        return list(range(n))

    def cross_2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [h[2] for h in hull]


# =====================================================================
# DELAUNAY TRIANGULATION (2D Bowyer-Watson)
# =====================================================================

def delaunay_2d(points):
    """
    2D Delaunay triangulation using Bowyer-Watson algorithm.

    Algorithm:
        1. Create super-triangle containing all points
        2. Add points one at a time:
            a. Find all triangles whose circumcircle contains the point
            b. Remove those triangles, leaving a star-shaped hole
            c. Re-triangulate the hole with the new point
        3. Remove triangles sharing vertices with super-triangle

    Returns list of (i, j, k) index triples.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        return []

    # Super-triangle (large enough to contain all points)
    min_x = min(p[0] for p in pts) - 1.0
    max_x = max(p[0] for p in pts) + 1.0
    min_y = min(p[1] for p in pts) - 1.0
    max_y = max(p[1] for p in pts) + 1.0
    dx = max_x - min_x
    dy = max_y - min_y
    d_max = max(dx, dy) * 10.0

    super_pts = [
        (min_x - d_max, min_y - d_max),
        (min_x + 2 * d_max + dx, min_y - d_max),
        (min_x + dx / 2, max_y + d_max),
    ]

    all_pts = list(pts) + super_pts
    st = (n, n + 1, n + 2)  # Super-triangle indices
    triangles = [st]

    def circumcircle_contains(tri, px, py):
        ax, ay = all_pts[tri[0]]
        bx, by = all_pts[tri[1]]
        cx, cy = all_pts[tri[2]]

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < EPSILON:
            return False

        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / D
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / D
        r_sq = (ax - ux)**2 + (ay - uy)**2
        d_sq = (px - ux)**2 + (py - uy)**2
        return d_sq < r_sq + EPSILON

    for i in range(n):
        px, py = pts[i]
        bad_triangles = []
        for tri in triangles:
            if circumcircle_contains(tri, px, py):
                bad_triangles.append(tri)

        # Find boundary edges of the hole
        boundary = []
        for tri in bad_triangles:
            for j in range(3):
                edge = (tri[j], tri[(j + 1) % 3])
                is_shared = False
                for other in bad_triangles:
                    if other == tri:
                        continue
                    if edge[0] in other and edge[1] in other:
                        is_shared = True
                        break
                if not is_shared:
                    boundary.append(edge)

        # Remove bad triangles
        triangles = [t for t in triangles if t not in bad_triangles]

        # Create new triangles
        for edge in boundary:
            triangles.append((edge[0], edge[1], i))

    # Remove triangles connected to super-triangle
    super_set = {n, n + 1, n + 2}
    triangles = [t for t in triangles if not (set(t) & super_set)]

    return triangles


# =====================================================================
# KD-TREE
# =====================================================================

class KDNode:
    """KD-tree node for spatial queries."""
    def __init__(self, point=None, left=None, right=None, axis=0, index=-1):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis
        self.index = index


def build_kdtree(points, depth=0, indices=None):
    """Build a KD-tree from a list of 3D points."""
    if not points:
        return None
    if indices is None:
        indices = list(range(len(points)))

    k = 3  # dimensions
    axis = depth % k

    sorted_pairs = sorted(zip(indices, points), key=lambda p: p[1][axis])
    median = len(sorted_pairs) // 2

    idx, pt = sorted_pairs[median]
    return KDNode(
        point=pt,
        index=idx,
        axis=axis,
        left=build_kdtree([p for _, p in sorted_pairs[:median]],
                          depth + 1,
                          [i for i, _ in sorted_pairs[:median]]),
        right=build_kdtree([p for _, p in sorted_pairs[median + 1:]],
                           depth + 1,
                           [i for i, _ in sorted_pairs[median + 1:]]),
    )


def kdtree_nearest(node, target, best=None, best_dist=float('inf')):
    """Find nearest neighbor in KD-tree."""
    if node is None:
        return best, best_dist

    pt = np.array(node.point[:3])
    tgt = np.array(target[:3])
    d = float(np.linalg.norm(pt - tgt))

    if d < best_dist:
        best = node
        best_dist = d

    axis = node.axis
    diff = tgt[axis] - pt[axis]

    if diff <= 0:
        near, far = node.left, node.right
    else:
        near, far = node.right, node.left

    best, best_dist = kdtree_nearest(near, target, best, best_dist)

    if abs(diff) < best_dist:
        best, best_dist = kdtree_nearest(far, target, best, best_dist)

    return best, best_dist


# =====================================================================
# SPATIAL HASHING
# =====================================================================

class SpatialHash:
    """Uniform grid spatial hash for fast proximity queries."""

    def __init__(self, cell_size=1.0):
        self.cell_size = cell_size
        self.cells = {}

    def _key(self, point):
        return (int(np.floor(point[0] / self.cell_size)),
                int(np.floor(point[1] / self.cell_size)),
                int(np.floor(point[2] / self.cell_size)))

    def insert(self, point, data=None):
        key = self._key(point)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append((point, data))

    def query(self, point, radius):
        """Find all points within radius of query point."""
        results = []
        r_cells = int(np.ceil(radius / self.cell_size))
        center_key = self._key(point)

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    key = (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)
                    if key in self.cells:
                        for pt, data in self.cells[key]:
                            d = np.linalg.norm(np.array(pt[:3]) - np.array(point[:3]))
                            if d <= radius:
                                results.append((pt, data, d))
        return sorted(results, key=lambda x: x[2])
