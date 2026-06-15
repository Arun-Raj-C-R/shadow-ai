"""
CSG Boolean Operations via BSP Trees
=======================================
Union, subtract, intersect — the core of constructive solid geometry.
Boolean geometry is notoriously horrible. This is the hard part.

Uses BSP (Binary Space Partition) trees for robust boolean operations.

Algorithm:
    1. Convert solid meshes to polygon lists
    2. Build BSP tree for each solid
    3. Clip polygons of A against BSP of B (and vice versa)
    4. Combine/invert clipped polygons based on operation
    5. Rebuild solid from resulting polygons
"""

import numpy as np
from .constants import EPSILON, BOOLEAN_TOLERANCE
from .vector_math import Vec3, _to_array


class BSPPolygon:
    """A convex polygon with vertices, normal, and plane constant."""

    def __init__(self, vertices, shared=None):
        self.vertices = [Vec3(v) for v in vertices]
        self.shared = shared  # Material/face reference
        self._compute_plane()

    def _compute_plane(self):
        if len(self.vertices) >= 3:
            a = self.vertices[1] - self.vertices[0]
            b = self.vertices[2] - self.vertices[0]
            self.normal = a.cross(b).normalize()
            self.w = self.normal.dot(self.vertices[0])
        else:
            self.normal = Vec3(0, 0, 1)
            self.w = 0.0

    def clone(self):
        return BSPPolygon([v.copy() for v in self.vertices], self.shared)

    def flip(self):
        """Reverse orientation."""
        self.vertices.reverse()
        self.normal = -self.normal
        self.w = -self.w


class BSPNode:
    """
    BSP tree node.

    Each node has a dividing plane (from one polygon).
    Polygons are classified as:
        - coplanar_front / coplanar_back
        - front (positive side)
        - back (negative side)
        - spanning (split across the plane)
    """

    COPLANAR = 0
    FRONT = 1
    BACK = 2
    SPANNING = 3

    def __init__(self, polygons=None):
        self.plane_normal = None
        self.plane_w = None
        self.front = None
        self.back = None
        self.polygons = []

        if polygons:
            self.build(polygons)

    def clone(self):
        node = BSPNode()
        if self.plane_normal:
            node.plane_normal = self.plane_normal.copy()
            node.plane_w = self.plane_w
        node.front = self.front.clone() if self.front else None
        node.back = self.back.clone() if self.back else None
        node.polygons = [p.clone() for p in self.polygons]
        return node

    def invert(self):
        """Flip all polygons and swap front/back subtrees."""
        for poly in self.polygons:
            poly.flip()
        if self.plane_normal:
            self.plane_normal = -self.plane_normal
            self.plane_w = -self.plane_w
        if self.front: self.front.invert()
        if self.back: self.back.invert()
        self.front, self.back = self.back, self.front

    def _classify_vertex(self, vertex):
        t = self.plane_normal.dot(vertex) - self.plane_w
        if t < -BOOLEAN_TOLERANCE:
            return self.BACK
        elif t > BOOLEAN_TOLERANCE:
            return self.FRONT
        return self.COPLANAR

    def _split_polygon(self, polygon):
        """Classify and optionally split a polygon against this node's plane."""
        types = []
        polygon_type = 0
        for v in polygon.vertices:
            t = self._classify_vertex(v)
            polygon_type |= t
            types.append(t)

        coplanar_front = []
        coplanar_back = []
        front = []
        back = []

        if polygon_type == self.COPLANAR:
            if self.plane_normal.dot(polygon.normal) > 0:
                coplanar_front.append(polygon)
            else:
                coplanar_back.append(polygon)

        elif polygon_type == self.FRONT:
            front.append(polygon)

        elif polygon_type == self.BACK:
            back.append(polygon)

        else:  # SPANNING — need to split
            f_verts = []
            b_verts = []
            n = len(polygon.vertices)
            for i in range(n):
                j = (i + 1) % n
                ti = types[i]
                tj = types[j]
                vi = polygon.vertices[i]
                vj = polygon.vertices[j]

                if ti != self.BACK:
                    f_verts.append(vi.copy())
                if ti != self.FRONT:
                    b_verts.append(vi.copy())

                if (ti | tj) == self.SPANNING:
                    # Compute intersection point
                    t_val = (self.plane_w - self.plane_normal.dot(vi))
                    denom = self.plane_normal.dot(vj - vi)
                    if abs(denom) > EPSILON:
                        t_val /= denom
                        t_val = max(0.0, min(1.0, t_val))
                    else:
                        t_val = 0.5
                    intersection = vi + (vj - vi) * t_val
                    f_verts.append(intersection.copy())
                    b_verts.append(intersection.copy())

            if len(f_verts) >= 3:
                front.append(BSPPolygon(f_verts, polygon.shared))
            if len(b_verts) >= 3:
                back.append(BSPPolygon(b_verts, polygon.shared))

        return coplanar_front, coplanar_back, front, back

    def clip_polygons(self, polygons):
        """Recursively clip polygons against this BSP tree."""
        if not self.plane_normal:
            return list(polygons)

        front_list = []
        back_list = []

        for poly in polygons:
            cf, cb, f, b = self._split_polygon(poly)
            front_list.extend(cf)
            front_list.extend(f)
            back_list.extend(cb)
            back_list.extend(b)

        front_list = self.front.clip_polygons(front_list) if self.front else front_list
        back_list = self.back.clip_polygons(back_list) if self.back else []

        return front_list + back_list

    def clip_to(self, bsp):
        """Clip this tree's polygons using another BSP tree."""
        self.polygons = bsp.clip_polygons(self.polygons)
        if self.front: self.front.clip_to(bsp)
        if self.back: self.back.clip_to(bsp)

    def all_polygons(self):
        """Collect all polygons from the tree."""
        result = list(self.polygons)
        if self.front: result.extend(self.front.all_polygons())
        if self.back: result.extend(self.back.all_polygons())
        return result

    def build(self, polygons):
        """Build BSP tree from polygon list."""
        if not polygons:
            return

        if not self.plane_normal:
            self.plane_normal = polygons[0].normal.copy()
            self.plane_w = polygons[0].w

        front_list = []
        back_list = []

        for poly in polygons:
            cf, cb, f, b = self._split_polygon(poly)
            self.polygons.extend(cf)
            self.polygons.extend(cb)
            front_list.extend(f)
            back_list.extend(b)

        if front_list:
            if not self.front:
                self.front = BSPNode()
            self.front.build(front_list)

        if back_list:
            if not self.back:
                self.back = BSPNode()
            self.back.build(back_list)


# =====================================================================
# BOOLEAN OPERATIONS
# =====================================================================

def _solid_to_polygons(solid):
    """Convert a B-Rep solid to BSP polygons."""
    polygons = []
    for face in solid.all_faces():
        tris = face.triangulate()
        for v0, v1, v2 in tris:
            poly = BSPPolygon([v0.point, v1.point, v2.point], shared=face.id)
            polygons.append(poly)
    return polygons


def _polygons_to_mesh(polygons):
    """Convert BSP polygons back to mesh arrays."""
    vertices = []
    vert_map = {}
    triangles = []

    for poly in polygons:
        # Fan triangulate the polygon
        if len(poly.vertices) < 3:
            continue
        for i in range(1, len(poly.vertices) - 1):
            for v in [poly.vertices[0], poly.vertices[i], poly.vertices[i + 1]]:
                key = (round(v.x, 6), round(v.y, 6), round(v.z, 6))
                if key not in vert_map:
                    vert_map[key] = len(vertices)
                    vertices.append(v)
            tri = (
                vert_map[(round(poly.vertices[0].x, 6), round(poly.vertices[0].y, 6), round(poly.vertices[0].z, 6))],
                vert_map[(round(poly.vertices[i].x, 6), round(poly.vertices[i].y, 6), round(poly.vertices[i].z, 6))],
                vert_map[(round(poly.vertices[i+1].x, 6), round(poly.vertices[i+1].y, 6), round(poly.vertices[i+1].z, 6))],
            )
            triangles.append(tri)

    return vertices, triangles


def boolean_union(solid_a, solid_b):
    """
    A ∪ B: Combine two solids.

    Algorithm:
        a = BSP(A), b = BSP(B)
        a.clip_to(b)        — remove parts of A inside B
        b.clip_to(a)        — remove parts of B inside A
        b.invert()           — invert B
        b.clip_to(a)         — remove coplanar faces
        b.invert()
        result = a + b       — combine remaining polygons
    """
    polys_a = _solid_to_polygons(solid_a)
    polys_b = _solid_to_polygons(solid_b)

    a = BSPNode(polys_a)
    b = BSPNode(polys_b)

    a.clip_to(b)
    b.clip_to(a)
    b.invert()
    b.clip_to(a)
    b.invert()

    result_polys = a.all_polygons() + b.all_polygons()
    return result_polys


def boolean_subtract(solid_a, solid_b):
    """
    A - B: Remove B from A.

    Algorithm:
        a = BSP(A), b = BSP(B)
        a.invert()
        a.clip_to(b)
        b.clip_to(a)
        b.invert()
        b.clip_to(a)
        b.invert()
        a.build(b.all_polygons())
        a.invert()
    """
    polys_a = _solid_to_polygons(solid_a)
    polys_b = _solid_to_polygons(solid_b)

    a = BSPNode(polys_a)
    b = BSPNode(polys_b)

    a.invert()
    a.clip_to(b)
    b.clip_to(a)
    b.invert()
    b.clip_to(a)
    b.invert()
    a.build(b.all_polygons())
    a.invert()

    return a.all_polygons()


def boolean_intersect(solid_a, solid_b):
    """
    A ∩ B: Keep only the intersection.

    Algorithm:
        Intersect = complement of (complement(A) ∪ complement(B))
    """
    polys_a = _solid_to_polygons(solid_a)
    polys_b = _solid_to_polygons(solid_b)

    a = BSPNode(polys_a)
    b = BSPNode(polys_b)

    a.invert()
    b.clip_to(a)
    b.invert()
    a.clip_to(b)
    b.clip_to(a)
    a.build(b.all_polygons())
    a.invert()

    return a.all_polygons()
