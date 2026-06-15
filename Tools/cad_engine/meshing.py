"""
Meshing Engine
================
Surface tessellation and volumetric meshing for rendering, FEM, and CFD.

Implements:
    - Surface triangulation (fan, ear-clipping)
    - Adaptive mesh refinement (midpoint subdivision)
    - Laplacian mesh smoothing
    - Mesh quality metrics (aspect ratio, skewness)
    - Basic tetrahedral meshing (Delaunay-based)
    - STL-compatible mesh export
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE
from .vector_math import Vec3, _to_array


class TriMesh:
    """Triangle mesh data structure."""

    def __init__(self, vertices=None, triangles=None, normals=None):
        self.vertices = vertices or []  # List of Vec3
        self.triangles = triangles or []  # List of (i, j, k)
        self.normals = normals or []  # Per-vertex or per-face normals

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.triangles)

    def compute_face_normals(self):
        """Compute per-face normals."""
        self.face_normals = []
        for tri in self.triangles:
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            n = e1.cross(e2).normalize()
            self.face_normals.append(n)
        return self.face_normals

    def compute_vertex_normals(self):
        """Compute per-vertex normals by averaging adjacent face normals."""
        if not hasattr(self, 'face_normals') or not self.face_normals:
            self.compute_face_normals()

        self.normals = [Vec3(0, 0, 0) for _ in self.vertices]
        counts = [0] * len(self.vertices)

        for fi, tri in enumerate(self.triangles):
            for vi in tri:
                self.normals[vi] = self.normals[vi] + self.face_normals[fi]
                counts[vi] += 1

        for i in range(len(self.normals)):
            if counts[i] > 0:
                self.normals[i] = self.normals[i].normalize()

        return self.normals

    def surface_area(self):
        """Total surface area."""
        area = 0.0
        for tri in self.triangles:
            v0, v1, v2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            area += e1.cross(e2).length() * 0.5
        return area

    def volume(self):
        """Volume (for closed meshes) using divergence theorem."""
        vol = 0.0
        for tri in self.triangles:
            p0 = self.vertices[tri[0]].array
            p1 = self.vertices[tri[1]].array
            p2 = self.vertices[tri[2]].array
            vol += np.dot(p0, np.cross(p1, p2)) / 6.0
        return abs(vol)

    def bounding_box(self):
        if not self.vertices:
            return Vec3(0, 0, 0), Vec3(0, 0, 0)
        pts = np.array([v.array for v in self.vertices])
        return Vec3(pts.min(axis=0)), Vec3(pts.max(axis=0))

    def centroid(self):
        if not self.vertices:
            return Vec3(0, 0, 0)
        pts = np.array([v.array for v in self.vertices])
        return Vec3(pts.mean(axis=0))

    def get_mesh_arrays(self):
        """Export as flat arrays for rendering."""
        x = [v.x for v in self.vertices]
        y = [v.y for v in self.vertices]
        z = [v.z for v in self.vertices]
        i_list = [t[0] for t in self.triangles]
        j_list = [t[1] for t in self.triangles]
        k_list = [t[2] for t in self.triangles]
        intensity = [v.z for v in self.vertices]
        return {'x': x, 'y': y, 'z': z, 'i': i_list, 'j': j_list, 'k': k_list,
                'intensity': intensity}

    def __repr__(self):
        return f"TriMesh(V={self.num_vertices}, F={self.num_faces})"


# =====================================================================
# MESH OPERATIONS
# =====================================================================

def subdivide_mesh(mesh, iterations=1):
    """
    Loop subdivision — split each triangle into 4 by adding midpoints.
    Each iteration quadruples the face count.
    """
    for _ in range(iterations):
        new_verts = list(mesh.vertices)
        new_tris = []
        edge_midpoints = {}

        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (new_verts[i] + new_verts[j]) / 2.0
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
            return idx

        for tri in mesh.triangles:
            i, j, k = tri
            ij = get_midpoint(i, j)
            jk = get_midpoint(j, k)
            ki = get_midpoint(k, i)

            new_tris.append((i, ij, ki))
            new_tris.append((j, jk, ij))
            new_tris.append((k, ki, jk))
            new_tris.append((ij, jk, ki))

        mesh = TriMesh(new_verts, new_tris)

    mesh.compute_vertex_normals()
    return mesh


def smooth_mesh(mesh, iterations=3, factor=0.5):
    """
    Laplacian smoothing — move each vertex toward the average of its neighbors.

    v_new = v + λ * (v_avg_neighbors - v)
    """
    # Build adjacency
    n = len(mesh.vertices)
    neighbors = [set() for _ in range(n)]
    for tri in mesh.triangles:
        for a in range(3):
            for b in range(3):
                if a != b:
                    neighbors[tri[a]].add(tri[b])

    for _ in range(iterations):
        new_verts = [v.copy() for v in mesh.vertices]
        for i in range(n):
            if not neighbors[i]:
                continue
            avg = Vec3(0, 0, 0)
            for ni in neighbors[i]:
                avg = avg + mesh.vertices[ni]
            avg = avg / len(neighbors[i])
            new_verts[i] = mesh.vertices[i] + (avg - mesh.vertices[i]) * factor
        mesh.vertices = new_verts

    mesh.compute_vertex_normals()
    return mesh


def mesh_quality(mesh):
    """
    Compute mesh quality metrics for each triangle.
    Returns dict with aspect_ratios, min_angles, skewness.
    """
    aspect_ratios = []
    min_angles = []

    for tri in mesh.triangles:
        v0, v1, v2 = mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]
        edges = [v0.distance_to(v1), v1.distance_to(v2), v2.distance_to(v0)]
        longest = max(edges)
        shortest = min(edges)
        aspect_ratios.append(longest / max(shortest, EPSILON))

        # Angles
        e01 = (v1 - v0).normalize()
        e02 = (v2 - v0).normalize()
        angle0 = e01.angle_to(e02)

        e10 = (v0 - v1).normalize()
        e12 = (v2 - v1).normalize()
        angle1 = e10.angle_to(e12)

        angle2 = np.pi - angle0 - angle1
        min_angles.append(min(angle0, angle1, angle2))

    return {
        'num_vertices': mesh.num_vertices,
        'num_faces': mesh.num_faces,
        'aspect_ratio_avg': float(np.mean(aspect_ratios)),
        'aspect_ratio_max': float(np.max(aspect_ratios)),
        'min_angle_avg_deg': float(np.degrees(np.mean(min_angles))),
        'min_angle_min_deg': float(np.degrees(np.min(min_angles))),
        'surface_area': mesh.surface_area(),
    }


def solid_to_trimesh(solid):
    """Convert a B-Rep Solid to a TriMesh."""
    mesh_data = solid.get_mesh_data()
    vertices = [Vec3(mesh_data['x'][i], mesh_data['y'][i], mesh_data['z'][i])
                for i in range(mesh_data['num_vertices'])]
    triangles = list(zip(mesh_data['i'], mesh_data['j'], mesh_data['k']))
    mesh = TriMesh(vertices, triangles)
    mesh.compute_vertex_normals()
    return mesh


def adaptive_subdivide(mesh, max_edge_length):
    """Subdivide triangles with edges longer than max_edge_length."""
    changed = True
    while changed:
        changed = False
        new_tris = []
        new_verts = list(mesh.vertices)
        edge_mids = {}

        for tri in mesh.triangles:
            i, j, k = tri
            edges = [(i, j), (j, k), (k, i)]
            lengths = [new_verts[a].distance_to(new_verts[b]) for a, b in edges]

            if max(lengths) > max_edge_length:
                changed = True
                # Split longest edge
                longest_idx = np.argmax(lengths)
                a, b = edges[longest_idx]
                c = [k, i, j][longest_idx]  # Opposite vertex

                key = (min(a, b), max(a, b))
                if key not in edge_mids:
                    mid = (new_verts[a] + new_verts[b]) / 2.0
                    edge_mids[key] = len(new_verts)
                    new_verts.append(mid)
                m = edge_mids[key]

                new_tris.append((a, m, c))
                new_tris.append((m, b, c))
            else:
                new_tris.append(tri)

        mesh = TriMesh(new_verts, new_tris)

    mesh.compute_vertex_normals()
    return mesh
