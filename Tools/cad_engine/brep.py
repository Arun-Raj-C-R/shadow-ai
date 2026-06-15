"""
B-Rep Solid Modeling Kernel
=============================
The BIG one. Boundary Representation — the industry standard for solid modeling.

Implements a half-edge data structure:
    Vertex → Edge → Face → Shell → Solid

Topology graph:
    Solid contains Shells
    Shell contains Faces
    Face bounded by Wires (loops of Edges)
    Edge connects two Vertices
    HalfEdge provides traversal direction

Euler's formula for valid manifolds: V - E + F = 2(S - G)
    V = vertices, E = edges, F = faces, S = shells, G = genus (holes)
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE, MERGE_TOLERANCE
from .vector_math import Vec3, _to_array
import uuid


def _uid():
    return str(uuid.uuid4())[:8]


class Vertex:
    """Topological vertex — a point in 3D space."""

    def __init__(self, point, vid=None):
        self.id = vid or _uid()
        self.point = Vec3(point)
        self.half_edges = []  # HalfEdges starting from this vertex

    def is_coincident(self, other, tol=LINEAR_TOLERANCE):
        return self.point.distance_to(other.point) < tol

    def __repr__(self):
        return f"Vertex({self.id}, {self.point})"


class HalfEdge:
    """
    Half-edge: directed edge used for traversal.

    Each geometric edge has two half-edges (one per adjacent face).
    Navigation:
        next  → next half-edge in the face loop
        prev  → previous half-edge in the face loop
        twin  → opposite half-edge (same geometric edge, other face)
        vertex → origin vertex
        face  → face this half-edge bounds
    """

    def __init__(self, vertex=None, heid=None):
        self.id = heid or _uid()
        self.vertex = vertex      # Origin vertex
        self.twin = None          # Opposite half-edge
        self.next = None          # Next in face loop
        self.prev = None          # Previous in face loop
        self.face = None          # Adjacent face
        self.edge = None          # Parent Edge

    @property
    def end_vertex(self):
        return self.next.vertex if self.next else None

    def __repr__(self):
        v0 = self.vertex.id if self.vertex else '?'
        v1 = self.end_vertex.id if self.end_vertex else '?'
        return f"HE({v0}→{v1})"


class Edge:
    """Geometric edge — connects two vertices. Has two half-edges."""

    def __init__(self, he1=None, he2=None, eid=None):
        self.id = eid or _uid()
        self.half_edge_1 = he1
        self.half_edge_2 = he2
        if he1: he1.edge = self
        if he2: he2.edge = self

    @property
    def start_vertex(self):
        return self.half_edge_1.vertex if self.half_edge_1 else None

    @property
    def end_vertex(self):
        return self.half_edge_2.vertex if self.half_edge_2 else None

    def length(self):
        if self.start_vertex and self.end_vertex:
            return self.start_vertex.point.distance_to(self.end_vertex.point)
        return 0.0

    def midpoint(self):
        if self.start_vertex and self.end_vertex:
            return (self.start_vertex.point + self.end_vertex.point) / 2.0
        return Vec3(0, 0, 0)

    def __repr__(self):
        return f"Edge({self.id}, len={self.length():.4f})"


class Wire:
    """A closed loop of half-edges bounding a face."""

    def __init__(self, half_edges=None, wid=None):
        self.id = wid or _uid()
        self.half_edges = half_edges or []

    def is_closed(self):
        if not self.half_edges:
            return False
        return self.half_edges[-1].next == self.half_edges[0]

    def vertices(self):
        return [he.vertex for he in self.half_edges]

    def __repr__(self):
        return f"Wire({self.id}, {len(self.half_edges)} edges)"


class Face:
    """
    Topological face — a bounded region of a surface.
    Outer boundary = outer wire (CCW), holes = inner wires (CW).
    """

    def __init__(self, outer_wire=None, inner_wires=None, fid=None, normal=None):
        self.id = fid or _uid()
        self.outer_wire = outer_wire
        self.inner_wires = inner_wires or []
        self._normal = normal
        self.surface = None  # Optional NURBS surface geometry

    @property
    def normal(self):
        if self._normal:
            return self._normal
        # Compute from vertices using Newell's method
        verts = self.outer_wire.vertices() if self.outer_wire else []
        if len(verts) < 3:
            return Vec3(0, 0, 1)
        n = Vec3(0, 0, 0)
        for i in range(len(verts)):
            v_curr = verts[i].point
            v_next = verts[(i + 1) % len(verts)].point
            n = n + Vec3(
                (v_curr.y - v_next.y) * (v_curr.z + v_next.z),
                (v_curr.z - v_next.z) * (v_curr.x + v_next.x),
                (v_curr.x - v_next.x) * (v_curr.y + v_next.y)
            )
        return n.normalize()

    def area(self):
        """Compute face area using the shoelace-like formula in 3D."""
        verts = self.outer_wire.vertices() if self.outer_wire else []
        if len(verts) < 3:
            return 0.0
        total = Vec3(0, 0, 0)
        v0 = verts[0].point
        for i in range(1, len(verts) - 1):
            edge1 = verts[i].point - v0
            edge2 = verts[i + 1].point - v0
            total = total + edge1.cross(edge2)
        return total.length() * 0.5

    def centroid(self):
        verts = self.outer_wire.vertices() if self.outer_wire else []
        if not verts:
            return Vec3(0, 0, 0)
        c = Vec3(0, 0, 0)
        for v in verts:
            c = c + v.point
        return c / len(verts)

    def triangulate(self):
        """Simple fan triangulation (works for convex faces)."""
        verts = self.outer_wire.vertices() if self.outer_wire else []
        if len(verts) < 3:
            return []
        triangles = []
        for i in range(1, len(verts) - 1):
            triangles.append((verts[0], verts[i], verts[i + 1]))
        return triangles

    def __repr__(self):
        return f"Face({self.id}, area={self.area():.4f})"


class Shell:
    """A connected set of faces forming a closed or open surface."""

    def __init__(self, faces=None, sid=None):
        self.id = sid or _uid()
        self.faces = faces or []

    def is_closed(self):
        """Check if shell is watertight (every edge has twin)."""
        for face in self.faces:
            if face.outer_wire:
                for he in face.outer_wire.half_edges:
                    if he.twin is None:
                        return False
        return True

    def surface_area(self):
        return sum(f.area() for f in self.faces)

    def __repr__(self):
        return f"Shell({self.id}, {len(self.faces)} faces)"


class Solid:
    """
    A closed volume bounded by one or more shells.
    Outer shell = external boundary.
    Inner shells = voids/cavities.
    """

    def __init__(self, outer_shell=None, inner_shells=None, name="Solid", material=None):
        self.id = _uid()
        self.name = name
        self.outer_shell = outer_shell
        self.inner_shells = inner_shells or []
        self.material = material
        self.transform = None

    def all_vertices(self):
        verts = set()
        for shell in [self.outer_shell] + self.inner_shells:
            if shell:
                for face in shell.faces:
                    if face.outer_wire:
                        for v in face.outer_wire.vertices():
                            verts.add(v)
        return list(verts)

    def all_faces(self):
        faces = []
        for shell in [self.outer_shell] + self.inner_shells:
            if shell:
                faces.extend(shell.faces)
        return faces

    def all_edges(self):
        edges = set()
        for face in self.all_faces():
            if face.outer_wire:
                for he in face.outer_wire.half_edges:
                    if he.edge:
                        edges.add(he.edge)
        return list(edges)

    def surface_area(self):
        return sum(f.area() for f in self.all_faces())

    def volume(self):
        """Compute volume using divergence theorem (signed tetrahedron volumes)."""
        total = 0.0
        for face in self.all_faces():
            tris = face.triangulate()
            for v0, v1, v2 in tris:
                p0, p1, p2 = v0.point.array, v1.point.array, v2.point.array
                total += np.dot(p0, np.cross(p1, p2)) / 6.0
        return abs(total)

    def centroid(self):
        """Volume-weighted centroid."""
        verts = self.all_vertices()
        if not verts:
            return Vec3(0, 0, 0)
        c = Vec3(0, 0, 0)
        for v in verts:
            c = c + v.point
        return c / len(verts)

    def bounding_box(self):
        verts = self.all_vertices()
        if not verts:
            return Vec3(0, 0, 0), Vec3(0, 0, 0)
        pts = np.array([v.point.array for v in verts])
        return Vec3(pts.min(axis=0)), Vec3(pts.max(axis=0))

    def euler_characteristic(self):
        """V - E + F (should be 2 for simple closed solid)."""
        V = len(self.all_vertices())
        E = len(self.all_edges())
        F = len(self.all_faces())
        return V - E + F

    def is_valid(self):
        """Basic manifold validation."""
        chi = self.euler_characteristic()
        is_closed = self.outer_shell.is_closed() if self.outer_shell else False
        return is_closed and chi == 2

    def get_mesh_data(self):
        """Export as flat mesh arrays for rendering (x, y, z, i, j, k)."""
        all_verts = []
        vert_map = {}
        triangles = []

        for face in self.all_faces():
            tris = face.triangulate()
            for v0, v1, v2 in tris:
                for v in [v0, v1, v2]:
                    if v.id not in vert_map:
                        vert_map[v.id] = len(all_verts)
                        all_verts.append(v.point)
                triangles.append((vert_map[v0.id], vert_map[v1.id], vert_map[v2.id]))

        x = [v.x for v in all_verts]
        y = [v.y for v in all_verts]
        z = [v.z for v in all_verts]
        i_list = [t[0] for t in triangles]
        j_list = [t[1] for t in triangles]
        k_list = [t[2] for t in triangles]
        intensity = [v.z for v in all_verts]  # Color by Z

        return {
            'x': x, 'y': y, 'z': z,
            'i': i_list, 'j': j_list, 'k': k_list,
            'intensity': intensity,
            'num_vertices': len(all_verts),
            'num_faces': len(triangles),
        }

    def info(self):
        V = len(self.all_vertices())
        E = len(self.all_edges())
        F = len(self.all_faces())
        return (f"Solid '{self.name}': V={V}, E={E}, F={F}, "
                f"χ={self.euler_characteristic()}, "
                f"Volume={self.volume():.4f}, Area={self.surface_area():.4f}")

    def __repr__(self):
        return self.info()


# =====================================================================
# B-REP BUILDER UTILITIES
# =====================================================================

def make_face_from_points(points, fid=None):
    """Create a planar B-Rep face from an ordered list of 3D points."""
    vertices = [Vertex(p) for p in points]
    n = len(vertices)
    half_edges = []
    for i in range(n):
        he = HalfEdge(vertex=vertices[i])
        half_edges.append(he)
        vertices[i].half_edges.append(he)

    for i in range(n):
        half_edges[i].next = half_edges[(i + 1) % n]
        half_edges[i].prev = half_edges[(i - 1) % n]

    edges = []
    for i in range(n):
        edge = Edge(he1=half_edges[i])
        edges.append(edge)

    wire = Wire(half_edges)
    return Face(outer_wire=wire, fid=fid), vertices, edges


def make_box(lx, ly, lz, origin=None):
    """Create a B-Rep box solid. lx, ly, lz = dimensions."""
    if origin is None:
        origin = Vec3(0, 0, 0)
    o = origin
    # 8 vertices of the box
    v = [
        Vertex(Vec3(o.x,      o.y,      o.z)),       # 0: bottom-front-left
        Vertex(Vec3(o.x + lx, o.y,      o.z)),       # 1: bottom-front-right
        Vertex(Vec3(o.x + lx, o.y + ly, o.z)),       # 2: bottom-back-right
        Vertex(Vec3(o.x,      o.y + ly, o.z)),       # 3: bottom-back-left
        Vertex(Vec3(o.x,      o.y,      o.z + lz)),  # 4: top-front-left
        Vertex(Vec3(o.x + lx, o.y,      o.z + lz)),  # 5: top-front-right
        Vertex(Vec3(o.x + lx, o.y + ly, o.z + lz)),  # 6: top-back-right
        Vertex(Vec3(o.x,      o.y + ly, o.z + lz)),  # 7: top-back-left
    ]

    # 6 faces (each defined by 4 vertices in CCW order when viewed from outside)
    face_indices = [
        [0, 3, 2, 1],  # bottom (-Z)
        [4, 5, 6, 7],  # top (+Z)
        [0, 1, 5, 4],  # front (-Y)
        [2, 3, 7, 6],  # back (+Y)
        [0, 4, 7, 3],  # left (-X)
        [1, 2, 6, 5],  # right (+X)
    ]

    faces = []
    all_edges = []
    for fi in face_indices:
        pts = [v[i].point for i in fi]
        face, _, edges = make_face_from_points(pts)
        faces.append(face)
        all_edges.extend(edges)

    shell = Shell(faces)
    solid = Solid(outer_shell=shell, name=f"Box({lx}x{ly}x{lz})")
    return solid


def make_cylinder(radius, height, num_sides=32, origin=None):
    """Create a B-Rep cylinder."""
    if origin is None:
        origin = Vec3(0, 0, 0)

    bottom_pts = []
    top_pts = []
    for i in range(num_sides):
        angle = 2.0 * np.pi * i / num_sides
        x = origin.x + radius * np.cos(angle)
        y = origin.y + radius * np.sin(angle)
        bottom_pts.append(Vec3(x, y, origin.z))
        top_pts.append(Vec3(x, y, origin.z + height))

    faces = []

    # Bottom face (reversed for inward normal)
    face_b, _, _ = make_face_from_points(list(reversed(bottom_pts)))
    faces.append(face_b)

    # Top face
    face_t, _, _ = make_face_from_points(top_pts)
    faces.append(face_t)

    # Side faces (quads)
    for i in range(num_sides):
        j = (i + 1) % num_sides
        quad = [bottom_pts[i], bottom_pts[j], top_pts[j], top_pts[i]]
        face_s, _, _ = make_face_from_points(quad)
        faces.append(face_s)

    shell = Shell(faces)
    return Solid(outer_shell=shell, name=f"Cylinder(r={radius}, h={height})")


def make_sphere(radius, num_lat=16, num_lon=32, center=None):
    """Create a B-Rep sphere approximation."""
    if center is None:
        center = Vec3(0, 0, 0)

    vertices = []
    # Generate vertex grid
    for i in range(num_lat + 1):
        phi = np.pi * i / num_lat
        for j in range(num_lon):
            theta = 2.0 * np.pi * j / num_lon
            x = center.x + radius * np.sin(phi) * np.cos(theta)
            y = center.y + radius * np.sin(phi) * np.sin(theta)
            z = center.z + radius * np.cos(phi)
            vertices.append(Vec3(x, y, z))

    faces = []
    for i in range(num_lat):
        for j in range(num_lon):
            j_next = (j + 1) % num_lon
            i0 = i * num_lon + j
            i1 = i * num_lon + j_next
            i2 = (i + 1) * num_lon + j_next
            i3 = (i + 1) * num_lon + j

            if i == 0:
                # Triangle at north pole
                face, _, _ = make_face_from_points([vertices[i0], vertices[i2], vertices[i3]])
                faces.append(face)
            elif i == num_lat - 1:
                # Triangle at south pole
                face, _, _ = make_face_from_points([vertices[i0], vertices[i1], vertices[i3]])
                faces.append(face)
            else:
                # Quad
                face, _, _ = make_face_from_points([vertices[i0], vertices[i1], vertices[i2], vertices[i3]])
                faces.append(face)

    shell = Shell(faces)
    return Solid(outer_shell=shell, name=f"Sphere(r={radius})")


def make_cone(radius, height, num_sides=32, origin=None):
    """Create a B-Rep cone."""
    if origin is None:
        origin = Vec3(0, 0, 0)
    apex = Vec3(origin.x, origin.y, origin.z + height)

    base_pts = []
    for i in range(num_sides):
        angle = 2.0 * np.pi * i / num_sides
        base_pts.append(Vec3(origin.x + radius * np.cos(angle),
                             origin.y + radius * np.sin(angle), origin.z))

    faces = []
    # Base
    face_b, _, _ = make_face_from_points(list(reversed(base_pts)))
    faces.append(face_b)
    # Side triangles
    for i in range(num_sides):
        j = (i + 1) % num_sides
        face_s, _, _ = make_face_from_points([base_pts[i], base_pts[j], apex])
        faces.append(face_s)

    shell = Shell(faces)
    return Solid(outer_shell=shell, name=f"Cone(r={radius}, h={height})")


def make_torus(major_radius, minor_radius, num_major=32, num_minor=16, center=None):
    """Create a B-Rep torus."""
    if center is None:
        center = Vec3(0, 0, 0)

    vertices = []
    for i in range(num_major):
        theta = 2.0 * np.pi * i / num_major
        for j in range(num_minor):
            phi = 2.0 * np.pi * j / num_minor
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
            y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
            z = minor_radius * np.sin(phi)
            vertices.append(center + Vec3(x, y, z))

    faces = []
    for i in range(num_major):
        i_next = (i + 1) % num_major
        for j in range(num_minor):
            j_next = (j + 1) % num_minor
            v0 = vertices[i * num_minor + j]
            v1 = vertices[i_next * num_minor + j]
            v2 = vertices[i_next * num_minor + j_next]
            v3 = vertices[i * num_minor + j_next]
            face, _, _ = make_face_from_points([v0, v1, v2, v3])
            faces.append(face)

    shell = Shell(faces)
    return Solid(outer_shell=shell, name=f"Torus(R={major_radius}, r={minor_radius})")
