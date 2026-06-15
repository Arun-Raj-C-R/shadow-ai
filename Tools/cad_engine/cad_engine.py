"""
Main CAD Engine — Orchestrator
================================
High-level interface for parametric CAD operations.

Usage:
    from cad_engine.cad_engine import CADSession

    session = CADSession("MyPart")
    session.add_box(10, 20, 5)
    session.add_cylinder(3, 15, origin=[5, 10, 5])
    session.boolean_subtract(0, 1)
    session.export_stl("part.stl")
    session.render_html("part.html")
"""

import os
import json
import time
import webbrowser
from datetime import datetime

from .constants import MATERIALS, get_material
from .vector_math import Vec3
from .transforms import Transform
from .brep import (Solid, make_box, make_cylinder, make_sphere,
                   make_cone, make_torus, make_face_from_points)
from .sketch import Sketch, Point2D
from .constraints import ConstraintSolver
from .cad_operations import extrude, revolve, loft, sweep
from .features import FeatureTree
from .meshing import TriMesh, solid_to_trimesh, subdivide_mesh, smooth_mesh, mesh_quality
from .renderer import render_to_html
from .file_io import (export_stl_ascii, export_stl_binary, export_obj,
                      export_dxf, export_step, import_stl)
from .assembly import Assembly, Part


class CADSession:
    """
    Main CAD session — single entry point for all operations.

    Manages:
        - Solid bodies (B-Rep)
        - Feature history (parametric timeline)
        - Assembly management
        - File I/O
        - 3D rendering
    """

    def __init__(self, name="Part", material=None, output_dir=None):
        self.name = name
        self.material = material
        self.solids = []
        self.sketches = []
        self.feature_tree = FeatureTree(name)
        self.assembly = Assembly(name)
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cad_runs"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._start_time = time.time()

    # ── Primitive Creation ────────────────────────────────────────

    def add_box(self, lx, ly, lz, origin=None, name=None):
        """Add a box solid."""
        solid = make_box(lx, ly, lz, Vec3(origin) if origin else None)
        if name:
            solid.name = name
        if self.material:
            solid.material = self.material
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Box({lx}×{ly}×{lz})",
            lambda p, d: self.solids[-1],
            {'lx': lx, 'ly': ly, 'lz': lz}
        )
        return len(self.solids) - 1

    def add_cylinder(self, radius, height, origin=None, num_sides=32, name=None):
        """Add a cylinder solid."""
        solid = make_cylinder(radius, height, num_sides, Vec3(origin) if origin else None)
        if name:
            solid.name = name
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Cylinder(r={radius}, h={height})",
            lambda p, d: self.solids[-1],
            {'radius': radius, 'height': height}
        )
        return len(self.solids) - 1

    def add_sphere(self, radius, center=None, name=None):
        """Add a sphere solid."""
        solid = make_sphere(radius, center=Vec3(center) if center else None)
        if name:
            solid.name = name
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Sphere(r={radius})",
            lambda p, d: self.solids[-1],
            {'radius': radius}
        )
        return len(self.solids) - 1

    def add_cone(self, radius, height, origin=None, name=None):
        """Add a cone solid."""
        solid = make_cone(radius, height, origin=Vec3(origin) if origin else None)
        if name:
            solid.name = name
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Cone(r={radius}, h={height})",
            lambda p, d: self.solids[-1],
            {'radius': radius, 'height': height}
        )
        return len(self.solids) - 1

    def add_torus(self, major_radius, minor_radius, center=None, name=None):
        """Add a torus solid."""
        solid = make_torus(major_radius, minor_radius,
                           center=Vec3(center) if center else None)
        if name:
            solid.name = name
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Torus(R={major_radius}, r={minor_radius})",
            lambda p, d: self.solids[-1],
            {'major_radius': major_radius, 'minor_radius': minor_radius}
        )
        return len(self.solids) - 1

    # ── Sketch-Based Operations ───────────────────────────────────

    def new_sketch(self, plane_origin=None, plane_normal=None, name=None):
        """Create a new sketch on a plane."""
        sketch = Sketch(plane_origin, plane_normal, name or f"Sketch{len(self.sketches)+1}")
        self.sketches.append(sketch)
        return sketch, len(self.sketches) - 1

    def extrude_sketch(self, sketch_idx, distance, direction=None):
        """Extrude a sketch to create a solid."""
        sketch = self.sketches[sketch_idx]
        solid = extrude(sketch, distance, direction)
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Extrude({sketch.name}, d={distance})",
            lambda p, d: solid,
            {'sketch_idx': sketch_idx, 'distance': distance}
        )
        return len(self.solids) - 1

    def revolve_sketch(self, sketch_idx, axis_origin=None, axis_direction=None,
                       angle=None, num_steps=32):
        """Revolve a sketch around an axis."""
        sketch = self.sketches[sketch_idx]
        solid = revolve(sketch, axis_origin, axis_direction, angle, num_steps)
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Revolve({sketch.name})",
            lambda p, d: solid,
            {'sketch_idx': sketch_idx}
        )
        return len(self.solids) - 1

    def loft_profiles(self, profiles, num_steps=20):
        """Loft between multiple profiles."""
        solid = loft(profiles, num_steps)
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Loft({len(profiles)} profiles)",
            lambda p, d: solid,
            {'num_profiles': len(profiles)}
        )
        return len(self.solids) - 1

    def sweep_profile(self, profile_points, path_points):
        """Sweep a profile along a path."""
        solid = sweep(profile_points, path_points)
        self.solids.append(solid)

        self.feature_tree.add_feature(
            f"Sweep",
            lambda p, d: solid,
        )
        return len(self.solids) - 1

    # ── Boolean Operations ────────────────────────────────────────

    def boolean_union(self, idx_a, idx_b):
        """Boolean union of two solids."""
        from .boolean_ops import boolean_union, _polygons_to_mesh
        polys = boolean_union(self.solids[idx_a], self.solids[idx_b])
        # Store result polygons for rendering
        verts, tris = _polygons_to_mesh(polys)
        self.solids[idx_a]._boolean_result = {'vertices': verts, 'triangles': tris}

        self.feature_tree.add_feature(
            f"Union({idx_a}, {idx_b})",
            lambda p, d: polys,
            {'idx_a': idx_a, 'idx_b': idx_b}
        )
        return idx_a

    def boolean_subtract(self, idx_a, idx_b):
        """Boolean subtraction: A - B."""
        from .boolean_ops import boolean_subtract, _polygons_to_mesh
        polys = boolean_subtract(self.solids[idx_a], self.solids[idx_b])
        verts, tris = _polygons_to_mesh(polys)
        self.solids[idx_a]._boolean_result = {'vertices': verts, 'triangles': tris}

        self.feature_tree.add_feature(
            f"Subtract({idx_a} - {idx_b})",
            lambda p, d: polys,
        )
        return idx_a

    def boolean_intersect(self, idx_a, idx_b):
        """Boolean intersection: A ∩ B."""
        from .boolean_ops import boolean_intersect, _polygons_to_mesh
        polys = boolean_intersect(self.solids[idx_a], self.solids[idx_b])
        verts, tris = _polygons_to_mesh(polys)
        self.solids[idx_a]._boolean_result = {'vertices': verts, 'triangles': tris}

        self.feature_tree.add_feature(
            f"Intersect({idx_a} ∩ {idx_b})",
            lambda p, d: polys,
        )
        return idx_a

    # ── Mesh & Analysis ───────────────────────────────────────────

    def get_combined_mesh(self):
        """Get combined mesh data from all solids."""
        x, y, z, i_l, j_l, k_l, intensity = [], [], [], [], [], [], []
        offset = 0

        for solid in self.solids:
            # Check for boolean result first
            if hasattr(solid, '_boolean_result') and solid._boolean_result:
                br = solid._boolean_result
                for v in br['vertices']:
                    x.append(v.x); y.append(v.y); z.append(v.z)
                    intensity.append(v.z)
                for t in br['triangles']:
                    i_l.append(t[0] + offset)
                    j_l.append(t[1] + offset)
                    k_l.append(t[2] + offset)
                offset += len(br['vertices'])
            else:
                mesh = solid.get_mesh_data()
                x.extend(mesh['x']); y.extend(mesh['y']); z.extend(mesh['z'])
                i_l.extend([idx + offset for idx in mesh['i']])
                j_l.extend([idx + offset for idx in mesh['j']])
                k_l.extend([idx + offset for idx in mesh['k']])
                intensity.extend(mesh.get('intensity', [0] * mesh['num_vertices']))
                offset += mesh['num_vertices']

        return {
            'x': x, 'y': y, 'z': z,
            'i': i_l, 'j': j_l, 'k': k_l,
            'intensity': intensity,
            'num_vertices': len(x),
            'num_faces': len(i_l),
        }

    def analyze(self, solid_idx=None):
        """Analyze a solid or all solids."""
        results = []
        targets = [self.solids[solid_idx]] if solid_idx is not None else self.solids
        for solid in targets:
            info = {
                'name': solid.name,
                'vertices': len(solid.all_vertices()),
                'edges': len(solid.all_edges()),
                'faces': len(solid.all_faces()),
                'euler_characteristic': solid.euler_characteristic(),
                'volume': solid.volume(),
                'surface_area': solid.surface_area(),
            }
            bb_min, bb_max = solid.bounding_box()
            info['bounding_box'] = {
                'min': bb_min.to_list(),
                'max': bb_max.to_list(),
                'size': (bb_max - bb_min).to_list()
            }
            if solid.material and solid.material in MATERIALS:
                mat = MATERIALS[solid.material]
                vol_m3 = solid.volume() * 1e-9  # mm³ → m³
                info['mass_kg'] = vol_m3 * mat['density_kg_m3']
            results.append(info)
        return results

    # ── Feature History ───────────────────────────────────────────

    def show_timeline(self):
        """Print the parametric feature timeline."""
        return self.feature_tree.timeline()

    def rebuild(self, verbose=False):
        """Rebuild all dirty features."""
        return self.feature_tree.rebuild(verbose)

    # ── Export / Render ───────────────────────────────────────────

    def render_html(self, filepath=None, open_browser=True):
        """Render to interactive 3D HTML viewer."""
        mesh_data = self.get_combined_mesh()
        if not filepath:
            timestamp = int(datetime.now().timestamp())
            filepath = os.path.join(self.output_dir, f"{self.name}_{timestamp}.html")

        stats = {
            'Part': self.name,
            'Vertices': mesh_data['num_vertices'],
            'Faces': mesh_data['num_faces'],
            'Bodies': len(self.solids),
            'Features': len(self.feature_tree.features),
        }

        html = render_to_html(
            mesh_data,
            title=f"CAD — {self.name}",
            subtitle=f"Shadow AI CAD Engine · {len(self.solids)} bodies · Parametric History",
            stats=stats,
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        if open_browser:
            webbrowser.open(f"file:///{filepath.replace(os.sep, '/')}")

        return filepath

    def export_stl(self, filepath=None, binary=True):
        """Export to STL."""
        mesh_data = self.get_combined_mesh()
        if not filepath:
            filepath = os.path.join(self.output_dir, f"{self.name}.stl")
        if binary:
            return export_stl_binary(mesh_data, filepath, self.name)
        return export_stl_ascii(mesh_data, filepath, self.name)

    def export_obj(self, filepath=None):
        """Export to Wavefront OBJ."""
        mesh_data = self.get_combined_mesh()
        if not filepath:
            filepath = os.path.join(self.output_dir, f"{self.name}.obj")
        return export_obj(mesh_data, filepath, self.name)

    def export_step(self, filepath=None):
        """Export to STEP (AP203)."""
        mesh_data = self.get_combined_mesh()
        if not filepath:
            filepath = os.path.join(self.output_dir, f"{self.name}.step")
        return export_step(mesh_data, filepath, self.name)

    def export_dxf(self, filepath=None):
        """Export to DXF."""
        mesh_data = self.get_combined_mesh()
        if not filepath:
            filepath = os.path.join(self.output_dir, f"{self.name}.dxf")
        return export_dxf(mesh_data, filepath)

    def export_all(self, directory=None):
        """Export to all supported formats."""
        d = directory or self.output_dir
        results = {}
        results['stl'] = self.export_stl(os.path.join(d, f"{self.name}.stl"))
        results['obj'] = self.export_obj(os.path.join(d, f"{self.name}.obj"))
        results['step'] = self.export_step(os.path.join(d, f"{self.name}.step"))
        results['dxf'] = self.export_dxf(os.path.join(d, f"{self.name}.dxf"))
        results['html'] = self.render_html(os.path.join(d, f"{self.name}.html"), open_browser=False)
        return results

    # ── Summary ───────────────────────────────────────────────────

    def summary(self):
        """Print session summary."""
        elapsed = time.time() - self._start_time
        mesh = self.get_combined_mesh()

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  CAD SESSION SUMMARY")
        lines.append(f"{'='*60}")
        lines.append(f"  Part:       {self.name}")
        lines.append(f"  Material:   {self.material or 'None'}")
        lines.append(f"  Bodies:     {len(self.solids)}")
        lines.append(f"  Sketches:   {len(self.sketches)}")
        lines.append(f"  Features:   {len(self.feature_tree.features)}")
        lines.append(f"  Vertices:   {mesh['num_vertices']}")
        lines.append(f"  Faces:      {mesh['num_faces']}")
        lines.append(f"  Time:       {elapsed:.2f} s")

        for i, solid in enumerate(self.solids):
            lines.append(f"\n  Body {i}: {solid.name}")
            lines.append(f"    V={len(solid.all_vertices())}, "
                        f"E={len(solid.all_edges())}, "
                        f"F={len(solid.all_faces())}, "
                        f"χ={solid.euler_characteristic()}")
            lines.append(f"    Volume={solid.volume():.4f} mm³, "
                        f"Area={solid.surface_area():.4f} mm²")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def __repr__(self):
        return f"CADSession('{self.name}', {len(self.solids)} bodies, {len(self.feature_tree.features)} features)"
