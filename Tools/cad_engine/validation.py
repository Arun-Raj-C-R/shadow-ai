"""
CAD Engine — Automated Validation Suite
==========================================
Tests all core modules for correctness.
"""

import numpy as np
import sys
import time


def run_all_tests(verbose=True):
    """Run the full CAD engine validation suite."""
    passed = 0
    failed = 0
    total_start = time.time()
    test_results = []

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        test_results.append((name, status, detail))
        if verbose:
            print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))

    print("\n" + "=" * 60)
    print("  CAD ENGINE VALIDATION SUITE")
    print("=" * 60 + "\n")

    # ── 1. Vector Math ────────────────────────────────────────────
    print("▶ Vector Math")
    from .vector_math import Vec3, dot, cross, normalize, distance, lerp

    v1 = Vec3(1, 0, 0)
    v2 = Vec3(0, 1, 0)
    v3 = Vec3(1, 1, 0)

    check("Vec3 creation", v1.x == 1.0 and v1.y == 0.0 and v1.z == 0.0)
    check("Dot product orthogonal", abs(v1.dot(v2)) < 1e-10, f"got {v1.dot(v2)}")
    check("Dot product parallel", abs(v1.dot(v1) - 1.0) < 1e-10)
    check("Cross product X×Y=Z", v1.cross(v2).is_equal_to(Vec3(0, 0, 1)))
    check("Cross product anti-commutative", v2.cross(v1).is_equal_to(Vec3(0, 0, -1)))
    check("Normalize unit length", abs(v3.normalize().length() - 1.0) < 1e-10)
    check("Distance", abs(distance(Vec3(0,0,0), Vec3(3,4,0)) - 5.0) < 1e-10)
    check("Lerp midpoint", lerp(Vec3(0,0,0), Vec3(10,0,0), 0.5).is_equal_to(Vec3(5,0,0)))

    # Rodrigues rotation
    v = Vec3(1, 0, 0)
    rotated = v.rotate_axis_angle(Vec3(0, 0, 1), np.pi / 2)
    check("Rodrigues rotation 90°", rotated.is_equal_to(Vec3(0, 1, 0), tol=1e-6),
          f"got {rotated}")

    check("Projection onto axis",
          Vec3(3, 4, 0).project_onto(Vec3(1, 0, 0)).is_equal_to(Vec3(3, 0, 0)))

    # ── 2. Transforms ────────────────────────────────────────────
    print("\n▶ Transforms")
    from .transforms import Transform, CoordinateSystem

    T = Transform.identity()
    check("Identity transform", T.is_identity())

    p = Vec3(1, 2, 3)
    T_trans = Transform.translation(10, 20, 30)
    p_moved = T_trans.apply_point(p)
    check("Translation", p_moved.is_equal_to(Vec3(11, 22, 33)),
          f"got {p_moved}")

    T_scale = Transform.scaling(2.0)
    p_scaled = T_scale.apply_point(Vec3(1, 1, 1))
    check("Uniform scaling", p_scaled.is_equal_to(Vec3(2, 2, 2)))

    T_rot = Transform.rotation_z(np.pi / 2)
    p_rot = T_rot.apply_point(Vec3(1, 0, 0))
    check("Rotation Z 90°", p_rot.is_equal_to(Vec3(0, 1, 0), tol=1e-6),
          f"got {p_rot}")

    # Compose: translate then rotate
    T_comp = T_rot.compose(T_trans)
    check("Compose is different from identity", not T_comp.is_identity())

    # Inverse
    T_inv = T_trans.inverse
    p_round = T_inv.apply_point(T_trans.apply_point(p))
    check("Inverse roundtrip", p_round.is_equal_to(p, tol=1e-6))

    # Coordinate system
    cs = CoordinateSystem()
    check("Default CS at origin", cs.origin.is_equal_to(Vec3(0, 0, 0)))

    # ── 3. Tolerance ──────────────────────────────────────────────
    print("\n▶ Tolerance & Predicates")
    from .tolerance import float_eq, orient2d, points_are_collinear

    check("Float equality 0.1+0.2≈0.3", float_eq(0.1 + 0.2, 0.3, tol=1e-9))
    check("Float inequality", not float_eq(1.0, 2.0))

    orient = orient2d([0, 0], [1, 0], [0.5, 1])
    check("Orient2D CCW > 0", orient > 0, f"got {orient}")

    orient_cw = orient2d([0, 0], [0.5, 1], [1, 0])
    check("Orient2D CW < 0", orient_cw < 0)

    check("Collinear points", points_are_collinear([0,0,0], [1,0,0], [2,0,0]))
    check("Non-collinear points", not points_are_collinear([0,0,0], [1,0,0], [0,1,0]))

    # ── 4. NURBS ──────────────────────────────────────────────────
    print("\n▶ NURBS & Splines")
    from .nurbs import BezierCurve, NURBSCurve, NURBSSurface, nurbs_circle

    # Bézier line
    bez = BezierCurve([Vec3(0,0,0), Vec3(10,0,0)])
    check("Bézier start", bez.evaluate(0.0).is_equal_to(Vec3(0,0,0)))
    check("Bézier end", bez.evaluate(1.0).is_equal_to(Vec3(10,0,0)))
    check("Bézier midpoint", bez.evaluate(0.5).is_equal_to(Vec3(5,0,0)))

    # Quadratic Bézier
    bez2 = BezierCurve([Vec3(0,0,0), Vec3(5,10,0), Vec3(10,0,0)])
    mid = bez2.evaluate(0.5)
    check("Quadratic Bézier apex", abs(mid.y - 5.0) < 0.01, f"y={mid.y}")

    # NURBS circle
    circ = nurbs_circle(radius=5.0)
    p_start = circ.evaluate(0.0)
    check("NURBS circle start on x-axis", abs(p_start.x - 5.0) < 0.1 and abs(p_start.y) < 0.1,
          f"got {p_start}")
    check("NURBS circle length ≈ 2πr", abs(circ.length() - 2 * np.pi * 5) < 1.0,
          f"got {circ.length():.2f}, expected {2*np.pi*5:.2f}")

    # ── 5. B-Rep Primitives ───────────────────────────────────────
    print("\n▶ B-Rep Solids")
    from .brep import make_box, make_cylinder, make_sphere, make_cone, make_torus

    box = make_box(10, 20, 5)
    check("Box has 24 vertices (4 per face)", len(box.all_vertices()) == 24,
          f"got {len(box.all_vertices())}")  # Each face creates its own verts
    check("Box has 6 faces", len(box.all_faces()) == 6)
    check("Box volume > 0", box.volume() > 0, f"vol={box.volume():.2f}")

    cyl = make_cylinder(5, 10, num_sides=16)
    check("Cylinder faces = sides + 2 caps", len(cyl.all_faces()) == 18,
          f"got {len(cyl.all_faces())}")
    check("Cylinder volume > 0", cyl.volume() > 0)

    sph = make_sphere(5, num_lat=8, num_lon=16)
    check("Sphere has faces", len(sph.all_faces()) > 0,
          f"got {len(sph.all_faces())} faces")
    check("Sphere volume > 0", sph.volume() > 0)

    cone = make_cone(5, 10, num_sides=16)
    check("Cone has faces", len(cone.all_faces()) > 0)

    torus = make_torus(10, 3, num_major=16, num_minor=8)
    check("Torus has faces", len(torus.all_faces()) > 0,
          f"got {len(torus.all_faces())} faces")

    # ── 6. Sketch Engine ──────────────────────────────────────────
    print("\n▶ Sketch Engine")
    from .sketch import Sketch, SketchLine, SketchCircle, SketchArc

    s = Sketch(name="TestSketch")
    l1 = s.add_line((0, 0), (10, 0))
    l2 = s.add_line((10, 0), (10, 5))
    l3 = s.add_line((10, 5), (0, 5))
    l4 = s.add_line((0, 5), (0, 0))
    check("Sketch has 4 entities", len(s.entities) == 4)
    check("Line length", abs(l1.length() - 10.0) < 1e-10)

    c = s.add_circle((5, 2.5), 2)
    check("Circle circumference", abs(c.length() - 2 * np.pi * 2) < 1e-6)
    check("Circle area", abs(c.area() - np.pi * 4) < 1e-6)

    rect = Sketch()
    rect.add_rectangle(0, 0, 20, 10)
    check("Rectangle = 4 lines", len(rect.entities) == 4)

    # ── 7. Constraints ────────────────────────────────────────────
    print("\n▶ Constraint Solver")
    from .constraints import (ConstraintSolver, CoincidentConstraint,
                               HorizontalConstraint, PerpendicularConstraint)
    from .sketch import Point2D

    p1 = Point2D(0, 0)
    p2 = Point2D(0.1, 0.1)  # Slightly off
    solver = ConstraintSolver()
    solver.add_constraint(CoincidentConstraint(p1, p2))
    result = solver.solve()
    check("Coincident constraint converges", result['converged'],
          f"err={result['error']:.2e}")
    check("Points coincident after solve", p1.distance_to(p2) < 1e-6,
          f"dist={p1.distance_to(p2):.2e}")

    # ── 8. Computational Geometry ─────────────────────────────────
    print("\n▶ Computational Geometry")
    from .computational_geom import (point_in_polygon_2d, ray_triangle_intersection,
                                      convex_hull_2d, delaunay_2d)

    # Point in polygon
    square = [(0,0), (10,0), (10,10), (0,10)]
    check("Point inside polygon", point_in_polygon_2d((5, 5), square))
    check("Point outside polygon", not point_in_polygon_2d((15, 5), square))

    # Ray-triangle
    hit, t, u, v = ray_triangle_intersection(
        [0, 0, 10], [0, 0, -1],
        [0, 0, 0], [10, 0, 0], [0, 10, 0]
    )
    check("Ray hits triangle", hit, f"t={t:.4f}")

    hit2, _, _, _ = ray_triangle_intersection(
        [100, 100, 10], [0, 0, -1],
        [0, 0, 0], [10, 0, 0], [0, 10, 0]
    )
    check("Ray misses triangle", not hit2)

    # Convex hull
    pts_2d = [(0,0), (10,0), (5,5), (10,10), (0,10), (5,3)]
    hull_idx = convex_hull_2d(pts_2d)
    check("Convex hull has <= n points", len(hull_idx) <= len(pts_2d))
    check("Convex hull has >= 3 points", len(hull_idx) >= 3)

    # Delaunay
    tri_pts = [(0,0), (10,0), (10,10), (0,10), (5,5)]
    tris = delaunay_2d(tri_pts)
    check("Delaunay produces triangles", len(tris) > 0, f"got {len(tris)} triangles")

    # ── 9. Meshing ────────────────────────────────────────────────
    print("\n▶ Meshing")
    from .meshing import solid_to_trimesh, subdivide_mesh, mesh_quality

    mesh = solid_to_trimesh(box)
    check("Box mesh has vertices", mesh.num_vertices > 0)
    check("Box mesh has faces", mesh.num_faces > 0)

    quality = mesh_quality(mesh)
    check("Mesh quality computable", quality['aspect_ratio_avg'] > 0)
    check("Mesh has valid min angle", quality['min_angle_min_deg'] > 0)

    # Subdivision
    sub_mesh = subdivide_mesh(solid_to_trimesh(box), iterations=1)
    check("Subdivision increases faces", sub_mesh.num_faces > mesh.num_faces,
          f"{mesh.num_faces} → {sub_mesh.num_faces}")

    # ── 10. Feature History ───────────────────────────────────────
    print("\n▶ Feature History")
    from .features import FeatureTree

    ft = FeatureTree("TestPart")
    f1 = ft.add_feature("Sketch", lambda p, d: "sketch_result")
    f2 = ft.add_feature("Extrude", lambda p, d: f"extruded_{d.get(f1.id, 'none')}",
                         dependencies=[f1])
    ft_results = ft.rebuild()
    check("Feature tree rebuilds", f1.result == "sketch_result")
    check("Feature dependency works", f2.result is not None)

    ft.suppress(f2.id)
    check("Feature suppression", f2.suppressed)
    check("Final result after suppress", ft.get_final_result() == "sketch_result")

    # ── 11. CAD Session Integration ───────────────────────────────
    print("\n▶ CAD Session Integration")
    from .cad_engine import CADSession

    session = CADSession("ValidationPart")
    idx_box = session.add_box(10, 20, 5)
    idx_cyl = session.add_cylinder(3, 15)
    check("Session has 2 solids", len(session.solids) == 2)

    mesh_data = session.get_combined_mesh()
    check("Combined mesh has data", mesh_data['num_vertices'] > 0 and mesh_data['num_faces'] > 0,
          f"V={mesh_data['num_vertices']}, F={mesh_data['num_faces']}")

    analysis = session.analyze()
    check("Analysis returns results", len(analysis) == 2)
    check("Volume is positive", all(a['volume'] > 0 for a in analysis))

    timeline = session.show_timeline()
    check("Timeline is printable", len(timeline) > 0)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - total_start
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION COMPLETE: {passed}/{total} passed, {failed} failed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'=' * 60}\n")

    return {
        'passed': passed, 'failed': failed, 'total': total,
        'elapsed': elapsed, 'results': test_results
    }


if __name__ == "__main__":
    run_all_tests()
