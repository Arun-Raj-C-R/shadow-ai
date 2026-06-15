"""
CAD Engine â€” First-Principles Computer-Aided Design in Python
=============================================================

A from-scratch, production-quality parametric CAD kernel built entirely
from first principles. No external CAD libraries. Pure computational
geometry, NURBS mathematics, and solid modeling.

Architecture inspired by OpenCASCADE, Parasolid, and ACIS.

Modules:
    constants           â€” Tolerances, material properties, physical constants
    vector_math         â€” Vec3 operations: dot, cross, normalize, projections, rotations
    transforms          â€” 4Ã—4 homogeneous transformation matrices
    tolerance           â€” Epsilon comparisons, robust numerical predicates
    nurbs               â€” BÃ©zier curves, B-splines, NURBS curves & surfaces
    brep                â€” B-Rep kernel: Vertex, Edge, Face, Shell, Solid (half-edge)
    sketch              â€” 2D parametric sketch primitives (line, arc, circle, spline)
    constraints         â€” Geometric constraint solver (Newton-Raphson, Gauss-Newton)
    boolean_ops         â€” CSG boolean operations via BSP trees
    cad_operations      â€” Extrude, revolve, loft, sweep, fillet, chamfer, shell, pattern
    features            â€” Parametric feature history / dependency graph / recomputation
    computational_geom  â€” Triangulation, convex hull, point-in-polygon, ray casting
    assembly            â€” Constraint-based assembly: mates, joints, kinematics
    meshing             â€” Surface tessellation, tetrahedral meshing, adaptive refinement
    renderer            â€” Three.js WebGL HTML export with PBR, camera, modes
    file_io             â€” STEP, STL, OBJ, DXF export/import
    cad_engine          â€” Main CAD calculation orchestrator
    validation          â€” Automated test suite
"""

__version__ = "1.0.0"
__author__ = "Shadow AI / SHADOW CAD Engine"
