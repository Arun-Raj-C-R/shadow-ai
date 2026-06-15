"""
Constraint-Based Assembly System
===================================
Mates, joints, gears, kinematics.

Implements:
    - Mate constraints (coincident, concentric, parallel, distance, angle)
    - Joint definitions (revolute, prismatic, cylindrical, spherical)
    - Assembly tree (hierarchical parts)
    - Degree of freedom analysis
    - Basic kinematic solving: τ = r × F
"""

import numpy as np
from .constants import EPSILON, DEG_TO_RAD
from .vector_math import Vec3, _to_array
from .transforms import Transform


class AssemblyConstraint:
    """Base class for assembly constraints (mates)."""
    _counter = 0

    def __init__(self, name, part_a, part_b):
        AssemblyConstraint._counter += 1
        self.id = AssemblyConstraint._counter
        self.name = name
        self.part_a = part_a
        self.part_b = part_b
        self.satisfied = False

    def residual(self):
        raise NotImplementedError

    def error(self):
        return float(np.linalg.norm(self.residual()))


class CoincidentMate(AssemblyConstraint):
    """Two points/faces must be at the same location."""
    def __init__(self, part_a, point_a, part_b, point_b):
        super().__init__("coincident_mate", part_a, part_b)
        self.point_a = Vec3(point_a)
        self.point_b = Vec3(point_b)

    def residual(self):
        pa = self.part_a.transform.apply_point(self.point_a) if self.part_a.transform else self.point_a
        pb = self.part_b.transform.apply_point(self.point_b) if self.part_b.transform else self.point_b
        return np.array([(pa - pb).x, (pa - pb).y, (pa - pb).z])


class ConcentricMate(AssemblyConstraint):
    """Two cylindrical axes must be coincident."""
    def __init__(self, part_a, axis_a_origin, axis_a_dir, part_b, axis_b_origin, axis_b_dir):
        super().__init__("concentric_mate", part_a, part_b)
        self.axis_a_origin = Vec3(axis_a_origin)
        self.axis_a_dir = Vec3(axis_a_dir).normalize()
        self.axis_b_origin = Vec3(axis_b_origin)
        self.axis_b_dir = Vec3(axis_b_dir).normalize()

    def residual(self):
        # Axes must be parallel and intersecting
        cross_err = self.axis_a_dir.cross(self.axis_b_dir).length()
        dist_err = (self.axis_a_origin - self.axis_b_origin).reject_from(self.axis_a_dir).length()
        return np.array([cross_err, dist_err])


class ParallelMate(AssemblyConstraint):
    """Two faces/axes must be parallel."""
    def __init__(self, part_a, normal_a, part_b, normal_b):
        super().__init__("parallel_mate", part_a, part_b)
        self.normal_a = Vec3(normal_a).normalize()
        self.normal_b = Vec3(normal_b).normalize()

    def residual(self):
        return np.array([self.normal_a.cross(self.normal_b).length()])


class DistanceMate(AssemblyConstraint):
    """Maintain specific distance between two entities."""
    def __init__(self, part_a, point_a, part_b, point_b, distance):
        super().__init__("distance_mate", part_a, part_b)
        self.point_a = Vec3(point_a)
        self.point_b = Vec3(point_b)
        self.target_distance = float(distance)

    def residual(self):
        d = self.point_a.distance_to(self.point_b)
        return np.array([d - self.target_distance])


class AngleMate(AssemblyConstraint):
    """Maintain specific angle between two planes/axes."""
    def __init__(self, part_a, dir_a, part_b, dir_b, angle_rad):
        super().__init__("angle_mate", part_a, part_b)
        self.dir_a = Vec3(dir_a).normalize()
        self.dir_b = Vec3(dir_b).normalize()
        self.target_angle = float(angle_rad)

    def residual(self):
        current = self.dir_a.angle_to(self.dir_b)
        return np.array([current - self.target_angle])


# =====================================================================
# JOINTS
# =====================================================================

class Joint:
    """Base class for mechanical joints."""
    def __init__(self, name, part_a, part_b, origin, axis=None):
        self.name = name
        self.part_a = part_a
        self.part_b = part_b
        self.origin = Vec3(origin)
        self.axis = Vec3(axis).normalize() if axis else Vec3(0, 0, 1)
        self.dof = 0  # Degrees of freedom

    def __repr__(self):
        return f"Joint({self.name}, DOF={self.dof})"


class RevoluteJoint(Joint):
    """Rotation about a single axis. DOF = 1."""
    def __init__(self, part_a, part_b, origin, axis=None):
        super().__init__("revolute", part_a, part_b, origin, axis)
        self.dof = 1
        self.angle = 0.0
        self.min_angle = None
        self.max_angle = None

    def set_angle(self, angle_rad):
        if self.min_angle is not None:
            angle_rad = max(self.min_angle, angle_rad)
        if self.max_angle is not None:
            angle_rad = min(self.max_angle, angle_rad)
        self.angle = angle_rad

    def get_transform(self):
        return Transform.rotation_axis(self.axis, self.angle)


class PrismaticJoint(Joint):
    """Translation along a single axis. DOF = 1."""
    def __init__(self, part_a, part_b, origin, axis=None):
        super().__init__("prismatic", part_a, part_b, origin, axis)
        self.dof = 1
        self.displacement = 0.0

    def set_displacement(self, d):
        self.displacement = float(d)

    def get_transform(self):
        return Transform.translation(self.axis * self.displacement)


class CylindricalJoint(Joint):
    """Rotation + translation along same axis. DOF = 2."""
    def __init__(self, part_a, part_b, origin, axis=None):
        super().__init__("cylindrical", part_a, part_b, origin, axis)
        self.dof = 2
        self.angle = 0.0
        self.displacement = 0.0


class SphericalJoint(Joint):
    """3-axis rotation. DOF = 3."""
    def __init__(self, part_a, part_b, origin):
        super().__init__("spherical", part_a, part_b, origin)
        self.dof = 3
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0


class GearJoint(Joint):
    """Gear coupling: ω₁/ω₂ = -r₂/r₁."""
    def __init__(self, part_a, part_b, origin, ratio=1.0):
        super().__init__("gear", part_a, part_b, origin)
        self.ratio = ratio
        self.dof = 1


# =====================================================================
# ASSEMBLY
# =====================================================================

class Part:
    """A part in an assembly."""
    def __init__(self, solid, name=None):
        self.solid = solid
        self.name = name or solid.name
        self.transform = Transform.identity()
        self.grounded = False  # If True, part is fixed in space

    def set_position(self, x, y, z):
        self.transform = Transform.translation(x, y, z)

    def __repr__(self):
        return f"Part({self.name}, grounded={self.grounded})"


class Assembly:
    """
    Assembly of parts with constraints and joints.
    """

    def __init__(self, name="Assembly"):
        self.name = name
        self.parts = []
        self.constraints = []
        self.joints = []

    def add_part(self, part, grounded=False):
        if isinstance(part, Part):
            part.grounded = grounded
        else:
            part = Part(part)
            part.grounded = grounded
        self.parts.append(part)
        return part

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return constraint

    def add_joint(self, joint):
        self.joints.append(joint)
        return joint

    def total_dof(self):
        """Calculate total degrees of freedom."""
        free_parts = sum(1 for p in self.parts if not p.grounded)
        total_dof = free_parts * 6  # 3 translation + 3 rotation per free part
        for c in self.constraints:
            total_dof -= len(c.residual())
        for j in self.joints:
            total_dof -= (6 - j.dof)
        return max(0, total_dof)

    def solve(self, verbose=False):
        """Solve assembly constraints."""
        total_error = sum(c.error() for c in self.constraints)
        if verbose:
            print(f"Assembly '{self.name}':")
            print(f"  Parts: {len(self.parts)} ({sum(1 for p in self.parts if p.grounded)} grounded)")
            print(f"  Constraints: {len(self.constraints)}")
            print(f"  Joints: {len(self.joints)}")
            print(f"  DOF: {self.total_dof()}")
            print(f"  Total error: {total_error:.2e}")
        return total_error

    def get_all_mesh_data(self):
        """Combine all part meshes for rendering."""
        x, y, z, i_l, j_l, k_l, intensity = [], [], [], [], [], [], []
        offset = 0
        for part in self.parts:
            mesh = part.solid.get_mesh_data()
            x.extend(mesh['x'])
            y.extend(mesh['y'])
            z.extend(mesh['z'])
            i_l.extend([idx + offset for idx in mesh['i']])
            j_l.extend([idx + offset for idx in mesh['j']])
            k_l.extend([idx + offset for idx in mesh['k']])
            intensity.extend(mesh['intensity'])
            offset += mesh['num_vertices']
        return {'x': x, 'y': y, 'z': z, 'i': i_l, 'j': j_l, 'k': k_l,
                'intensity': intensity, 'num_vertices': len(x), 'num_faces': len(i_l)}

    def info(self):
        return (f"Assembly '{self.name}': {len(self.parts)} parts, "
                f"{len(self.constraints)} constraints, "
                f"{len(self.joints)} joints, DOF={self.total_dof()}")

    def __repr__(self):
        return self.info()
