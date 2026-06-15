"""
Geometric Constraint Solver
==============================
Solves geometric constraints using numerical optimization.

Constraint types:
    - Coincident (two points same location)
    - Parallel (two lines same direction)
    - Perpendicular (two lines at 90°)
    - Tangent (curve-curve tangency)
    - Equal length (two segments same length)
    - Fixed angle (line at specific angle)
    - Dimensional (distance, angle, radius)
    - Horizontal / Vertical
    - Symmetric
    - Concentric

Solver methods:
    - Newton-Raphson:     x_{n+1} = x_n - f(x_n) / f'(x_n)
    - Gauss-Newton:       Δx = -(J^T J)^{-1} J^T r
    - Levenberg-Marquardt: Δx = -(J^T J + λI)^{-1} J^T r
"""

import numpy as np
from .constants import (EPSILON, CONSTRAINT_TOLERANCE, MAX_SOLVER_ITERATIONS,
                         DAMPING_FACTOR, PI)
from .sketch import Point2D, SketchLine, SketchArc, SketchCircle


# =====================================================================
# CONSTRAINT DEFINITIONS
# =====================================================================

class Constraint:
    """Base class for geometric constraints."""
    _counter = 0

    def __init__(self, name="constraint"):
        Constraint._counter += 1
        self.id = Constraint._counter
        self.name = name
        self.satisfied = False

    def residual(self):
        """Return residual vector (should be ~0 when satisfied)."""
        raise NotImplementedError

    def error(self):
        """Scalar error magnitude."""
        r = self.residual()
        return np.linalg.norm(r)


class CoincidentConstraint(Constraint):
    """Two points must be at the same location."""
    def __init__(self, p1, p2):
        super().__init__("coincident")
        self.p1 = p1
        self.p2 = p2

    def residual(self):
        return np.array([self.p1.x - self.p2.x, self.p1.y - self.p2.y])

    def apply_correction(self, weight=0.5):
        mx = (self.p1.x + self.p2.x) / 2
        my = (self.p1.y + self.p2.y) / 2
        if not self.p1.fixed:
            self.p1.x = mx; self.p1.y = my
        if not self.p2.fixed:
            self.p2.x = mx; self.p2.y = my


class DistanceConstraint(Constraint):
    """Distance between two points = target."""
    def __init__(self, p1, p2, target_distance):
        super().__init__("distance")
        self.p1 = p1
        self.p2 = p2
        self.target = float(target_distance)

    def residual(self):
        d = self.p1.distance_to(self.p2)
        return np.array([d - self.target])


class HorizontalConstraint(Constraint):
    """Line must be horizontal (Δy = 0)."""
    def __init__(self, line):
        super().__init__("horizontal")
        self.line = line

    def residual(self):
        return np.array([self.line.p1.y - self.line.p2.y])


class VerticalConstraint(Constraint):
    """Line must be vertical (Δx = 0)."""
    def __init__(self, line):
        super().__init__("vertical")
        self.line = line

    def residual(self):
        return np.array([self.line.p1.x - self.line.p2.x])


class ParallelConstraint(Constraint):
    """Two lines must be parallel: cross product of directions = 0."""
    def __init__(self, line1, line2):
        super().__init__("parallel")
        self.line1 = line1
        self.line2 = line2

    def residual(self):
        dx1 = self.line1.p2.x - self.line1.p1.x
        dy1 = self.line1.p2.y - self.line1.p1.y
        dx2 = self.line2.p2.x - self.line2.p1.x
        dy2 = self.line2.p2.y - self.line2.p1.y
        return np.array([dx1 * dy2 - dy1 * dx2])


class PerpendicularConstraint(Constraint):
    """Two lines must be perpendicular: dot product of directions = 0."""
    def __init__(self, line1, line2):
        super().__init__("perpendicular")
        self.line1 = line1
        self.line2 = line2

    def residual(self):
        dx1 = self.line1.p2.x - self.line1.p1.x
        dy1 = self.line1.p2.y - self.line1.p1.y
        dx2 = self.line2.p2.x - self.line2.p1.x
        dy2 = self.line2.p2.y - self.line2.p1.y
        return np.array([dx1 * dx2 + dy1 * dy2])


class EqualLengthConstraint(Constraint):
    """Two lines must have equal length."""
    def __init__(self, line1, line2):
        super().__init__("equal_length")
        self.line1 = line1
        self.line2 = line2

    def residual(self):
        return np.array([self.line1.length() - self.line2.length()])


class FixedAngleConstraint(Constraint):
    """Line at a fixed angle (radians) from horizontal."""
    def __init__(self, line, angle_rad):
        super().__init__("fixed_angle")
        self.line = line
        self.target_angle = float(angle_rad)

    def residual(self):
        dx = self.line.p2.x - self.line.p1.x
        dy = self.line.p2.y - self.line.p1.y
        current_angle = np.arctan2(dy, dx)
        diff = current_angle - self.target_angle
        # Wrap to [-π, π]
        while diff > PI: diff -= 2 * PI
        while diff < -PI: diff += 2 * PI
        return np.array([diff])


class TangentConstraint(Constraint):
    """Line tangent to arc/circle at connection point."""
    def __init__(self, line, arc):
        super().__init__("tangent")
        self.line = line
        self.arc = arc

    def residual(self):
        # Line direction
        dx = self.line.p2.x - self.line.p1.x
        dy = self.line.p2.y - self.line.p1.y
        # Radial direction at connection (p1 of line assumed on arc)
        rx = self.line.p1.x - self.arc.center.x
        ry = self.line.p1.y - self.arc.center.y
        # Tangent ⟹ line direction ⊥ radial direction
        return np.array([dx * rx + dy * ry])


class ConcentricConstraint(Constraint):
    """Two circles/arcs share the same center."""
    def __init__(self, entity1, entity2):
        super().__init__("concentric")
        self.e1 = entity1
        self.e2 = entity2

    def residual(self):
        return np.array([self.e1.center.x - self.e2.center.x,
                         self.e1.center.y - self.e2.center.y])


class RadiusConstraint(Constraint):
    """Circle/arc has specific radius."""
    def __init__(self, entity, target_radius):
        super().__init__("radius")
        self.entity = entity
        self.target = float(target_radius)

    def residual(self):
        return np.array([self.entity.radius - self.target])


class SymmetricConstraint(Constraint):
    """Two points symmetric about a line."""
    def __init__(self, p1, p2, sym_line):
        super().__init__("symmetric")
        self.p1 = p1
        self.p2 = p2
        self.sym_line = sym_line

    def residual(self):
        mid = Point2D((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)
        # Midpoint must lie on the symmetry line
        closest, _ = self.sym_line.closest_point(mid)
        # And the connecting line must be perpendicular to sym_line
        dx, dy = self.sym_line.direction()
        px = self.p2.x - self.p1.x
        py = self.p2.y - self.p1.y
        return np.array([
            mid.distance_to(closest),
            dx * px + dy * py
        ])


# =====================================================================
# CONSTRAINT SOLVER
# =====================================================================

class ConstraintSolver:
    """
    Numerical constraint solver using Levenberg-Marquardt.

    Collects all free variables (unfixed point coordinates),
    builds the residual vector from all constraints,
    and iteratively adjusts variables to minimize residuals.

    Algorithm:
        1. Pack free variables into vector x
        2. Compute residual r(x) from all constraints
        3. Compute Jacobian J = ∂r/∂x (numerical)
        4. Solve (J^T J + λI) Δx = -J^T r
        5. Update x ← x + Δx
        6. Repeat until ||r|| < tolerance
    """

    def __init__(self, constraints=None, tolerance=CONSTRAINT_TOLERANCE,
                 max_iter=MAX_SOLVER_ITERATIONS):
        self.constraints = constraints or []
        self.tolerance = tolerance
        self.max_iter = max_iter
        self._points = []
        self._free_indices = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        return constraint

    def _collect_free_points(self):
        """Gather all unique unfixed points from constraints."""
        seen = set()
        self._points = []
        for c in self.constraints:
            for attr_name in ['p1', 'p2']:
                p = getattr(c, attr_name, None)
                if p and isinstance(p, Point2D) and not p.fixed and id(p) not in seen:
                    seen.add(id(p))
                    self._points.append(p)
            # Lines
            line = getattr(c, 'line', None) or getattr(c, 'line1', None)
            if line:
                for attr in ['p1', 'p2']:
                    p = getattr(line, attr, None)
                    if p and isinstance(p, Point2D) and not p.fixed and id(p) not in seen:
                        seen.add(id(p))
                        self._points.append(p)
            line2 = getattr(c, 'line2', None)
            if line2:
                for attr in ['p1', 'p2']:
                    p = getattr(line2, attr, None)
                    if p and isinstance(p, Point2D) and not p.fixed and id(p) not in seen:
                        seen.add(id(p))
                        self._points.append(p)
            # Arc/circle centers
            for attr_name in ['arc', 'entity', 'e1', 'e2']:
                ent = getattr(c, attr_name, None)
                if ent and hasattr(ent, 'center') and id(ent.center) not in seen:
                    if not ent.center.fixed:
                        seen.add(id(ent.center))
                        self._points.append(ent.center)

    def _pack(self):
        """Pack free point coordinates into a vector."""
        return np.array([coord for p in self._points for coord in [p.x, p.y]])

    def _unpack(self, x):
        """Unpack vector back into point coordinates."""
        for i, p in enumerate(self._points):
            p.x = x[2 * i]
            p.y = x[2 * i + 1]

    def _residual_vector(self):
        """Build full residual vector from all constraints."""
        residuals = []
        for c in self.constraints:
            residuals.append(c.residual())
        return np.concatenate(residuals) if residuals else np.array([])

    def _jacobian(self, x, h=1e-7):
        """Numerical Jacobian via central differences."""
        r0 = self._residual_vector()
        n_vars = len(x)
        n_res = len(r0)
        J = np.zeros((n_res, n_vars))

        for j in range(n_vars):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h

            self._unpack(x_plus)
            r_plus = self._residual_vector()

            self._unpack(x_minus)
            r_minus = self._residual_vector()

            J[:, j] = (r_plus - r_minus) / (2.0 * h)

        self._unpack(x)  # Restore
        return J

    def solve(self, verbose=False):
        """
        Solve all constraints using Levenberg-Marquardt.

        Returns dict with convergence info.
        """
        self._collect_free_points()
        if not self._points:
            return {'converged': True, 'iterations': 0, 'error': 0.0, 'message': 'No free points'}

        x = self._pack()
        n_vars = len(x)
        lam = 1.0  # Damping parameter

        for iteration in range(self.max_iter):
            self._unpack(x)
            r = self._residual_vector()
            err = np.linalg.norm(r)

            if verbose:
                print(f"  Solver iter {iteration}: error = {err:.2e}")

            if err < self.tolerance:
                for c in self.constraints:
                    c.satisfied = True
                return {
                    'converged': True, 'iterations': iteration,
                    'error': err, 'message': 'Converged'
                }

            J = self._jacobian(x)
            JtJ = J.T @ J
            Jtr = J.T @ r

            # Levenberg-Marquardt: (J^T J + λI) Δx = -J^T r
            A = JtJ + lam * np.eye(n_vars)
            try:
                dx = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, -Jtr, rcond=None)[0]

            # Line search with damping
            x_new = x + DAMPING_FACTOR * dx
            self._unpack(x_new)
            r_new = self._residual_vector()
            err_new = np.linalg.norm(r_new)

            if err_new < err:
                x = x_new
                lam = max(lam * 0.5, 1e-10)
            else:
                lam = min(lam * 2.0, 1e6)

        self._unpack(x)
        final_err = np.linalg.norm(self._residual_vector())
        return {
            'converged': False, 'iterations': self.max_iter,
            'error': final_err, 'message': f'Did not converge (err={final_err:.2e})'
        }

    def status(self):
        """Report constraint satisfaction status."""
        lines = [f"Constraint Solver: {len(self.constraints)} constraints, {len(self._points)} free points"]
        for c in self.constraints:
            err = c.error()
            status = "✅" if err < self.tolerance else "❌"
            lines.append(f"  {status} {c.name} (id={c.id}): error={err:.2e}")
        return "\n".join(lines)
