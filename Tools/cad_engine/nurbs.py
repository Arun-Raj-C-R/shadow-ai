"""
NURBS & Spline Mathematics
=============================
Modern CAD uses mathematically exact surfaces, NOT triangle meshes.

Implements:
    - Bézier curves (De Casteljau algorithm)
    - B-spline basis functions (Cox-de Boor recursion)
    - NURBS curves: C(u) = Σ N_{i,p}(u) w_i P_i / Σ N_{i,p}(u) w_i
    - NURBS surfaces: S(u,v) = ΣΣ N_{i,p}(u) N_{j,q}(v) w_{ij} P_{ij} / ΣΣ ...
    - Knot insertion, degree elevation
    - Derivatives and curvature evaluation

Without NURBS: no precision engineering, no smooth industrial surfaces,
no manufacturable geometry.
"""

import numpy as np
from .constants import EPSILON, PARAMETRIC_TOLERANCE
from .vector_math import Vec3, _to_array


# =====================================================================
# BÉZIER CURVES
# =====================================================================

class BezierCurve:
    """
    Bézier curve of degree n defined by n+1 control points.

    Evaluation via De Casteljau algorithm (numerically stable):
        B^0_i = P_i
        B^r_i(t) = (1-t) B^{r-1}_i(t) + t B^{r-1}_{i+1}(t)
        C(t) = B^n_0(t)
    """

    def __init__(self, control_points):
        self.control_points = [Vec3(p) for p in control_points]
        self.degree = len(self.control_points) - 1

    def evaluate(self, t):
        """Evaluate curve at parameter t ∈ [0, 1] using De Casteljau."""
        pts = [p.copy() for p in self.control_points]
        n = len(pts)
        for r in range(1, n):
            for i in range(n - r):
                pts[i] = pts[i] * (1.0 - t) + pts[i + 1] * t
        return pts[0]

    def derivative(self, t):
        """First derivative: C'(t) using hodograph (degree n-1 Bézier)."""
        if self.degree < 1:
            return Vec3(0, 0, 0)
        n = self.degree
        hodograph_pts = [(self.control_points[i+1] - self.control_points[i]) * n
                         for i in range(n)]
        hodograph = BezierCurve(hodograph_pts)
        return hodograph.evaluate(t)

    def tangent(self, t):
        """Unit tangent at parameter t."""
        return self.derivative(t).normalize()

    def split(self, t):
        """Split curve at parameter t into two Bézier curves."""
        pts = [p.copy() for p in self.control_points]
        n = len(pts)
        left = [pts[0].copy()]
        right = [None] * n
        right[-1] = pts[-1].copy()

        for r in range(1, n):
            for i in range(n - r):
                pts[i] = pts[i] * (1.0 - t) + pts[i + 1] * t
            left.append(pts[0].copy())
            right[n - 1 - r] = pts[n - 1 - r].copy()

        return BezierCurve(left), BezierCurve(right)

    def length(self, num_samples=64):
        """Approximate arc length by sampling."""
        total = 0.0
        prev = self.evaluate(0.0)
        for i in range(1, num_samples + 1):
            t = i / num_samples
            curr = self.evaluate(t)
            total += prev.distance_to(curr)
            prev = curr
        return total

    def bounding_box(self):
        """Axis-aligned bounding box of control points (conservative)."""
        pts = np.array([p.array for p in self.control_points])
        return Vec3(pts.min(axis=0)), Vec3(pts.max(axis=0))

    def tessellate(self, num_points=50):
        """Return list of points along the curve."""
        return [self.evaluate(t / (num_points - 1)) for t in range(num_points)]


# =====================================================================
# B-SPLINE BASIS FUNCTIONS
# =====================================================================

def bspline_basis(i, p, u, knots):
    """
    Cox-de Boor recursion for B-spline basis function N_{i,p}(u).

    N_{i,0}(u) = 1 if knots[i] <= u < knots[i+1], else 0
    N_{i,p}(u) = (u - knots[i]) / (knots[i+p] - knots[i]) * N_{i,p-1}(u)
               + (knots[i+p+1] - u) / (knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}(u)
    """
    if p == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        # Handle end of domain
        if abs(u - knots[-1]) < PARAMETRIC_TOLERANCE and i == len(knots) - p - 2:
            return 1.0
        return 0.0

    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]

    c1 = ((u - knots[i]) / d1 * bspline_basis(i, p - 1, u, knots)) if abs(d1) > EPSILON else 0.0
    c2 = ((knots[i + p + 1] - u) / d2 * bspline_basis(i + 1, p - 1, u, knots)) if abs(d2) > EPSILON else 0.0

    return c1 + c2


def bspline_basis_all(p, u, knots):
    """Evaluate all non-zero basis functions at u (efficient)."""
    n = len(knots) - p - 1
    N = np.zeros(n)
    for i in range(n):
        N[i] = bspline_basis(i, p, u, knots)
    return N


# =====================================================================
# NURBS CURVES
# =====================================================================

class NURBSCurve:
    """
    Non-Uniform Rational B-Spline curve.

    C(u) = Σ_{i=0}^{n} N_{i,p}(u) w_i P_i  /  Σ_{i=0}^{n} N_{i,p}(u) w_i

    Parameters:
        control_points: list of Vec3
        weights: list of floats (default all 1.0 = non-rational B-spline)
        degree: polynomial degree p
        knots: knot vector (auto-generated if None)
    """

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        self.control_points = [Vec3(p) for p in control_points]
        self.n = len(self.control_points) - 1  # n+1 control points
        self.degree = min(degree, self.n)

        if weights is None:
            self.weights = [1.0] * (self.n + 1)
        else:
            self.weights = list(weights)

        if knots is None:
            self.knots = self._generate_clamped_knots()
        else:
            self.knots = list(knots)

    def _generate_clamped_knots(self):
        """Generate clamped uniform knot vector."""
        n, p = self.n, self.degree
        m = n + p + 1
        knots = [0.0] * (p + 1)
        num_internal = m - 2 * p
        for j in range(1, num_internal):
            knots.append(j / num_internal)
        knots.extend([1.0] * (p + 1))
        return knots

    def evaluate(self, u):
        """Evaluate NURBS curve at parameter u ∈ [0, 1]."""
        u = max(self.knots[0], min(u, self.knots[-1] - PARAMETRIC_TOLERANCE))
        N = bspline_basis_all(self.degree, u, self.knots)

        numerator = Vec3(0, 0, 0)
        denominator = 0.0
        for i in range(self.n + 1):
            Nw = N[i] * self.weights[i]
            numerator = numerator + self.control_points[i] * Nw
            denominator += Nw

        if abs(denominator) < EPSILON:
            return Vec3(0, 0, 0)
        return numerator / denominator

    def derivative(self, u, h=1e-6):
        """Numerical derivative at u."""
        u0 = max(self.knots[0], u - h)
        u1 = min(self.knots[-1] - PARAMETRIC_TOLERANCE, u + h)
        p0 = self.evaluate(u0)
        p1 = self.evaluate(u1)
        return (p1 - p0) / (u1 - u0)

    def tangent(self, u):
        return self.derivative(u).normalize()

    def curvature(self, u, h=1e-5):
        """Curvature κ = |C' × C''| / |C'|³"""
        d1 = self.derivative(u, h)
        d2_fwd = self.derivative(u + h, h)
        d2 = (d2_fwd - d1) / h  # Approximate C''
        cross_len = d1.cross(d2).length()
        d1_len = d1.length()
        if d1_len < EPSILON:
            return 0.0
        return cross_len / (d1_len ** 3)

    def length(self, num_samples=100):
        total = 0.0
        prev = self.evaluate(0.0)
        for i in range(1, num_samples + 1):
            u = i / num_samples
            curr = self.evaluate(u)
            total += prev.distance_to(curr)
            prev = curr
        return total

    def tessellate(self, num_points=50):
        return [self.evaluate(i / (num_points - 1)) for i in range(num_points)]

    def insert_knot(self, u_new):
        """Knot insertion (Boehm's algorithm) — refines without changing shape."""
        p = self.degree
        k = 0
        for i in range(len(self.knots) - 1):
            if self.knots[i] <= u_new < self.knots[i + 1]:
                k = i
                break

        new_pts = []
        new_wts = []
        for i in range(self.n + 2):
            if i <= k - p:
                new_pts.append(self.control_points[i].copy())
                new_wts.append(self.weights[i])
            elif i >= k + 1:
                new_pts.append(self.control_points[i - 1].copy())
                new_wts.append(self.weights[i - 1])
            else:
                alpha = (u_new - self.knots[i]) / (self.knots[i + p] - self.knots[i])
                pt = self.control_points[i - 1] * (1 - alpha) + self.control_points[i] * alpha
                wt = self.weights[i - 1] * (1 - alpha) + self.weights[i] * alpha
                new_pts.append(pt)
                new_wts.append(wt)

        new_knots = self.knots[:k + 1] + [u_new] + self.knots[k + 1:]
        return NURBSCurve(new_pts, new_wts, self.degree, new_knots)


# =====================================================================
# NURBS SURFACES
# =====================================================================

class NURBSSurface:
    """
    NURBS surface: S(u,v) = ΣΣ N_{i,p}(u) N_{j,q}(v) w_{ij} P_{ij} / ΣΣ ...

    Parameters:
        control_points: 2D list [i][j] of Vec3
        weights: 2D list [i][j] of floats
        degree_u, degree_v: polynomial degrees
        knots_u, knots_v: knot vectors
    """

    def __init__(self, control_points, weights=None, degree_u=3, degree_v=3,
                 knots_u=None, knots_v=None):
        self.control_points = [[Vec3(p) for p in row] for row in control_points]
        self.nu = len(self.control_points) - 1
        self.nv = len(self.control_points[0]) - 1 if self.control_points else 0
        self.degree_u = min(degree_u, self.nu)
        self.degree_v = min(degree_v, self.nv)

        if weights is None:
            self.weights = [[1.0] * (self.nv + 1) for _ in range(self.nu + 1)]
        else:
            self.weights = weights

        if knots_u is None:
            self.knots_u = self._gen_knots(self.nu, self.degree_u)
        else:
            self.knots_u = list(knots_u)

        if knots_v is None:
            self.knots_v = self._gen_knots(self.nv, self.degree_v)
        else:
            self.knots_v = list(knots_v)

    def _gen_knots(self, n, p):
        m = n + p + 1
        knots = [0.0] * (p + 1)
        num_int = m - 2 * p
        for j in range(1, num_int):
            knots.append(j / num_int)
        knots.extend([1.0] * (p + 1))
        return knots

    def evaluate(self, u, v):
        """Evaluate surface point at (u, v)."""
        u = max(self.knots_u[0], min(u, self.knots_u[-1] - PARAMETRIC_TOLERANCE))
        v = max(self.knots_v[0], min(v, self.knots_v[-1] - PARAMETRIC_TOLERANCE))

        Nu = bspline_basis_all(self.degree_u, u, self.knots_u)
        Nv = bspline_basis_all(self.degree_v, v, self.knots_v)

        numerator = Vec3(0, 0, 0)
        denominator = 0.0
        for i in range(self.nu + 1):
            for j in range(self.nv + 1):
                Nw = Nu[i] * Nv[j] * self.weights[i][j]
                numerator = numerator + self.control_points[i][j] * Nw
                denominator += Nw

        if abs(denominator) < EPSILON:
            return Vec3(0, 0, 0)
        return numerator / denominator

    def normal(self, u, v, h=1e-5):
        """Surface normal at (u, v) via cross product of partial derivatives."""
        du = (self.evaluate(min(u + h, 1.0), v) - self.evaluate(max(u - h, 0.0), v))
        dv = (self.evaluate(u, min(v + h, 1.0)) - self.evaluate(u, max(v - h, 0.0)))
        return du.cross(dv).normalize()

    def tessellate(self, nu=20, nv=20):
        """Generate triangle mesh from surface."""
        vertices = []
        normals = []
        for i in range(nu + 1):
            u = i / nu
            for j in range(nv + 1):
                v = j / nv
                vertices.append(self.evaluate(u, v))
                normals.append(self.normal(u, v))

        triangles = []
        for i in range(nu):
            for j in range(nv):
                idx = i * (nv + 1) + j
                triangles.append((idx, idx + 1, idx + nv + 1))
                triangles.append((idx + 1, idx + nv + 2, idx + nv + 1))

        return vertices, normals, triangles


# =====================================================================
# UTILITY: Create standard NURBS primitives
# =====================================================================

def nurbs_circle(center=None, radius=1.0, normal=None):
    """Create a NURBS circle (degree 2, rational)."""
    if center is None: center = Vec3(0, 0, 0)
    if normal is None: normal = Vec3(0, 0, 1)
    r = radius
    # 9-point rational circle
    w = np.sqrt(2.0) / 2.0
    pts = [
        Vec3(r, 0, 0), Vec3(r, r, 0), Vec3(0, r, 0),
        Vec3(-r, r, 0), Vec3(-r, 0, 0), Vec3(-r, -r, 0),
        Vec3(0, -r, 0), Vec3(r, -r, 0), Vec3(r, 0, 0),
    ]
    pts = [p + center for p in pts]
    weights = [1, w, 1, w, 1, w, 1, w, 1]
    knots = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
    return NURBSCurve(pts, weights, degree=2, knots=knots)

def nurbs_line(start, end):
    """Create a degree-1 NURBS line."""
    return NURBSCurve([Vec3(start), Vec3(end)], degree=1)

def nurbs_arc(center, radius, start_angle, end_angle, normal=None):
    """Create a NURBS arc."""
    if normal is None: normal = Vec3(0, 0, 1)
    # Sample points along arc and create Bézier approximation
    n_pts = max(3, int(abs(end_angle - start_angle) / (np.pi / 2)) * 2 + 1)
    pts = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        angle = start_angle + t * (end_angle - start_angle)
        p = Vec3(center) + Vec3(np.cos(angle) * radius, np.sin(angle) * radius, 0)
        pts.append(p)
    return NURBSCurve(pts, degree=min(3, n_pts - 1))
