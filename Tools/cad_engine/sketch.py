"""
2D Parametric Sketch Engine
==============================
THIS is where actual CAD begins.

Implements 2D sketch primitives on a work plane:
    - Line segments
    - Arcs (center + start/end angles)
    - Circles
    - Splines (B-spline through points)
    - Rectangles, polygons
    - Offset curves
    - Trim / extend operations

Sketches live in a 2D coordinate system on a reference plane
and are the input to 3D operations (extrude, revolve, loft).
"""

import numpy as np
from .constants import EPSILON, LINEAR_TOLERANCE, PI, TWO_PI, DEG_TO_RAD
from .vector_math import Vec3, _to_array


class Point2D:
    """2D point on a sketch plane."""
    __slots__ = ('x', 'y', 'fixed')

    def __init__(self, x=0.0, y=0.0, fixed=False):
        self.x = float(x)
        self.y = float(y)
        self.fixed = fixed

    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def to_vec3(self, z=0.0):
        return Vec3(self.x, self.y, z)

    def to_array(self):
        return np.array([self.x, self.y], dtype=np.float64)

    def copy(self):
        return Point2D(self.x, self.y, self.fixed)

    def __add__(self, o):
        return Point2D(self.x + o.x, self.y + o.y)
    def __sub__(self, o):
        return Point2D(self.x - o.x, self.y - o.y)
    def __mul__(self, s):
        return Point2D(self.x * float(s), self.y * float(s))
    def __repr__(self):
        return f"Pt({self.x:.4f}, {self.y:.4f})"


class SketchEntity:
    """Base class for all sketch entities."""
    _counter = 0

    def __init__(self):
        SketchEntity._counter += 1
        self.id = SketchEntity._counter
        self.construction = False  # Construction geometry (not used in profiles)

    def start_point(self): raise NotImplementedError
    def end_point(self): raise NotImplementedError
    def midpoint(self): raise NotImplementedError
    def length(self): raise NotImplementedError
    def sample(self, num_points=20): raise NotImplementedError
    def bounding_box(self): raise NotImplementedError


class SketchLine(SketchEntity):
    """Line segment between two points."""

    def __init__(self, p1, p2):
        super().__init__()
        self.p1 = Point2D(*p1) if not isinstance(p1, Point2D) else p1
        self.p2 = Point2D(*p2) if not isinstance(p2, Point2D) else p2

    def start_point(self): return self.p1
    def end_point(self): return self.p2

    def midpoint(self):
        return Point2D((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)

    def length(self):
        return self.p1.distance_to(self.p2)

    def direction(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        L = np.sqrt(dx*dx + dy*dy)
        return (dx/L, dy/L) if L > EPSILON else (1.0, 0.0)

    def normal(self):
        dx, dy = self.direction()
        return (-dy, dx)

    def point_at(self, t):
        return Point2D(self.p1.x + t * (self.p2.x - self.p1.x),
                       self.p1.y + t * (self.p2.y - self.p1.y))

    def closest_point(self, pt):
        a = self.p1.to_array()
        b = self.p2.to_array()
        p = np.array([pt.x, pt.y])
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < EPSILON:
            return self.p1.copy(), 0.0
        t = max(0.0, min(1.0, np.dot(p - a, ab) / ab_sq))
        return self.point_at(t), t

    def sample(self, num_points=20):
        return [self.point_at(i / (num_points - 1)) for i in range(num_points)]

    def bounding_box(self):
        return (Point2D(min(self.p1.x, self.p2.x), min(self.p1.y, self.p2.y)),
                Point2D(max(self.p1.x, self.p2.x), max(self.p1.y, self.p2.y)))

    def __repr__(self):
        return f"Line({self.p1} → {self.p2})"


class SketchArc(SketchEntity):
    """Circular arc defined by center, radius, start/end angles (radians)."""

    def __init__(self, center, radius, start_angle, end_angle):
        super().__init__()
        self.center = Point2D(*center) if not isinstance(center, Point2D) else center
        self.radius = float(radius)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)

    def start_point(self):
        return Point2D(self.center.x + self.radius * np.cos(self.start_angle),
                       self.center.y + self.radius * np.sin(self.start_angle))

    def end_point(self):
        return Point2D(self.center.x + self.radius * np.cos(self.end_angle),
                       self.center.y + self.radius * np.sin(self.end_angle))

    def midpoint(self):
        mid_angle = (self.start_angle + self.end_angle) / 2
        return Point2D(self.center.x + self.radius * np.cos(mid_angle),
                       self.center.y + self.radius * np.sin(mid_angle))

    def sweep_angle(self):
        da = self.end_angle - self.start_angle
        while da < 0: da += TWO_PI
        while da > TWO_PI: da -= TWO_PI
        return da

    def length(self):
        return abs(self.sweep_angle()) * self.radius

    def point_at(self, t):
        angle = self.start_angle + t * self.sweep_angle()
        return Point2D(self.center.x + self.radius * np.cos(angle),
                       self.center.y + self.radius * np.sin(angle))

    def tangent_at(self, t):
        angle = self.start_angle + t * self.sweep_angle()
        return (-np.sin(angle), np.cos(angle))

    def sample(self, num_points=20):
        return [self.point_at(i / (num_points - 1)) for i in range(num_points)]

    def bounding_box(self):
        pts = self.sample(50)
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return Point2D(min(xs), min(ys)), Point2D(max(xs), max(ys))

    def __repr__(self):
        return f"Arc(c={self.center}, r={self.radius:.4f}, {np.degrees(self.start_angle):.1f}°→{np.degrees(self.end_angle):.1f}°)"


class SketchCircle(SketchEntity):
    """Full circle."""

    def __init__(self, center, radius):
        super().__init__()
        self.center = Point2D(*center) if not isinstance(center, Point2D) else center
        self.radius = float(radius)

    def start_point(self):
        return Point2D(self.center.x + self.radius, self.center.y)

    def end_point(self):
        return self.start_point()

    def midpoint(self):
        return Point2D(self.center.x, self.center.y + self.radius)

    def length(self):
        return TWO_PI * self.radius

    def area(self):
        return PI * self.radius ** 2

    def point_at(self, t):
        angle = TWO_PI * t
        return Point2D(self.center.x + self.radius * np.cos(angle),
                       self.center.y + self.radius * np.sin(angle))

    def sample(self, num_points=50):
        return [self.point_at(i / num_points) for i in range(num_points)]

    def bounding_box(self):
        return (Point2D(self.center.x - self.radius, self.center.y - self.radius),
                Point2D(self.center.x + self.radius, self.center.y + self.radius))

    def __repr__(self):
        return f"Circle(c={self.center}, r={self.radius:.4f})"


class SketchSpline(SketchEntity):
    """B-spline through fit points (interpolating spline)."""

    def __init__(self, fit_points, degree=3):
        super().__init__()
        self.fit_points = [Point2D(*p) if not isinstance(p, Point2D) else p for p in fit_points]
        self.degree = min(degree, len(self.fit_points) - 1)
        self._parameterize()

    def _parameterize(self):
        """Chord-length parameterization."""
        n = len(self.fit_points)
        self._params = [0.0]
        for i in range(1, n):
            d = self.fit_points[i - 1].distance_to(self.fit_points[i])
            self._params.append(self._params[-1] + d)
        total = self._params[-1]
        if total > EPSILON:
            self._params = [t / total for t in self._params]
        else:
            self._params = [i / (n - 1) for i in range(n)]

    def start_point(self): return self.fit_points[0]
    def end_point(self): return self.fit_points[-1]
    def midpoint(self): return self.point_at(0.5)

    def point_at(self, t):
        """Evaluate using Catmull-Rom-style interpolation."""
        n = len(self.fit_points)
        if n < 2: return self.fit_points[0].copy()

        # Find segment
        for i in range(n - 1):
            if t <= self._params[i + 1] or i == n - 2:
                local_t = (t - self._params[i]) / max(self._params[i+1] - self._params[i], EPSILON)
                local_t = max(0.0, min(1.0, local_t))
                p0 = self.fit_points[max(0, i-1)]
                p1 = self.fit_points[i]
                p2 = self.fit_points[min(n-1, i+1)]
                p3 = self.fit_points[min(n-1, i+2)]
                # Catmull-Rom
                tt = local_t
                tt2 = tt * tt
                tt3 = tt2 * tt
                x = 0.5 * ((2*p1.x) + (-p0.x + p2.x)*tt +
                    (2*p0.x - 5*p1.x + 4*p2.x - p3.x)*tt2 +
                    (-p0.x + 3*p1.x - 3*p2.x + p3.x)*tt3)
                y = 0.5 * ((2*p1.y) + (-p0.y + p2.y)*tt +
                    (2*p0.y - 5*p1.y + 4*p2.y - p3.y)*tt2 +
                    (-p0.y + 3*p1.y - 3*p2.y + p3.y)*tt3)
                return Point2D(x, y)
        return self.fit_points[-1].copy()

    def length(self):
        pts = self.sample(100)
        return sum(pts[i].distance_to(pts[i+1]) for i in range(len(pts)-1))

    def sample(self, num_points=50):
        return [self.point_at(i / (num_points - 1)) for i in range(num_points)]

    def bounding_box(self):
        pts = self.sample(100)
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return Point2D(min(xs), min(ys)), Point2D(max(xs), max(ys))

    def __repr__(self):
        return f"Spline({len(self.fit_points)} pts, deg={self.degree})"


# =====================================================================
# SKETCH CONTAINER
# =====================================================================

class Sketch:
    """
    Container for 2D sketch entities on a reference plane.
    Provides methods to add geometry, detect profiles, and convert to 3D.
    """

    def __init__(self, plane_origin=None, plane_normal=None, name="Sketch"):
        self.name = name
        self.plane_origin = Vec3(plane_origin) if plane_origin else Vec3(0, 0, 0)
        self.plane_normal = Vec3(plane_normal).normalize() if plane_normal else Vec3(0, 0, 1)
        self.entities = []
        self.constraints = []

    def add_line(self, p1, p2):
        line = SketchLine(p1, p2)
        self.entities.append(line)
        return line

    def add_arc(self, center, radius, start_angle, end_angle):
        arc = SketchArc(center, radius, start_angle, end_angle)
        self.entities.append(arc)
        return arc

    def add_circle(self, center, radius):
        circle = SketchCircle(center, radius)
        self.entities.append(circle)
        return circle

    def add_spline(self, points, degree=3):
        spline = SketchSpline(points, degree)
        self.entities.append(spline)
        return spline

    def add_rectangle(self, x, y, width, height):
        """Add a rectangle as 4 lines."""
        p1, p2, p3, p4 = ((x,y), (x+width,y), (x+width,y+height), (x,y+height))
        lines = [self.add_line(p1, p2), self.add_line(p2, p3),
                 self.add_line(p3, p4), self.add_line(p4, p1)]
        return lines

    def add_polygon(self, center, radius, num_sides):
        """Add a regular polygon."""
        pts = []
        for i in range(num_sides):
            angle = TWO_PI * i / num_sides
            pts.append((center[0] + radius * np.cos(angle),
                        center[1] + radius * np.sin(angle)))
        lines = []
        for i in range(num_sides):
            lines.append(self.add_line(pts[i], pts[(i+1) % num_sides]))
        return lines

    def get_profile_points(self, entity_filter=None):
        """Get all points from non-construction entities for profile creation."""
        all_pts = []
        for e in self.entities:
            if e.construction:
                continue
            if entity_filter and not entity_filter(e):
                continue
            pts = e.sample(20)
            for p in pts:
                all_pts.append(p.to_vec3())
        return all_pts

    def get_closed_profiles(self):
        """
        Detect closed wire profiles from connected entities.
        Returns list of lists of Point2D forming closed loops.
        """
        profiles = []
        # Simple: if only circles, each circle is a profile
        circles = [e for e in self.entities if isinstance(e, SketchCircle) and not e.construction]
        for c in circles:
            profiles.append(c.sample(50))

        # For connected lines/arcs: try to find closed chains
        lines = [e for e in self.entities if isinstance(e, (SketchLine, SketchArc, SketchSpline)) and not e.construction]
        if lines:
            # Simple chain detection: assume entities are in order
            profile_pts = []
            for e in lines:
                pts = e.sample(20)
                profile_pts.extend(pts)
            if profile_pts:
                profiles.append(profile_pts)

        return profiles

    def to_3d_points(self, points_2d):
        """Convert 2D sketch points to 3D using the sketch plane."""
        # Build local coordinate system
        n = self.plane_normal
        if abs(n.dot(Vec3(0, 0, 1))) < 0.9:
            u = Vec3(0, 0, 1).cross(n).normalize()
        else:
            u = Vec3(1, 0, 0).cross(n).normalize()
        v = n.cross(u).normalize()

        result = []
        for p in points_2d:
            if isinstance(p, Point2D):
                pt3d = self.plane_origin + u * p.x + v * p.y
            else:
                pt3d = self.plane_origin + u * p.x + v * p.y
            result.append(pt3d)
        return result

    def bounding_box(self):
        all_pts = []
        for e in self.entities:
            bb_min, bb_max = e.bounding_box()
            all_pts.extend([bb_min, bb_max])
        if not all_pts:
            return Point2D(0, 0), Point2D(0, 0)
        xs = [p.x for p in all_pts]
        ys = [p.y for p in all_pts]
        return Point2D(min(xs), min(ys)), Point2D(max(xs), max(ys))

    def info(self):
        counts = {}
        for e in self.entities:
            name = type(e).__name__
            counts[name] = counts.get(name, 0) + 1
        return f"Sketch '{self.name}': {counts}"

    def __repr__(self):
        return self.info()
